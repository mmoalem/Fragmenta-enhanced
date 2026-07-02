"""Monkey-patch hook manager for per-step self-attention KV injection in SA3.

Adapted from the ComfyUI ace_step_reference HookManager pattern
for SA3's `Attention` class (transformer.py).

Two-pass per diffusion step:
  1. Capture pass — run noised reference latent through model (null cond),
     collect K/V after RoPE from each self-attention layer.
  2. Injection pass — run generation latent through model, blend captured
     K/V with computed K/V per-layer before attention.
"""

import torch
import torch.nn.functional as F
from einops import rearrange
from typing import Optional

from ..transformer import Attention, apply_rotary_pos_emb


def _match_length(ref_k, gen_t):
    """Trim or repeat-pad ref_k along sequence dim to match gen_t."""
    ref_t = ref_k.shape[2]
    if ref_t == gen_t:
        return ref_k
    if ref_t > gen_t:
        return ref_k[:, :, :gen_t, :]
    repeats = (gen_t + ref_t - 1) // ref_t
    return ref_k.repeat(1, 1, repeats, 1)[:, :, :gen_t, :]


def _make_capture_forward(attn_mod, cache, layer_idx):
    """Build a patched forward that captures K/V after RoPE into cache[layer_idx]."""
    orig_forward = attn_mod.forward

    def capture_forward(
        self,
        x,
        context=None,
        rotary_pos_emb=None,
        rotary_pos_emb_k=None,
        causal=None,
        flex_attention_block_mask=None,
        flex_attention_score_mod=None,
        flash_attn_sliding_window=None,
        padding_mask=None,
        varlen_metadata=None,
    ):
        h, kv_h, has_context = self.num_heads, self.kv_heads, context is not None
        kv_input = context if has_context else x

        # --- replicate Attention forward up to the QKV projections ---
        if hasattr(self, 'to_q'):
            q = self.to_q(x)
            q = rearrange(q, 'b n (h d) -> b h n d', h=h)
            k, v = self.to_kv(kv_input).chunk(2, dim=-1)
            k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=kv_h), (k, v))
        else:
            q, k, v = self.to_qkv(x).chunk(3, dim=-1)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        # QK norm
        if self.qk_norm == "l2":
            q = F.normalize(q, dim=-1, eps=self.qk_norm_eps)
            k = F.normalize(k, dim=-1, eps=self.qk_norm_eps)
        elif self.qk_norm != "none":
            q, k = self.apply_qk_layernorm(q, k)

        # RoPE
        if rotary_pos_emb is not None:
            freqs, _ = rotary_pos_emb
            q_dtype, k_dtype = q.dtype, k.dtype
            q, k = q.to(torch.float32), k.to(torch.float32)
            freqs = freqs.to(torch.float32)
            q_freqs = freqs
            if rotary_pos_emb_k is not None:
                k_freqs, _ = rotary_pos_emb_k
                k_freqs = k_freqs.to(torch.float32)
            else:
                k_freqs = q_freqs
                if q.shape[-2] >= k.shape[-2]:
                    ratio = q.shape[-2] / k.shape[-2]
                    q_freqs, k_freqs = freqs, ratio * freqs
                else:
                    ratio = k.shape[-2] / q.shape[-2]
                    q_freqs, k_freqs = ratio * freqs, freqs
            q = apply_rotary_pos_emb(q, q_freqs)
            k = apply_rotary_pos_emb(k, k_freqs)
            q, k = q.to(v.dtype), k.to(v.dtype)

        # --- capture K/V into cache ---
        # Only capture self-attention (no context) K/V
        if not has_context:
            cache[layer_idx] = {
                "k": k.detach().cpu(),
                "v": v.detach().cpu(),
            }

        # --- complete the original forward ---
        return orig_forward(
            x,
            context=context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_emb_k=rotary_pos_emb_k,
            causal=causal,
            flex_attention_block_mask=flex_attention_block_mask,
            flex_attention_score_mod=flex_attention_score_mod,
            flash_attn_sliding_window=flash_attn_sliding_window,
            padding_mask=padding_mask,
            varlen_metadata=varlen_metadata,
        )

    return capture_forward


def _make_inject_forward(attn_mod, cache, layer_idx, strength, mode="inject", time_taper="none"):
    """Build a patched forward that injects captured K/V into self-attention.

    Args:
        strength: Per-layer strength multiplier.
        mode: "inject" (concat), "replace" (hard overwrite), or "threshold".
        time_taper: Temporal fade pattern across the audio sequence.
    """
    orig_forward = attn_mod.forward

    def inject_forward(
        self,
        x,
        context=None,
        rotary_pos_emb=None,
        rotary_pos_emb_k=None,
        causal=None,
        flex_attention_block_mask=None,
        flex_attention_score_mod=None,
        flash_attn_sliding_window=None,
        padding_mask=None,
        varlen_metadata=None,
    ):
        ref_data = cache.get(layer_idx)
        if ref_data is None or strength <= 0.0 or context is not None:
            # Skip injection for cross-attention layers or if no cache
            return orig_forward(
                x, context=context, rotary_pos_emb=rotary_pos_emb,
                rotary_pos_emb_k=rotary_pos_emb_k, causal=causal,
                flex_attention_block_mask=flex_attention_block_mask,
                flex_attention_score_mod=flex_attention_score_mod,
                flash_attn_sliding_window=flash_attn_sliding_window,
                padding_mask=padding_mask, varlen_metadata=varlen_metadata,
            )

        actual_mode = "replace" if (mode == "threshold" and strength > 1.0) else mode

        h, kv_h, has_context = self.num_heads, self.kv_heads, False
        kv_input = x

        # --- replicate Attention forward up to QKV projections ---
        if hasattr(self, 'to_q'):
            q = self.to_q(x)
            q = rearrange(q, 'b n (h d) -> b h n d', h=h)
            k, v = self.to_kv(kv_input).chunk(2, dim=-1)
            k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=kv_h), (k, v))
        else:
            q, k, v = self.to_qkv(x).chunk(3, dim=-1)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        # QK norm
        if self.qk_norm == "l2":
            q = F.normalize(q, dim=-1, eps=self.qk_norm_eps)
            k = F.normalize(k, dim=-1, eps=self.qk_norm_eps)
        elif self.qk_norm != "none":
            q, k = self.apply_qk_layernorm(q, k)

        # RoPE
        if rotary_pos_emb is not None:
            freqs, _ = rotary_pos_emb
            q_dtype, k_dtype = q.dtype, k.dtype
            q, k = q.to(torch.float32), k.to(torch.float32)
            freqs = freqs.to(torch.float32)
            q_freqs = freqs
            if rotary_pos_emb_k is not None:
                k_freqs, _ = rotary_pos_emb_k
                k_freqs = k_freqs.to(torch.float32)
            else:
                k_freqs = q_freqs
                if q.shape[-2] >= k.shape[-2]:
                    q_freqs, k_freqs = freqs, k.shape[-2] / q.shape[-2] * freqs
                else:
                    q_freqs, k_freqs = q.shape[-2] / k.shape[-2] * freqs, freqs
            q = apply_rotary_pos_emb(q, q_freqs)
            k = apply_rotary_pos_emb(k, k_freqs)
            q, k = q.to(v.dtype), k.to(v.dtype)

        # --- inject reference K/V ---
        gen_t = k.shape[2]
        ref_k = ref_data["k"].to(device=k.device, dtype=k.dtype)
        ref_v = ref_data["v"].to(device=v.device, dtype=v.dtype)

        # Match reference length to generation
        ref_k = _match_length(ref_k, gen_t)
        ref_v = _match_length(ref_v, gen_t)

        # Expand B=1 reference to match generation batch
        bsz = k.shape[0]
        if ref_k.shape[0] == 1 and bsz > 1:
            ref_k = ref_k.expand(bsz, -1, -1, -1)
            ref_v = ref_v.expand(bsz, -1, -1, -1)

        # Time taper across audio sequence
        if time_taper != "none":
            time_mult = _compute_time_multiplier(ref_k.shape[2], time_taper, ref_k.device, ref_k.dtype)
            ref_k = ref_k * time_mult
            ref_v = ref_v * time_mult

        if actual_mode == "replace":
            k = ref_k * strength
            v = ref_v * strength
        else:  # inject (concat)
            ref_k = ref_k * strength
            ref_v = ref_v * strength
            k = torch.cat([k, ref_k], dim=2)
            v = torch.cat([v, ref_v], dim=2)

        # --- complete attention with injected K/V ---
        # Apply GQA repeat interleave if needed
        if self.num_heads != self.kv_heads:
            heads_per_kv = self.num_heads // self.kv_heads
            k, v = map(lambda t: t.repeat_interleave(heads_per_kv, dim=1), (k, v))

        # Run attention with injected K/V
        out = self.apply_attn(
            q, k, v, causal=causal,
            flex_attention_block_mask=flex_attention_block_mask,
            flex_attention_score_mod=flex_attention_score_mod,
            flash_attn_sliding_window=flash_attn_sliding_window,
            padding_mask=padding_mask,
            varlen_metadata=varlen_metadata,
        )
        out = rearrange(out, ' b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

    return inject_forward


def _compute_time_multiplier(seq_len, taper, device, dtype):
    """Build a (1, 1, T, 1) multiplier for temporal fading."""
    t = torch.arange(seq_len, device=device, dtype=dtype)
    if taper == "fade_out":
        mult = 1.0 - t / (seq_len - 1)
    elif taper == "fade_in":
        mult = t / (seq_len - 1)
    elif taper == "cosine_fade_out":
        mult = torch.cos(t / (seq_len - 1) * (torch.pi / 2))
    elif taper == "cosine_fade_in":
        mult = torch.sin(t / (seq_len - 1) * (torch.pi / 2))
    elif taper == "cosine_bell":
        mult = 0.5 * (1 - torch.cos(2 * torch.pi * t / (seq_len - 1)))
    else:
        mult = torch.ones_like(t)
    return mult.view(1, 1, -1, 1)


class InjectionHookManager:
    """Context manager for monkey-patching SA3 Attention layers for KV injection.

    Usage:
        model = dit_wrapper.model  # DiffusionTransformer

        with InjectionHookManager(model, active_layers=[0,1,2,3]) as hm:
            # Capture pass
            hm.capture(noised_ref, t, null_cond_inputs)
            # Injection pass
            output = hm.inject(x, t, cond_inputs, layer_strengths={0: 0.8, 1: 0.5, ...})
    """

    def __init__(self, model, active_layers=None):
        self.model = model
        self.transformer = model.transformer if hasattr(model, 'transformer') else model
        self.layers = self.transformer.layers
        self.num_layers = len(self.layers)
        self.active_layers = active_layers or list(range(self.num_layers))
        self._cache = {}
        self._patched = []

    def _find_self_attn_modules(self):
        """Return list of (layer_idx, self_attn_module) for all layers."""
        modules = []
        for idx, block in enumerate(self.layers):
            if hasattr(block, 'self_attn'):
                modules.append((idx, block.self_attn))
        return modules

    def _patch_all(self, make_fn, **fn_kwargs):
        """Apply a monkey-patch factory to all self-attention modules."""
        self._unpatch()
        for idx, attn_mod in self._find_self_attn_modules():
            if idx not in self.active_layers:
                continue
            patched = make_fn(attn_mod, idx=idx, **fn_kwargs)
            # Bind the patched function as an instance method
            bound = patched.__get__(attn_mod, type(attn_mod))
            attn_mod.forward = bound
            self._patched.append(attn_mod)

    def _unpatch(self):
        """Remove all monkey-patches and restore original forward methods."""
        for mod in self._patched:
            if hasattr(mod, 'forward'):
                # Delete the instance attribute to restore class method
                try:
                    del mod.forward
                except AttributeError:
                    pass
        self._patched.clear()

    def capture(self, *args, **kwargs):
        """Run a forward pass that captures K/V into cache.

        Accepts same args as DiffusionTransformer.forward().
        Returns the model output (which is discarded).
        """
        self._cache.clear()
        self._patch_all(
            lambda mod, idx: _make_capture_forward(mod, self._cache, idx),
        )
        try:
            return self.model(*args, **kwargs)
        finally:
            self._unpatch()

    def inject(self, *args, layer_strengths=None, mode="inject", time_taper="none", **kwargs):
        """Run a forward pass with K/V injection.

        Args:
            layer_strengths: dict mapping layer_idx -> strength float.
                Layers not in dict or with strength <= 0 are skipped.
            mode: "inject" (concat), "replace", or "threshold".
            time_taper: temporal fade pattern.
            *args, **kwargs: passed to DiffusionTransformer.forward().
        """
        ls = layer_strengths or {}
        self._patch_all(
            lambda mod, idx: _make_inject_forward(
                mod, self._cache, idx,
                strength=ls.get(idx, 0.0),
                mode=mode,
                time_taper=time_taper,
            ),
        )
        try:
            return self.model(*args, **kwargs)
        finally:
            self._unpatch()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._unpatch()
        self._cache.clear()

    @property
    def cache_size(self):
        return len(self._cache)


# Convenience wrappers for the two-pass sampling pattern

def injection_capture(model, *args, active_layers=None, **kwargs):
    """One-shot capture: returns (model_output, cache_dict)."""
    hm = InjectionHookManager(model, active_layers=active_layers)
    output = hm.capture(*args, **kwargs)
    cache = dict(hm._cache)
    hm._cache.clear()
    return output, cache


def injection_inject(model, cache, *args, layer_strengths=None, mode="inject", time_taper="none", **kwargs):
    """One-shot inject: returns model output with K/V injection."""
    hm = InjectionHookManager(model, active_layers=list(cache.keys()))
    return hm.inject(*args, layer_strengths=layer_strengths, mode=mode, time_taper=time_taper, **kwargs)
