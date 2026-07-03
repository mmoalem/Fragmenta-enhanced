"""Reference audio per-step self-attention KV injection wrapper.

Wraps a DiffusionTransformer model to perform two-pass sampling:
  1. Capture pass: run noised reference latent through model with null
     conditioning, collect K/V from all self-attention layers.
  2. Injection pass: run generation latent through model, blending
     captured reference K/V with computed K/V per-layer.

Designed to be passed as the `model` argument to `sample_diffusion()`,
transparent to all sampler types (euler, rk4, dpmpp, pingpong).
"""

from typing import Optional

import torch

from stable_audio_3.models.injection import InjectionHookManager


def _nullify_cond(kwargs: dict) -> dict:
    """Return a copy of kwargs with text-carrying conds zeroed for capture pass."""
    null = dict(kwargs)
    for key in ("cross_attn_cond", "prepend_cond", "global_embed"):
        if key in null and null[key] is not None:
            null[key] = torch.zeros_like(null[key])
    null["cfg_scale"] = 1.0
    # Strip length-dependent masks computed for the generation latent
    null.pop("padding_mask", None)
    null.pop("inpaint_mask", None)
    null.pop("inpaint_masked_input", None)
    return null


class RefInjectModelWrapper(torch.nn.Module):
    """Wraps a DiT/transformer model to inject reference audio KV at each step.

    Usage:
        wrapper = RefInjectModelWrapper(model, ref_latent, layer_strengths={0: 0.5, ...})
        output = sample_diffusion(model=wrapper, ...)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        ref_latent: torch.Tensor,
        layer_strengths: Optional[dict] = None,
        inject_mode: str = "inject",
        step_taper: str = "none",
        time_taper: str = "none",
        active_layers: Optional[list] = None,
    ):
        super().__init__()
        self._dit = model  # DiffusionTransformer
        self._ref = ref_latent  # (1, C, T) latent
        self._strengths = layer_strengths or {}
        self._active = active_layers or [i for i, s in self._strengths.items() if s > 0]
        self._mode = inject_mode
        self._time_taper = time_taper

    def _diffusion_objective_sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Compute sigma (noise level) from timestep t for the RF objective."""
        return t  # For rectified_flow / rf_denoiser: sigma = t

    def __call__(self, x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        device, dtype = x.device, x.dtype
        batch_size = x.shape[0]

        # Noised reference latent for current timestep
        sigma = t.view(-1, 1, 1) if t.dim() == 1 else t[0]
        noise = torch.randn_like(self._ref).to(device, dtype)
        ref_noised = self._ref.to(device, dtype) * (1 - sigma.to(dtype)) + noise * sigma

        # Null conditioning for capture pass
        null_kwargs = _nullify_cond(kwargs)

        # Two-pass: capture then inject within a single HookManager scope
        # so the cache is shared between passes.
        hm = InjectionHookManager(self._dit, active_layers=self._active)
        with hm:
            # Capture pass (output discarded)
            hm.capture(ref_noised, t, **null_kwargs)
            # Injection pass (output returned to sampler)
            return hm.inject(
                x, t, **kwargs,
                layer_strengths=self._strengths,
                mode=self._mode,
                time_taper=self._time_taper,
            )
