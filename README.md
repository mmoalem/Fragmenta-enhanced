<div align="center">

> **⚠️ Work in progress — experimental code. Expect rough edges, incomplete features, and breaking changes.**

# Fragmenta Enhanced

[![License](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

**Extended fork of [Fragmenta](https://github.com/MAz-Codes/fragmenta) — the open-source text-to-audio pipeline for musicians.**

</div>

Fragmenta Enhanced is a feature fork built on [Fragmenta](https://github.com/MAz-Codes/fragmenta) by [Misagh Azimi](https://www.misaghazimi.com). It extends the original with experimental generation modes, additional sampler options, and deeper model-level control — while remaining fully compatible with the upstream workflow.

---

## Additions vs. Upstream

### Per-Step Self-Attention KV Injection (Reference Injection)

A new generation mode that guides diffusion by injecting self-attention Key/Value pairs from a reference audio clip into each transformer layer at every denoising step.

- **Two-pass sampling**: reference audio → null-conditioned capture pass → generation pass with injected K/V states
- **Per-layer strength controls**: 12 individual sliders (0–3) for each SA3 DiT transformer layer
- **Injection modes**: `inject` (blend), `replace` (overwrite), `threshold` (attention-score gated)
- **Step / time tapering**: cosine, linear, or none — control how injection strength evolves across denoising steps and the generated timeline

Ported from ComfyUI's `per_step_inject_per_layer` node architecture, adapted for SA3's 12-layer DiT with fused QKV projections.

### Sampler Selection

Exposes the full suite of ODE solvers available in SA3:

- **Euler** — simple, fast, lowest quality
- **RK4** — 4th-order Runge-Kutta, slower but more accurate trajectories
- **DPM++** — recommended quality/speed balance
- **PingPong** — alternates forward/backward trajectories for smoother convergence

### LoRA System Fix

Circular import between `model.py` and `utils.py` in the vendored SA3 LoRA module resolved — required for the LoRA system to load at all.

---

## Features (Upstream)

- **Desktop app** with a lightweight `pywebview` window and a pre-built React frontend
- **Bulk auto-annotation** — generate text prompts for your audio files via DSP analysis (Basic) or AI tagging with LAION-CLAP (Rich), with optional user-defined vocabulary
- **Project-aware LoRA training** with configurable rank, steps, learning rate, batch size, checkpoint frequency, and precision — trains directly on a Dataset Workbench project
- **LoRA adapters** — train LoRA, DoRA, or BoRA adapters (plus low-VRAM `-xs` variants) on top of a frozen `*-base` checkpoint for consumer GPUs; stack up to 4 at once with per-slot strength, bypass, and reorder at generation time
- **Text-to-audio generation** — variable-length clips (up to 120s small / 380s medium), with CFG scale, inference steps, seed control, and a multi-LoRA stack
- **Audio editing (Edit tab)** — style transfer (audio-to-audio), region inpainting, and clip extension/continuation
- **Checkpoint Manager** — pick and download individual SA3 checkpoints (Small Music/SFX, Medium, and the matching `*-base` models) with per-item progress and hardware-compatibility hints
- **Performance Mode** — a 4-channel live sampler: per-channel effects (gain, pan, filter, delay, reverb), master dBFS metering, bars-mode generation, launch quantization (standalone or via **Ableton Link**), persistent sessions, named presets, and MIDI learn
- **Real-time GPU memory monitoring****


---

## Getting Started

Same setup as upstream Fragmenta. Requirements: **Python 3.11**.

```bash
git clone https://github.com/mmoalem/Fragmenta-enhanced.git
cd fragmenta
```

Then run the installer for your platform:

| Platform | Command |
|---|---|
| **Linux** | `./fragmenta.sh` |
| **macOS** | `fragmenta.command` |
| **Windows** | `./fragmenta.bat` |

The first run downloads and installs everything; subsequent launches are faster. All data stays local.

---

## Project Structure

```
fragmenta/
├── app/
│   ├── backend/             # Flask API server
│   ├── frontend/
│   │   ├── build/           # Pre-built React app (served by Flask)
│   │   └── src/             # React source (development only)
│   └── core/
│       └── generation/
│           ├── audio_generator.py   # Sampler selection, ref-injection wiring
│           └── ref_inject.py        # RefInjectModelWrapper (two-pass DiT wrapper)
├── vendor/
│   └── stable-audio-3/
│       └── models/
│           ├── lora/                # (circular import fix)
│           └── injection/           # InjectionHookManager (per-step KV capture/inject)
├── models/
├── config/
├── fragmenta.sh / .bat / .command
├── install.py
└── start.py
```

---

## License

This fork is licensed under the **GNU Affero General Public License v3.0**, inheriting the upstream terms. See [LICENSE](LICENSE).

### Upstream

[Fragmenta](https://github.com/MAz-Codes/fragmenta) by [Misagh Azimi](https://www.misaghazimi.com) — [DOI: 10.5281/zenodo.20692998](https://doi.org/10.5281/zenodo.20692998).

### Third-Party

- **Stable Audio 3** by Stability AI — model weights under the [Stability AI Community License](https://stability.ai/community-license-agreement); vendored inference code under MIT.
- **Per-step injection pattern** ported from [ComfyUI BitsAndBots / ace_step_reference](https://github.com/BitsAndBobs-LLC/ComfyuAudioNodes-BitsAndBobs) (architecture reference, not vendored code).
