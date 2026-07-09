<div align="center">

> **⚠️ Work in progress — experimental code. Expect rough edges, incomplete features, and breaking changes.**

# Fragmenta Enhanced

[![License](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

**Extended fork of [Fragmenta](https://github.com/MAz-Codes/fragmenta) — the open-source text-to-audio pipeline for musicians.**

</div>

Fragmenta Enhanced is a feature fork built on [Fragmenta](https://github.com/MAz-Codes/fragmenta) by [Misagh Azimi](https://www.misaghazimi.com). It extends the original with additional ODE solvers, advanced sigma schedulers, deeper model-level control, and quality-of-life UI improvements — while remaining fully compatible with the upstream workflow.

---

## Additions vs. Upstream

All enhancements described below are concentrated in the **Generation tab**. Every other tab — Dataset, Training, Performance — behaves identically to upstream Fragmenta.

### Samplers

ODE solvers for diffusion sampling. Listed in increasing order of quality / cost:

| Sampler | Type | Profile |
|---|---|---|
| **Euler** | 1st-order | Fastest baseline |
| **Heun** | 2nd-order improved Euler | Cleaner than Euler, 2 NFE/step |
| **Midpoint** | 2nd-order RK | Efficient, 1.5 NFE/step average |
| **RK4** | 4th-order Runge-Kutta | Most accurate fixed-step |
| **DPM++** | Multi-step predictor-corrector | Recommended quality/speed balance |
| **PingPong** | Re-noising ODE | Distilled-model default; re-noises each step to stay on-manifold |
| **STORM** | Adaptive stiffness-switching hybrid | Dispatches per-step between STORK (stiff RK2-5) and DPM++3M (smooth); best quality, auto-calibrated — [upstream](https://github.com/MDMAchine/STORM-Sampler) by Alexander Allan (MDMAchine) |

> **Note on sampler behaviour by model type.**  
> In our tests on `sa3-medium` (the post-trained 24-layer distilled model), **PingPong** is the only sampler that consistently produces clean output. Euler, Heun, Midpoint, RK4, and DPM++ produce varying degrees of amplification / compression artefacts.  
> *Why?* The distilled model was trained on a trajectory that expects a re-noising step at each denoising iteration — `x = (1−t)·denoised + t·noise` — which PingPong provides. Solvers that integrate the velocity directly (`x += dt·v`) drift off the training manifold, and the model outputs incorrect velocity on inputs it was never trained on, causing error accumulation.  
> On `sa3-medium-base` (the non-distilled 20-layer rectified-flow model), all samplers work correctly because the base model learns the full continuous velocity field. The other solvers are included for experimentation on base models and future distilled checkpoints that may use different training objectives.

### Schedules

Sigma schedule warping that controls how denoising steps are distributed:

| Schedule | Description |
|---|---|
| **Linear** | Even spacing (default) |
| **Karras** | ρ=7 power-law; concentrates steps at low noise for detail |
| **Beta** | Beta(0.7, 0.7) CDF warp; gentle U-shape |
| **LogSNR** | Uniform in log-SNR space; concentrates at low noise |
| **Flux** | Flux-style parametric shift; concentrates at high noise |
| **HAP** | Hamiltonian Action-Principle physics simulation | Particle-in-potential-well curve (ω=1.5, γ=3.0) — [upstream](https://github.com/MDMAchine/MD-HAP-Scheduler) by Alexander Allan (MDMAchine) |

### Generation UI

- **Metadata inspection panel**: click a fragment in the Generated Fragments list to expand its full generation parameters (sampler, schedule, steps, CFG, seed, model, duration).
- **Unlocked controls**: CFG scale, steps, and sampler dropdown are fully adjustable on all models (including distilled). Model switch sets sensible defaults; user overrides are preserved.
- **Continuous steps slider**: step=1 granularity (not just integers) for fine-grained quality/speed tuning.

### Alternative Reference Audio

Available in the **Edit tab**, the **Alternative Ref Input** button opens a dialog that generates clean synthetic reference audio — useful as a style-transfer or inpainting source. Pure sine-wave references let the model focus on the harmonic structure (key, chord progression, register) without the clutter of real-instrument timbre, background noise, or mix artefacts.

Two modes:

| Mode | What it does |
|---|---|
| **MIDI to Sine** | Import a `.mid` file → render each note as a pure sine or triangle wave. Adjustable waveform and transposition (±12 semitones). Good for transferring a specific melodic/harmonic idea into the model. |
| **Convert Audio to Chord Progression** | Upload any audio file → AI chord extraction (madmom CNN+CRF, librosa fallback) → editable chord text → render as sine/triangle wave. A **mix slider** blends the original audio with the sine output so you can inject a controlled amount of the original timbre alongside the clean harmonic guide. |

Both modes produce a `.wav` that is automatically uploaded as the Edit tab's source clip via **Use as source**.

### LoRA System Fix

Circular import between `model.py` and `utils.py` in the vendored SA3 LoRA module resolved — required for the LoRA system to load at all.

> **LoRA architecture compatibility.** The app pairs LoRAs to checkpoints by stripping the `-base` suffix before comparing, so a LoRA trained on `sa3-medium` works with both `sa3-medium` and `sa3-medium-base` (same DiT backbone, only CFG distillation state differs). The same logic applies to the small variants. What *is* incompatible: `*-small-music` LoRAs will not load onto `*-small-sfx` checkpoints, or vice versa, because music and SFX have fundamentally different conditioning objectives despite sharing the same layer/embed dimensions. When in doubt, check the LoRA's `base_model` metadata key — the app shows a clear error on true mismatches.

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

### Where to put LoRAs

Place `.safetensors` LoRA checkpoint files under:

```
models/fine_tuned/<run_name>/checkpoints/
```

- `<run_name>` is any folder name you choose (e.g. `my-lora` or `saxbarblues`).
- The file must carry a `base_model` key in its safetensors metadata (e.g. `sa3-medium` or `sa3-small-music`). The app reads this automatically.
- No extra config files are needed; drop the files in and the LoRA picker in the **Performance** tab (and Generation tab LoRA stack) will pick them up on next launch.

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
│           └── audio_generator.py   # Sampler/schedule dispatch, model loading
├── vendor/
│   └── stable-audio-3/
│       └── models/
│           ├── lora/                # (circular import fix)
├── models/               # Checkpoint storage (HF cache + flat layout)
├── config/                # LoRA config, app settings
├── output/                # Generated audio, uploads, recordings
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
- **[STORM Sampler](https://github.com/MDMAchine/STORM-Sampler)** by Alexander Allan (MDMAchine) — adaptive stiffness-switching ODE solver, GPL v3.
- **[MD-HAP Scheduler](https://github.com/MDMAchine/MD-HAP-Scheduler)** by Alexander Allan (MDMAchine) — Hamiltonian action-principle sigma schedule, GPL v3.
