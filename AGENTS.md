# Fragmenta Enhanced — Project Handoff

> **Generated**: 2026-07-26 | **Version**: 1.0.2  
> **Repo**: `https://github.com/mmoalem/Fragmenta-enhanced.git` (branch `main`)

---

## Project Overview

Fragmenta Enhanced is an extended fork of [Fragmenta](https://github.com/MAz-Codes/fragmenta) — an open-source text-to-audio desktop app for musicians. Built with:
- **Backend**: Python 3.11 (Flask API) + `pywebview` desktop wrapper
- **Frontend**: React + Vite (built into `app/frontend/build/`)
- **ML**: Stable Audio 3 (SA3) models via vendored `vendor/stable-audio-3/`

All enhancements are scoped to the **Generation tab** — Dataset, Training, and Performance tabs are identical to upstream Fragmenta.

---

## Repository Layout

```
E:\AI\fragmenta\                         # Project root
├── app/
│   ├── backend/app.py                   # Flask API server (main entry point)
│   ├── core/
│   │   ├── audio/
│   │   │   ├── midi_synth.py            # MIDI renderer (mido-based, handles high tick offsets)
│   │   │   ├── chord_to_sine/           # Audio→chord→sine pipeline
│   │   │   ├── midi_input.py            # MIDI device input
│   │   │   └── link_sync.py             # Ableton Link sync
│   │   └── training/
│   │       ├── sa3_trainer.py           # Training orchestrator (subprocess-based)
│   │       ├── sa3_lora_runner.py       # CLI command builder + model pre-staging
│   │       └── hyperparam_suggester.py  # Auto-suggest training params
│   └── frontend/
│       ├── build/                       # Vite build output (served by Flask)
│       └── src/
│           ├── App.js                   # Main React app
│           └── components/
│               ├── EditPanel.js          # Edit tab (style transfer, inpaint, extend)
│               ├── RefInjectPanel.js     # Self-attention reference injection (Generation tab)
│               ├── ChordToSineModal.js   # MIDI/audio→sine reference audio dialog
│               ├── GeneratedFragmentsWindow.js
│               ├── PerformancePanel.js   # Live performance mode
│               ├── PerformanceChannel.js
│               ├── AudioWaveform.js
│               └── ...
├── models/
│   ├── checkpoints/                     # SA3 base model checkpoints
│   └── fine_tuned/                      # Trained LoRA adapters
│       └── stable_audio3_medium/        # Example LoRA root (contains others/ subdirs)
├── vendor/stable-audio-3/               # Vendored SA3 library + train_lora.py
├── config/                              # App configuration
├── output/                              # Generated audio output
├── projects/                            # Dataset workbench projects
├── uploads/                             # Uploaded audio files
├── VERSION                              # "1.0.2"
└── start.py                             # Desktop app launcher
```

---

## Completed Work

### 1. MIDI Renderer — `pretty_midi` → `mido`
**File**: `app/core/audio/midi_synth.py`

Replaced `pretty_midi` (which has a hardcoded `MAX_TICK = 1e7` limit) with `mido` for parsing MIDI files. Handles ACE Studio exports that embed notes at ~4.3B absolute ticks (near `2^32`) by detecting and subtracting the offset. Includes a 10-minute render cap.

### 2. LoRA Selector — Recursive Scan
**File**: `app/backend/app.py` (~line 853)

Changed LoRA discovery from `d.glob` to `d.rglob` in fallback mode so `.safetensors` files in subdirectories (e.g., `others/`) are found.

### 3. Reference Audio Normalization
**Files**: `app/backend/app.py` (endpoint), `app/frontend/src/components/RefInjectPanel.js` (button)

- Added `/api/audio/normalize` endpoint — peak-normalizes to -1 dBFS in place
- Added "Normalise" button in RefInjectPanel (Generation tab, next to Clear)
- The EditPanel already had a client-side normalise (Web Audio API, peaks to 0.95)
- All three input sources (file upload, MIDI-to-sine, chord-to-sine) converge on the EditPanel's `sourcePath` and the existing normalise works on all of them

### 4. ComfyUI Audio Nodes
**File**: `app/backend/comfyui_integration.py`

ComfyUI integration for audio processing nodes. ComfyUI runs from embedded Python at `E:\AI\ComfyUI_windows_portable\python_embeded\` — NOT the project's pixi env.

### 5. Chord-to-Sine Pipeline
**Files**: `app/core/audio/chord_to_sine/`, `ChordToSineModal.js`

Two modes:
- **MIDI to Sine**: Import `.mid` → render notes as pure sine/triangle waves
- **Audio to Chord Progression**: Upload audio → AI chord extraction (madmom CNN+CRF, librosa fallback) → editable chord text → render as sine/triangle

### 6. Frontend Build
After any `src/` changes, run: `npx vite build` in `app/frontend/`

---

## In Progress / Next Steps

### Phase 1: Underfit Training Engine Merge (HIGH PRIORITY)

The user has an enhanced training fork at `E:\AI\underfit-test` that needs to be merged into Fragmenta to replace the current subprocess-based training.

**Current Fragmenta training** (`app/core/training/`):
- `sa3_trainer.py` — orchestrator, dispatches `vendor/stable-audio-3/scripts/train_lora.py` as subprocess, parses tqdm stdout + metrics.csv
- `sa3_lora_runner.py` — builds CLI command, pre-stages base model in HF cache, converts .ckpt → .safetensors
- Config is a flat JSON dict (modelName, baseModel, steps, batchSize, lr, etc.)

**Underfit training** (`underfit/`):
- `training/loop.py` — raw PyTorch training loop (no Lightning), direct control over forward/backward pass
- `training/lora.py`, `training/loss.py`, `training/optim.py`, `training/timestep.py` — modular training components
- `training/demo_step.py` — demo MP3 generation at checkpoints
- `backends/sa3.py` — SA3 backend adapter (abstracted backend interface)
- `backends/sat.py` — SAT backend adapter
- `lora_train.py` — CLI entry point, reads JSON config
- `defaults.ini` — default config values

**Key Underfit advantages**:
- Raw PyTorch (no Lightning dependency overhead)
- Backend abstraction (SA3/SAT)
- Best checkpoint tracking by EMA loss
- Demo MP3 generation per checkpoint
- Gradio per checkpoint
- Oversampling for tiny datasets
- TensorBoard logging
- Kill diagnostics

**Merge plan**:
1. Copy `underfit/training/` + `underfit/backends/sa3.py` into Fragmenta
2. Rewrite `sa3_lora_runner.py` to generate JSON configs for Underfit's `lora_train.py`
3. Keep Fragmenta's React UI + Flask backend
4. Do NOT merge the standalone dashboard UI (it will be replaced by React GUI features)

### Phase 2: Underfit Dashboard Features → React GUI (MEDIUM PRIORITY)

Port these Underfit features to Fragmenta's React frontend:
- **Live console log viewer** (highest priority — currently no way to see training output)
- **Best checkpoint badges** (EMA loss tracking)
- **Demo MP3 generation** at checkpoints
- **Kill diagnostics** (what state was the training in when stopped)
- **Oversampling helper** for tiny datasets
- **TensorBoard toggle**

---

## Key Technical Details

### Model Compatibility
- `sa3-medium` ↔ `sa3-medium-base`: cross-compatible (same DiT backbone). App strips `-base` suffix before comparing.
- LoRAs: `.safetensors` files with `base_model` nested inside `lora_config` JSON need fallback `lora_config.get("base_model")`.
- `-small-music` LoRAs are NOT compatible with `-small-sfx` checkpoints (different conditioning objectives).

### Samplers
- **PingPong** is the only sampler that works on `sa3-medium` (distilled model). Other samplers produce amplified/distorted output.
- All samplers work on `sa3-medium-base` (non-distilled rectified-flow model).
- Samplers: Euler, Heun, Midpoint, RK4, DPM++, PingPong, STORM.

### MIDI Tempo
- Ableton exports embed project tempo (not clip tempo). BPM modifier overrides the file's tempo.
- LoRA negative strengths: slider range -2 to 2. Negative values invert adapter effect (`W' = W - α·BA`).

### Flask Serving
- Flask serves frontend from `app/frontend/build/` (Vite built output).
- `app/backend/app.py` is the main API server.

### Environment
- Python 3.11 required
- ComfyUI: embedded Python at `E:\AI\ComfyUI_windows_portable\python_embeded\`
- LoRA root path: `<project_root>/models/fine_tuned/<run_name>/checkpoints/` (or `FRAGMENTA_FINE_TUNED_DIR` env var)

---

## Git History (Recent)

```
ee66353 fix: use rglob to find .safetensors in subdirectories of fine_tuned
25ed667 fix: switch MIDI renderer from pretty_midi to mido, handle high tick offsets
fbe1fe3 feat: MIDI/chord-to-sine reference audio, detail panel, version bump
d1d9938 feat: MIDI/chord-to-sine reference audio, detail panel, version bump
cc024ee Add note: all enhancements are Generation-tab-only
cd5453b Fix LoRA dropdown: add overflowY:auto so maxHeight actually scrolls
553ffd2 Fix LoRA dropdown scroll: stop wheel propagation to parent column
```

---

## Build & Run

```bash
# Start the desktop app
python start.py

# Or just the Flask dev server
python -m app.backend.app

# Rebuild frontend after src/ changes
cd app/frontend && npx vite build
```
