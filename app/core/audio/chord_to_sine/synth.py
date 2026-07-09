import numpy as np
from .chords import parse_chord_label, midi_to_hz


def synthesize_chord(
    frequencies: list[float],
    duration: float,
    sample_rate: int = 44100,
    amplitude: float = 0.2,
) -> np.ndarray:
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = np.zeros_like(t)
    n = max(len(frequencies), 1)
    for freq in frequencies:
        wave += (amplitude / n) * np.sin(2.0 * np.pi * freq * t)
    return wave


def build_sine_audio(
    chord_segments: list[tuple[float, float, str]],
    sample_rate: int = 44100,
    amplitude: float = 0.2,
    fade_samples: int = 128,
    base_midi: int = 48,
) -> np.ndarray:
    if not chord_segments:
        return np.array([], dtype=np.float64)

    total_duration = chord_segments[-1][1]
    total_samples = int(sample_rate * total_duration)
    out = np.zeros(total_samples, dtype=np.float64)

    last_freqs: list[float] = []

    for start, end, label in chord_segments:
        duration = end - start
        if duration <= 0:
            continue

        midi_notes = parse_chord_label(label, base_midi=base_midi)
        if midi_notes:
            last_freqs = [midi_to_hz(m) for m in midi_notes]
        elif label == "N" and last_freqs:
            pass
        elif label == "N":
            last_freqs = []
        else:
            last_freqs = [midi_to_hz(m) for m in midi_notes] if midi_notes else []

        freqs = last_freqs if last_freqs else []
        block = synthesize_chord(freqs, duration, sample_rate, amplitude)

        if fade_samples > 0 and len(block) > fade_samples * 2:
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            block[:fade_samples] *= fade_in
            block[-fade_samples:] *= fade_out

        sample_start = int(sample_rate * start)
        sample_end = sample_start + len(block)
        out[sample_start:sample_end] += block[: min(len(block), len(out) - sample_start)]

    return out
