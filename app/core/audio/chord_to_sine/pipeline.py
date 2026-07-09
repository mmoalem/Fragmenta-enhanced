from pathlib import Path

import soundfile as sf

from .extract import extract_chords
from .synth import build_sine_audio


def extract_only(
    input_path: str | Path,
) -> list[tuple[float, float, str]]:
    return extract_chords(str(input_path))


def synthesize(
    chord_segments: list[tuple[float, float, str]],
    output_path: str | Path,
    sample_rate: int = 44100,
    amplitude: float = 0.2,
    base_midi: int = 48,
) -> None:
    audio = build_sine_audio(chord_segments, sample_rate, amplitude, base_midi=base_midi)
    sf.write(str(output_path), audio, sample_rate)


def format_chords(chord_segments: list[tuple[float, float, str]]) -> str:
    lines: list[str] = []
    for start, end, label in chord_segments:
        lines.append(f"{start:.1f} {end:.1f} {label}")
    return "\n".join(lines)


def parse_chord_text(text: str) -> list[tuple[float, float, str]]:
    segments: list[tuple[float, float, str]] = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 2)
        if len(parts) >= 3:
            try:
                start = float(parts[0])
                end = float(parts[1])
                segments.append((start, end, parts[2]))
            except ValueError:
                pass
    return segments
