import numpy as np
import soundfile as sf
import pretty_midi


def midi_to_hz(midi_note: float) -> float:
    return 440.0 * (2.0 ** ((midi_note - 69.0) / 12.0))


def _sine_wave(freq: float, t: np.ndarray, amplitude: float) -> np.ndarray:
    return amplitude * np.sin(2.0 * np.pi * freq * t)


def _triangle_wave(freq: float, t: np.ndarray, amplitude: float) -> np.ndarray:
    phase = (freq * t) % 1.0
    return amplitude * (2.0 * np.abs(2.0 * phase - 1.0) - 1.0)


_WAVEFORM_FUNCS = {
    "sine": _sine_wave,
    "triangle": _triangle_wave,
}


def render_midi(
    midi_path: str,
    output_path: str,
    waveform: str = "sine",
    transpose: int = 0,
    sample_rate: int = 44100,
    amplitude: float = 0.2,
) -> float:
    pm = pretty_midi.PrettyMIDI(midi_path)

    notes = []
    for inst in pm.instruments:
        if not inst.is_drum:
            for note in inst.notes:
                notes.append({
                    "pitch": note.pitch + transpose,
                    "start": note.start,
                    "end": note.end,
                    "velocity": note.velocity / 127.0,
                })

    if not notes:
        raise ValueError("No non-drum notes found in MIDI file")

    notes.sort(key=lambda n: n["start"])

    total_duration = max(n["end"] for n in notes)
    total_samples = int(sample_rate * total_duration) + 1
    audio = np.zeros(total_samples, dtype=np.float64)

    wave_fn = _WAVEFORM_FUNCS.get(waveform, _sine_wave)
    fade_samples = 128

    for n in notes:
        start_sample = int(n["start"] * sample_rate)
        end_sample = int(n["end"] * sample_rate)
        dur_samples = end_sample - start_sample
        if dur_samples <= 0:
            continue

        t = np.linspace(0, (end_sample - start_sample) / sample_rate, dur_samples, endpoint=False)
        freq = midi_to_hz(n["pitch"])
        vel = n["velocity"]
        wave = wave_fn(freq, t, amplitude * vel)

        if dur_samples > fade_samples * 2:
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            wave[:fade_samples] *= fade_in
            wave[-fade_samples:] *= fade_out

        end = min(start_sample + dur_samples, len(audio))
        actual_len = end - start_sample
        audio[start_sample:end] += wave[:actual_len]

    peak = np.abs(audio).max()
    if peak > 1e-10:
        audio = audio / peak * amplitude

    sf.write(str(output_path), audio, sample_rate)
    return total_duration
