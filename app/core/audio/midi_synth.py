import logging

import numpy as np
import soundfile as sf
import mido

logger = logging.getLogger(__name__)


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


def _parse_midi_with_mido(midi_path: str, transpose: int = 0) -> list[dict]:
    """Parse MIDI using mido (no tick limit) and return a list of notes.

    Returns list of dicts with keys: pitch, start, end, velocity (0-127).
    Only non-drum notes are included.
    """
    mid = mido.MidiFile(midi_path)
    ticks_per_beat = mid.ticks_per_beat

    # Collect tempo changes across all tracks
    tempo_events: list[tuple[int, float]] = []
    notes: list[dict] = []

    for track in mid.tracks:
        abs_tick = 0
        tempo = 500000.0  # default 120 BPM (microseconds per beat)
        # track active notes: {(channel, note): (start_tick, velocity)}
        active: dict[tuple[int, int], tuple[int, int]] = {}

        for msg in track:
            abs_tick += msg.time

            if msg.type == "set_tempo":
                tempo = msg.tempo
                tempo_events.append((abs_tick, tempo))
            elif msg.type == "note_on" and msg.velocity > 0:
                active[(msg.channel, msg.note)] = (abs_tick, msg.velocity)
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                key = (msg.channel, msg.note)
                if key in active:
                    start_tick, vel = active.pop(key)
                    notes.append({
                        "start_tick": start_tick,
                        "end_tick": abs_tick,
                        "pitch": msg.note + transpose,
                        "velocity": vel,
                        "channel": msg.channel,
                    })

    if not notes:
        raise ValueError("No non-drum notes found in MIDI file")

    # Some exporters (e.g. ACE Studio) embed notes at very high absolute
    # tick values (close to 2^32).  Detect and strip the offset so the
    # first note starts near tick 0 and the file renders in reasonable time.
    min_tick = min(n["start_tick"] for n in notes)
    max_tick = max(n["end_tick"] for n in notes)
    if max_tick > MAX_TICK:
        for n in notes:
            n["start_tick"] -= min_tick
            n["end_tick"] -= min_tick
        tempo_events = [(t - min_tick, bpm) for t, bpm in tempo_events
                        if t - min_tick >= 0]
        logger.info("Shifted MIDI ticks by -%d to normalize high-offset file", min_tick)

    # Convert ticks to seconds
    # Build a list of (tick, tempo) for tempo lookup
    tempo_events.sort(key=lambda x: x[0])

    def ticks_to_seconds(tick: int) -> float:
        """Convert absolute tick to seconds, applying tempo map."""
        if not tempo_events:
            return tick * (500000.0 / (ticks_per_beat * 1_000_000.0))

        # Find the last tempo change at or before this tick
        current_tempo = 500000.0
        last_tick = 0
        elapsed = 0.0
        for te_tick, te_tempo in tempo_events:
            if te_tick > tick:
                break
            elapsed += (te_tick - last_tick) * (current_tempo / (ticks_per_beat * 1_000_000.0))
            current_tempo = te_tempo
            last_tick = te_tick
        elapsed += (tick - last_tick) * (current_tempo / (ticks_per_beat * 1_000_000.0))
        return elapsed

    # Get original BPM from first tempo event
    orig_bpm = 120.0
    if tempo_events:
        orig_bpm = mido.tempo2bpm(tempo_events[0][1])

    result = []
    for n in notes:
        result.append({
            "pitch": n["pitch"],
            "start": ticks_to_seconds(n["start_tick"]),
            "end": ticks_to_seconds(n["end_tick"]),
            "velocity": n["velocity"] / 127.0,
        })

    return result, orig_bpm


MAX_RENDER_SECONDS = 600  # 10 minutes — anything beyond this is likely stray tick data
MAX_TICK = 1_000_000  # ticks above this trigger offset normalization


def render_midi(
    midi_path: str,
    output_path: str,
    waveform: str = "sine",
    transpose: int = 0,
    bpm: float | None = None,
    sample_rate: int = 44100,
    amplitude: float = 0.2,
) -> float:
    parsed_notes, orig_bpm = _parse_midi_with_mido(midi_path, transpose=transpose)

    time_scale = orig_bpm / bpm if (bpm is not None and bpm > 0 and abs(bpm - orig_bpm) > 0.5) else 1.0

    notes = []
    for n in parsed_notes:
        notes.append({
            "pitch": n["pitch"],
            "start": n["start"] * time_scale,
            "end": n["end"] * time_scale,
            "velocity": n["velocity"],
        })

    if not notes:
        raise ValueError("No non-drum notes found in MIDI file")

    notes.sort(key=lambda n: n["start"])

    # Strip notes that land beyond MAX_RENDER_SECONDS — stray high-tick
    # events can map to thousands of hours of empty audio.
    filtered = [n for n in notes if n["start"] < MAX_RENDER_SECONDS]
    if len(filtered) < len(notes):
        logger.warning(
            "Filtered %d/%d notes beyond %ds — likely stray high-tick data",
            len(notes) - len(filtered), len(notes), MAX_RENDER_SECONDS,
        )
    notes = filtered
    if not notes:
        raise ValueError("All notes exceed the maximum render duration")

    total_duration = max(n["end"] for n in notes)
    # Hard cap on render length to prevent OOM on stray tick data
    if total_duration > MAX_RENDER_SECONDS:
        total_duration = MAX_RENDER_SECONDS
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
