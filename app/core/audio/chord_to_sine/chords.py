CHORD_INTERVALS = {
    "maj":  [0, 4, 7],
    "min":  [0, 3, 7],
    "dim":  [0, 3, 6],
    "aug":  [0, 4, 8],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
    "6":    [0, 4, 7, 9],
    "7":    [0, 4, 7, 10],
    "maj7": [0, 4, 7, 11],
    "min7": [0, 3, 7, 10],
}

ROOT_OFFSETS = {
    "C": 0, "C#": 1, "Db": 1,
    "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "Fb": 4, "E#": 5,
    "F": 5, "F#": 6, "Gb": 6,
    "G": 7, "G#": 8, "Ab": 8,
    "A": 9, "A#": 10, "Bb": 10,
    "B": 11, "Cb": 11, "B#": 12,
}

QUALITY_ALIASES = {
    "m": "min",
    "M": "maj",
    "Maj": "maj", "MAJ": "maj", "MAJOR": "maj",
    "Min": "min", "MIN": "min", "MINOR": "min",
    "o": "dim", "\u00b0": "dim", "dim7": "dim",
    "+": "aug", "aug7": "aug",
    "M7": "maj7",
    "m7": "min7",
    "dom7": "7",
    "sus": "sus4",
}

QUALITY_SEPARATORS = ":-"

_ROOTS_SORTED = sorted(ROOT_OFFSETS.items(), key=lambda x: -len(x[0]))


def _normalize_quality(q: str) -> str:
    q = q.strip()
    for sep in QUALITY_SEPARATORS:
        q = q.lstrip(sep)
    if not q:
        return "maj"
    return q


def _match_root(label: str):
    for root_name, offset in _ROOTS_SORTED:
        if label.startswith(root_name):
            return root_name, offset
    lower = label.lower()
    for root_name, offset in _ROOTS_SORTED:
        if lower.startswith(root_name.lower()):
            return root_name, offset
    return None


def _parse_simple(label: str, base_midi: int) -> list[float]:
    match = _match_root(label)
    if match is None:
        return []
    root_name, offset = match
    raw_quality = label[len(root_name):].strip()
    quality = _normalize_quality(raw_quality)
    quality = QUALITY_ALIASES.get(quality, quality)
    intervals = CHORD_INTERVALS.get(quality, CHORD_INTERVALS["maj"])
    return [base_midi + offset + i for i in intervals]


def parse_chord_label(label: str, base_midi: int = 48) -> list[float]:
    label = label.strip()
    if label == "N" or not label:
        return []

    if "/" in label:
        parts = label.rsplit("/", 1)
        chord_notes = _parse_simple(parts[0].strip(), base_midi)
        if not chord_notes:
            return []
        bass_match = _match_root(parts[1].strip())
        if bass_match is None:
            return chord_notes
        _, bass_off = bass_match
        bass_midi = base_midi + bass_off
        notes = sorted(chord_notes)
        b = bass_midi
        while b > notes[0]:
            b -= 12
        while b + 12 <= notes[0]:
            b += 12
        if b < notes[0]:
            notes.insert(0, b)
        return notes

    return _parse_simple(label, base_midi)


def midi_to_hz(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))
