import numpy as np
from ._compat import *

import collections.abc
import collections

import warnings

import librosa
from madmom.features.chords import CNNChordFeatureProcessor, CRFChordRecognitionProcessor
from madmom.processors import SequentialProcessor

ROOT_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

CHORD_TEMPLATES = {
    'maj':  [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    'min':  [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    'dim':  [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    'aug':  [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    '7':    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    'maj7': [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    'min7': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
}


def _build_template_matrix():
    from scipy.ndimage import gaussian_filter1d
    names = []
    rows = []
    for root in range(12):
        for quality, vec in CHORD_TEMPLATES.items():
            arr = np.array(vec, dtype=np.float64)
            arr = gaussian_filter1d(arr, sigma=0.5, mode='constant')
            arr /= arr.max()
            rows.append(np.roll(arr, root))
            names.append(f"{ROOT_NAMES[root]}:{quality}")
    mat = np.array(rows, dtype=np.float64)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    return mat, names


def _extract_librosa(audio_path: str) -> list[tuple[float, float, str]]:
    y, sr = librosa.load(audio_path, sr=22050)
    hop = 512
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop, n_chroma=12)
    n_frames = chroma.shape[1]
    chroma_norm = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-10)
    mat, names = _build_template_matrix()
    times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop)
    frame_labels = []

    for i in range(n_frames):
        energy = np.linalg.norm(chroma_norm[:, i])
        if energy < 0.1:
            frame_labels.append("N")
        else:
            sims = mat @ chroma_norm[:, i]
            frame_labels.append(names[int(np.argmax(sims))])

    segments = []
    cur = frame_labels[0]
    start = times[0]
    for i in range(1, n_frames):
        if frame_labels[i] != cur:
            segments.append((float(start), float(times[i]), str(cur)))
            cur = frame_labels[i]
            start = times[i]
    segments.append((float(start), float(times[-1]), str(cur)))

    min_dur = hop / sr * 4
    filtered = []
    i = 0
    while i < len(segments):
        s, e, l = segments[i]
        if e - s < min_dur and filtered:
            filtered[-1] = (filtered[-1][0], e, filtered[-1][2])
        elif e - s < min_dur and i + 1 < len(segments):
            segments[i + 1] = (s, segments[i + 1][1], segments[i + 1][2])
        else:
            filtered.append((s, e, l))
        i += 1
    return filtered


def extract_chords(
    audio_path: str,
) -> list[tuple[float, float, str]]:
    try:
        featproc = CNNChordFeatureProcessor()
        decode = CRFChordRecognitionProcessor()
        chordrec = SequentialProcessor([featproc, decode])
        chords = chordrec(audio_path)
        result = [(float(s), float(e), str(l)) for s, e, l in chords]

        non_n = sum(1 for _, _, l in result if l != "N")
        if non_n == 0:
            warnings.warn("madmom returned all 'N' — falling back to librosa")
            return _extract_librosa(audio_path)
        return result
    except Exception as e:
        warnings.warn(f"madmom failed: {e} — falling back to librosa")
        return _extract_librosa(audio_path)
