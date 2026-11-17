import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def resample(x: np.ndarray, target_len: int) -> np.ndarray:
    """Nearest-neighbor resample along time axis."""
    T = x.shape[0]
    if T == target_len:
        return x
    idx = np.linspace(0, T - 1, target_len).round().astype(int)
    return x[idx]

def dtw_align(vision: np.ndarray, audio: np.ndarray, text: np.ndarray):
    """
    Align all streams to common frame rate (assumed pre-resampled to 30 Hz).
    DTW between vision and audio (Sec. 3.2.2); text already aligned by word timestamps.
    Returns re-indexed tensors of the same length.
    """
    T = min(len(vision), len(audio), len(text))
    vision, audio, text = vision[:T], audio[:T], text[:T]

    # DTW path between vision and audio features
    _, path = fastdtw(vision, audio, dist=euclidean)
    v_idx = np.array([p[0] for p in path])
    a_idx = np.array([p[1] for p in path])
    # re-index audio onto vision timeline
    audio_aligned = audio[a_idx]
    # vision is reference; keep as is
    vision_aligned = vision[v_idx]
    # text aligned to vision timeline (nearest in path)
    text_aligned = text[v_idx]
    # ensure same length
    L = min(len(vision_aligned), len(audio_aligned), len(text_aligned))
    return vision_aligned[:L], audio_aligned[:L], text_aligned[:L]