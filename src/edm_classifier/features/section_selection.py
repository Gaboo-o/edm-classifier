"""Section-selection helpers aligned to fixed-duration embedding windows."""
from __future__ import annotations

import math
import numpy as np


def robust_unit_scale(values: np.ndarray) -> np.ndarray:
    """Map a 1-D track-local descriptor to approximately [0, 1]."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"expected 1-D values, got {x.shape}")
    if x.size == 0:
        return x.astype(np.float32)
    lo, hi = np.percentile(x, [10.0, 90.0])
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("non-finite section descriptor")
    if hi <= lo + 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def center_fraction_bounds(
    window_count: int,
    *,
    start_fraction: float = 0.35,
    end_fraction: float = 0.65,
) -> tuple[int, int]:
    """Return half-open window bounds for the central fraction of a track."""
    if window_count < 1:
        raise ValueError("window_count must be positive")
    if not 0.0 <= start_fraction < end_fraction <= 1.0:
        raise ValueError("invalid center fractions")
    start = int(math.floor(window_count * start_fraction))
    end = int(math.ceil(window_count * end_fraction))
    start = min(max(start, 0), window_count - 1)
    end = min(max(end, start + 1), window_count)
    return start, end


def best_contiguous_bounds(
    scores: np.ndarray,
    *,
    block_windows: int,
    tie_fraction: float = 0.01,
) -> tuple[int, int]:
    """Select the strongest contiguous block, using centrality only as a tie-break.

    Blocks whose mean score lies within ``tie_fraction`` of the maximum are
    considered effectively tied; among them the block nearest the track center
    wins. This avoids a strong structural prior while making ties deterministic.
    """
    x = np.asarray(scores, dtype=np.float64)
    if x.ndim != 1 or x.size < 1:
        raise ValueError("scores must be a non-empty 1-D array")
    if not np.isfinite(x).all():
        raise ValueError("scores contain non-finite values")
    if block_windows < 1:
        raise ValueError("block_windows must be positive")
    width = min(int(block_windows), int(x.size))
    if width == x.size:
        return 0, int(x.size)

    kernel = np.ones(width, dtype=np.float64) / width
    block_scores = np.convolve(x, kernel, mode="valid")
    best = float(block_scores.max())
    tolerance = max(1e-9, abs(best) * float(tie_fraction))
    candidates = np.flatnonzero(block_scores >= best - tolerance)

    track_center = x.size / 2.0
    block_centers = candidates + width / 2.0
    chosen = int(candidates[np.argmin(np.abs(block_centers - track_center))])
    return chosen, chosen + width
