"""Mean-maximal power (MMP) curves and training-load metrics from 1 Hz power."""
from __future__ import annotations

import numpy as np

# Duration grid used by GoldenCheetah OpenData's *_mmp.csv index files [s].
GC_GRID = np.array([
    1, 15, 30, 45, 60, 90, 120, 150, 180, 210, 240, 270, 300, 360, 420, 480, 540, 600, 660,
    720, 780, 840, 900, 960, 1020, 1080, 1140, 1200, 1260, 1320, 1380, 1440, 1500, 1560,
    1620, 1680, 1740, 1800, 1920, 2040, 2160, 2280, 2400, 2520, 2640, 2760, 2880, 3000,
    3120, 3240, 3360, 3480, 3600, 3900, 4200, 4500, 4800, 5220, 5520, 5820, 6120, 6420,
    6720, 7020, 7200, 10800, 14400, 18000, 21600, 25200, 28800, 32400, 36000], dtype=float)

DEFAULT_GRID = np.array([1, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 300, 420, 600, 720,
                         900, 1200, 1800, 2400, 3600, 4800, 5400, 7200], dtype=float)


def mmp(power, durations=DEFAULT_GRID) -> np.ndarray:
    """Exact MMP of a 1 Hz power series via cumulative sums.

    ``power`` must already be on a uniform 1 s grid (see :func:`resample_1hz`).
    Durations longer than the series give NaN. Cost O(N * len(durations)).
    """
    x = np.nan_to_num(np.asarray(power, float), nan=0.0)
    x = np.clip(x, 0.0, None)
    c = np.concatenate([[0.0], np.cumsum(x)])
    out = np.full(len(durations), np.nan)
    for i, d in enumerate(np.asarray(durations, int)):
        if 0 < d <= len(x):
            out[i] = np.max(c[d:] - c[:-d]) / d
    return out


def monotone_envelope(p) -> np.ndarray:
    """Enforce monotonicity of a mean-maximal curve: P(t1) >= P(t2) for t1 < t2.

    Any window of length t2 with mean P contains a sub-window of length t1 < t2
    whose mean is >= P, so a valid MMP curve is non-increasing. Rounded or
    resampled index files occasionally violate this; we repair with
    env(t) = max_{t' >= t} raw(t'). Durations with no finite value at or beyond
    them stay NaN.
    """
    p = np.asarray(p, float)
    q = np.where(np.isfinite(p), p, -np.inf)
    env = np.maximum.accumulate(q[::-1])[::-1]
    env[~np.isfinite(env)] = np.nan
    return env


def resample_1hz(t_s, power, max_gap_s: float = 10.0):
    """Put irregular (t, P) samples on a 1 s grid.

    Gaps <= ``max_gap_s`` are linearly interpolated (dropouts); longer gaps are
    treated as stopped (P = 0), which is what an auto-pause looks like.
    """
    t = np.asarray(t_s, float); p = np.asarray(power, float)
    ok = np.isfinite(t) & np.isfinite(p)
    t, p = t[ok], p[ok]
    if len(t) == 0:
        return np.zeros(0)
    order = np.argsort(t); t, p = t[order], p[order]
    t = t - t[0]
    grid = np.arange(0, int(np.floor(t[-1])) + 1, dtype=float)
    out = np.interp(grid, t, p)
    gaps = np.diff(t)
    for i in np.where(gaps > max_gap_s)[0]:
        out[(grid > t[i]) & (grid < t[i + 1])] = 0.0
    return out


def normalized_power(power_1hz) -> float:
    """Coggan Normalized Power: 4th-root of the mean of (30 s rolling mean)^4."""
    x = np.nan_to_num(np.asarray(power_1hz, float))
    if len(x) < 30:
        return float(np.mean(x)) if len(x) else 0.0
    c = np.concatenate([[0.0], np.cumsum(x)])
    roll = (c[30:] - c[:-30]) / 30.0
    return float(np.mean(roll ** 4) ** 0.25)


def training_load(duration_s: float, np_w: float, threshold_w: float) -> float:
    """TSS-style load: 100 * duration[h] * IF^2, IF = NP / threshold.

    We use the athlete's fitted CP as the threshold (not a self-reported FTP).
    """
    if threshold_w <= 0:
        raise ValueError("threshold must be > 0")
    return 100.0 * duration_s / 3600.0 * (np_w / threshold_w) ** 2
