"""Out-of-sample protocols and error summaries."""
from __future__ import annotations

import numpy as np

from .models import MODELS, fit

PROTOCOLS = {
    # classic CP testing window -> extrapolate both ways
    "P1_extrapolate": {
        "train": [120, 180, 240, 300, 420, 600, 720, 900, 1200],
        "test": [30, 1800, 3600],
        "desc": "fit 2-20 min, predict 30 s, 30 min, 60 min",
    },
    # sparse points across the whole curve -> interpolate in between
    "P2_interpolate": {
        "train": [15, 60, 120, 300, 600, 1200, 2400, 4800],
        "test": [30, 90, 180, 420, 900, 1800, 3600],
        "desc": "fit 8 durations spanning 15 s-80 min, predict 7 durations in between",
    },
}


def evaluate_athlete(dur, p, protocol, models=tuple(MODELS)):
    """Signed relative errors (pred - obs) / obs at each test duration, per model."""
    spec = PROTOCOLS[protocol]
    idx = {float(d): i for i, d in enumerate(dur)}
    tt = np.array(spec["train"], float); te = np.array(spec["test"], float)
    pt_ = np.array([p[idx[d]] for d in tt]); pe = np.array([p[idx[d]] for d in te])
    out = {}
    for m in models:
        r = fit(m, tt, pt_)
        pred = r.predict(te)
        out[m] = {"err": (pred - pe) / pe, "fit": r}
    return te, out


def summarize(errs, rng=None, n_boot=2000):
    """MAPE / median APE / bias with a percentile bootstrap CI over athletes."""
    e = np.asarray(errs, float)
    e = e[np.isfinite(e)]
    rng = rng or np.random.default_rng(0)
    a = np.abs(e)
    boot = np.array([a[rng.integers(0, len(a), len(a))].mean() for _ in range(n_boot)])
    return {
        "n": int(len(e)),
        "mape": 100 * a.mean(),
        "mape_lo": 100 * np.quantile(boot, 0.025),
        "mape_hi": 100 * np.quantile(boot, 0.975),
        "mdape": 100 * np.median(a),
        "bias": 100 * e.mean(),
        "median_signed": 100 * np.median(e),
        "p90_ape": 100 * np.quantile(a, 0.9),
    }
