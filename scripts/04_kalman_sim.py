"""Kalman fitness-fatigue: parameter recovery + interval calibration on simulated data.

1. One detailed run (full grid) -> figures/fig4_kalman_sim.png
2. 100 independent replicates (coarser grid) -> empirical coverage of 95% intervals
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pdcurve.banister import BanisterKF, banister_mean, default_grids, default_prior, ewma_state, simulate_load
from pdcurve.plots import INK, INK2, style

ROOT = Path(__file__).resolve().parents[1]
TRUE = dict(p0=250.0, a1=0.9, a2=1.0, tau1=40.0, tau2=6.0, sigma=5.0)
TRUE["k1"] = TRUE["a1"] * (1 - np.exp(-1 / TRUE["tau1"]))
TRUE["k2"] = TRUE["a2"] * (1 - np.exp(-1 / TRUE["tau2"]))
T, P_OBS = 365, 0.35


def make(seed):
    rng = np.random.default_rng(seed)
    w = simulate_load(T, rng)
    mu = banister_mean(w, TRUE["p0"], TRUE["a1"], TRUE["a2"], TRUE["tau1"], TRUE["tau2"])
    y = np.full(T, np.nan); obs = rng.random(T) < P_OBS
    y[obs] = mu[obs] + rng.normal(0, TRUE["sigma"], obs.sum())
    return w, mu, y


# ---------------- 1. detailed run
w, mu, y = make(1)
kf = BanisterKF(*default_grids(y), default_prior(y))
res = kf.run(w, y)
summ = res.summary()
states = kf.smooth_states(w, y, res)
ok = np.isfinite(y)
detail = {k: {"true": TRUE[k], "median": float(v[1]), "lo95": float(v[0]), "hi95": float(v[2]),
              "covered": bool(v[0] <= TRUE[k] <= v[2])} for k, v in summ.items()}
detail["one_step_PI95_coverage"] = float(np.mean((y[ok] >= res.pred_lo[ok]) & (y[ok] <= res.pred_hi[ok])))
detail["n_obs"] = int(res.n_obs); detail["grid_points"] = int(len(kf.tau1))

style()
days = np.arange(T)
fig, ax = plt.subplots(2, 2, figsize=(13, 7.5))
a = ax[0, 0]
a.bar(days, w, color="#c9c8c2", width=1.0, label="daily load")
a.set_ylabel("load [TSS/day]"); a.set_title("A. simulated periodised load", loc="left")
a = ax[0, 1]
a.fill_between(days, res.pred_lo, res.pred_hi, color="#2a78d6", alpha=0.2, lw=0, label="95% one-step predictive")
a.plot(days, res.pred_mean, color="#2a78d6", lw=1.5, label="one-step predictive mean")
a.plot(days, mu, color=INK2, lw=1, ls="--", label="true noiseless performance")
a.plot(days[ok], y[ok], "o", ms=3, color=INK, label=f"observations (n={ok.sum()})")
a.set_ylabel("performance [W]"); a.legend(fontsize=8, loc="lower right")
a.set_title(f"B. online prediction (coverage {100 * detail['one_step_PI95_coverage']:.0f}% of 95% PI)", loc="left")
a = ax[1, 0]
for key, c, tau, gain in (("fitness", "#2a78d6", TRUE["tau1"], TRUE["a1"]), ("fatigue", "#eb6834", TRUE["tau2"], TRUE["a2"])):
    s = states[key]
    a.fill_between(days, s[:, 0], s[:, 2], color=c, alpha=0.2, lw=0)
    a.plot(days, s[:, 1], color=c, lw=1.8, label=f"{key} (smoothed, 95% band)")
    a.plot(days, gain * ewma_state(w, tau), color=c, lw=1, ls="--")
a.plot([], [], color=INK2, ls="--", lw=1, label="truth")
a.set_ylabel("contribution to performance [W]"); a.set_xlabel("day"); a.legend(fontsize=8)
a.set_title("C. RTS-smoothed fitness and fatigue", loc="left")
a = ax[1, 1]
a.plot(days, res.tau1_mean, color="#2a78d6", label="E[tau1 | y_1:t]")
a.plot(days, res.tau2_mean, color="#eb6834", label="E[tau2 | y_1:t]")
a.axhline(TRUE["tau1"], color="#2a78d6", ls="--", lw=1); a.axhline(TRUE["tau2"], color="#eb6834", ls="--", lw=1)
a.set_ylabel("days"); a.set_xlabel("day"); a.legend(fontsize=8)
a.set_title("D. online posterior mean of the time constants (dashed = truth)", loc="left")
fig.tight_layout(); fig.savefig(ROOT / "figures" / "fig4_kalman_sim.png"); plt.close(fig)

# ---------------- 2. calibration over replicates
R = 100
names = ["p0", "a1", "a2", "k1", "k2", "tau1", "tau2", "sigma"]
cover = {k: 0 for k in names}; relerr = {k: [] for k in names}; pic = []
for r in range(R):
    w, mu, y = make(1000 + r)
    kf = BanisterKF(*default_grids(y), default_prior(y))
    res = kf.run(w, y); s = res.summary()
    for k in names:
        cover[k] += s[k][0] <= TRUE[k] <= s[k][2]
        relerr[k].append((s[k][1] - TRUE[k]) / TRUE[k])
    ok = np.isfinite(y)
    pic.append(np.mean((y[ok] >= res.pred_lo[ok]) & (y[ok] <= res.pred_hi[ok])))
calib = {"replicates": R, "coverage_95": {k: cover[k] / R for k in names},
         "median_rel_error": {k: float(np.median(relerr[k])) for k in names},
         "median_abs_rel_error": {k: float(np.median(np.abs(relerr[k]))) for k in names},
         "one_step_PI95_coverage_mean": float(np.mean(pic))}
out = {"truth": TRUE, "detailed_run": detail, "calibration": calib}
(ROOT / "results" / "kalman_sim.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
