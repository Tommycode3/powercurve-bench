"""Kalman fitness-fatigue on one real OpenData athlete-season (data/demo/athlete_season.csv).

Evaluations (all causal):
  * online one-step-ahead prediction over the season vs two naive baselines
  * hold-out forecast: filter days 1-270, forecast days 271-365 given the realised load
"""
import csv
import json
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pdcurve.banister import BanisterKF, default_grids, default_prior
from pdcurve.plots import INK, INK2, style

ROOT = Path(__file__).resolve().parents[1]
rows = list(csv.DictReader(l for l in open(ROOT / "data" / "demo" / "athlete_season.csv") if not l.startswith("#")))
days = np.array([date.fromisoformat(r["date"]) for r in rows])
load = np.array([float(r["load"]) for r in rows])
y = np.array([float(r["perf"]) if r["perf"] else np.nan for r in rows])
T = len(y); BURN = 30; SPLIT = 270
obs = np.where(np.isfinite(y))[0]
ev = obs[obs >= BURN]


def naive(y, t, k):
    prev = np.where(np.isfinite(y[:t]))[0]
    return np.mean(y[prev[-k:]]) if len(prev) else np.nan


out = {"n_days": T, "n_obs": int(len(obs)), "burn_in_days": BURN}
variants = {"static": (0.0, 0.0, 0.0), "p0_random_walk": (1.0, 0.0, 0.0)}
results = {}
for name, q in variants.items():
    kf = BanisterKF(*default_grids(y), default_prior(y), q=q)
    res = kf.run(load, y)
    e = y[ev] - res.pred_mean[ev]
    cov = np.mean((y[ev] >= res.pred_lo[ev]) & (y[ev] <= res.pred_hi[ev]))
    s = res.summary()
    out[name] = {"q_sd_per_day": q, "one_step_MAE_W": float(np.mean(np.abs(e))),
                 "one_step_PI95_coverage": float(cov),
                 "posterior": {k: [float(x) for x in v] for k, v in s.items()}}
    results[name] = (kf, res)
for k in (1, 5):
    nb = np.array([naive(y, t, k) for t in ev])
    out[f"naive_last{k}_MAE_W"] = float(np.mean(np.abs(y[ev] - nb)))

# hold-out forecast with the static model
kf_s = BanisterKF(*default_grids(y[:SPLIT]), default_prior(y[:SPLIT]))
res_s = kf_s.run(load[:SPLIT], y[:SPLIT])
fc = kf_s.forecast(res_s, load[:SPLIT], load[SPLIT:])
ho = np.where(np.isfinite(y[SPLIT:]))[0]
fe = y[SPLIT:][ho] - fc[ho, 1]
nb = naive(y, SPLIT, 5)
out["holdout"] = {"train_days": SPLIT, "test_obs": int(len(ho)),
                  "forecast_MAE_W": float(np.mean(np.abs(fe))),
                  "forecast_PI95_coverage": float(np.mean((y[SPLIT:][ho] >= fc[ho, 0]) & (y[SPLIT:][ho] <= fc[ho, 2]))),
                  "naive_last5_MAE_W": float(np.mean(np.abs(y[SPLIT:][ho] - nb)))}
(ROOT / "results" / "kalman_real.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))

# ---------------- figure
kf, res = results["static"]
states = kf.smooth_states(load, y, res)
style()
fig, ax = plt.subplots(3, 1, figsize=(12, 9.5), sharex=True, gridspec_kw={"height_ratios": [1, 2.2, 1.8]})
ax[0].bar(days, load, width=1.0, color="#c9c8c2")
ax[0].set_ylabel("load\n[100 h (NP/CP)^2]"); ax[0].set_title("A. daily training load (computed from power)", loc="left")
a = ax[1]
a.fill_between(days, res.pred_lo, res.pred_hi, color="#2a78d6", alpha=0.18, lw=0, label="95% one-step predictive (online)")
a.plot(days, res.pred_mean, color="#2a78d6", lw=1.4, label="one-step predictive mean")
a.fill_between(days[SPLIT:], fc[:, 0], fc[:, 2], color="#eb6834", alpha=0.18, lw=0,
               label=f"95% forecast from day {SPLIT} (given realised load)")
a.plot(days[SPLIT:], fc[:, 1], color="#eb6834", lw=1.4)
ok = np.isfinite(y)
a.plot(days[ok], y[ok], "o", ms=3.2, color=INK, label="best 5-min power on test days")
a.axvline(days[SPLIT], color=INK2, lw=0.8, ls=":")
a.set_ylabel("5-min power [W]"); a.legend(fontsize=8, loc="lower left", ncol=2)
o = out["static"]; h = out["holdout"]
a.set_title(f"B. online one-step MAE {o['one_step_MAE_W']:.1f} W (naive last-5 mean: {out['naive_last5_MAE_W']:.1f} W), "
            f"PI coverage {100 * o['one_step_PI95_coverage']:.0f}%; hold-out forecast MAE {h['forecast_MAE_W']:.1f} W "
            f"(naive {h['naive_last5_MAE_W']:.1f} W)", loc="left", fontsize=10)
a = ax[2]
for key, c in (("fitness", "#2a78d6"), ("fatigue", "#eb6834")):
    s = states[key]
    a.fill_between(days, s[:, 0], s[:, 2], color=c, alpha=0.2, lw=0)
    a.plot(days, s[:, 1], color=c, lw=1.8, label=f"{key} = a{'1' if key == 'fitness' else '2'} x EWMA(load; tau{'1' if key == 'fitness' else '2'})")
p = o["posterior"]
a.set_title(f"C. RTS-smoothed fitness and fatigue, 95% bands (tau1 {p['tau1'][1]:.0f} d [{p['tau1'][0]:.0f}-{p['tau1'][2]:.0f}], "
            f"tau2 {p['tau2'][1]:.1f} d [{p['tau2'][0]:.1f}-{p['tau2'][2]:.1f}])", loc="left", fontsize=10)
a.set_ylabel("contribution [W]"); a.legend(fontsize=8)
fig.tight_layout(); fig.savefig(ROOT / "figures" / "fig5_kalman_real.png"); plt.close(fig)
