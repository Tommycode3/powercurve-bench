"""pdcurve command line.

  pdcurve mmp FILE.fit [...]                 MMP envelope of one or more FIT files (CSV to stdout)
  pdcurve fit FILE.fit [...] [--plot x.png]  fit all power-duration models to the envelope
  pdcurve load FILE.fit [...] --cp 250       daily training load (TSS-style, threshold = CP)
  pdcurve banister daily.csv [--plot x.png]  Kalman fitness-fatigue on a date,load,perf CSV
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict

import numpy as np

from .mmp import DEFAULT_GRID, mmp, monotone_envelope, normalized_power, training_load
from .models import MODELS, fit


def _envelope(files, grid):
    from .fitio import read_fit
    curves = []
    for f in files:
        a = read_fit(f)
        curves.append(mmp(a.power_1hz, grid))
        print(f"# {f}: sport={a.sport} duration={a.duration_s}s", file=sys.stderr)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            env = np.nanmax(np.stack(curves), axis=0)
    return monotone_envelope(env)


def _read_mmp_csv(path):
    with open(path) as f:
        rows = [r for r in csv.DictReader(f)]
    return (np.array([float(r["duration_s"]) for r in rows]),
            np.array([float(r["power_w"]) for r in rows]))


def cmd_mmp(a):
    grid = DEFAULT_GRID
    env = _envelope(a.files, grid)
    w = csv.writer(sys.stdout); w.writerow(["duration_s", "power_w"])
    for d, p in zip(grid, env):
        if np.isfinite(p):
            w.writerow([int(d), round(float(p), 1)])


def cmd_fit(a):
    if a.mmp_csv:
        dur, env = _read_mmp_csv(a.mmp_csv)
    else:
        dur = DEFAULT_GRID; env = _envelope(a.files, dur)
    ok = np.isfinite(env) & (dur >= a.min_s) & (dur <= a.max_s)
    t, p = dur[ok], env[ok]
    models = a.models.split(",") if a.models else list(MODELS)
    fits = {}
    print(f"fitting {len(t)} durations in [{t.min():g}, {t.max():g}] s")
    for m in models:
        r = fit(m, t, p); fits[m] = r
        ps = "  ".join(f"{k}={v:.4g}" for k, v in r.params.items())
        flag = f"  [at bound: {','.join(r.at_bound)}]" if r.at_bound else ""
        flag += f"  [fixed: {r.fixed}]" if r.fixed else ""
        print(f"{MODELS[m].label:18s} {ps}   rms rel err={100 * r.rmse_rel:.2f}%{flag}")
    if a.plot:
        from .plots import plot_fits
        plot_fits(dur, env, fits, train=t, title="power-duration fit", path=a.plot)
        print(f"wrote {a.plot}")


def cmd_load(a):
    from .fitio import read_fit
    daily = defaultdict(float)
    for f in a.files:
        act = read_fit(f)
        L = training_load(act.duration_s, normalized_power(act.power_1hz), a.cp)
        key = act.start.date().isoformat() if act.start else f
        daily[key] += L
    w = csv.writer(sys.stdout); w.writerow(["date", "load"])
    for k in sorted(daily):
        w.writerow([k, round(daily[k], 1)])


def cmd_banister(a):
    from datetime import date, timedelta
    from .banister import BanisterKF, default_grids, default_prior

    with open(a.daily) as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    d0 = date.fromisoformat(rows[0]["date"]); d1 = date.fromisoformat(rows[-1]["date"])
    T = (d1 - d0).days + 1
    load = np.zeros(T); y = np.full(T, np.nan)
    for r in rows:
        i = (date.fromisoformat(r["date"]) - d0).days
        load[i] += float(r.get("load") or 0)
        if r.get(a.perf_col) not in (None, ""):
            y[i] = float(r[a.perf_col])
    kf = BanisterKF(*default_grids(y), default_prior(y), q=(a.q_p0, a.q_gain, a.q_gain))
    res = kf.run(load, y)
    print(f"{res.n_obs} performance observations over {T} days")
    for k, (lo, med, hi) in res.summary().items():
        print(f"{k:6s} {med:10.4g}   95% [{lo:.4g}, {hi:.4g}]")
    if a.forecast_days:
        fut = np.full(a.forecast_days, np.mean(load[-28:]))
        fc = kf.forecast(res, load, fut)
        print(f"forecast day +{a.forecast_days} at constant load {fut[0]:.0f}/day: "
              f"{fc[-1, 1]:.1f} [{fc[-1, 0]:.1f}, {fc[-1, 2]:.1f}]")
    if a.plot:
        from .plots import style
        import matplotlib.pyplot as plt
        style()
        days = [d0 + timedelta(days=i) for i in range(T)]
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.fill_between(days, res.pred_lo, res.pred_hi, color="#2a78d6", alpha=0.18, lw=0,
                        label="95% one-step predictive")
        ax.plot(days, res.pred_mean, color="#2a78d6", lw=1.5, label="predictive mean")
        ok = np.isfinite(y)
        ax.plot(np.array(days)[ok], y[ok], "o", ms=3.5, color="#0b0b0b", label="observed")
        ax.legend(fontsize=8); ax.set_ylabel(a.perf_col)
        fig.savefig(a.plot); print(f"wrote {a.plot}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pdcurve", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("mmp", help="MMP envelope of FIT files"); s.add_argument("files", nargs="+")
    s.set_defaults(func=cmd_mmp)
    s = sub.add_parser("fit", help="fit power-duration models")
    s.add_argument("files", nargs="*"); s.add_argument("--mmp-csv", help="duration_s,power_w CSV instead of FIT")
    s.add_argument("--models", help=f"comma list from {','.join(MODELS)}")
    s.add_argument("--min-s", type=float, default=120); s.add_argument("--max-s", type=float, default=1200)
    s.add_argument("--plot"); s.set_defaults(func=cmd_fit)
    s = sub.add_parser("load", help="daily training load from FIT files")
    s.add_argument("files", nargs="+"); s.add_argument("--cp", type=float, required=True)
    s.set_defaults(func=cmd_load)
    s = sub.add_parser("banister", help="Kalman fitness-fatigue model")
    s.add_argument("daily"); s.add_argument("--perf-col", default="perf")
    s.add_argument("--q-p0", type=float, default=0.0); s.add_argument("--q-gain", type=float, default=0.0)
    s.add_argument("--forecast-days", type=int, default=0); s.add_argument("--plot")
    s.set_defaults(func=cmd_banister)
    a = ap.parse_args(argv)
    if a.cmd == "fit" and not a.files and not a.mmp_csv:
        ap.error("fit: give FIT files or --mmp-csv")
    a.func(a)


if __name__ == "__main__":
    main()
