"""Pick one OpenData athlete-season for the Kalman demo -> data/demo/athlete_season.csv.

Criterion (causal, no look-ahead in the *values*): a day is a "performance test" if its
best 5-min power >= 90% of the trailing 42-day max of 5-min power (incl. that day).
We pick the 365-day window with >= 150 riding days that has the most such days.
"""
import csv
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "derived" / "daily"


def series(path):
    rows = list(csv.DictReader(open(path)))
    d0 = date.fromisoformat(rows[0]["date"]); T = (date.fromisoformat(rows[-1]["date"]) - d0).days + 1
    load = np.zeros(T); p5 = np.full(T, np.nan); rode = np.zeros(T, bool)
    for r in rows:
        i = (date.fromisoformat(r["date"]) - d0).days
        load[i] = float(r["load"]); rode[i] = True
        if r["p300"]:
            p5[i] = float(r["p300"])
    return d0, load, p5, rode


def hard_days(p5, frac=0.9, win=42):
    out = np.full(len(p5), np.nan)
    for t in range(len(p5)):
        if np.isfinite(p5[t]):
            m = np.nanmax(p5[max(0, t - win + 1):t + 1])
            if p5[t] >= frac * m:
                out[t] = p5[t]
    return out


best = None
for f in sorted(D.glob("*.csv")):
    d0, load, p5, rode = series(f)
    if len(load) < 365:
        continue
    perf = hard_days(p5)
    for s in range(0, len(load) - 364, 7):
        sl = slice(s, s + 365)
        if rode[sl].sum() < 150 or np.nanmax(load[sl]) > 1000:
            continue
        n = int(np.isfinite(perf[sl]).sum())
        if best is None or n > best[0]:
            best = (n, f.stem, d0 + timedelta(days=s), load[sl], perf[sl], int(rode[sl].sum()))
n, aid, start, load, perf, nr = best
with open(ROOT / "data" / "demo" / "athlete_season.csv", "w") as f:
    f.write(f"# GoldenCheetah OpenData (CC0) athlete {aid}; load = 100*h*(NP/CP)^2; perf = best 5-min power on test days\n")
    f.write("date,load,perf\n")
    for i in range(365):
        f.write(f"{start + timedelta(days=i)},{load[i]:.1f},{'' if not np.isfinite(perf[i]) else perf[i]}\n")
print(f"athlete {aid}: {n} test days, {nr} riding days, from {start}")
