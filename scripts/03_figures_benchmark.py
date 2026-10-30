"""Benchmark figures + paired tests from results/*.csv (no refitting except the example athlete)."""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from scipy.stats import wilcoxon

from pdcurve.benchmark import PROTOCOLS
from pdcurve.models import MODELS, fit
from pdcurve.opendata import load_envelopes
from pdcurve.plots import COLORS, INK2, fmt_dur, plot_fits, style

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"; FIG.mkdir(exist_ok=True)
style()
ids, dur, P = load_envelopes(ROOT / "data" / "derived" / "envelopes.csv")

err = defaultdict(list)                       # (prot, test_s, model) -> errors
per_ath = defaultdict(lambda: defaultdict(list))  # (prot, model) -> athlete -> |e|
for r in csv.DictReader(open(RES / "per_athlete_errors.csv")):
    e = float(r["rel_err"])
    err[(r["protocol"], int(r["test_s"]), r["model"])].append(e)
    per_ath[(r["protocol"], r["model"])][r["athlete"]].append(abs(e))
params = defaultdict(dict)
for r in csv.DictReader(open(RES / "fit_params.csv")):
    if r["protocol"] == "P1_extrapolate":
        params[r["model"]][r["athlete"]] = [float(x) for x in (r["p1"], r["p2"], r["p3"], r["p4"]) if x != ""]

# ---------------- fig 1: example athlete (median CP2 W') with both protocols
wp = {a: v[1] for a, v in params["cp2"].items()}
med = sorted(wp, key=wp.get)[len(wp) // 2]
i = ids.index(med); p = P[i]
fig, axes = plt.subplots(2, 2, figsize=(12, 6.4), sharex=True, gridspec_kw={"height_ratios": [3, 1.4]})
for col, prot in enumerate(PROTOCOLS):
    spec = PROTOCOLS[prot]
    tr = np.array(spec["train"], float); te = np.array(spec["test"], float)
    fits = {m: fit(m, tr, np.interp(tr, dur, p)) for m in MODELS}
    plot_fits(dur, p, fits, train=tr, test=te, title=f"{prot}: {spec['desc']}",
              ax_top=axes[0, col], ax_bot=axes[1, col])
fig.suptitle("One GoldenCheetah OpenData athlete (median W'), 90-day MMP envelope", x=0.01, ha="left",
             fontsize=11, color=INK2)
fig.savefig(FIG / "fig1_example_fits.png"); plt.close(fig)

# ---------------- fig 2: out-of-sample signed error distributions
fig, axes = plt.subplots(2, 1, figsize=(12, 7.5))
for ax, prot in zip(axes, PROTOCOLS):
    tests = PROTOCOLS[prot]["test"]; k = len(MODELS); wdt = 0.8 / k
    for j, m in enumerate(MODELS):
        data = [100 * np.array(err[(prot, d, m)]) for d in tests]
        pos = np.arange(len(tests)) + (j - (k - 1) / 2) * wdt
        bp = ax.boxplot(data, positions=pos, widths=wdt * 0.85, whis=(5, 95), showfliers=False,
                        patch_artist=True, medianprops=dict(color="#0b0b0b", lw=1.2))
        for b in bp["boxes"]:
            b.set(facecolor=COLORS[m], edgecolor=COLORS[m], alpha=0.85)
        for el in bp["whiskers"] + bp["caps"]:
            el.set(color=COLORS[m], lw=1.2)
        ax.plot([], [], "s", color=COLORS[m], label=MODELS[m].label)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(range(len(tests))); ax.set_xticklabels([fmt_dur(d) for d in tests])
    ax.set_ylabel("(predicted - observed) / observed [%]")
    ax.set_title(f"{prot}: {PROTOCOLS[prot]['desc']}  (box = IQR, whiskers = 5-95th pct, N = {len(ids)})", loc="left")
    ax.legend(ncol=5, fontsize=8, loc="upper right")
axes[0].set_ylim(-60, 80); axes[1].set_ylim(-30, 30)
fig.tight_layout(); fig.savefig(FIG / "fig2_oos_error.png"); plt.close(fig)

# ---------------- fig 3: CP / W' distributions (P1 = classic 2-20 min testing window)
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
bins_cp = np.linspace(80, 420, 35); bins_w = np.linspace(0, 60, 31)
for m in ("cp2", "cp3", "omni"):
    cp = np.array([v[0] for v in params[m].values()]); w = np.array([v[1] for v in params[m].values()]) / 1000
    axes[0].hist(cp, bins_cp, histtype="step", lw=2, color=COLORS[m], label=f"{MODELS[m].label} (median {np.median(cp):.0f} W)")
    axes[1].hist(np.clip(w, 0, 60), bins_w, histtype="step", lw=2, color=COLORS[m],
                 label=f"{MODELS[m].label} (median {np.median(w):.1f} kJ)")
axes[0].set_xlabel("CP [W]"); axes[0].set_ylabel("athletes"); axes[0].legend(fontsize=8)
axes[1].set_xlabel("W' [kJ] (clipped at 60)"); axes[1].legend(fontsize=8)
a = sorted(set(params["cp2"]) & set(params["cp3"]))
x = np.array([params["cp2"][k][0] for k in a]); y = np.array([params["cp3"][k][0] for k in a])
axes[2].plot(x, y, "o", ms=3, alpha=0.5, color=COLORS["cp3"], mec="none")
lim = [min(x.min(), y.min()), max(x.max(), y.max())]
axes[2].plot(lim, lim, color=INK2, lw=0.8)
axes[2].set_xlabel("CP, 2-param [W]"); axes[2].set_ylabel("CP, 3-param [W]")
axes[2].set_title(f"same data, different model: median diff {np.median(y - x):+.1f} W", loc="left", fontsize=10)
fig.suptitle("Parameter estimates from the 2-20 min window (P1), N = %d" % len(ids), x=0.01, ha="left")
fig.tight_layout(); fig.savefig(FIG / "fig3_cp_wprime.png"); plt.close(fig)

# ---------------- paired tests on per-athlete pooled APE
lines = ["# Paired comparisons (per-athlete mean APE over the protocol's test durations)\n",
         "Wilcoxon signed-rank, two-sided; Holm correction within each protocol.\n"]
for prot in PROTOCOLS:
    mape = {m: np.array([np.mean(per_ath[(prot, m)][a]) for a in ids]) for m in MODELS}
    best = min(MODELS, key=lambda m: mape[m].mean())
    others = [m for m in MODELS if m != best]
    ps = [wilcoxon(mape[best], mape[m]).pvalue for m in others]
    order = np.argsort(ps); adj = np.empty(len(ps)); run = 0
    for r, j in enumerate(order):
        run = max(run, min(1, ps[j] * (len(ps) - r))); adj[j] = run
    lines.append(f"\n## {prot} (best: {MODELS[best].label})\n")
    lines.append("| vs | athletes where best wins | median paired diff (pp) | Holm p |")
    lines.append("|---|---|---|---|")
    for m, pa in zip(others, adj):
        d = 100 * (mape[m] - mape[best])
        lines.append(f"| {MODELS[m].label} | {np.mean(d > 0) * 100:.0f}% | {np.median(d):+.2f} | {pa:.1e} |")
(RES / "paired_tests.md").write_text("\n".join(lines) + "\n")
print((RES / "paired_tests.md").read_text())
print("example athlete:", med)
