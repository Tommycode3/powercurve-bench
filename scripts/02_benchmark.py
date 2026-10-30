"""Run both out-of-sample protocols on every eligible athlete (one at a time).

Outputs (results/):
  per_athlete_errors.csv   athlete, protocol, model, test_s, obs_w, pred_w, rel_err
  fit_params.csv           athlete, protocol, model, params..., rmse_rel, at_bound
  benchmark_table.csv/.md  per protocol x test duration x model summary (+ pooled)
"""
import csv
import json
from pathlib import Path

import numpy as np

from pdcurve.benchmark import PROTOCOLS, evaluate_athlete, summarize
from pdcurve.models import MODELS
from pdcurve.opendata import load_envelopes

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"; RES.mkdir(exist_ok=True)
ids, dur, P = load_envelopes(ROOT / "data" / "derived" / "envelopes.csv")
idx = {float(d): i for i, d in enumerate(dur)}

ferr = open(RES / "per_athlete_errors.csv", "w", newline="")
we = csv.writer(ferr); we.writerow(["athlete", "protocol", "model", "test_s", "obs_w", "pred_w", "rel_err"])
fpar = open(RES / "fit_params.csv", "w", newline="")
wp = csv.writer(fpar); wp.writerow(["athlete", "protocol", "model", "p1", "p2", "p3", "p4", "rmse_rel", "at_bound", "fixed"])
errs = {}
for a, (aid, p) in enumerate(zip(ids, P)):
    for prot in PROTOCOLS:
        te, out = evaluate_athlete(dur, p, prot)
        for m, o in out.items():
            r = o["fit"]
            th = list(np.round(r.theta, 4)) + [""] * (4 - len(r.theta))
            wp.writerow([aid, prot, m, *th, round(r.rmse_rel, 5), "|".join(r.at_bound),
                         "|".join(f"{k}={v}" for k, v in r.fixed.items())])
            for d, e in zip(te, o["err"]):
                obs = p[idx[d]]
                we.writerow([aid, prot, m, int(d), round(obs, 2), round(obs * (1 + e), 2), round(e, 6)])
                errs.setdefault((prot, int(d), m), []).append(e)
                errs.setdefault((prot, "all", m), []).append(e)
    if a % 100 == 0:
        print(f"{a}/{len(ids)}", flush=True)
ferr.close(); fpar.close()

rows = []
for (prot, d, m), e in errs.items():
    s = summarize(e)
    rows.append({"protocol": prot, "test_s": d, "model": m, **{k: (round(v, 2) if isinstance(v, float) else v) for k, v in s.items()}})
order = {m: i for i, m in enumerate(MODELS)}
rows.sort(key=lambda r: (r["protocol"], 10**9 if r["test_s"] == "all" else r["test_s"], order[r["model"]]))
with open(RES / "benchmark_table.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def fmt_d(d):
    if d == "all":
        return "**pooled**"
    d = int(d)
    return f"{d} s" if d < 60 else f"{d // 60} min" if d % 60 == 0 else f"{d / 60:.1f} min"


md = []
for prot, spec in PROTOCOLS.items():
    md.append(f"### {prot}: {spec['desc']}\n")
    md.append("MAPE % [95% bootstrap CI over athletes] / bias % (mean signed error, + = over-prediction)\n")
    md.append("| test duration | " + " | ".join(MODELS[m].label for m in MODELS) + " |")
    md.append("|---|" + "---|" * len(MODELS))
    for d in spec["test"] + ["all"]:
        cells = []
        rs = {r["model"]: r for r in rows if r["protocol"] == prot and str(r["test_s"]) == str(d)}
        best = min(rs.values(), key=lambda r: r["mape"])["model"]
        for m in MODELS:
            r = rs[m]
            c = f"{r['mape']:.1f} [{r['mape_lo']:.1f}-{r['mape_hi']:.1f}] / {r['bias']:+.1f}"
            cells.append(f"**{c}**" if m == best else c)
        md.append(f"| {fmt_d(d)} | " + " | ".join(cells) + " |")
    md.append("")
n = len(ids)
(RES / "benchmark_table.md").write_text(f"N = {n} athletes (one 90-day MMP envelope each).\n\n" + "\n".join(md))
print((RES / "benchmark_table.md").read_text())
