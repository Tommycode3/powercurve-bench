"""GoldenCheetah OpenData (OSF 10.17605/OSF.IO/6HFPZ, CC0-1.0): download + streaming prep.

We only use two *index* files published in the project's ``INDEX`` folder
(~79 MB compressed in total), not the ~116 GB of per-athlete ride archives:

* ``activities.csv.zip``      one row per activity: sport, duration, Normalized Power...
* ``activities_mmp.csv.zip``  one row per activity: mean-maximal power on a 74-point grid

Both are grouped by athlete, so everything below streams one athlete at a time
(peak RAM stays in the tens of MB). The join key is (athlete id, start time).
"""
from __future__ import annotations

import csv
import hashlib
import io
import os
import sqlite3
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .mmp import GC_GRID, monotone_envelope, training_load
from .models import fit

OSF_FILES = {
    # name: (download url, expected bytes)
    "activities.csv.zip": ("https://osf.io/download/4mazr/", 55130196),
    "activities_mmp.csv.zip": ("https://osf.io/download/5unzr/", 24170102),
}
MAX_DOWNLOAD_BYTES = 1_500_000_000   # hard cap requested for this project

BIKE_SPORTS = {"Bike"}
WINDOW_DAYS = 90
MIN_RIDES_IN_WINDOW = 10
KEEP_DURATIONS = GC_GRID[GC_GRID <= 7200]


def download(raw_dir: str | Path, force: bool = False) -> dict:
    raw = Path(raw_dir); raw.mkdir(parents=True, exist_ok=True)
    total = sum(b for _, b in OSF_FILES.values())
    if total > MAX_DOWNLOAD_BYTES:
        raise RuntimeError("download plan exceeds cap")
    info = {}
    for name, (url, nbytes) in OSF_FILES.items():
        dst = raw / name
        if force or not dst.exists() or dst.stat().st_size != nbytes:
            tmp = dst.with_suffix(".part")
            with urllib.request.urlopen(url, timeout=600) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            os.replace(tmp, dst)
        h = hashlib.sha256()
        with open(dst, "rb") as f:
            while chunk := f.read(1 << 20):
                h.update(chunk)
        info[name] = {"bytes": dst.stat().st_size, "sha256": h.hexdigest(), "url": url}
    return info


def _csv_stream(zpath: Path):
    z = zipfile.ZipFile(zpath)
    inner = z.namelist()[0]
    return csv.reader(io.TextIOWrapper(z.open(inner), encoding="utf-8", errors="replace"))


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return np.nan
    return v if v > 0 else np.nan


def _parse_date(s):
    return datetime.strptime(s.replace(" UTC", ""), "%Y/%m/%d %H:%M:%S").replace(tzinfo=timezone.utc)


def build_activity_index(raw_dir: Path, db_path: Path) -> int:
    """activities.csv -> on-disk sqlite (id, date, seconds, NP) for rides with power."""
    if db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(db_path)
    con.execute("CREATE TABLE a (id TEXT, date TEXT, secs REAL, np REAL)")
    r = _csv_stream(raw_dir / "activities.csv.zip")
    h = next(r); ix = {k: i for i, k in enumerate(h)}
    batch, n = [], 0
    for row in r:
        if len(row) < len(h) - 1 or row[ix["sport"]] not in BIKE_SPORTS:
            continue
        secs, npw = _f(row[ix["workout_time"]]), _f(row[ix["coggan_np"]])
        if not (np.isfinite(secs) and np.isfinite(npw)):
            continue
        batch.append((row[ix["id"]], row[ix["date"]], secs, npw)); n += 1
        if len(batch) >= 20000:
            con.executemany("INSERT INTO a VALUES (?,?,?,?)", batch); batch.clear()
    con.executemany("INSERT INTO a VALUES (?,?,?,?)", batch)
    con.execute("CREATE INDEX ia ON a(id)")
    con.commit(); con.close()
    return n


def _iter_athletes_mmp(raw_dir: Path):
    r = _csv_stream(raw_dir / "activities_mmp.csv.zip")
    header = next(r)
    grid = np.array([float(x) for x in header[2:] if x.strip()])
    assert np.allclose(grid, GC_GRID), "unexpected MMP grid in activities_mmp.csv"
    cur, rows = None, []
    for row in r:
        if row[0] != cur:
            if rows:
                yield cur, rows
            cur, rows = row[0], []
        rows.append(row)
    if rows:
        yield cur, rows


def clean_ride(p):
    """Reject rides with power-meter spikes (values a cyclist cannot produce)."""
    if np.nanmax(p) > 2500 or (np.isfinite(p[1]) and p[1] > 2000):   # 1 s > 2500 W, 15 s > 2000 W
        return None
    if np.count_nonzero(np.isfinite(p)) < 5:
        return None
    return p


def best_window(dates, n_min=MIN_RIDES_IN_WINDOW, days=WINDOW_DAYS):
    """End index of the `days`-long window with most rides (latest on ties)."""
    d = np.asarray(dates, float)
    starts = np.searchsorted(d, d - days, side="right")
    counts = np.arange(len(d)) - starts + 1
    if counts.max() < n_min:
        return None
    j = len(d) - 1 - int(np.argmax(counts[::-1]))
    return int(starts[j]), j


def process_athlete(aid, rows, rides):
    """Return (envelope_record | None, daily_rows | None, reason)."""
    recs = []
    for row in rows:
        meta = rides.get(row[1])
        if meta is None:
            continue
        p = np.array([_f(x) for x in row[2:2 + len(GC_GRID)]])
        if len(p) < len(GC_GRID):
            continue
        p = clean_ride(p)
        if p is None:
            continue
        recs.append((_parse_date(row[1]), monotone_envelope(p), meta[0], meta[1]))
    if len(recs) < MIN_RIDES_IN_WINDOW:
        return None, None, "too_few_rides"
    recs.sort(key=lambda r: r[0])
    days = np.array([r[0].timestamp() / 86400.0 for r in recs])
    bw = best_window(days)
    if bw is None:
        return None, None, "no_dense_window"
    i0, i1 = bw
    env = monotone_envelope(np.nanmax(np.stack([r[1] for r in recs[i0:i1 + 1]]), axis=0))
    keep = GC_GRID <= 7200
    env = env[keep]
    gi = {d: i for i, d in enumerate(KEEP_DURATIONS)}
    need = [15, 30, 60, 90, 120, 180, 300, 420, 600, 900, 1200, 1800, 2400, 3600, 4800]
    if not all(np.isfinite(env[gi[d]]) for d in need):
        return None, None, "envelope_incomplete"
    p20 = env[gi[1200]]
    if not (60 <= p20 <= 550) or env[gi[15]] / p20 > 6:
        return None, None, "implausible_envelope"
    # athlete threshold for load: CP of the 2-parameter model on 2-20 min of the envelope
    tt = np.array([120, 180, 300, 420, 600, 900, 1200.0])
    cp = fit("cp2", tt, [env[gi[d]] for d in tt]).params["CP"]
    record = {"athlete": aid, "n_rides_window": i1 - i0 + 1, "cp_for_load": round(cp, 2),
              **{f"p{int(d)}": round(float(v), 2) for d, v in zip(KEEP_DURATIONS, env)}}
    # daily series over the athlete's whole history (load + best 5-min power that day)
    daily = {}
    i300 = int(np.where(GC_GRID == 300)[0][0])
    for (dt, p, secs, npw) in recs:
        key = dt.date().isoformat()
        L, P5, n = daily.get(key, (0.0, np.nan, 0))
        daily[key] = (L + training_load(secs, npw, cp), np.fmax(P5, p[i300]), n + 1)
    return record, daily, "ok"


def prepare(raw_dir: str | Path, out_dir: str | Path, log=print) -> dict:
    raw, out = Path(raw_dir), Path(out_dir)
    (out / "tmp").mkdir(parents=True, exist_ok=True); (out / "daily").mkdir(exist_ok=True)
    db = out / "tmp" / "bike_activities.sqlite"
    n = build_activity_index(raw, db)
    log(f"indexed {n} bike activities with power")
    con = sqlite3.connect(db)
    stats, records = {}, []
    for aid, rows in _iter_athletes_mmp(raw):
        rides = {d: (s, p) for d, s, p in con.execute("SELECT date, secs, np FROM a WHERE id=?", (aid,))}
        if not rides:
            stats["no_bike_power"] = stats.get("no_bike_power", 0) + 1
            continue
        rec, daily, why = process_athlete(aid, rows, rides)
        stats[why] = stats.get(why, 0) + 1
        if rec is None:
            continue
        records.append(rec)
        with open(out / "daily" / f"{aid}.csv", "w") as f:
            f.write("date,load,p300,n_rides\n")
            for k in sorted(daily):
                L, P5, nr = daily[k]
                f.write(f"{k},{L:.2f},{'' if not np.isfinite(P5) else round(P5, 1)},{nr}\n")
    con.close()
    db.unlink()
    cols = list(records[0].keys())
    with open(out / "envelopes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(records)
    log(f"athletes: {stats}")
    return stats


def load_envelopes(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    dur = np.array([float(k[1:]) for k in rows[0] if k.startswith("p") and k[1:].isdigit()])
    P = np.array([[float(r[f"p{int(d)}"]) if r[f"p{int(d)}"] not in ("", "nan") else np.nan
                   for d in dur] for r in rows])
    ids = [r["athlete"] for r in rows]
    return ids, dur, P
