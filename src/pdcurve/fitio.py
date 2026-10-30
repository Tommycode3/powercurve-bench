"""Read power from FIT files (COROS, Garmin, Wahoo, Stryd...) with fitdecode."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from .mmp import resample_1hz


@dataclass
class Activity:
    path: str
    start: datetime | None
    sport: str
    power_1hz: np.ndarray  # W, uniform 1 s grid

    @property
    def duration_s(self) -> int:
        return int(len(self.power_1hz))


def _record_power(frame):
    """Native ``power`` field first, then any developer field named like power."""
    if frame.has_field("power"):
        v = frame.get_value("power")
        if v is not None:
            return float(v)
    for f in frame.fields:
        name = (f.name or "").lower()
        if "power" in name and f.value is not None and isinstance(f.value, (int, float)):
            if "accumulated" in name or "zone" in name:
                continue
            return float(f.value)
    return None


def read_fit(path: str | Path, max_gap_s: float = 10.0) -> Activity:
    import fitdecode

    ts, pw, sport, start = [], [], "unknown", None
    with fitdecode.FitReader(str(path), check_crc=fitdecode.CrcCheck.WARN) as fr:
        for frame in fr:
            if not isinstance(frame, fitdecode.FitDataMessage):
                continue
            if frame.name == "sport" and frame.has_field("sport"):
                sport = str(frame.get_value("sport"))
            elif frame.name == "session" and frame.has_field("sport") and sport == "unknown":
                sport = str(frame.get_value("sport"))
            elif frame.name == "record" and frame.has_field("timestamp"):
                p = _record_power(frame)
                if p is None:
                    continue
                t = frame.get_value("timestamp")
                if start is None:
                    start = t
                ts.append((t - start).total_seconds() if hasattr(t, "tzinfo") else float(t))
                pw.append(p)
    if not pw:
        raise ValueError(f"{path}: no power samples found")
    return Activity(str(path), start, sport, resample_1hz(ts, pw, max_gap_s))
