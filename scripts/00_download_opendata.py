"""Download the two GoldenCheetah OpenData index files (~79 MB) into data/raw/."""
import json
from pathlib import Path

from pdcurve.opendata import download

ROOT = Path(__file__).resolve().parents[1]
info = download(ROOT / "data" / "raw")
tot = sum(v["bytes"] for v in info.values())
print(json.dumps(info, indent=2))
print(f"total downloaded: {tot / 1e6:.1f} MB")
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results" / "dataset_manifest.json").write_text(json.dumps(
    {"source": "GoldenCheetah OpenData, OSF project 6hfpz, DOI 10.17605/OSF.IO/6HFPZ",
     "license": "CC0-1.0", "files": info, "total_bytes": tot}, indent=2) + "\n")
