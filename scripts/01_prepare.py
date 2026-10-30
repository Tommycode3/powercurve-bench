"""Stream the index files one athlete at a time -> data/derived/envelopes.csv + daily/*.csv."""
import json
from pathlib import Path

from pdcurve.opendata import prepare

ROOT = Path(__file__).resolve().parents[1]
stats = prepare(ROOT / "data" / "raw", ROOT / "data" / "derived")
(ROOT / "results" / "prepare_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
