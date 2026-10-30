#!/usr/bin/env bash
# Reproduce every number and figure. Downloads ~79 MB (two OpenData index files).
set -euo pipefail
cd "$(dirname "$0")/.."
export MPLBACKEND=Agg
python scripts/00_download_opendata.py
python scripts/01_prepare.py
python scripts/02_benchmark.py
python scripts/03_figures_benchmark.py
python scripts/04_kalman_sim.py
python scripts/05a_select_demo_athlete.py
python scripts/05_kalman_real.py
