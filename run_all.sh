#!/usr/bin/env bash
# Entry point: check the dataset is present, then run the experiment suite
# (Plan-A streaming pipeline; see README.md).
# Usage:
#   ./run_all.sh            full run using config.yaml (train cap 4M rows/arm)
#   ./run_all.sh --quick    smoke run (200k train rows/arm, 1 epoch)
# Extra args are forwarded to src.run_experiments (e.g. --train-cap,
# --results-dir, --resume, --force-prepare).
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f data/raw/train.csv ] && [ ! -f data/raw/train.csv.gz ] && [ ! -f data/raw/train.gz ]; then
  echo "ERROR: Avazu training file not found at data/raw/train.csv" >&2
  echo "Download it first (see README.md, 'Dataset' section), e.g.:" >&2
  echo "  kaggle competitions download -c avazu-ctr-prediction -p data/raw" >&2
  echo "  # then unzip/gunzip so that data/raw/train.csv exists" >&2
  exit 1
fi

PY="python3"
if [ -x .venv/bin/python ]; then PY=".venv/bin/python"; fi
exec "$PY" -m src.run_experiments --config config.yaml "$@"
