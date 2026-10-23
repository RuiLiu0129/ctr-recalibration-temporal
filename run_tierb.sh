#!/bin/bash
# Tier-B run chain (2026-10-22): origin 2, origin-1 seed-42 instrumented
# reproduction, origin-1 seed replicates 43/44. Sequential (2 CPUs).
# Each step logs to tierb_<name>.log and drops a tierb_<name>.done marker.
cd "$(dirname "$0")" || exit 1
PY=.venv/bin/python

run() {
  name="$1"; shift
  echo "=== $name started $(date -Is) ==="
  "$@" > "tierb_${name}.log" 2>&1
  rc=$?
  echo "=== $name finished rc=$rc $(date -Is) ==="
  echo "$rc" > "tierb_${name}.done"
}

run o2 $PY -m src.run_experiments --config config_tierb_o2.yaml --arm temporal \
  --processed-dir data/processed_o2 --results-dir results/origin2 --save-predictions

run o1repro $PY -m src.run_experiments --config config.yaml --arm temporal \
  --processed-dir data/processed_o1repro --results-dir results/o1_repro --save-predictions

run seed43 $PY -m src.run_experiments --config config_tierb_s43.yaml --arm temporal \
  --processed-dir data/processed_seed43 --results-dir results/seeds/seed43

run seed44 $PY -m src.run_experiments --config config_tierb_s44.yaml --arm temporal \
  --processed-dir data/processed_seed44 --results-dir results/seeds/seed44

echo "TIERB CHAIN COMPLETE $(date -Is)"
