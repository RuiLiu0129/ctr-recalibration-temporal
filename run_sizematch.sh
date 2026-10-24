#!/bin/bash
# Size-match chain (2026-10-24): retrain the RQ1 random-split arm at a
# training cap of exactly 3,338,217 rows (origin 1's temporal-arm size),
# 3 models x 3 seeds, via the project's own src.run_experiments random
# path -- the ONLY change vs the existing 4M random runs is the cap
# (passed via --train-cap) plus output locations and --save-predictions.
#
# Same conventions as run_taobao.sh / run_tierf.sh: each step logs to
# sizematch_<name>.log, drops a sizematch_<name>.done marker with the
# exit code, and is SKIPPED on rerun when its marker already reads 0 --
# after a VM/runtime restart, simply relaunching this script resumes
# the chain. A STATUS file (results/sizematch/STATUS.txt) mirrors step
# boundaries with epoch timestamps so progress survives session loss.
# Launch detached:
#   setsid nohup ./run_sizematch.sh >> sizematch_chain.log 2>&1 < /dev/null &
cd "$(dirname "$0")" || exit 1
PY=.venv/bin/python
STATUS=results/sizematch/STATUS.txt
mkdir -p results/sizematch

status() { echo "$1" >> "$STATUS"; }

run() {
  name="$1"; shift
  if [ -f "sizematch_${name}.done" ] && [ "$(cat "sizematch_${name}.done")" = "0" ]; then
    echo "=== $name skipped (already rc=0) $(date -Is) ==="
    status "STEP $name SKIP $(date +%s) $(date -Is)"
    return 0
  fi
  rm -f "sizematch_${name}.done"
  echo "=== $name started $(date -Is) ==="
  status "STEP $name START $(date +%s) $(date -Is)"
  "$@" > "sizematch_${name}.log" 2>&1
  rc=$?
  echo "=== $name finished rc=$rc $(date -Is) ==="
  status "STEP $name END rc=$rc $(date +%s) $(date -Is)"
  echo "$rc" > "sizematch_${name}.done"
}

echo "=== SIZEMATCH chain started $(date -Is) ==="
status "CHAIN START $(date +%s) $(date -Is)"

run fit_s42 $PY -m src.run_experiments --config config.yaml \
  --arm random --train-cap 3338217 \
  --processed-dir data/processed_sizematch_s42 \
  --results-dir results/sizematch/seed42 --save-predictions

run fit_s43 $PY -m src.run_experiments --config config_tierc_rand_s43.yaml \
  --arm random --train-cap 3338217 \
  --processed-dir data/processed_sizematch_s43 \
  --results-dir results/sizematch/seed43 --save-predictions

run fit_s44 $PY -m src.run_experiments --config config_tierc_rand_s44.yaml \
  --arm random --train-cap 3338217 \
  --processed-dir data/processed_sizematch_s44 \
  --results-dir results/sizematch/seed44 --save-predictions

run gates   $PY sizematch_gates.py
run collate $PY sizematch_collate.py

echo "SIZEMATCH CHAIN COMPLETE $(date -Is)"
status "CHAIN COMPLETE $(date +%s) $(date -Is)"
