#!/bin/bash
# Tier-F chain (2026-10-23): Experiment B -- same-target-day
# identification control. For each (seed, origin): tierf_prep.py builds
# the random-draw arm (uniform draw without replacement from the
# origin's 5-day window, exactly the temporal arm's training size,
# encoder fit inside the arm), then tierf_same_day.py trains ONE model
# per cell (lr / deepfm / dcnv2) and evaluates RAW AUC/ECE-15 on the
# same four target test days as the temporal arm (which is reused from
# saved results, never retrained).
#
# Same conventions as run_tiere.sh: each step logs to
# tierf_<name>.log, drops a tierf_<name>.done marker with the exit
# code, and is SKIPPED on rerun when its marker already reads 0 -- so
# after a VM/runtime restart, simply relaunching this script resumes
# the chain. (Experiment A, the bidding sweep, is a standalone 70 s
# script -- tierf_bidsweep.py -- not part of this chain.)
# Launch detached:
#   setsid nohup ./run_tierf.sh >> tierf_chain.log 2>&1 < /dev/null &
cd "$(dirname "$0")" || exit 1
PY=.venv/bin/python

run() {
  name="$1"; shift
  if [ -f "tierf_${name}.done" ] && [ "$(cat "tierf_${name}.done")" = "0" ]; then
    echo "=== $name skipped (already rc=0) $(date -Is) ==="
    return 0
  fi
  rm -f "tierf_${name}.done"
  echo "=== $name started $(date -Is) ==="
  "$@" > "tierf_${name}.log" 2>&1
  rc=$?
  echo "=== $name finished rc=$rc $(date -Is) ==="
  echo "$rc" > "tierf_${name}.done"
}

echo "=== TIERF chain started $(date -Is) ==="

for s in 42 43 44; do
  for o in 1 2; do
    run prep_o${o}_s${s} $PY tierf_prep.py --origin $o --seed $s
    for m in lr deepfm dcnv2; do
      run o${o}_s${s}_${m} $PY tierf_same_day.py \
        --origin $o --seed $s --model $m
    done
  done
done

echo "TIERF CHAIN COMPLETE $(date -Is)"
