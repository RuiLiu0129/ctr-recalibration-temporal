#!/bin/bash
# Tier-E chain (2026-10-23): Experiment A (mechanism control at seeds
# 43/44, both origins; mech_slice_seeds.py -- train once per
# (seed, origin), validated against that seed's own sweep runs), then
# Experiment B (recency/staleness control, seed 42, both origins;
# recency_slice.py), then the combined analysis (analyze_tiere.py).
# Sequential (2 CPUs). Same conventions as run_tierd.sh: each step logs
# to tiere_<name>.log, drops a tiere_<name>.done marker with the exit
# code, and is SKIPPED on rerun when its marker already reads 0 -- so
# after a VM/runtime restart, simply relaunching this script resumes
# the chain. Launch detached:
#   setsid nohup ./run_tiere.sh >> tiere_chain.log 2>&1 < /dev/null &
cd "$(dirname "$0")" || exit 1
PY=.venv/bin/python

run() {
  name="$1"; shift
  if [ -f "tiere_${name}.done" ] && [ "$(cat "tiere_${name}.done")" = "0" ]; then
    echo "=== $name skipped (already rc=0) $(date -Is) ==="
    return 0
  fi
  rm -f "tiere_${name}.done"
  echo "=== $name started $(date -Is) ==="
  "$@" > "tiere_${name}.log" 2>&1
  rc=$?
  echo "=== $name finished rc=$rc $(date -Is) ==="
  echo "$rc" > "tiere_${name}.done"
}

echo "=== TIERE chain started $(date -Is) ==="

# --- Experiment A: mechanism control at seeds 43/44 -------------------------
for s in 43 44; do
  for o in 1 2; do
    run mech_s${s}_o${o} $PY mech_slice_seeds.py --seed $s --origin $o \
      --out-dir results/mechanism_seeds
  done
done

# --- Experiment B: recency / staleness control (seed 42) ---------------------
for o in 1 2; do
  run recency_o${o} $PY recency_slice.py --origin $o --out-dir results/recency
done

# --- Analysis ----------------------------------------------------------------
run analyze $PY analyze_tiere.py

echo "TIERE CHAIN COMPLETE $(date -Is)"
