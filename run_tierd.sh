#!/bin/bash
# Tier-D chain (2026-10-22): Experiment 2 (mechanism: slice size vs
# coverage, seed 42, both origins, models trained once per origin), then
# Experiment 1 (slice-length sweep at seeds 43/44: calib_hours {1,3,12,24}
# x origins {1,2} x seeds {43,44}; hours_6 already exists from Tier-B/C).
# Sequential (2 CPUs). Same conventions as run_tierc.sh /
# run_tierc_resume.sh: each step logs to tierd_<name>.log, drops a
# tierd_<name>.done marker with the exit code, and is SKIPPED on rerun
# when its marker already reads 0 -- so after a VM/runtime restart,
# simply relaunching this script resumes the chain. Launch detached:
#   setsid nohup ./run_tierd.sh >> tierd_chain.log 2>&1 < /dev/null &
cd "$(dirname "$0")" || exit 1
PY=.venv/bin/python

run() {
  name="$1"; shift
  if [ -f "tierd_${name}.done" ] && [ "$(cat "tierd_${name}.done")" = "0" ]; then
    echo "=== $name skipped (already rc=0) $(date -Is) ==="
    return 0
  fi
  rm -f "tierd_${name}.done"
  echo "=== $name started $(date -Is) ==="
  "$@" > "tierd_${name}.log" 2>&1
  rc=$?
  echo "=== $name finished rc=$rc $(date -Is) ==="
  echo "$rc" > "tierd_${name}.done"
}

echo "=== TIERD chain started $(date -Is) ==="

# --- staging: processed dirs + configs for the seed-43/44 sweep ----------
run stage $PY stage_tierd.py

# --- Experiment 2: mechanism (seed 42, train once per origin) ------------
run mech_o1 $PY mech_slice.py --origin 1 --out-dir results/mechanism_slice
run mech_o2 $PY mech_slice.py --origin 2 --out-dir results/mechanism_slice

# --- Experiment 1: slice-length sweep at seeds 43/44 ----------------------
for s in 43 44; do
  for o in 1 2; do
    for h in 1 3 12 24; do
      run sweep_s${s}_o${o}_h${h} $PY -m src.run_experiments \
        --config config_sweep_o${o}_h${h}_s${s}.yaml --arm temporal \
        --processed-dir data/processed_sweep_o${o}_h${h}_s${s} \
        --results-dir results/sweep_seeds/seed${s}/hours_${h}/origin${o}
    done
  done
done

echo "TIERD CHAIN COMPLETE $(date -Is)"
