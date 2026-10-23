#!/bin/bash
# Tier-C RESUME chain (2026-10-22): the runtime restart at ~18:22 UTC killed
# run_tierc.sh mid rand_seed43. o2_seed43/o2_seed44 finished rc=0 (markers
# present) and are NOT rerun. rand_seed43's partial outputs (processed dir
# held only dataset_index.json; results dir empty) were discarded. This
# script re-runs exactly the remaining steps of run_tierc.sh with identical
# commands, logging and .done-marker conventions; stdout is appended to
# tierc_chain.log so the TIERC CHAIN COMPLETE line lands there.
cd "$(dirname "$0")" || exit 1
PY=.venv/bin/python

run() {
  name="$1"; shift
  if [ -f "tierc_${name}.done" ] && [ "$(cat "tierc_${name}.done")" = "0" ]; then
    echo "=== $name skipped (already rc=0) $(date -Is) ==="
    return 0
  fi
  rm -f "tierc_${name}.done"
  echo "=== $name started $(date -Is) ==="
  "$@" > "tierc_${name}.log" 2>&1
  rc=$?
  echo "=== $name finished rc=$rc $(date -Is) ==="
  echo "$rc" > "tierc_${name}.done"
}

echo "=== RESUME chain started $(date -Is) (restart recovery; o2 seeds already rc=0) ==="

# --- WS3: seeds for the random arm, origin-1 window (fresh seed-dependent
# prepares in their own processed dirs) ---
run rand_seed43 $PY -m src.run_experiments --config config_tierc_rand_s43.yaml --arm random \
  --processed-dir data/processed_rand_seed43 --results-dir results/random_seeds/seed43

run rand_seed44 $PY -m src.run_experiments --config config_tierc_rand_s44.yaml --arm random \
  --processed-dir data/processed_rand_seed44 --results-dir results/random_seeds/seed44

# --- WS4: slice-length sweep, seed 42, both origins (calib_hours 6 already
# exists: o1_repro / origin2, copied into results/sweep/hours_6/) ---
for o in 1 2; do
  for h in 1 3 12 24; do
    run sweep_o${o}_h${h} $PY -m src.run_experiments --config config_sweep_o${o}_h${h}.yaml \
      --arm temporal --processed-dir data/processed_sweep_o${o}_h${h} \
      --results-dir results/sweep/hours_${h}/origin${o}
  done
done

echo "TIERC CHAIN COMPLETE $(date -Is)"
