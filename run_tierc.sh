#!/bin/bash
# Tier-C run chain (2026-10-22): origin-2 seed replicates, random-arm seed
# replicates (origin-1 window), then the slice-length sweep on both origins.
# Sequential (2 CPUs). Each step logs to tierc_<name>.log and drops a
# tierc_<name>.done marker with the exit code. Launched detached (setsid).
cd "$(dirname "$0")" || exit 1
PY=.venv/bin/python

run() {
  name="$1"; shift
  echo "=== $name started $(date -Is) ==="
  "$@" > "tierc_${name}.log" 2>&1
  rc=$?
  echo "=== $name finished rc=$rc $(date -Is) ==="
  echo "$rc" > "tierc_${name}.done"
}

# --- WS3: seeds for origin 2 (staged splits: copied from data/processed_o2,
# manifest fingerprint seed patched + md5-verified; temporal prepare is
# seed-independent) ---
run o2_seed43 $PY -m src.run_experiments --config config_tierc_o2_s43.yaml --arm temporal \
  --processed-dir data/processed_o2_seed43 --results-dir results/origin2_seeds/seed43

run o2_seed44 $PY -m src.run_experiments --config config_tierc_o2_s44.yaml --arm temporal \
  --processed-dir data/processed_o2_seed44 --results-dir results/origin2_seeds/seed44

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
