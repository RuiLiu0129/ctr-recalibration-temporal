"""Cost probe for the 'Recalibration Is Not Free' claim.

Times the project's OWN components (src/calibration.py classes/functions) on
the project's REAL saved artifacts:
  - calibration slice scores: results/o1_repro/preds/temporal_lr_gap0_calib_raw.npy
    (687,198 rows, float32 downcast of the run's float64 scores)
  - calibration slice labels: data/processed_o1repro/temporal/calib_gap0_y.npy
  - test bucket scores:      results/o1_repro/preds/temporal_lr_gap0_raw.npy
    (3,148,694 rows)
Secondary slice (gap3, 1,007,461 rows) timed for fits only.

All results/ and data/ access is read-only. Outputs go to cost_probe/.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

sys.path.insert(0, ".venv/lib/python3.12/site-packages")  # no-op safety
sys.path.insert(0, ".")

from src.calibration import (  # noqa: E402
    IsotonicRecalibrator,
    PlattRecalibrator,
    expected_calibration_error,
)

ROOT = "."
N_RUNS = 7


def timeit(fn, n=N_RUNS):
    """Median wall-clock of n runs (1 warm-up run first), plus min/max."""
    fn()  # warm-up
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts = np.array(ts)
    return {"median_s": float(np.median(ts)), "min_s": float(ts.min()),
            "max_s": float(ts.max()), "runs_s": [float(t) for t in ts]}


def main():
    out = {}

    # ---- load gap0 slice (primary) ----
    p_slice = np.load(f"{ROOT}/results/o1_repro/preds/temporal_lr_gap0_calib_raw.npy")
    y_slice = np.asarray(np.load(f"{ROOT}/data/processed_o1repro/temporal/calib_gap0_y.npy"),
                         dtype=np.float64)
    p_test = np.load(f"{ROOT}/results/o1_repro/preds/temporal_lr_gap0_raw.npy")
    out["slice"] = {"bucket": "lr_gap0", "n_slice": int(len(p_slice)),
                    "n_test": int(len(p_test))}

    # ---- alignment validation vs slice_stats_temporal.csv (lr, gap0) ----
    ece_check = expected_calibration_error(y_slice, p_slice.astype(np.float64), 15)
    out["validation"] = {
        "obs_ctr_from_labels": float(y_slice.mean()),
        "obs_ctr_slice_stats": 0.19388444087439136,
        "ece_from_saved_preds": float(ece_check),
        "ece_slice_stats": 0.013024564403363064,
        "note": "preds were saved as float32 downcasts of float64 run scores, "
                "so ECE may differ from the logged value in later decimals",
    }

    # ---- 1) Platt fit on the slice ----
    out["platt_fit_gap0"] = timeit(lambda: PlattRecalibrator().fit(p_slice, y_slice))

    # ---- 2) Isotonic fit on the slice ----
    out["isotonic_fit_gap0"] = timeit(lambda: IsotonicRecalibrator().fit(p_slice, y_slice))

    # ---- 3) ECE evaluation pass (the gate's check): ECE + signed bias ----
    def ece_pass():
        e = expected_calibration_error(y_slice, p_slice.astype(np.float64), 15)
        b = float(np.clip(p_slice, 1e-7, 1 - 1e-7).mean() - y_slice.mean())
        return e, b
    out["ece_check_gap0"] = timeit(ece_pass)

    # ---- 4) Apply on the full test bucket (fit once, time transform) ----
    platt = PlattRecalibrator().fit(p_slice, y_slice)
    iso = IsotonicRecalibrator().fit(p_slice, y_slice)
    out["platt_apply_gap0"] = timeit(lambda: platt.transform(p_test))
    out["isotonic_apply_gap0"] = timeit(lambda: iso.transform(p_test))

    # ---- secondary: fits on the gap3 slice (1,007,461 rows) ----
    p3 = np.load(f"{ROOT}/results/o1_repro/preds/temporal_lr_gap3_calib_raw.npy")
    y3 = np.asarray(np.load(f"{ROOT}/data/processed_o1repro/temporal/calib_gap3_y.npy"),
                     dtype=np.float64)
    out["slice_gap3"] = {"n_slice": int(len(p3))}
    out["platt_fit_gap3"] = timeit(lambda: PlattRecalibrator().fit(p3, y3))
    out["isotonic_fit_gap3"] = timeit(lambda: IsotonicRecalibrator().fit(p3, y3))
    out["ece_check_gap3"] = timeit(
        lambda: expected_calibration_error(y3, p3.astype(np.float64), 15))

    with open("cost_probe/timings.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
