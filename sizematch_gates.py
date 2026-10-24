"""Size-match run: sanity gates (asserted, hard-fail).

Checks, per seed (42/43/44), for the random arm retrained at cap
3,338,217 via the project's own src.run_experiments path:

- manifest fingerprint cap == 3,338,217 and seed matches;
- exact split counts: train 3,338,217 / val 476,888 / test 953,777
  (pool 4,768,882 = ceil(cap / 0.7); int() split arithmetic of
  src.data_pipeline._prepare_random);
- labels binary {0,1} on every split; split CTR within +/-0.5 pp of
  the Avazu overall CTR 16.98%;
- saved test predictions (raw/platt/isotonic) and val (calib) raw
  predictions: exact lengths, finite, within [0, 1];
- metrics_random.csv present with 9 rows (3 models x 3 variants);
- raw metrics recomputed from the saved float32 test predictions +
  labels reproduce metrics_random.csv within 1e-4 (float32 tolerance).

Writes results/sizematch/gates_report.json; exit code 1 on any failure.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.train_eval import evaluate_predictions  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
SEEDS = (42, 43, 44)
MODELS = ("lr", "deepfm", "dcnv2")
EXPECTED = {"train": 3_338_217, "val": 476_888, "test": 953_777}
OVERALL_CTR = 0.1698
CTR_TOL = 0.005
ECE_BINS = 15


def main() -> None:
    report = {"seeds": {}, "failures": []}

    def fail(msg: str) -> None:
        report["failures"].append(msg)
        print(f"GATE FAIL: {msg}", flush=True)

    for seed in SEEDS:
        arm_dir = os.path.join(ROOT, "data", f"processed_sizematch_s{seed}", "random")
        res_dir = os.path.join(ROOT, "results", "sizematch", f"seed{seed}")
        srep: dict = {}
        with open(os.path.join(arm_dir, "manifest.json")) as f:
            manifest = json.load(f)
        fp = manifest["fingerprint"]
        if fp["cap"] != 3_338_217:
            fail(f"seed {seed}: fingerprint cap {fp['cap']} != 3338217")
        if fp["seed"] != seed:
            fail(f"seed {seed}: fingerprint seed {fp['seed']} != {seed}")
        srep["total_vocab"] = manifest["total_vocab"]
        srep["field_dims"] = manifest["field_dims"]

        # -- split counts + label sanity -----------------------------------
        ys = {}
        for name, want in EXPECTED.items():
            got = manifest["splits"][name]["n"]
            if got != want:
                fail(f"seed {seed}: split {name} n={got:,} != {want:,}")
            y = np.load(os.path.join(arm_dir, f"{name}_y.npy"))
            if len(y) != want:
                fail(f"seed {seed}: {name}_y length {len(y):,} != {want:,}")
            vals = np.unique(y)
            if not set(vals.tolist()) <= {0, 1}:
                fail(f"seed {seed}: {name}_y not binary: {vals[:5]}")
            ctr = float(y.mean())
            if abs(ctr - OVERALL_CTR) > CTR_TOL:
                fail(f"seed {seed}: {name} CTR {ctr:.4f} outside "
                     f"{OVERALL_CTR} +/- {CTR_TOL}")
            srep[f"{name}_n"] = int(len(y))
            srep[f"{name}_ctr"] = ctr
            ys[name] = y

        # -- metrics file shape --------------------------------------------
        metrics = pd.read_csv(os.path.join(res_dir, "metrics_random.csv"))
        if len(metrics) != 9:
            fail(f"seed {seed}: metrics_random.csv has {len(metrics)} rows != 9")

        # -- saved predictions ----------------------------------------------
        preds_dir = os.path.join(res_dir, "preds")
        max_dev = 0.0
        for model in MODELS:
            for variant in ("raw", "platt", "isotonic"):
                p = np.load(os.path.join(
                    preds_dir, f"random_{model}_test_{variant}.npy"))
                if len(p) != EXPECTED["test"]:
                    fail(f"seed {seed} {model} {variant}: test preds "
                         f"{len(p):,} != {EXPECTED['test']:,}")
                if not np.isfinite(p).all():
                    fail(f"seed {seed} {model} {variant}: non-finite preds")
                if p.min() < 0.0 or p.max() > 1.0:
                    fail(f"seed {seed} {model} {variant}: preds outside [0,1]")
            p_val = np.load(os.path.join(
                preds_dir, f"random_{model}_test_calib_raw.npy"))
            if len(p_val) != EXPECTED["val"]:
                fail(f"seed {seed} {model}: val preds {len(p_val):,} "
                     f"!= {EXPECTED['val']:,}")
            if not np.isfinite(p_val).all():
                fail(f"seed {seed} {model}: non-finite val preds")

            # recompute raw test metrics from saved preds
            p_raw = np.load(os.path.join(
                preds_dir, f"random_{model}_test_raw.npy"))
            m = evaluate_predictions(ys["test"].astype(np.int64),
                                     p_raw.astype(np.float64), ECE_BINS)
            row = metrics[(metrics.model == model)
                          & (metrics.variant == "raw")].iloc[0]
            for k in ("auc", "logloss", "ece"):
                dev = abs(m[k] - float(row[k]))
                max_dev = max(max_dev, dev)
                if dev > 1e-4:
                    fail(f"seed {seed} {model}: recomputed {k} {m[k]:.6f} "
                         f"vs csv {float(row[k]):.6f} (dev {dev:.2e})")
        srep["max_recompute_dev"] = max_dev
        report["seeds"][str(seed)] = srep
        print(f"seed {seed}: gates checked "
              f"(vocab {manifest['total_vocab']:,}, "
              f"max recompute dev {max_dev:.2e})", flush=True)

    out = os.path.join(ROOT, "results", "sizematch", "gates_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    if report["failures"]:
        print(f"{len(report['failures'])} GATE FAILURE(S); see {out}", flush=True)
        sys.exit(1)
    print(f"ALL GATES PASSED; report at {out}", flush=True)


if __name__ == "__main__":
    main()
