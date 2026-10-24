"""Tier-F Experiment B (2026-10-23): same-target-day identification
control -- ONE random-arm fit (one origin, one seed, one model).

The temporal arm's structure is mirrored exactly: one fit per
(origin, seed, model) on the origin's 5-day history window, evaluated
RAW (pre-calibration) on the same four target test days. The ONLY
difference from the temporal arm is how the training rows are drawn
from the window: a uniform random draw without replacement of exactly
the temporal arm's training size (tierf_prep.py), with the encoder fit
inside the arm -- instead of the deterministic stride sample. The
temporal arm itself is NOT retrained; its saved metrics are loaded
next to ours for the paired comparison.

Usage:
    .venv/bin/python tierf_same_day.py --origin 1 --seed 42 --model lr

Outputs:
    results/tierf/cells/o{origin}_s{seed}_{model}.csv   (4 rows, gaps 0-3,
        with the temporal arm's raw AUC/ECE/logloss alongside)
    results/tierf/preds/o{origin}_s{seed}/tierf_{model}_gap{g}_raw.npy
"""

from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import pandas as pd
import yaml

from src.data_pipeline import load_split
from src.data_prep import VocabEncoder
from src.train_eval import evaluate_predictions, train_model

ROOT = os.path.dirname(os.path.abspath(__file__))
TEMPORAL_METRICS = {
    (1, 42): "results/o1_repro/metrics_temporal.csv",
    (1, 43): "results/seeds/seed43/metrics_temporal.csv",
    (1, 44): "results/seeds/seed44/metrics_temporal.csv",
    (2, 42): "results/origin2/metrics_temporal.csv",
    (2, 43): "results/origin2_seeds/seed43/metrics_temporal.csv",
    (2, 44): "results/origin2_seeds/seed44/metrics_temporal.csv",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--origin", type=int, required=True, choices=(1, 2))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--model", required=True,
                    choices=("lr", "deepfm", "dcnv2"))
    args = ap.parse_args()
    origin, seed, model_name = args.origin, args.seed, args.model

    with open(os.path.join(ROOT, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    cfg["seed"] = seed  # mirror the per-seed runs: config seed = run seed
    ece_bins = int(cfg["calibration"]["ece_bins"])
    gaps = [int(g) for g in cfg["split"]["test_gaps_days"]]

    arm_dir = os.path.join(
        ROOT, "data", "processed_tierf", f"o{origin}_s{seed}", "tierf")
    with open(os.path.join(arm_dir, "manifest.json")) as f:
        manifest = json.load(f)
    encoder = VocabEncoder.load(manifest["encoder_path"])
    X_train, y_train = load_split(arm_dir, "train")
    print(f"[tierf o{origin} s{seed}] train rows: {len(y_train):,} "
          f"(temporal arm: {manifest['splits']['train']['n']:,})",
          flush=True)

    t0 = time.time()
    model = train_model(model_name, encoder, X_train, y_train, cfg,
                        seed=seed)
    train_s = time.time() - t0
    print(f"[tierf o{origin} s{seed}] {model_name}: trained in "
          f"{train_s:.1f}s", flush=True)

    tmet = pd.read_csv(os.path.join(ROOT, TEMPORAL_METRICS[(origin, seed)]))
    tmet = tmet[(tmet["model"] == model_name) & (tmet["variant"] == "raw")]
    tmet = {int(r["bucket"].replace("gap", "")): r
            for _, r in tmet.iterrows()}

    preds_dir = os.path.join(ROOT, "results", "tierf", "preds",
                             f"o{origin}_s{seed}")
    os.makedirs(preds_dir, exist_ok=True)
    rows = []
    for g in gaps:
        X_test, y_test = load_split(arm_dir, f"test_gap{g}")
        t0 = time.time()
        p = model.predict_proba(X_test)
        m = evaluate_predictions(np.asarray(y_test), p, ece_bins)
        np.save(os.path.join(
            preds_dir, f"tierf_{model_name}_gap{g}_raw.npy"),
            p.astype(np.float32))
        t = tmet[g]
        assert int(t["n_test"]) == len(y_test), "test size mismatch"
        rows.append({
            "origin": origin, "seed": seed, "model": model_name,
            "gap": g, "arm": "tierf_random_draw",
            "n_train": int(len(y_train)), "n_test": int(len(y_test)),
            "auc": m["auc"], "logloss": m["logloss"], "ece": m["ece"],
            "temporal_auc": float(t["auc"]),
            "temporal_logloss": float(t["logloss"]),
            "temporal_ece": float(t["ece"]),
            "train_time_s": train_s, "eval_time_s": time.time() - t0,
        })
        print(f"[tierf o{origin} s{seed}] {model_name} gap{g}: "
              f"AUC {m['auc']:.4f} (temporal {float(t['auc']):.4f}) "
              f"ECE {m['ece']:.4f} (temporal {float(t['ece']):.4f})",
              flush=True)
    out_dir = os.path.join(ROOT, "results", "tierf", "cells")
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(
        out_dir, f"o{origin}_s{seed}_{model_name}.csv"), index=False)
    print(f"[tierf o{origin} s{seed}] {model_name}: cell written", flush=True)


if __name__ == "__main__":
    main()
