"""Tier-E Experiment A (2026-10-23): mechanism control at seeds 43/44.

Exact replication of the Tier-D Experiment 2 design (mech_slice.py, seed
42) at training seeds 43 and 44, both origins. Per bucket, fit Platt /
isotonic on:

  (a) full12  -- the full 12 h calibration slice,
  (b) hour1   -- the actual 1 h slice (exact row subset of the 12 h
                 slice: FIRST n1 rows for gap 0, LAST n1 rows for
                 gaps 1-3; verified elementwise per seed),
  (c) sub1..5 -- K=5 uniform random subsamples of the 12 h slice drawn
                 down to exactly the 1 h slice's impression count n1.

WHY RETRAINING: no per-impression predictions were saved for the seed
43/44 runs (results/seeds/, results/origin2_seeds/, results/sweep_seeds/
contain metrics / reliability / slice-stats CSVs only; .npy preds exist
solely for the seed-42 instrumented runs o1_repro / origin2, base 6 h
protocol). The seed-42 control itself was produced by this same
train-once-per-origin harness (deterministic training; the temporal
prepare is seed-independent, so the staged *_s{seed} processed dirs are
copies of the seed-42 ones), and it validated against the sweep runs to
~1e-13. This script does the identical thing per seed: train once per
(seed, origin) with run_experiments' seeding replicated, and validate
(a)/(b) against THAT seed's own sweep runs
(results/sweep_seeds/seed{seed}/hours_{12,1}/origin{origin}).

Subsample RNG: identical to seed 42 -- np.random.default_rng(20261023 +
origin*100000 + gap*1000 + k), k = 1..5, indices drawn once per
(origin, gap, k) and shared across models and calibrators (paired
design). The data splits are seed-independent, so the drawn index sets
are the same ones the seed-42 control used; only the trained model
differs. Every subsample seed is recorded in the output CSV/JSON.

All conditions are evaluated on the SAME fixed test set per bucket: the
12 h run's test split (identical to seed 42's design, including the
documented gap-0 deviation: the 12 h run scores hours 12-23 of the
bucket day, the 1 h run hours 1-23).
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

from src.calibration import expected_calibration_error, fit_recalibrator
from src.data_pipeline import load_split, prepare_arm
from src.data_prep import VocabEncoder
from src.models import MODEL_NAMES
from src.run_experiments import load_config
from src.train_eval import train_model

SUB_SEED_BASE = 20261023
K_SUB = 5
ECE_BINS = 15


def subsample_seed(origin: int, gap: int, k: int) -> int:
    return SUB_SEED_BASE + origin * 100000 + gap * 1000 + k


def sweep_delta_ece(seed: int, origin: int, hours: int) -> dict:
    """Per-cell dECE of the seed's own sweep run."""
    df = pd.read_csv(
        f"results/sweep_seeds/seed{seed}/hours_{hours}/origin{origin}/"
        "metrics_temporal.csv")
    raw = df[df.variant == "raw"].set_index(["model", "bucket"])["ece"]
    out = {}
    for _, r in df[df.variant != "raw"].iterrows():
        out[(r.model, r.bucket, r.variant)] = r.ece - raw.loc[(r.model, r.bucket)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, choices=(43, 44), required=True)
    ap.add_argument("--origin", type=int, choices=(1, 2), required=True)
    ap.add_argument("--out-dir", default="results/mechanism_seeds")
    args = ap.parse_args()
    seed, o = args.seed, args.origin
    out_dir = os.path.join(args.out_dir, f"seed{seed}")
    os.makedirs(out_dir, exist_ok=True)

    cfg = load_config(f"config_sweep_o{o}_h12_s{seed}.yaml")
    assert int(cfg["seed"]) == seed, f"config seed {cfg['seed']} != {seed}"
    cap = int(cfg["data"]["train_cap_rows"])

    # replicate run_experiments.main seeding exactly
    np.random.seed(seed)
    import torch
    torch.set_num_threads(int(cfg.get("train", {}).get("torch_threads", 2)))
    torch.manual_seed(seed)

    root12 = f"data/processed_sweep_o{o}_h12_s{seed}"
    root1 = f"data/processed_sweep_o{o}_h1_s{seed}"
    manifest = prepare_arm("temporal", cfg, root12, cap)  # fingerprint reuse
    encoder = VocabEncoder.load(manifest["encoder_path"])
    arm12 = os.path.join(root12, "temporal")
    arm1 = os.path.join(root1, "temporal")

    gaps = [int(g) for g in cfg["split"]["test_gaps_days"]]
    X_train, y_train = load_split(arm12, "train")
    print(f"[mech s{seed} o{o}] train rows: {len(y_train):,}", flush=True)

    # ---- per-gap slice bookkeeping + subset verification -------------------
    gap_info: dict[int, dict] = {}
    for g in gaps:
        Xc12, yc12 = load_split(arm12, f"calib_gap{g}")
        Xc1, yc1 = load_split(arm1, f"calib_gap{g}")
        n12, n1 = len(yc12), len(yc1)
        if g == 0:
            hour_idx = np.arange(n1)
        else:
            hour_idx = np.arange(n12 - n1, n12)
        eq_X = bool(np.array_equal(np.asarray(Xc12[hour_idx]), np.asarray(Xc1)))
        eq_y = bool(np.array_equal(np.asarray(yc12[hour_idx]), np.asarray(yc1)))
        assert eq_X and eq_y, f"1h slice is not the expected subset of the 12h slice (s{seed} o{o} gap{g})"
        subs = {}
        for k in range(1, K_SUB + 1):
            rng = np.random.default_rng(subsample_seed(o, g, k))
            subs[k] = np.sort(rng.choice(n12, size=n1, replace=False))
        gap_info[g] = {"n12": n12, "n1": n1, "hour_idx": hour_idx, "subs": subs,
                       "subset_X_equal": eq_X, "subset_y_equal": eq_y}
        print(f"[mech s{seed} o{o}] gap{g}: n12={n12:,} n1={n1:,} subset verified", flush=True)

    sweep12 = sweep_delta_ece(seed, o, 12)
    sweep1 = sweep_delta_ece(seed, o, 1)

    rows = []
    val = {"origin": o, "seed": seed, "sub_seed_base": SUB_SEED_BASE,
           "gaps": {str(g): {"n12": gi["n12"], "n1": gi["n1"],
                             "subset_X_equal": gi["subset_X_equal"],
                             "subset_y_equal": gi["subset_y_equal"],
                             "subsample_seeds": {str(k): subsample_seed(o, g, k)
                                                 for k in range(1, K_SUB + 1)}}
                    for g, gi in gap_info.items()},
           "full12_vs_sweep12_max_abs_diff": 0.0,
           "hour1_vs_sweep1_gaps123_max_abs_diff": 0.0,
           "hour1_gap0_vs_sweep1_diff_by_cell": {}}

    for model_name in MODEL_NAMES:
        t0 = time.time()
        model = train_model(model_name, encoder, X_train, y_train, cfg, seed=seed)
        print(f"[mech s{seed} o{o}] {model_name}: trained in {time.time() - t0:.1f}s", flush=True)
        for g in gaps:
            gi = gap_info[g]
            Xc, yc = load_split(arm12, f"calib_gap{g}")
            Xt, yt = load_split(arm12, f"test_gap{g}")
            yc = np.asarray(yc)
            yt = np.asarray(yt)
            p_slice = model.predict_proba(Xc)
            p_test = model.predict_proba(Xt)
            ece_raw = expected_calibration_error(yt, p_test, ECE_BINS)
            conds: dict[str, np.ndarray] = {"full12": np.arange(gi["n12"]),
                                            "hour1": gi["hour_idx"]}
            for k in range(1, K_SUB + 1):
                conds[f"sub{k}"] = gi["subs"][k]
            for cond, idx in conds.items():
                for kind in ("platt", "isotonic"):
                    # wall-clock fit/apply, cost_probe convention
                    t0 = time.perf_counter()
                    recal = fit_recalibrator(kind, p_slice[idx], yc[idx])
                    fit_s = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    p_cal = recal.transform(p_test)
                    apply_s = time.perf_counter() - t0
                    ece_cal = expected_calibration_error(yt, p_cal, ECE_BINS)
                    d = ece_cal - ece_raw
                    rows.append({
                        "origin": o, "seed": seed, "model": model_name, "gap": g,
                        "bucket": f"gap{g}", "calibrator": kind,
                        "condition": cond, "n_fit": int(len(idx)),
                        "n_slice": gi["n12"], "n_test": int(len(yt)),
                        "ece_raw": ece_raw, "ece_cal": ece_cal, "delta_ece": d,
                        "fit_time_s": fit_s, "apply_time_s": apply_s,
                        "subsample_seed": (subsample_seed(o, g, int(cond[3:]))
                                           if cond.startswith("sub") else None),
                    })
                    # validation bookkeeping
                    key = (model_name, f"gap{g}", kind)
                    if cond == "full12":
                        val["full12_vs_sweep12_max_abs_diff"] = max(
                            val["full12_vs_sweep12_max_abs_diff"],
                            abs(d - sweep12[key]))
                    elif cond == "hour1":
                        if g > 0:
                            val["hour1_vs_sweep1_gaps123_max_abs_diff"] = max(
                                val["hour1_vs_sweep1_gaps123_max_abs_diff"],
                                abs(d - sweep1[key]))
                        else:
                            val["hour1_gap0_vs_sweep1_diff_by_cell"][
                                f"{model_name}/{kind}"] = d - sweep1[key]
            print(f"[mech s{seed} o{o}] {model_name} gap{g}: raw ECE {ece_raw:.4f} done", flush=True)
        del model

    out_csv = os.path.join(out_dir, f"mechanism_o{o}.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    with open(os.path.join(out_dir, f"mechanism_o{o}_validation.json"), "w") as f:
        json.dump(val, f, indent=2)
    print(f"[mech s{seed} o{o}] wrote {out_csv}; validation: "
          f"full12 max|diff|={val['full12_vs_sweep12_max_abs_diff']:.3e}, "
          f"hour1 gaps1-3 max|diff|={val['hour1_vs_sweep1_gaps123_max_abs_diff']:.3e}",
          flush=True)


if __name__ == "__main__":
    main()
