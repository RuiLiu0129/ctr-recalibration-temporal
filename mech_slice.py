"""Tier-D Experiment 2 (2026-10-22): slice SIZE vs diurnal COVERAGE.

Question: in the Tier-C slice-length sweep the 12 h calibration slice
beats the 1 h slice. Is that because the 12 h slice has ~12-26x more
impressions (size), or because it covers more of the day's cycle
(coverage)?

Design (per origin, seed 42 — the sweep's seed): train each of the three
models ONCE on the origin's train split (identical protocol to the
sweep runs; training is invariant to calib_hours — verified: raw bucket
metrics for gaps 1-3 are bit-identical across all sweep hour settings,
and train_X / encoder are md5-identical across the h1/h12 processed
dirs). For every test bucket, take the bucket's 12 h calibration slice
and fit Platt / isotonic on:

  (a) full12  -- the full 12 h slice (the sweep's 12 h condition),
  (b) hour1   -- the actual 1 h slice (the sweep's 1 h condition): the
                 1 h slice is an exact row subset of the 12 h slice
                 (verified elementwise: FIRST n1 rows for gap 0, where
                 the slice is the bucket day's first hours; LAST n1 rows
                 for gaps 1-3, where the slice is the previous day's
                 tail),
  (c) sub1..5 -- K=5 uniform random subsamples of the 12 h slice drawn
                 down to exactly the 1 h slice's impression count n1.

All conditions are evaluated on the SAME fixed test set per bucket: the
12 h run's test split (for gaps 1-3 this is the full bucket day,
identical in the 1 h run; for gap 0 it is hours 12-23 of the bucket day,
whereas the sweep's 1 h run scored hours 1-23 — the single documented
deviation, needed so the three conditions share one evaluation set).

Verdict logic: if (c) ~= (b), coverage is the mechanism (same size, but
spread over 12 h, performs like the contiguous last hour -> the extra
hours' coverage is what the full slice adds); if (c) ~= (a), size is
the mechanism (n1 impressions drawn from anywhere in the 12 h window
perform like the full 12 h slice... i.e. the gain is not about n).
More precisely we report all three means and the (c)-(b) / (a)-(b)
recovery fraction.

Subsample RNG: np.random.default_rng(SUB_SEED_BASE + origin*100000 +
gap*1000 + k), SUB_SEED_BASE = 20261023, k = 1..5. Indices are drawn
once per (origin, gap, k) and shared across models and calibrators
(paired design); every seed is recorded in the output CSV/JSON.

Validation (written to mechanism_o{o}_validation.json): condition (a)
must reproduce the sweep h12 run's per-cell dECE and condition (b) the
sweep h1 run's for gaps 1-3, from results/sweep/hours_{12,1}/origin{o}/
metrics_temporal.csv, to float precision; gap-0 (b) is expected to
differ (different test remainder, see above).
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


def sweep_delta_ece(origin: int, hours: int) -> pd.DataFrame:
    """Per-cell dECE of the original sweep run (seed 42)."""
    df = pd.read_csv(f"results/sweep/hours_{hours}/origin{origin}/metrics_temporal.csv")
    raw = df[df.variant == "raw"].set_index(["model", "bucket"])["ece"]
    out = {}
    for _, r in df[df.variant != "raw"].iterrows():
        out[(r.model, r.bucket, r.variant)] = r.ece - raw.loc[(r.model, r.bucket)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--origin", type=int, choices=(1, 2), required=True)
    ap.add_argument("--out-dir", default="results/mechanism_slice")
    args = ap.parse_args()
    o = args.origin
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = load_config(f"config_sweep_o{o}_h12.yaml")
    seed = int(cfg["seed"])
    assert seed == 42
    cap = int(cfg["data"]["train_cap_rows"])

    # replicate run_experiments.main seeding exactly
    np.random.seed(seed)
    import torch
    torch.set_num_threads(int(cfg.get("train", {}).get("torch_threads", 2)))
    torch.manual_seed(seed)

    root12 = f"data/processed_sweep_o{o}_h12"
    root1 = f"data/processed_sweep_o{o}_h1"
    manifest = prepare_arm("temporal", cfg, root12, cap)  # fingerprint reuse
    encoder = VocabEncoder.load(manifest["encoder_path"])
    arm12 = os.path.join(root12, "temporal")
    arm1 = os.path.join(root1, "temporal")

    gaps = [int(g) for g in cfg["split"]["test_gaps_days"]]
    X_train, y_train = load_split(arm12, "train")
    print(f"[mech o{o}] train rows: {len(y_train):,}", flush=True)

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
        assert eq_X and eq_y, f"1h slice is not the expected subset of the 12h slice (o{o} gap{g})"
        subs = {}
        for k in range(1, K_SUB + 1):
            rng = np.random.default_rng(subsample_seed(o, g, k))
            subs[k] = np.sort(rng.choice(n12, size=n1, replace=False))
        gap_info[g] = {"n12": n12, "n1": n1, "hour_idx": hour_idx, "subs": subs,
                       "subset_X_equal": eq_X, "subset_y_equal": eq_y}
        print(f"[mech o{o}] gap{g}: n12={n12:,} n1={n1:,} subset verified", flush=True)

    sweep12 = sweep_delta_ece(o, 12)
    sweep1 = sweep_delta_ece(o, 1)

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
        print(f"[mech o{o}] {model_name}: trained in {time.time() - t0:.1f}s", flush=True)
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
                    recal = fit_recalibrator(kind, p_slice[idx], yc[idx])
                    ece_cal = expected_calibration_error(
                        yt, recal.transform(p_test), ECE_BINS)
                    d = ece_cal - ece_raw
                    rows.append({
                        "origin": o, "model": model_name, "gap": g,
                        "bucket": f"gap{g}", "calibrator": kind,
                        "condition": cond, "n_fit": int(len(idx)),
                        "n_slice": gi["n12"], "n_test": int(len(yt)),
                        "ece_raw": ece_raw, "ece_cal": ece_cal, "delta_ece": d,
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
            print(f"[mech o{o}] {model_name} gap{g}: raw ECE {ece_raw:.4f} done", flush=True)
        del model

    out_csv = os.path.join(args.out_dir, f"mechanism_o{o}.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    with open(os.path.join(args.out_dir, f"mechanism_o{o}_validation.json"), "w") as f:
        json.dump(val, f, indent=2)
    print(f"[mech o{o}] wrote {out_csv}; validation: "
          f"full12 max|diff|={val['full12_vs_sweep12_max_abs_diff']:.3e}, "
          f"hour1 gaps1-3 max|diff|={val['hour1_vs_sweep1_gaps123_max_abs_diff']:.3e}",
          flush=True)


if __name__ == "__main__":
    main()
