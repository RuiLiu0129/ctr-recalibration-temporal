"""Size-match run: collate per-cell metrics + pre-specified contrasts.

Outputs under results/sizematch/:
- sizematch_cells.csv     one row per seed x model x variant (raw/platt/
                          isotonic) for the NEW random arm at cap 3,338,217.
- sizematch_contrasts.csv the two pre-specified contrasts, raw variant:
    (a) size-free split contrast = new-random(3,338,217) test metric
        minus origin-1 temporal gap-0 metric of the SAME seed;
    (b) pure size effect = new-random(3,338,217) test metric minus
        existing random(4,000,000) test metric of the SAME seed;
  per model x seed, plus the across-seed mean per model (seed == 'mean').
  ECE ratio column: new-random test ECE / temporal gap-0 ECE (contrast a).
- preds/                  audit bundle: per seed, test predictions
                          (raw/platt/isotonic) + test labels, val raw
                          predictions + val labels, copied from the run
                          outputs / prepared splits.

Reference sources (exact files):
- temporal gap-0, seed 42: results/metrics_temporal.csv
- temporal gap-0, seed 43: results/seeds/seed43/metrics_temporal.csv
- temporal gap-0, seed 44: results/seeds/seed44/metrics_temporal.csv
- random 4M, seed 42:      results/metrics_random.csv
- random 4M, seed 43:      results/random_seeds/seed43/metrics_random.csv
- random 4M, seed 44:      results/random_seeds/seed44/metrics_random.csv
"""

from __future__ import annotations

import json
import os
import shutil

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
SEEDS = (42, 43, 44)
MODELS = ("lr", "deepfm", "dcnv2")
OUT = os.path.join(ROOT, "results", "sizematch")

TEMPORAL_REF = {
    42: "results/metrics_temporal.csv",
    43: "results/seeds/seed43/metrics_temporal.csv",
    44: "results/seeds/seed44/metrics_temporal.csv",
}
RANDOM4M_REF = {
    42: "results/metrics_random.csv",
    43: "results/random_seeds/seed43/metrics_random.csv",
    44: "results/random_seeds/seed44/metrics_random.csv",
}


def raw_row(path: str, model: str, bucket: str | None) -> pd.Series:
    df = pd.read_csv(os.path.join(ROOT, path))
    df = df[(df.model == model) & (df.variant == "raw")]
    if bucket is not None:
        df = df[df.bucket == bucket]
    assert len(df) == 1, f"{path} {model} raw {bucket}: {len(df)} rows"
    return df.iloc[0]


def main() -> None:
    os.makedirs(os.path.join(OUT, "preds"), exist_ok=True)

    # ---- cells -------------------------------------------------------------
    cell_frames = []
    for seed in SEEDS:
        arm_dir = os.path.join(ROOT, "data", f"processed_sizematch_s{seed}",
                               "random")
        with open(os.path.join(arm_dir, "manifest.json")) as f:
            manifest = json.load(f)
        m = pd.read_csv(os.path.join(OUT, f"seed{seed}", "metrics_random.csv"))
        m.insert(0, "seed", seed)
        m.insert(1, "cap", 3_338_217)
        m["n_train"] = manifest["splits"]["train"]["n"]
        m["total_vocab"] = manifest["total_vocab"]
        cell_frames.append(m)
    cells = pd.concat(cell_frames, ignore_index=True)
    cells = cells[["seed", "cap", "model", "bucket", "variant", "n_train",
                   "n_calib", "n_test", "auc", "logloss", "ece", "mean_pred",
                   "obs_ctr", "signed_bias", "total_vocab"]]
    cells.to_csv(os.path.join(OUT, "sizematch_cells.csv"), index=False)
    print(f"cells: {len(cells)} rows", flush=True)

    # ---- prediction audit bundle -------------------------------------------
    for seed in SEEDS:
        src_preds = os.path.join(OUT, f"seed{seed}", "preds")
        arm_dir = os.path.join(ROOT, "data", f"processed_sizematch_s{seed}",
                               "random")
        for model in MODELS:
            for variant in ("raw", "platt", "isotonic"):
                shutil.copyfile(
                    os.path.join(src_preds, f"random_{model}_test_{variant}.npy"),
                    os.path.join(OUT, "preds",
                                 f"sizematch_s{seed}_{model}_test_{variant}.npy"))
            shutil.copyfile(
                os.path.join(src_preds, f"random_{model}_test_calib_raw.npy"),
                os.path.join(OUT, "preds",
                             f"sizematch_s{seed}_{model}_val_raw.npy"))
        for name in ("test", "val"):
            shutil.copyfile(
                os.path.join(arm_dir, f"{name}_y.npy"),
                os.path.join(OUT, "preds", f"sizematch_s{seed}_{name}_y.npy"))
    print("preds bundle copied", flush=True)

    # ---- contrasts (raw variant) --------------------------------------------
    rows = []
    for model in MODELS:
        for seed in SEEDS:
            new = cells[(cells.seed == seed) & (cells.model == model)
                        & (cells.variant == "raw")].iloc[0]
            temp = raw_row(TEMPORAL_REF[seed], model, "gap0")
            old = raw_row(RANDOM4M_REF[seed], model, None)
            rows.append({
                "contrast": "a_sizefree_split", "model": model, "seed": seed,
                "new_random_auc": new.auc, "ref_auc": temp.auc,
                "d_auc": new.auc - temp.auc,
                "d_logloss": new.logloss - temp.logloss,
                "d_ece": new.ece - temp.ece,
                "ece_ratio": new.ece / temp.ece,
                "ref_source": TEMPORAL_REF[seed],
            })
            rows.append({
                "contrast": "b_pure_size", "model": model, "seed": seed,
                "new_random_auc": new.auc, "ref_auc": old.auc,
                "d_auc": new.auc - old.auc,
                "d_logloss": new.logloss - old.logloss,
                "d_ece": new.ece - old.ece,
                "ece_ratio": new.ece / old.ece,
                "ref_source": RANDOM4M_REF[seed],
            })
    contrasts = pd.DataFrame(rows)
    means = (contrasts.groupby(["contrast", "model"], as_index=False)
             [["d_auc", "d_logloss", "d_ece", "ece_ratio"]].mean())
    means["seed"] = "mean"
    means["new_random_auc"] = np.nan
    means["ref_auc"] = np.nan
    means["ref_source"] = ""
    contrasts = pd.concat([contrasts, means[contrasts.columns]],
                          ignore_index=True)
    contrasts.to_csv(os.path.join(OUT, "sizematch_contrasts.csv"), index=False)
    print(contrasts.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
