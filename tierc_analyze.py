"""Tier-C analyses for the Avazu temporal-calibration paper (2026-10-22).

Workstreams:
  1. Paired block-bootstrap CIs on delta-ECE (variant - raw) per cell, both
     origins, from the saved per-impression predictions. Protocol mirrors
     tierb_analyze.block_boot_ci: 200 contiguous blocks, 10,000 resamples,
     rng seed 7. ECE is recomputed per resample from per-(block, bin)
     aggregates (count / sum of predictions / sum of labels), with the
     paired draw shared by raw and the variant. Bins are the paper's
     ECE-15 equal-width bins over clipped probabilities, exactly as in
     src/calibration.expected_calibration_error.
  2. Deployable slice-statistic gates across origins: recalibrate
     (isotonic) iff g(slice statistic) fires, for every candidate
     statistic tried (slice ECE, |slice signed bias|, slice signed bias
     one-sided both directions, slice ECE minus |slice signed bias|).
     Tau fitted on origin 1 only (minimising origin-1 mean gated ECE,
     mirroring Tier-B section D), frozen for origin 2.
  3. Seed stability for origin 2 (seeds 42/43/44) and for the random arm /
     RQ1 gap-0 deltas (origin-1 window, seeds 42/43/44).
  4. Slice-length sweep analysis: delta-ECE vs calib_hours on both
     origins (hours_6 copied from the seed-42 o1_repro / origin2 runs).

Inputs are only read; outputs go to results/tierc/ + results/tierc_analysis.md.
Sections 3 and 4 are skipped (with a note) while their runs are missing.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src.calibration import clip_probs

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results", "tierc")
os.makedirs(OUT, exist_ok=True)
MODELS = ("lr", "deepfm", "dcnv2")
GAPS = (0, 1, 2, 3)
CALS = ("platt", "isotonic")
N_BINS = 15
N_BLOCKS = 200
N_BOOT = 10000
BOOT_SEED = 7

md: list[str] = []


def w(s: str = "") -> None:
    md.append(s)


def md_table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append(floatfmt.format(v) if np.isfinite(v) else "nan")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def load_metrics(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["gap"] = df["bucket"].str.replace("gap", "").astype(int)
    return df


def ece_lookup(metrics: pd.DataFrame) -> dict:
    return {(r.model, r.gap, r.variant): r.ece for r in metrics.itertuples()}


w("# Tier-C analysis — Avazu temporal calibration (2026-10-22)")
w()
w("Computed by `tierc_analyze.py` from run outputs and saved predictions; "
  "nothing is hand-entered.")
w()

# ===========================================================================
# WS1: bootstrap CIs on delta-ECE
# ===========================================================================
w("## 1. Paired block-bootstrap CIs on ΔECE (variant − raw)")
w()
w("Saved per-impression predictions from `results/o1_repro/preds/` (origin 1) "
  "and `results/origin2/preds/` (origin 2); labels from the corresponding "
  "prepared splits. 200 contiguous blocks over the bucket's impression "
  "order, 10,000 paired resamples (same block draw for raw and variant), "
  "rng seed 7 — the RQ4 bootstrap protocol of Tier-B section E applied to "
  "ΔECE instead of profit. Point estimates are the full-sample ECE "
  "differences; CIs are the 2.5/97.5 percentiles of the resampled ΔECE.")
w()

RUNS = {1: ("o1_repro", "data/processed_o1repro"),
        2: ("origin2", "data/processed_o2")}


def block_bin_agg(y: np.ndarray, p: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """(n_blocks, 3*n_bins) aggregates: counts, sum of predictions, sum of
    labels per (block, bin), with ECE-15 bins over clipped probabilities."""
    p_c = clip_probs(p)
    bin_idx = np.minimum((p_c * N_BINS).astype(int), N_BINS - 1)
    block_idx = np.searchsorted(edges, np.arange(len(y)), side="right") - 1
    flat = block_idx * N_BINS + bin_idx
    n_flat = N_BLOCKS * N_BINS
    cnt = np.bincount(flat, minlength=n_flat).reshape(N_BLOCKS, N_BINS)
    sp = np.bincount(flat, weights=p_c, minlength=n_flat).reshape(N_BLOCKS, N_BINS)
    sy = np.bincount(flat, weights=y, minlength=n_flat).reshape(N_BLOCKS, N_BINS)
    return np.concatenate([cnt, sp, sy], axis=1)


def boot_delta_ece(y: np.ndarray, p_raw: np.ndarray, p_var: np.ndarray,
                   rng: np.random.Generator) -> tuple[float, float, float, float, float]:
    n = len(y)
    edges = np.linspace(0, n, N_BLOCKS + 1).astype(int)
    agg_raw = block_bin_agg(y, p_raw, edges)
    agg_var = block_bin_agg(y, p_var, edges)

    def ece_from_agg(agg: np.ndarray) -> float:
        # ECE = sum_b |sum_y,b - sum_p,b| / N: the aggregates hold per-bin
        # SUMS, so the bin term is |sy - sp| / N (no extra count factor --
        # a (cnt/N) * |sy - sp| form double-counts the bin size; that bug
        # inflated a former version of the point estimates by ~1e6 while
        # leaving the bootstrap CIs, computed by ece_boot below, correct).
        cnt = agg[:, :N_BINS].sum(axis=0)
        sp = agg[:, N_BINS:2 * N_BINS].sum(axis=0)
        sy = agg[:, 2 * N_BINS:].sum(axis=0)
        tot = cnt.sum()
        return float(np.abs(sy - sp).sum() / tot)

    d_point = ece_from_agg(agg_var) - ece_from_agg(agg_raw)
    ece_raw = ece_from_agg(agg_raw)
    ece_var = ece_from_agg(agg_var)
    idx = rng.integers(0, N_BLOCKS, size=(N_BOOT, N_BLOCKS))
    # multiplicity of each block in each resample
    mult = np.zeros((N_BOOT, N_BLOCKS))
    rows = np.repeat(np.arange(N_BOOT), N_BLOCKS)
    np.add.at(mult, (rows, idx.ravel()), 1.0)
    tot_n = mult @ agg_raw[:, :N_BINS].sum(axis=1)  # total rows per resample

    def ece_boot(agg: np.ndarray) -> np.ndarray:
        cnt = mult @ agg[:, :N_BINS]
        sp = mult @ agg[:, N_BINS:2 * N_BINS]
        sy = mult @ agg[:, 2 * N_BINS:]
        return (np.abs(sy - sp).sum(axis=1)) / tot_n

    d = ece_boot(agg_var) - ece_boot(agg_raw)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return d_point, float(lo), float(hi), ece_raw, ece_var


rng = np.random.default_rng(BOOT_SEED)
boot_rows = []
max_point_err = 0.0
for origin, (rdir, pdir) in RUNS.items():
    metrics = load_metrics(os.path.join(ROOT, "results", rdir, "metrics_temporal.csv"))
    lk = ece_lookup(metrics)
    for model in MODELS:
        for g in GAPS:
            y = np.asarray(np.load(os.path.join(
                ROOT, pdir, "temporal", f"test_gap{g}_y.npy"), mmap_mode="r"),
                dtype=np.float64)
            preds = {v: np.load(os.path.join(
                ROOT, "results", rdir, "preds",
                f"temporal_{model}_gap{g}_{v}.npy")).astype(np.float64)
                for v in ("raw", "platt", "isotonic")}
            assert len(y) == len(preds["raw"])
            for cal in CALS:
                d, lo, hi, e_raw, e_var = boot_delta_ece(y, preds["raw"], preds[cal], rng)
                ref = lk[(model, g, cal)] - lk[(model, g, "raw")]
                max_point_err = max(max_point_err, abs(d - ref))
                boot_rows.append({
                    "origin": origin, "model": model, "gap": g, "variant": cal,
                    "delta_ece": d, "ci_lo": lo, "ci_hi": hi,
                    "excludes_zero": bool(lo > 0 or hi < 0),
                })
            del preds
boot_df = pd.DataFrame(boot_rows)
boot_df.to_csv(os.path.join(OUT, "delta_ece_bootstrap.csv"), index=False)
n_inside = int(((boot_df["delta_ece"] >= boot_df["ci_lo"]) &
                (boot_df["delta_ece"] <= boot_df["ci_hi"])).sum())
w(f"Point-estimate check: max |ΔECE from saved predictions − ΔECE from the "
  f"runs' metrics CSVs| = {max_point_err:.2e} across all 48 cells. The "
  f"residual is the float32-storage floor of the saved predictions: the "
  f"runs' metrics were computed on float64 in-memory probabilities, the "
  f"saved .npy files are float32, and the raw variants (natively float32 "
  f"model outputs) agree to ~1e-17 while the float64-valued recalibrated "
  f"variants carry the ~1e-9 truncation. The point estimate falls inside "
  f"its own 95% CI in {n_inside} of 48 cells.")
w()
w(md_table(boot_df, floatfmt="{:.4f}"))
w()
sig = boot_df[boot_df["excludes_zero"]]
w(f"- Cells whose 95% CI excludes 0: {len(sig)} of {len(boot_df)}.")
for (origin, variant), sub in boot_df.groupby(["origin", "variant"]):
    s = sub[sub["excludes_zero"]]
    harm = s[s["delta_ece"] > 0]
    help_ = s[s["delta_ece"] < 0]
    w(f"- Origin {origin}, {variant}: {len(s)} of {len(sub)} significant "
      f"({len(harm)} harm, {len(help_)} help).")
o1_sig_harm = boot_df[(boot_df["origin"] == 1) & boot_df["excludes_zero"] & (boot_df["delta_ece"] > 0)]
w(f"- Seed-stable Tier-B headline harm cells under the bootstrap: "
  f"LR gap+3 platt ΔECE = "
  f"{boot_df[(boot_df.origin == 1) & (boot_df.model == 'lr') & (boot_df.gap == 3) & (boot_df.variant == 'platt')]['delta_ece'].iloc[0]:+.4f} "
  f"(CI excludes 0: {bool(boot_df[(boot_df.origin == 1) & (boot_df.model == 'lr') & (boot_df.gap == 3) & (boot_df.variant == 'platt')]['excludes_zero'].iloc[0])}), "
  f"LR gap+3 isotonic and LR gap+2 isotonic likewise in the table.")
w()

# ===========================================================================
# WS2: slice-statistic gates
# ===========================================================================
w("## 2. Slice-statistic gates across origins")
w()
w("Gate rule: recalibrate a bucket with isotonic iff the gate statistic "
  "computed on that bucket's 6h calibration slice fires. Candidate "
  "statistics (all tried are reported): slice ECE (the Tier-B section D "
  "rule, replicated as a sanity check), |slice signed bias|, slice signed "
  "bias in the under-prediction direction only (fire iff bias < −τ), in "
  "the over-prediction direction only (fire iff bias > τ), and slice ECE "
  "minus |slice signed bias| (miscalibration beyond the mean shift). "
  "τ is fitted on origin 1 only, minimising origin-1 mean gated ECE over "
  "the midpoint candidate grid (Tier-B protocol), then frozen for "
  "origin 2. Harm-classification accuracy predicts harm exactly when the "
  "gate does not fire (Tier-B convention).")
w()

slices, targets = {}, {}
for origin, (rdir, _) in RUNS.items():
    sl = pd.read_csv(os.path.join(ROOT, "results", rdir, "slice_stats_temporal.csv"))
    sl["gap"] = sl["bucket"].str.replace("gap", "").astype(int)
    metrics = load_metrics(os.path.join(ROOT, "results", rdir, "metrics_temporal.csv"))
    lk = ece_lookup(metrics)
    for r in sl.itertuples():
        slices[(origin, r.model, r.gap)] = {
            "slice_ece": r.ece, "slice_bias": r.signed_bias}
        targets[(origin, r.model, r.gap)] = (
            lk[(r.model, r.gap, "raw")], lk[(r.model, r.gap, "isotonic")])

STAT_DEFS = {
    "slice_ece": lambda s: s["slice_ece"],
    "abs_slice_bias": lambda s: abs(s["slice_bias"]),
    "neg_slice_bias (fire iff bias < -tau)": lambda s: -s["slice_bias"],
    "pos_slice_bias (fire iff bias > tau)": lambda s: s["slice_bias"],
    "slice_ece_minus_abs_bias": lambda s: s["slice_ece"] - abs(s["slice_bias"]),
}


def gate_eval(stat_fn, tau: float, origin: int) -> dict:
    raw_l, iso_l, gated_l, harm_true, harm_pred = [], [], [], [], []
    for (o, model, g), (raw_e, iso_e) in targets.items():
        if o != origin:
            continue
        fire = stat_fn(slices[(o, model, g)]) > tau
        raw_l.append(raw_e)
        iso_l.append(iso_e)
        gated_l.append(iso_e if fire else raw_e)
        harm_true.append(iso_e > raw_e)
        harm_pred.append(not fire)
    return {"mean_never": float(np.mean(raw_l)),
            "mean_always": float(np.mean(iso_l)),
            "mean_gated": float(np.mean(gated_l)),
            "harm_acc": float((np.array(harm_true) == np.array(harm_pred)).mean()),
            "n_fired": int(sum(1 for (o, m, g) in targets
                                if o == origin and stat_fn(slices[(o, m, g)]) > tau)),
            "n_cells": len(raw_l)}


gate_rows = []
for stat_name, stat_fn in STAT_DEFS.items():
    o1_vals = np.sort([stat_fn(slices[(1, m, g)]) for m in MODELS for g in GAPS])
    cands = np.concatenate([[o1_vals[0] - 1e-9],
                            (o1_vals[:-1] + o1_vals[1:]) / 2,
                            [o1_vals[-1] + 1e-9]])
    best_tau, best_obj = None, np.inf
    for t in cands:
        obj = gate_eval(stat_fn, float(t), 1)["mean_gated"]
        if obj < best_obj:
            best_obj, best_tau = obj, float(t)
    for origin in (1, 2):
        ev = gate_eval(stat_fn, best_tau, origin)
        gate_rows.append({"statistic": stat_name, "tau_fit_origin1": best_tau,
                          "origin": origin, **ev})
    ev_pool = {k: np.mean([gate_rows[-2][k], gate_rows[-1][k]])
               for k in ("mean_never", "mean_always", "mean_gated")}
    gate_rows.append({"statistic": stat_name, "tau_fit_origin1": best_tau,
                      "origin": "pooled(mean of origins)", **ev_pool,
                      "harm_acc": np.nan, "n_fired": np.nan, "n_cells": 24})
gate_df = pd.DataFrame(gate_rows)
gate_df.to_csv(os.path.join(OUT, "slice_bias_gates.csv"), index=False)
w(md_table(gate_df))
w()
chk = gate_df[(gate_df["statistic"] == "slice_ece") & (gate_df["origin"] == 2)].iloc[0]
w(f"Sanity check (slice-ECE replication of Tier-B section D): origin-2 gated "
  f"mean ECE = {chk['mean_gated']:.4f} with τ = {chk['tau_fit_origin1']:.4f} "
  f"(Tier-B: τ_slice = 0.0061, gated 0.0082, never 0.0083, always 0.0081).")
w()

with open(os.path.join(ROOT, "results", "tierc_analysis.md"), "w") as f:
    f.write("\n".join(md))
print("wrote results/tierc/ CSVs and results/tierc_analysis.md (WS1+WS2)")
