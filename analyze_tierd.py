"""Tier-D analyses (2026-10-22).

Experiment 1 -> results/sweep_seeds_analysis.md
  Slice-length sweep at seeds 43/44 (Tier-C ran seed 42 only): does the
  "harm is a short-slice phenomenon; ~12 h removes it" result replicate
  across seeds? Same cell definitions as Tier-C section 4: per
  (seed, origin, calib_hours), dECE = ECE(variant) - ECE(raw) on the
  metrics_temporal.csv cells (24 cells; 18 at 24 h, where gap 0 does
  not exist), harm = dECE > 0, plus per-calibrator means over 12 cells
  (9 at 24 h) and the canonical LR cells.

Experiment 2 -> results/mechanism_analysis.md
  Slice size vs diurnal coverage from results/mechanism_slice/
  mechanism_o{1,2}.csv (see mech_slice.py for the design). Per origin
  and calibrator: mean dECE of full12 (a), hour1 (b) and the K=5
  size-matched subsamples of the 12 h slice (c), over the 12
  model x gap cells, plus the recovery fraction
  (mean_c - mean_b) / (mean_a - mean_b): ~1 means the size-matched
  12 h-coverage subsample recovers the full slice's gain (coverage
  mechanism); ~0 means it behaves like the 1 h slice (size mechanism).

Each section is written only when its inputs exist, so the script can
be re-run as the chain progresses. Nothing is hand-entered.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

SEEDS = (42, 43, 44)
ORIGINS = (1, 2)
HOURS = (1, 3, 6, 12, 24)
CALS = ("platt", "isotonic")


def metrics_path(seed: int, origin: int, hours: int) -> str:
    if seed == 42:
        return f"results/sweep/hours_{hours}/origin{origin}/metrics_temporal.csv"
    if hours == 6:
        return (f"results/seeds/seed{seed}/metrics_temporal.csv" if origin == 1
                else f"results/origin2_seeds/seed{seed}/metrics_temporal.csv")
    return f"results/sweep_seeds/seed{seed}/hours_{hours}/origin{origin}/metrics_temporal.csv"


def cell_deltas(seed: int, origin: int, hours: int) -> pd.DataFrame:
    df = pd.read_csv(metrics_path(seed, origin, hours))
    raw = df[df.variant == "raw"].set_index(["model", "bucket"])["ece"]
    rows = []
    for _, r in df[df.variant != "raw"].iterrows():
        rows.append({"model": r.model, "bucket": r.bucket, "variant": r.variant,
                     "n_calib": r.n_calib,
                     "delta_ece": r.ece - raw.loc[(r.model, r.bucket)]})
    return pd.DataFrame(rows)


def fmt(x: float) -> str:
    return f"{x:+.4f}"


def sweep_section() -> str | None:
    if not all(os.path.exists(metrics_path(s, o, h))
               for s in (43, 44) for o in ORIGINS for h in (1, 3, 12, 24)):
        return None
    lines = ["# Tier-D Experiment 1 — slice-length sweep at seeds 43/44 (2026-10-22)\n",
             "Same cells and conventions as Tier-C section 4 (seed 42, recomputed "
             "here in the same code path; hours_6 for seeds 43/44 is the Tier-B/C "
             "seed run). dECE = ECE(calibrated) - ECE(raw); harm = dECE > 0. At "
             "24 h the gap-0 bucket does not exist by construction (18 of 24 "
             "cells, 9 per calibrator).\n"]
    data = {(s, o, h): cell_deltas(s, o, h) for s in SEEDS for o in ORIGINS for h in HOURS}

    lines.append("## Pooled (both calibrators): mean dECE and harm counts\n")
    lines.append("| seed | origin | calib_hours | mean_delta_ece | n_harm | n_cells |")
    lines.append("|---|---|---|---|---|---|")
    for s in SEEDS:
        for o in ORIGINS:
            for h in HOURS:
                d = data[(s, o, h)]["delta_ece"]
                lines.append(f"| {s} | {o} | {h} | {d.mean():+.4f} | "
                             f"{int((d > 0).sum())} | {len(d)} |")
    lines.append("")
    lines.append("## Per calibrator: mean dECE, harm/n\n")
    for s in SEEDS:
        for o in ORIGINS:
            for cal in CALS:
                cells = []
                for h in HOURS:
                    d = data[(s, o, h)]
                    d = d[d.variant == cal]["delta_ece"]
                    cells.append(f"{h}h: {d.mean():+.4f}, harm {int((d > 0).sum())}/{len(d)}")
                lines.append(f"- seed {s}, origin {o}, {cal}: " + "; ".join(cells) + ".")
    lines.append("")
    lines.append("## Canonical cells (dECE by slice length)\n")
    for (o, model, bucket, cal) in [(1, "lr", "gap3", "platt"), (1, "lr", "gap3", "isotonic"),
                                    (1, "lr", "gap2", "isotonic"),
                                    (2, "lr", "gap2", "platt"), (2, "lr", "gap2", "isotonic"),
                                    (2, "lr", "gap3", "platt"), (2, "lr", "gap3", "isotonic")]:
        for s in SEEDS:
            vals = []
            for h in HOURS:
                d = data[(s, o, h)]
                v = d[(d.model == model) & (d.bucket == bucket) & (d.variant == cal)]
                vals.append(f"{h}h {v.delta_ece.iloc[0]:+.4f}" if len(v) else f"{h}h n/a")
            lines.append(f"- O{o} {model} {bucket} {cal}, seed {s}: " + ", ".join(vals) + ".")
    lines.append("")
    lines.append("## Verdict inputs — harm at 1 h vs 12 h (per seed x origin, pooled calibrators)\n")
    lines.append("| seed | origin | harm@1h | harm@12h | mean dECE@1h | mean dECE@12h |")
    lines.append("|---|---|---|---|---|---|")
    for s in SEEDS:
        for o in ORIGINS:
            d1, d12 = data[(s, o, 1)]["delta_ece"], data[(s, o, 12)]["delta_ece"]
            lines.append(f"| {s} | {o} | {int((d1 > 0).sum())}/{len(d1)} | "
                         f"{int((d12 > 0).sum())}/{len(d12)} | {d1.mean():+.4f} | {d12.mean():+.4f} |")
    lines.append("")
    return "\n".join(lines)


def mechanism_section() -> str | None:
    paths = {o: f"results/mechanism_slice/mechanism_o{o}.csv" for o in ORIGINS}
    if not all(os.path.exists(p) for p in paths.values()):
        return None
    lines = ["# Tier-D Experiment 2 — slice size vs diurnal coverage (2026-10-22)\n",
             "Seed 42, models trained once per origin (see mech_slice.py for the "
             "design and the validation against the sweep runs). Conditions per "
             "bucket: (a) full12 = fit on the full 12 h slice; (b) hour1 = fit on "
             "the actual 1 h slice (size n1); (c) sub = K=5 uniform subsamples of "
             "the 12 h slice at exactly size n1 (subsample seeds "
             "20261023 + origin*100000 + gap*1000 + k, shared across models and "
             "calibrators). dECE = ECE(calibrated) - ECE(raw) on the bucket's "
             "fixed test set (the 12 h run's test split). Means are over the 12 "
             "model x gap cells per origin (sub: averaged over k within cell "
             "first). Recovery fraction = (mean_c - mean_b) / (mean_a - mean_b): "
             "~1 -> the size-matched 12 h-coverage subsample recovers the full "
             "slice's gain, i.e. coverage (temporal span), not impression count, "
             "is the mechanism; ~0 -> it behaves like the 1 h slice, i.e. size "
             "is the mechanism.\n"]
    frames = {o: pd.read_csv(p) for o, p in paths.items()}
    lines.append("| origin | calibrator | (a) full12 | (b) hour1 | (c) sub (mean of 5) | "
                 "(c) sd across k | recovery fraction |")
    lines.append("|---|---|---|---|---|---|---|")
    pooled = []
    for o in ORIGINS:
        df = frames[o]
        for cal in CALS:
            d = df[df.calibrator == cal]
            a = d[d.condition == "full12"]["delta_ece"].mean()
            b = d[d.condition == "hour1"]["delta_ece"].mean()
            sub = d[d.condition.str.startswith("sub")]
            c_cell = sub.groupby(["model", "gap"])["delta_ece"].mean()
            c = c_cell.mean()
            c_sd = sub.groupby(["model", "gap"])["delta_ece"].std().mean()
            frac = (c - b) / (a - b) if abs(a - b) > 1e-12 else float("nan")
            pooled.append((a, b, c))
            lines.append(f"| {o} | {cal} | {a:+.4f} | {b:+.4f} | {c:+.4f} | "
                         f"{c_sd:.4f} | {frac:.2f} |")
    a = np.mean([p[0] for p in pooled]); b = np.mean([p[1] for p in pooled]); c = np.mean([p[2] for p in pooled])
    lines.append(f"\nPooled over origins and calibrators: (a) {a:+.4f}, (b) {b:+.4f}, "
                 f"(c) {c:+.4f}; recovery fraction {(c - b) / (a - b):.2f}.\n")
    lines.append("## Per model (both origins pooled)\n")
    lines.append("| model | calibrator | (a) full12 | (b) hour1 | (c) sub | recovery fraction |")
    lines.append("|---|---|---|---|---|---|")
    allf = pd.concat(frames.values(), ignore_index=True)
    for model in ("lr", "deepfm", "dcnv2"):
        for cal in CALS:
            d = allf[(allf.model == model) & (allf.calibrator == cal)]
            a = d[d.condition == "full12"]["delta_ece"].mean()
            b = d[d.condition == "hour1"]["delta_ece"].mean()
            c = d[d.condition.str.startswith("sub")].groupby(
                ["origin", "gap"])["delta_ece"].mean().mean()
            frac = (c - b) / (a - b) if abs(a - b) > 1e-12 else float("nan")
            lines.append(f"| {model} | {cal} | {a:+.4f} | {b:+.4f} | {c:+.4f} | {frac:.2f} |")
    lines.append("")
    for o in ORIGINS:
        vp = f"results/mechanism_slice/mechanism_o{o}_validation.json"
        if os.path.exists(vp):
            v = json.load(open(vp))
            lines.append(
                f"Validation origin {o}: full12 vs sweep-12h max |ddECE| = "
                f"{v['full12_vs_sweep12_max_abs_diff']:.2e}; hour1 vs sweep-1h "
                f"(gaps 1-3) max |ddECE| = {v['hour1_vs_sweep1_gaps123_max_abs_diff']:.2e}; "
                f"1h-subset-of-12h verified elementwise for all gaps "
                f"({all(g['subset_X_equal'] and g['subset_y_equal'] for g in v['gaps'].values())}); "
                f"hour1 gap-0 vs sweep-1h differences (expected; different test "
                f"remainder): {v['hour1_gap0_vs_sweep1_diff_by_cell']}.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    sw = sweep_section()
    if sw:
        with open("results/sweep_seeds_analysis.md", "w") as f:
            f.write(sw)
        print("wrote results/sweep_seeds_analysis.md")
    else:
        print("sweep inputs incomplete -- skipped Experiment 1 analysis")
    me = mechanism_section()
    if me:
        with open("results/mechanism_analysis.md", "w") as f:
            f.write(me)
        print("wrote results/mechanism_analysis.md")
    else:
        print("mechanism inputs incomplete -- skipped Experiment 2 analysis")


if __name__ == "__main__":
    main()
