"""Tier-C finalize (2026-10-22): workstreams 3 (seeds) and 4 (slice sweep).

Run AFTER run_tierc.sh prints TIERC CHAIN COMPLETE. Reads only run outputs;
writes results/tierc/{seed_o2_ece,seed_random_arm,sweep_delta_ece}.csv and
inserts sections 3 and 4 into results/tierc_analysis.md before section 5
(the file's sections 1-2 come from tierc_analyze.py, section 5 from
tierc_ws5.py; this script preserves both and is idempotent via markers).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results", "tierc")
MODELS = ("lr", "deepfm", "dcnv2")
GAPS = (0, 1, 2, 3)
CALS = ("platt", "isotonic")
HOURS = (1, 3, 6, 12, 24)


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


def delta_table(seed_metrics: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Per-cell delta-ECE per seed with the Tier-B flag criterion:
    sign flip across seeds, or |dECE at the reference seed| < 2*sd."""
    seeds = sorted(seed_metrics)
    ref = seeds[0]
    lks = {s: ece_lookup(m) for s, m in seed_metrics.items()}
    rows = []
    for model in MODELS:
        for g in GAPS:
            for cal in CALS:
                d = {s: lks[s][(model, g, cal)] - lks[s][(model, g, "raw")]
                     for s in seeds}
                vals = np.array([d[s] for s in seeds])
                sd = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
                signs = np.sign(vals)
                flip = bool(len(set(signs)) > 1)
                rows.append({
                    "model": model, "gap": g, "calibrator": cal,
                    **{f"dECE_s{s}": d[s] for s in seeds},
                    "sd_dECE": sd,
                    "sign_flip": flip,
                    "within_seed_noise": bool(flip or abs(d[ref]) < 2 * sd),
                })
    return pd.DataFrame(rows)


md: list[str] = []


def w(s: str = "") -> None:
    md.append(s)

# ===========================================================================
# WS3a: origin-2 seeds
# ===========================================================================
o2_runs = {42: load_metrics(os.path.join(ROOT, "results", "origin2", "metrics_temporal.csv")),
           43: load_metrics(os.path.join(ROOT, "results", "origin2_seeds", "seed43", "metrics_temporal.csv")),
           44: load_metrics(os.path.join(ROOT, "results", "origin2_seeds", "seed44", "metrics_temporal.csv"))}
o2_seed_df = delta_table(o2_runs)
o2_seed_df.to_csv(os.path.join(OUT, "seed_o2_ece.csv"), index=False)

# origin-1 reference (Tier-B, seeds 42/43/44) for the noise-map comparison
o1_runs = {42: load_metrics(os.path.join(ROOT, "results", "o1_repro", "metrics_temporal.csv")),
           43: load_metrics(os.path.join(ROOT, "results", "seeds", "seed43", "metrics_temporal.csv")),
           44: load_metrics(os.path.join(ROOT, "results", "seeds", "seed44", "metrics_temporal.csv"))}
o1_seed_df = delta_table(o1_runs)

w("## 3. Seed stability — origin 2 and the random arm (seeds 42 / 43 / 44)")
w()
w("Same protocol and flag criterion as Tier-B section B: ΔECE = ECE(calibrated) "
  "− ECE(raw) per seed; a cell is *within seed noise* when the seed-42 sign "
  "flips across seeds or |ΔECE at seed 42| < 2·sd(ΔECE) across the three seeds "
  "(sd from 3 seeds is crude). Origin-2 seed-42 values are the `results/origin2/` "
  "run; origin-1 values are recomputed here from `o1_repro` + `results/seeds/` "
  "and reproduce the Tier-B map.")
w()
w("### 3a. Origin 2, ΔECE by seed")
w()
w(md_table(o2_seed_df))
w()
n_noise_o2 = int(o2_seed_df["within_seed_noise"].sum())
n_noise_o1 = int(o1_seed_df["within_seed_noise"].sum())
w(f"- Origin 2: {n_noise_o2} of 24 calibrator cells are within seed noise "
  f"(origin 1, Tier-B: {n_noise_o1} of 24).")
for model in MODELS:
    sub = o2_seed_df[o2_seed_df["model"] == model]
    raw_sd = []
    for g in GAPS:
        raws = [ece_lookup(o2_runs[s])[(model, g, "raw")] for s in (42, 43, 44)]
        raw_sd.append(float(np.std(raws, ddof=1)))
    w(f"- Origin 2 raw-ECE seed sd, {model}: max over gaps = {max(raw_sd):.4f} "
      f"(per-gap: " + ", ".join(f"{s:.4f}" for s in raw_sd) + ").")
stable_o2 = o2_seed_df[(~o2_seed_df["within_seed_noise"]) & (o2_seed_df["dECE_s42"].abs() >= 0.002)]
w("- Origin-2 cells that are seed-stable with |ΔECE| ≥ 0.002 at seed 42: " +
  ("; ".join(f"{r.model} gap+{r.gap} {r.calibrator} {r.dECE_s42:+.4f}"
             for r in stable_o2.itertuples()) or "none") + ".")
w()

# ===========================================================================
# WS3b: random arm seeds + RQ1 gap-0 contrasts
# ===========================================================================
rand_runs = {42: pd.read_csv(os.path.join(ROOT, "results", "metrics_random.csv")),
             43: pd.read_csv(os.path.join(ROOT, "results", "random_seeds", "seed43", "metrics_random.csv")),
             44: pd.read_csv(os.path.join(ROOT, "results", "random_seeds", "seed44", "metrics_random.csv"))}
rand_rows = []
for s, df in rand_runs.items():
    for r in df.itertuples():
        rand_rows.append({"seed": s, "model": r.model, "variant": r.variant,
                          "auc": r.auc, "logloss": r.logloss, "ece": r.ece})
rand_df = pd.DataFrame(rand_rows)
rand_df.to_csv(os.path.join(OUT, "seed_random_arm.csv"), index=False)

w("### 3b. Random arm by seed, and the RQ1 gap-0 contrasts per seed")
w()
w(md_table(rand_df))
w()
w("RQ1 contrasts per seed (random arm minus temporal gap+0, same seed; "
  "temporal gap+0 from `o1_repro` / `results/seeds/seed{43,44}`):")
w()
rq1_rows = []
for s in (42, 43, 44):
    lk = ece_lookup(o1_runs[s])
    tmp = pd.read_csv(os.path.join(ROOT, "results",
                      "o1_repro" if s == 42 else f"seeds/seed{s}",
                      "metrics_temporal.csv"))
    auc_t = {(r.model, r.variant): r.auc for r in tmp.itertuples()
             if r.bucket == "gap0"}
    for model in MODELS:
        rr = rand_df[(rand_df["seed"] == s) & (rand_df["model"] == model) &
                     (rand_df["variant"] == "raw")].iloc[0]
        rq1_rows.append({
            "seed": s, "model": model,
            "auc_inflation": rr["auc"] - auc_t[(model, "raw")],
            "ece_random_raw": rr["ece"],
            "ece_temporal_gap0_raw": lk[(model, 0, "raw")],
            "ece_ratio_rand_over_temporal": rr["ece"] / lk[(model, 0, "raw")],
        })
rq1_df = pd.DataFrame(rq1_rows)
w(md_table(rq1_df))
w()
for model in MODELS:
    sub = rq1_df[rq1_df["model"] == model]
    w(f"- {model}: AUC inflation across seeds = "
      + ", ".join(f"{v:+.4f}" for v in sub["auc_inflation"])
      + f"; random/temporal gap-0 raw-ECE ratio = "
      + ", ".join(f"{v:.2f}" for v in sub["ece_ratio_rand_over_temporal"]) + ".")
w()

# ===========================================================================
# WS4: slice-length sweep
# ===========================================================================
sweep_rows = []
for origin in (1, 2):
    for h in HOURS:
        m = load_metrics(os.path.join(
            ROOT, "results", "sweep", f"hours_{h}", f"origin{origin}",
            "metrics_temporal.csv"))
        lk = ece_lookup(m)
        ncal = {(r.model, r.gap): r.n_calib for r in m.itertuples()}
        # iterate the cells actually present: at calib_hours=24 the gap-0
        # bucket does not exist (the 24 h slice consumes the whole day,
        # leaving no test impressions), so hours_24 has gaps 1-3 only.
        for model in MODELS:
            for g in GAPS:
                if (model, g, "raw") not in lk:
                    continue
                for cal in CALS:
                    sweep_rows.append({
                        "origin": origin, "calib_hours": h, "model": model,
                        "gap": g, "variant": cal, "n_calib": ncal[(model, g)],
                        "ece_raw": lk[(model, g, "raw")],
                        "ece_cal": lk[(model, g, cal)],
                        "delta_ece": lk[(model, g, cal)] - lk[(model, g, "raw")],
                    })
sweep_df = pd.DataFrame(sweep_rows)
sweep_df.to_csv(os.path.join(OUT, "sweep_delta_ece.csv"), index=False)

w("## 4. Slice-length sweep — calibration-slice length {1, 3, 6, 12, 24} h")
w()
w("Seed 42, both origins; hours_6 is the paper's operating point (the "
  "`o1_repro` / `origin2` runs copied into `results/sweep/hours_6/`). "
  "ΔECE = ECE(calibrated) − ECE(raw) at that slice length; harm = ΔECE > 0. "
  "At 24 h the gap-0 bucket does not exist by construction (the slice is "
  "the bucket day's first 24 hours, i.e. the whole day, leaving no test "
  "impressions), so the 24 h rows cover gaps +1..+3 only (18 of 24 cells).")
w()
summ = (sweep_df.groupby(["origin", "calib_hours"])
        .agg(mean_delta_ece=("delta_ece", "mean"),
             n_harm=("delta_ece", lambda v: int((v > 0).sum())),
             n_cells=("delta_ece", "size"),
             mean_n_calib=("n_calib", "mean"))
        .reset_index())
w(md_table(summ))
w()
for (origin, cal), sub in sweep_df.groupby(["origin", "variant"]):
    line = []
    for h in HOURS:
        s2 = sub[sub["calib_hours"] == h]
        line.append(f"{h}h: mean ΔECE {s2['delta_ece'].mean():+.4f}, "
                    f"harm {int((s2['delta_ece'] > 0).sum())}/{len(s2)}")
    w(f"- Origin {origin}, {cal}: " + "; ".join(line) + ".")
# headline cells across slice lengths
w()
w("Headline cells across slice lengths (ΔECE):")
for origin, model, g, cal in [(1, "lr", 3, "platt"), (1, "lr", 3, "isotonic"),
                              (1, "lr", 2, "isotonic"), (1, "dcnv2", 1, "isotonic"),
                              (1, "deepfm", 1, "isotonic"),
                              (2, "lr", 3, "platt"), (2, "lr", 3, "isotonic"),
                              (2, "lr", 2, "platt")]:
    sub = sweep_df[(sweep_df["origin"] == origin) & (sweep_df["model"] == model) &
                   (sweep_df["gap"] == g) & (sweep_df["variant"] == cal)]
    vals = {int(r.calib_hours): r.delta_ece for r in sub.itertuples()}
    w(f"- O{origin} {model} gap+{g} {cal}: " +
      ", ".join(f"{h}h {vals[h]:+.4f}" for h in HOURS) + ".")
w()
w(md_table(sweep_df, floatfmt="{:.4f}"))
w()

# ===========================================================================
# assemble: insert sections 3-4 before section 5
# ===========================================================================
md_path = os.path.join(ROOT, "results", "tierc_analysis.md")
existing = open(md_path).read()
if "## 3. Seed stability" in existing:
    print("sections 3-4 already present; CSVs refreshed only")
else:
    marker = "## 5. RQ4 — isotonic-on-test-day oracle"
    assert marker in existing, "section 5 marker missing"
    head, tail = existing.split(marker, 1)
    with open(md_path, "w") as f:
        f.write(head.rstrip() + "\n\n" + "\n".join(md) + "\n" + marker + tail)
    print("inserted sections 3-4 into results/tierc_analysis.md")
print("wrote results/tierc/{seed_o2_ece,seed_random_arm,sweep_delta_ece}.csv")
