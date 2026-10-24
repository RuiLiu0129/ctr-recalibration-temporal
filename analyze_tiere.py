"""Tier-E analysis (2026-10-23): Exp A (mechanism control at seeds
43/44) -> results/mechanism_seeds_analysis.md; Exp B (recency control)
-> results/recency_analysis.md.

Cell conventions (identical to the seed-42 published analysis and to
v8's paragraph): a cell is (origin, model, gap, calibrator); the sub
condition's per-cell dECE is the mean over its K=5 subsamples.
dECE = ECE(calibrated) - ECE(raw); harm = dECE > 0. Recovery fraction
per group = (mean_c - mean_b) / (mean_a - mean_b) with a = full12,
b = hour1, c = sub.

Cell sets: ALL = gaps 0..3 (48 cells per condition per seed); FIXED =
gaps +1..+3 (36 cells per condition per seed) -- v8's paragraph set.
Self-check: recomputing seed 42 in this code path must reproduce the
published numbers (all-cells pooled -0.0020/+0.0007/-0.0016, recovery
0.88; fixed-set -0.0024/-0.0001/-0.0021, recovery 0.88, origin x
calibrator recovery range 0.80-0.99, harm full12/sub/hour1 = 4/4/15
of 36).
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

RES = "results"
CONDS = ("full12", "hour1", "sub")


def load_mech(seed: int) -> pd.DataFrame:
    if seed == 42:
        frames = [pd.read_csv(f"{RES}/mechanism_slice/mechanism_o{o}.csv")
                  for o in (1, 2)]
    else:
        frames = [pd.read_csv(f"{RES}/mechanism_seeds/seed{seed}/mechanism_o{o}.csv")
                  for o in (1, 2)]
    return pd.concat(frames, ignore_index=True)


def cell_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per cell with a/b/c dECE (sub averaged over k)."""
    sub = (df[df.condition.str.startswith("sub")]
           .groupby(["origin", "model", "gap", "calibrator"])["delta_ece"]
           .mean().rename("c"))
    piv = (df[df.condition.isin(["full12", "hour1"])]
           .pivot_table(index=["origin", "model", "gap", "calibrator"],
                        columns="condition", values="delta_ece")
           .rename(columns={"full12": "a", "hour1": "b"}))
    out = piv.join(sub).reset_index()
    assert out[["a", "b", "c"]].notna().all().all()
    return out


def recovery(g: pd.DataFrame) -> float:
    return (g["c"].mean() - g["b"].mean()) / (g["a"].mean() - g["b"].mean())


def fmt(x: float) -> str:
    return f"{x:+.4f}"


def mech_section(seed: int) -> tuple[str, dict]:
    ct = cell_table(load_mech(seed))
    lines = [f"## Seed {seed}\n"]
    summary = {}

    # ---- ALL cells (48 per condition) --------------------------------------
    lines.append("### All cells (gaps 0..+3; 48 cells per condition)\n")
    lines.append("| origin | calibrator | (a) full12 | (b) hour1 | (c) sub (mean of 5) | recovery |")
    lines.append("|---|---|---|---|---|---|")
    for (o, cal), g in ct.groupby(["origin", "calibrator"]):
        lines.append(f"| {o} | {cal} | {fmt(g['a'].mean())} | {fmt(g['b'].mean())} "
                     f"| {fmt(g['c'].mean())} | {recovery(g):.2f} |")
    pa, pb, pc = ct["a"].mean(), ct["b"].mean(), ct["c"].mean()
    rec_all = (pc - pb) / (pa - pb)
    harm_all = {c: int((ct[c] > 0).sum()) for c in ("a", "b", "c")}
    lines.append(f"\nPooled over origins and calibrators: (a) {fmt(pa)}, (b) {fmt(pb)}, "
                 f"(c) {fmt(pc)}; recovery fraction {rec_all:.2f}. "
                 f"Harm counts of 48: full12 {harm_all['a']}, hour1 {harm_all['b']}, "
                 f"sub {harm_all['c']}.\n")
    lines.append("Per model (both origins pooled), all cells:\n")
    lines.append("| model | calibrator | (a) full12 | (b) hour1 | (c) sub | recovery |")
    lines.append("|---|---|---|---|---|---|")
    for (m, cal), g in ct.groupby(["model", "calibrator"]):
        lines.append(f"| {m} | {cal} | {fmt(g['a'].mean())} | {fmt(g['b'].mean())} "
                     f"| {fmt(g['c'].mean())} | {recovery(g):.2f} |")
    summary["all"] = {"a": pa, "b": pb, "c": pc, "recovery": rec_all, "harm": harm_all}

    # ---- FIXED set (36 per condition) ---------------------------------------
    fx = ct[ct.gap >= 1]
    assert len(fx) == 36
    fa, fb, fc = fx["a"].mean(), fx["b"].mean(), fx["c"].mean()
    rec_fx = (fc - fb) / (fa - fb)
    recs_oc = {(o, cal): recovery(g) for (o, cal), g in fx.groupby(["origin", "calibrator"])}
    harm_fx = {c: int((fx[c] > 0).sum()) for c in ("a", "b", "c")}
    lines.append("\n### Fixed evaluation set (gaps +1..+3; 36 cells per condition) "
                 "-- v8's paragraph set\n")
    lines.append(f"Mean dECE: full12 {fmt(fa)}, hour1 {fmt(fb)}, sub {fmt(fc)}. "
                 f"Pooled recovery {rec_fx:.2f}; recovery across origin x calibrator: "
                 + ", ".join(f"o{o} {cal} {r:.2f}" for (o, cal), r in sorted(recs_oc.items()))
                 + f" (range {min(recs_oc.values()):.2f}-{max(recs_oc.values()):.2f}). "
                 f"Harm counts of 36: full12 {harm_fx['a']}, hour1 {harm_fx['b']}, "
                 f"sub {harm_fx['c']}.\n")
    lines.append("Per origin x calibrator (fixed set, 9 cells each):\n")
    lines.append("| origin | calibrator | (a) full12 | (b) hour1 | (c) sub | recovery | harm a/b/c |")
    lines.append("|---|---|---|---|---|---|---|")
    for (o, cal), g in sorted(fx.groupby(["origin", "calibrator"])):
        h = "/".join(str(int((g[c] > 0).sum())) for c in ("a", "b", "c"))
        lines.append(f"| {o} | {cal} | {fmt(g['a'].mean())} | {fmt(g['b'].mean())} "
                     f"| {fmt(g['c'].mean())} | {recovery(g):.2f} | {h} |")
    summary["fixed"] = {"a": fa, "b": fb, "c": fc, "recovery": rec_fx,
                        "rec_range": (min(recs_oc.values()), max(recs_oc.values())),
                        "harm": harm_fx}
    return "\n".join(lines), summary


def exp_a() -> None:
    lines = ["# Tier-E Experiment A — mechanism control at seeds 43/44 (2026-10-23)\n",
             "Exact replication of the Tier-D Exp 2 design (mech_slice.py) at training "
             "seeds 43 and 44: per bucket, Platt / isotonic fitted on (a) the full 12 h "
             "slice, (b) the true 1 h slice, (c) K=5 uniform subsamples of the 12 h slice "
             "at exactly the 1 h size (subsample RNG identical to seed 42: seeds 20261023 "
             "+ origin*100000 + gap*1000 + k, shared across models and calibrators; the "
             "temporal prepare is seed-independent, so the drawn index sets are the same "
             "as seed 42's -- only the trained model differs). No per-impression "
             "predictions were saved for the seed 43/44 runs, so -- exactly as for seed "
             "42 itself -- each (seed, origin) model set was retrained once with the "
             "run's own seeding replicated deterministically and validated against that "
             "seed's own sweep runs (see validation summary at the end). dECE = "
             "ECE(calibrated) - ECE(raw) on the bucket's fixed test set (the 12 h run's "
             "test split); harm = dECE > 0; the sub condition's per-cell dECE is the mean "
             "over its 5 subsamples. Recovery = (mean_c - mean_b) / (mean_a - mean_b): "
             "~1 -> the size-matched 12 h-span subsample recovers the full slice's gain "
             "(span/coverage is the mechanism); ~0 -> it behaves like the 1 h slice "
             "(size is the mechanism).\n"]
    summaries = {}
    for seed in (42, 43, 44):
        sec, summ = mech_section(seed)
        lines.append(sec)
        summaries[seed] = summ

    # ---- self-check: seed 42 must reproduce the published numbers ----------
    s42 = summaries[42]
    checks = [
        ("all pooled a", s42["all"]["a"], -0.0020, 6e-5),
        ("all pooled b", s42["all"]["b"], +0.0007, 6e-5),
        ("all pooled c", s42["all"]["c"], -0.0016, 6e-5),
        ("all recovery", s42["all"]["recovery"], 0.88, 0.011),
        ("fixed a", s42["fixed"]["a"], -0.0024, 6e-5),
        ("fixed b", s42["fixed"]["b"], -0.0001, 6e-5),
        ("fixed c", s42["fixed"]["c"], -0.0021, 6e-5),
        ("fixed recovery", s42["fixed"]["recovery"], 0.88, 0.011),
        ("fixed rec range lo", s42["fixed"]["rec_range"][0], 0.80, 0.011),
        ("fixed rec range hi", s42["fixed"]["rec_range"][1], 0.99, 0.011),
    ]
    lines.append("## Self-check (seed 42 recomputed in this code path vs published)\n")
    ok = True
    for name, got, want, tol in checks:
        good = abs(got - want) <= tol
        ok &= good
        lines.append(f"- {name}: recomputed {got:.4f} vs published {want:.4f} "
                     f"({'OK' if good else 'MISMATCH'})\n")
    harm42 = s42["fixed"]["harm"]
    harm_ok = (harm42["a"], harm42["b"], harm42["c"]) == (4, 15, 4)
    ok &= harm_ok
    lines.append(f"- fixed harm full12/hour1/sub: recomputed "
                 f"{harm42['a']}/{harm42['b']}/{harm42['c']} vs published 4/15/4 "
                 f"({'OK' if harm_ok else 'MISMATCH'})\n")
    assert ok, "seed-42 self-check failed; do not trust the 43/44 numbers"

    # ---- cross-seed verdict ---------------------------------------------------
    lines.append("## Cross-seed verdict\n")
    lines.append("| seed | set | (a) full12 | (b) hour1 | (c) sub | recovery | rec range (o x cal) | harm a/b/c |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for seed in (42, 43, 44):
        for setname in ("all", "fixed"):
            s = summaries[seed][setname]
            rng = (f"{s['rec_range'][0]:.2f}-{s['rec_range'][1]:.2f}"
                   if setname == "fixed" else "--")
            den = 48 if setname == "all" else 36
            lines.append(f"| {seed} | {setname} ({den}) | {fmt(s['a'])} | {fmt(s['b'])} "
                         f"| {fmt(s['c'])} | {s['recovery']:.2f} | {rng} "
                         f"| {s['harm']['a']}/{s['harm']['b']}/{s['harm']['c']} |")
    r43 = summaries[43]["fixed"]["recovery"]
    r44 = summaries[44]["fixed"]["recovery"]
    r43a = summaries[43]["all"]["recovery"]
    r44a = summaries[44]["all"]["recovery"]
    lines.append(f"\nSeed 42's coverage verdict (recovery 0.88 on both cell sets) "
                 f"{'HOLDS' if min(r43, r44, r43a, r44a) >= 0.7 else 'is WEAKENED'} "
                 f"at seeds 43/44: fixed-set recovery {r43:.2f} (seed 43) and {r44:.2f} "
                 f"(seed 44); all-cells recovery {r43a:.2f} and {r44a:.2f}. "
                 "A recovery near 1 means the size-matched 12 h-span subsample behaves "
                 "like the full 12 h slice, i.e. temporal span -- not impression count -- "
                 "is the operative variable.\n")

    # ---- validation summary ---------------------------------------------------
    lines.append("## Validation (retrain-once harness vs each seed's own sweep runs)\n")
    for seed in (43, 44):
        for o in (1, 2):
            with open(f"{RES}/mechanism_seeds/seed{seed}/mechanism_o{o}_validation.json") as f:
                v = json.load(f)
            lines.append(f"- seed {seed} origin {o}: full12 vs sweep-12h max |ddECE| = "
                         f"{v['full12_vs_sweep12_max_abs_diff']:.3e}; hour1 vs sweep-1h "
                         f"(gaps 1-3) max |ddECE| = "
                         f"{v['hour1_vs_sweep1_gaps123_max_abs_diff']:.3e}; 1h subset of "
                         "12h slice verified elementwise for all gaps.")
    with open(f"{RES}/mechanism_seeds_analysis.md", "w") as f:
        f.write("\n".join(lines))
    print("wrote results/mechanism_seeds_analysis.md; seed-42 self-check OK")


def exp_b() -> None:
    df = pd.concat([pd.read_csv(f"{RES}/recency/recency_o{o}.csv") for o in (1, 2)],
                   ignore_index=True)
    ages = ["age0", "age6", "age12", "age18"]
    lines = ["# Tier-E Experiment B — recency / staleness control (2026-10-23)\n",
             "Seed 42, both origins, fixed buckets gaps +1..+3 (the 24 h sweep has no "
             "gap-0 bucket). Per bucket, Platt / isotonic are fitted on four disjoint "
             "6-hour windows of the bucket's 24 h calibration slice (the whole day "
             "before the bucket), differing only in the data's age at serving time: "
             "age0 = slice-day hours 18-23 (the standard 6 h slice, [start-6h, start)), "
             "age6 = hours 12-17, age12 = hours 6-11, age18 = hours 0-5 "
             "([start-24h, start-18h)). Span (6 h) and size class are held fixed; the "
             "full24 condition (the sweep's 24 h slice) is the reference. Models were "
             "trained once per origin (train split md5-identical across calib-hours "
             "settings). dECE = ECE(calibrated) - ECE(raw) on the full bucket day; "
             "harm = dECE > 0. Denominators: per origin x age = 18 cells (3 gaps x 3 "
             "models x 2 calibrators); pooled per age = 36 cells.\n"]

    # window sizes
    wins = (df[df.condition != "full24"]
            .groupby(["origin", "gap", "condition"])["n_fit"].first().unstack())
    lines.append("## Window sizes (impressions fitted on)\n")
    lines.append("| origin | gap | age0 | age6 | age12 | age18 | full24 |")
    lines.append("|---|---|---|---|---|---|---|")
    full = (df[df.condition == "full24"].groupby(["origin", "gap"])["n_fit"].first())
    for (o, g), row in wins.iterrows():
        lines.append(f"| {o} | {g} | {int(row['age0']):,} | {int(row['age6']):,} "
                     f"| {int(row['age12']):,} | {int(row['age18']):,} "
                     f"| {int(full.loc[(o, g)]):,} |")

    def block(title: str, sub: pd.DataFrame, den: int) -> None:
        lines.append(f"\n## {title}\n")
        lines.append("| condition | mean dECE | harm |")
        lines.append("|---|---|---|")
        for cond in ages + ["full24"]:
            g = sub[sub.condition == cond]
            lines.append(f"| {cond} | {fmt(g['delta_ece'].mean())} "
                         f"| {int((g['delta_ece'] > 0).sum())}/{den} |")

    block("Pooled (both origins; 36 cells per age, 36 for full24)", df, 36)
    for o in (1, 2):
        block(f"Origin {o} (18 cells per age)", df[df.origin == o], 18)
    lines.append("\n## By calibrator (both origins pooled; 18 cells per age)\n")
    lines.append("| calibrator | age0 | age6 | age12 | age18 | full24 |")
    lines.append("|---|---|---|---|---|---|")
    for cal, g in df.groupby("calibrator"):
        cells = " | ".join(fmt(g[g.condition == c]["delta_ece"].mean())
                           for c in ages + ["full24"])
        lines.append(f"| {cal} | {cells} |")

    # paired contrasts vs age0
    piv = (df.pivot_table(index=["origin", "model", "gap", "calibrator"],
                          columns="condition", values="delta_ece"))
    lines.append("\n## Paired per-cell contrast vs age0 (36 cells)\n")
    lines.append("| contrast | mean d(dECE) | cells worse than age0 |")
    lines.append("|---|---|---|")
    for cond in ("age6", "age12", "age18", "full24"):
        d = piv[cond] - piv["age0"]
        lines.append(f"| {cond} - age0 | {fmt(d.mean())} | {int((d > 0).sum())}/36 |")

    pooled_means = [df[df.condition == c]["delta_ece"].mean() for c in ages]
    mono = all(pooled_means[i] <= pooled_means[i + 1] for i in range(3))
    per_origin_mono = {}
    for o in (1, 2):
        m = [df[(df.origin == o) & (df.condition == c)]["delta_ece"].mean() for c in ages]
        per_origin_mono[o] = all(m[i] <= m[i + 1] for i in range(3))
    f24 = df[df.condition == "full24"]["delta_ece"].mean()
    lines.append("\n## Verdict\n")
    lines.append(
        f"Pooled mean dECE by age: age0 {fmt(pooled_means[0])}, age6 "
        f"{fmt(pooled_means[1])}, age12 {fmt(pooled_means[2])}, age18 "
        f"{fmt(pooled_means[3])}; full24 reference {fmt(f24)}. The age sequence is "
        f"{'monotone non-decreasing (older = worse)' if mono else 'NOT monotone'} "
        f"pooled, and monotone in "
        f"{sum(per_origin_mono.values())} of 2 origins. "
        + ("Staleness at fixed span is therefore supported as the mechanism behind "
           "the 24 h rebound: the 24 h slice's oldest hours dilute its freshest "
           "ones, and a same-span window degrades as it ages."
           if mono else
           "Age alone at fixed 6 h span does not reproduce the 24 h rebound's "
           "degradation, so staleness by itself is NOT a sufficient explanation "
           "of the rebound; the rebound must also involve the span/size change "
           "itself (e.g. hour-mix composition of the longer window)."))

    # timings
    tim = (df.groupby("condition")[["fit_time_s", "apply_time_s"]].median())
    lines.append("\n## Refit wall-clock (median per fit, this experiment)\n")
    lines.append("| condition | fit_time_s | apply_time_s |")
    lines.append("|---|---|---|")
    for cond in ages + ["full24"]:
        lines.append(f"| {cond} | {tim.loc[cond, 'fit_time_s']:.3f} "
                     f"| {tim.loc[cond, 'apply_time_s']:.3f} |")

    lines.append("\n## Validation\n")
    for o in (1, 2):
        with open(f"{RES}/recency/recency_o{o}_validation.json") as f:
            v = json.load(f)
        lines.append(
            f"- origin {o}: train md5 h24 == h12: {v['train_md5_h24_eq_h12']}; "
            f"age0 window n == base 6 h slice n for all gaps: "
            f"{all(v['age0_n_eq_base_slice'].values())}; age0 vs base protocol "
            f"max |ddECE| = {v['age0_vs_base_max_abs_diff']:.3e}; full24 vs sweep-24h "
            f"max |ddECE| = {v['full24_vs_sweep24_max_abs_diff']:.3e}; raw ECE vs base "
            f"max |diff| = {v['raw_vs_base_max_abs_diff']:.3e}.")
    with open(f"{RES}/recency_analysis.md", "w") as f:
        f.write("\n".join(lines))
    print("wrote results/recency_analysis.md")


if __name__ == "__main__":
    exp_a()
    exp_b()
