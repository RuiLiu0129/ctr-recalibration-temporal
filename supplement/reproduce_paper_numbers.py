#!/usr/bin/env python3
"""One-command reproduction of the paper's quantitative claims (v19) from the
unrounded CSVs in this directory. Requires pandas and scipy. Run from this
directory:   python3 reproduce_paper_numbers.py

Each check prints the recomputed value next to the value printed in the paper
and marks it OK / MISMATCH. It also writes two derived tables that the paper
summarises but does not print in full:
  sweep_fixed_cells_three_seeds.csv   per seed x origin x calibrator x slice length
  sweep_harm_by_seed_origin.csv       harm counts per seed x origin x slice length
Exit code is the number of mismatches.
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

bad = 0
def check(label, got, want, tol=0.0):
    global bad
    ok = (abs(got - want) <= tol) if isinstance(want, (int, float)) else (got == want)
    bad += (not ok)
    print(f"[{'OK' if ok else 'MISMATCH'}] {label}: recomputed {got} | paper {want}")

# ---------------------------------------------------------------- 6-h grid
g = pd.read_csv("grid48_unrounded.csv")
g["harm"] = g.delta_ece > 0
check("harmful applications, seed 42, 6 h (of 48)", int(g.harm.sum()), 16)
rel = g.delta_ece / g.raw_ece
check("largest relative ECE reduction (%)", round(-rel.min() * 100), 61)
check("largest relative ECE increase (%)", round(rel.max() * 100), 107)
check("max |calibrated AUC - raw AUC|", round((g.cal_auc - g.raw_auc).abs().max(), 4), 0.0002)
dll = g.cal_logloss - g.raw_logloss
check("log-loss delta range", (round(dll.min(), 4), round(dll.max(), 4)), (-0.0009, 0.0006))
check("ECE and log loss move in opposite directions (of 48)", int(((g.delta_ece > 0) != (dll > 0)).sum()), 9)

h = pd.read_csv("seed_harm_counts.csv")
per_seed = h.groupby("seed").harm_cells.sum().to_dict()
check("harmful applications per seed (42/43/44)", tuple(per_seed[s] for s in (42, 43, 44)), (16, 14, 17))

# ------------------------------------------------- slice-vs-target bias rules
b = pd.read_csv("slice_target_bias.csv").merge(g, on=["origin", "model", "gap"])
sign_rule = np.sign(b.slice_signed_bias) != np.sign(b.target_signed_bias_raw)
over_rule = (b.target_signed_bias_raw - b.slice_signed_bias).abs() > b.target_signed_bias_raw.abs()
check("sign-flip rule correct (of 48)", int((sign_rule == b.harm).sum()), 40)
check("overshoot rule correct (of 48)", int((over_rule == b.harm).sum()), 42)
check("overshoot rule: flagged applications all harmed", (int(over_rule.sum()), int((over_rule & b.harm).sum())), (10, 10))

# ---------------------------------------------------- three-seed sweep (Table V)
S = pd.read_csv("sweep_seeds_cells.csv")
p = S.pivot_table(index=["seed", "calib_hours", "origin", "model", "bucket"], columns="variant", values="ece").reset_index()
parts = []
for c in ("platt", "isotonic"):
    t = p[["seed", "calib_hours", "origin", "model", "bucket"]].copy(); t["calibrator"] = c; t["delta_ece"] = p[c] - p["raw"]; parts.append(t)
D = pd.concat(parts); D["gap"] = D.bucket.str[-1].astype(int)
S42 = pd.read_csv("sweep_seed42_delta_ece.csv").rename(columns={"variant": "calibrator"}); S42["seed"] = 42
six = []
for o, f in ((1, pd.read_csv("seed_o1_delta_flags.csv")), (2, pd.read_csv("seed_o2_delta_ece.csv"))):
    for s in (43, 44):
        t = f[["model", "gap", "calibrator"]].copy(); t["delta_ece"] = f[f"dECE_s{s}"]; t["origin"] = o; t["seed"] = s; t["calib_hours"] = 6; six.append(t)
cols = ["seed", "calib_hours", "origin", "model", "gap", "calibrator", "delta_ece"]
A = pd.concat([D[cols], S42[cols], pd.concat(six)[cols]])
F = A[A.gap >= 1].copy(); F["harm"] = F.delta_ece > 0
F.sort_values(cols[:-1]).to_csv("sweep_fixed_cells_three_seeds.csv", index=False)
HSO = F.groupby(["seed", "origin", "calib_hours"]).harm.sum().unstack()
HSO.to_csv("sweep_harm_by_seed_origin.csv")
print("\nHarms per seed x origin (fixed cells, of 18):\n" + HSO.to_string())
hs = F.groupby(["seed", "calib_hours"]).harm.sum().unstack()
for hrs, want in ((1, (15, 15, 18)), (3, (16, 16, 20)), (6, (13, 14, 17)), (12, (4, 5, 3)), (24, (9, 11, 11))):
    check(f"Table V harms/seed at {hrs} h", tuple(int(hs.loc[s, hrs]) for s in (42, 43, 44)), want)
lr = F[F.model == "lr"].groupby(["seed", "calib_hours"]).harm.sum().unstack()
check("Table V LR harms/seed at 6 h", tuple(int(lr.loc[s, 6]) for s in (42, 43, 44)), (8, 8, 8))
check("Table V LR harms/seed at 12 h", tuple(int(lr.loc[s, 12]) for s in (42, 43, 44)), (2, 1, 1))
check("12 h has the fewest harms in every seed x origin", bool((HSO.idxmin(axis=1) == 12).all()), True)
mean_min = F.groupby(["seed", "origin", "calib_hours"]).delta_ece.mean().unstack().idxmin(axis=1)
check("seed x origin combinations where 12 h also has the lowest mean dECE", int((mean_min == 12).sum()), 5)
tv = (F.groupby(["origin", "calibrator", "calib_hours"]).delta_ece.mean().unstack() * 1e4).round(0)
want = {(1, "isotonic"): [-5, -13, -11, -38, -31], (1, "platt"): [-4, -2, -7, -17, -11],
        (2, "isotonic"): [18, 7, 2, -30, -30], (2, "platt"): [0, 6, 3, -14, -9]}
for k, v in want.items():
    check(f"Table V mean dECE x1e-4, origin {k[0]} {k[1]}", [int(x) + 0 for x in tv.loc[k].tolist()], v)

# ------------------------------------------------- size-matched control
M = pd.read_csv("mechanism_cells.csv"); M["cond"] = M.condition.where(~M.condition.str.startswith("sub"), "sub")
cell = M.groupby(["origin", "model", "gap", "calibrator", "cond"]).delta_ece.mean().unstack()
fx = cell[cell.index.get_level_values("gap") >= 1]
m = fx.mean()
check("control mean dECE full12 / hour1 / sub (gaps +1..+3)", tuple(round(m[c], 4) for c in ("full12", "hour1", "sub")), (-0.0024, -0.0001, -0.0021))
rec = (m["sub"] - m["hour1"]) / (m["full12"] - m["hour1"])
check("control recovery (gaps +1..+3)", round(rec, 2), 0.87)
ga = cell.mean(); rec_all = (ga["sub"] - ga["hour1"]) / (ga["full12"] - ga["hour1"])
check("control recovery incl. gap 0", round(rec_all, 2), 0.88, tol=0.005)
oc = fx.groupby(level=["origin", "calibrator"]).mean()
r = (oc["sub"] - oc["hour1"]) / (oc["full12"] - oc["hour1"])
check("control recovery range across origin x calibrator", (round(r.min(), 2), round(r.max(), 2)), (0.80, 0.99))
check("control harms full12 / hour1 / sub (of 36)", tuple(int((fx[c] > 0).sum()) for c in ("full12", "hour1", "sub")), (4, 15, 4))
Mt = M[M.gap >= 1]
ft = Mt.groupby(["cond", "calibrator"]).fit_time_s.median()
check("control same-run fit speed-up Platt / isotonic", (round(ft["full12", "platt"] / ft["sub", "platt"]), round(ft["full12", "isotonic"] / ft["sub", "isotonic"])), (20, 25))
check("control fit medians full12 Platt / iso (s)", (round(ft["full12", "platt"], 2), round(ft["full12", "isotonic"], 2)), (1.14, 0.58))
check("control fit medians sub Platt / iso (s)", (round(ft["sub", "platt"], 3), round(ft["sub", "isotonic"], 3)), (0.057, 0.023))

# ------------------------------------------------- timings (Table III)
T = pd.read_csv("sweep_seeds_timings.csv")
med = T.groupby(["calib_hours", "variant"]).fit_time_s.median()
for hrs, want in ((1, (0.085, 0.032)), (3, (0.26, 0.11)), (12, (1.41, 0.87)), (24, (2.80, 2.11))):
    nd = 3 if hrs == 1 else 2
    check(f"Table III fit medians at {hrs} h Platt / iso (s)", (round(med[hrs, "platt"], nd), round(med[hrs, "isotonic"], nd)), want)
ap = T.groupby("variant").apply_time_s.median()
check("apply-time medians within 0.2-0.5 s", bool(ap.between(0.2, 0.5).all()), True)

# ------------------------------------------------- RQ4
db = pd.read_csv("delta_ece_bootstrap.csv")
check("dECE bootstrap intervals excluding zero (of 48)", int(db.excludes_zero.sum()), 43)
rq = pd.read_csv("rq4_bootstrap.csv")
rec_ = rq[rq.comparison.isin(["platt-raw", "isotonic-raw"])]
check("profit intervals excluding zero (of 48)", int(rec_.ci_excludes_0.sum()), 24)
orc = pd.read_csv("rq4_binned_oracle.csv")
mm = rec_.assign(variant=rec_.comparison.str.replace("-raw", "", regex=False)).merge(
    orc[["origin", "model", "gap", "profit_raw"]], on=["origin", "model", "gap"]).merge(
    db[["origin", "model", "gap", "variant", "delta_ece"]], on=["origin", "model", "gap", "variant"])
mm["pct"] = mm.delta_profit / mm.profit_raw * 100
check("profit delta range (%)", (round(mm.pct.min(), 2), round(mm.pct.max(), 2)), (-0.21, 0.26))
check("profit delta range, three decimals (%)", (round(mm.pct.min(), 3), round(mm.pct.max(), 3)), (-0.207, 0.263))
check("sign agreement dECE vs dProfit (of 48)", int((mm.delta_ece * mm.pct < 0).sum()), 40)
for lab, sub, want in (("pooled", mm, -0.69), ("origin 1", mm[mm.origin == 1], -0.770), ("origin 2", mm[mm.origin == 2], -0.783)):
    nd = 2 if lab == "pooled" else 3
    check(f"Pearson r {lab}", round(pearsonr(sub.delta_ece, sub.pct)[0], nd), want)
co = rq[rq.comparison == "calib_oracle-raw"]
check("binned oracle below raw (point / significant)", (int((co.delta_profit < 0).sum()), int(((co.delta_profit < 0) & co.ci_excludes_0).sum())), (24, 23))
iso = pd.read_csv("rq4_isotonic_oracle.csv")
check("isotonic oracle median gain per 1,000", round(iso.delta_per_1000.median(), 3), 0.147)
lab = (orc.profit_label_oracle / orc.profit_raw)
check("label oracle multiple range", (round(lab.min(), 2), round(lab.max(), 2)), (1.75, 1.97))


# ------------------------------------------------- Tier-E: control at seeds 43/44
parts = [M.drop(columns=["cond"]).assign(seed=42)]
for sd in (43, 44):
    for o in (1, 2):
        parts.append(pd.read_csv(f"tiere/mechanism_seeds/seed{sd}/mechanism_o{o}.csv"))
MS = pd.concat(parts, ignore_index=True)
MS["cond"] = MS.condition.where(~MS.condition.str.startswith("sub"), "sub")
cs = MS[MS.gap >= 1].groupby(["seed", "origin", "model", "gap", "calibrator", "cond"]).delta_ece.mean().unstack()
def recov(x):
    mu = x.mean(); return (mu["sub"] - mu["hour1"]) / (mu["full12"] - mu["hour1"])
check("control recovery pooled over three seeds (gaps +1..+3)", round(recov(cs), 2), 0.88)
ps = [recov(cs.xs(sd, level="seed")) for sd in (42, 43, 44)]
check("control recovery per seed, range", (round(min(ps), 2), round(max(ps), 2)), (0.87, 0.88))
gso = cs.groupby(level=["seed", "origin", "calibrator"]).apply(recov)
check("control recovery per seed x origin x calibrator, range", (round(gso.min(), 2), round(gso.max(), 2)), (0.72, 1.00))
for c, want in (("sub", (4, 7, 4)), ("full12", (4, 5, 3)), ("hour1", (15, 15, 18))):
    check(f"control harms per seed, {c} (of 36)", tuple(int((cs.xs(sd, level="seed")[c] > 0).sum()) for sd in (42, 43, 44)), want)

# ------------------------------------------------- Tier-E: aged-window control (seed 42)
R = pd.concat([pd.read_csv(f"tiere/recency/recency_o{o}.csv") for o in (1, 2)])
R["harm"] = R.delta_ece > 0
agg = R.groupby("condition").agg(mean=("delta_ece", "mean"), harm=("harm", "sum"))
check("aged windows 0/6/12/18 h: mean dECE", tuple(round(agg.loc[f"age{a}", "mean"], 4) for a in (0, 6, 12, 18)), (-0.0003, -0.0018, 0.0004, 0.0004))
check("aged windows 0/6/12/18 h: harms (of 36)", tuple(int(agg.loc[f"age{a}", "harm"]) for a in (0, 6, 12, 18)), (13, 8, 19, 15))
check("full 24 h slice in the same runs: mean dECE, harms", (round(agg.loc["full24", "mean"], 4), int(agg.loc["full24", "harm"])), (-0.0019, 9))
hod = R.groupby("condition").window_hod.first()
check("window hours: age0 / age6 / age12 / age18", tuple(hod[f"age{a}"] for a in (0, 6, 12, 18)), ("18-23", "12-17", "6-11", "0-5"))
g12 = F[(F.seed == 42) & (F.calib_hours == 12)].delta_ece.mean()
check("afternoon 6 h window / 12 h slice gain (about three-quarters)", round(agg.loc["age6", "mean"] / g12, 2), 0.75, tol=0.05)
check("the two windows covering hours 00-11 are net-harmful", bool((agg.loc[["age12", "age18"], "mean"] > 0).all()), True)

# ------------------------------------------------- timing variation between runs (12 h fit)
var = (round((med[12, "platt"] / ft["full12", "platt"] - 1) * 100), round((med[12, "isotonic"] / ft["full12", "isotonic"] - 1) * 100))
check("12 h fit timing difference between runs, Platt / isotonic (%)", var, (23, 52))


# ------------------------------------------------- training sizes (Sec. III-B, IV-A, threat 4)
import re, math
def rows(log):
    return int(re.search(r"train rows: ([\d,]+)", open(log).read()).group(1).replace(",", ""))
o1, o2 = rows("logs/tierb_o1repro.log"), rows("logs/tierb_o2.log")
cap, tf, vf = 4_000_000, 0.7, 0.1
pool = math.ceil(cap / tf); rnd = int(pool * tf); rtest = pool - rnd - int(pool * vf)
check("training rows origin 1 / origin 2 (logs)", (o1, o2), (3338217, 3948439))
check("random arm: training rows / test rows (from the split rule)", (rnd, rtest), (4000000, 1142858))
check("random arm trains on X% more rows than origin 1 / origin 2", (round((rnd / o1 - 1) * 100), round((rnd / o2 - 1) * 100)), (20, 1))
wsize = R.drop_duplicates(["origin", "gap", "condition"]).groupby("condition").n_fit.median()
check("aged windows: median size afternoon (12-17) / evening (18-23), M", (round(wsize["age6"] / 1e6, 2), round(wsize["age0"] / 1e6, 2)), (1.22, 0.74))


# ------------------------------------------------- Tier-F: same-target-day control + bid sweep
tf = pd.read_csv("tierf/tierf_cells.csv")
check("Tier-F control: paired cells (2 origins x 3 seeds x 3 models x 4 gaps)", len(tf), 72)
check("Tier-F control: training rows per origin (exact size match)",
      tuple(int(tf[tf.origin == o].n_train.iloc[0]) for o in (1, 2)), (3338217, 3948439))
check("Tier-F control: pooled mean dAUC (draw - temporal)", round(tf.d_auc.mean(), 5), 0.00015)
check("Tier-F control: gap-0 mean dAUC", round(tf[tf.gap == 0].d_auc.mean(), 5), 0.00012)
check("Tier-F control: pooled mean dECE (draw - temporal)", round(tf.d_ece.mean(), 5), 0.00053)
check("Tier-F control: cells where the draw has lower ECE (of 72)", int((tf.d_ece < 0).sum()), 35)
bs = pd.read_csv("tierf/bidsweep_cells.csv")
check("bid sweep: overall profit-delta range across the 9 (mu, sigma) cells (%)",
      (round(bs.delta_pct.min(), 3), round(bs.delta_pct.max(), 3)), (-0.357, 0.343))
gm = bs.groupby(["mu", "sigma"]).delta_pct.mean()
check("bid sweep: per-cell mean profit-delta range (%)",
      (round(gm.min(), 3), round(gm.max(), 3)), (0.019, 0.040))
cen = bs[(bs.mu == -2.2) & (bs.sigma == 0.9)]
check("bid sweep: centre cell (mu=-2.2, sigma=0.9) profit-delta range (%)",
      (round(cen.delta_pct.min(), 3), round(cen.delta_pct.max(), 3)), (-0.207, 0.263))

# --------------------- v18: size-matched random-arm retraining (RQ1, threat 4)
# The random arm retrained at origin 1's exact training size (3,338,217 rows),
# 3 models x 3 seeds, same protocol with only the cap changed. Two pre-specified
# contrasts: (a) new random vs temporal gap 0, same seed (size-free split
# contrast); (b) new random vs the 4,000,000-row random arm, same seed (pure
# size effect). The capped-arm contrast the paper quotes alongside is (a) - (b).
sm = pd.read_csv("sizematch/sizematch_cells.csv")
check("size-matched cells (3 seeds x 3 models x 3 variants)", len(sm), 27)
check("size-matched retraining: cap / train / test rows (first cell)",
      (int(sm.cap.iloc[0]), int(sm.n_train.iloc[0]), int(sm.n_test.iloc[0])), (3338217, 3338217, 953777))
check("size-matched retraining: sizes identical across all 27 cells",
      (int(sm.cap.nunique()), int(sm.n_train.nunique()), int(sm.n_test.nunique())), (1, 1, 1))
sc = pd.read_csv("sizematch/sizematch_contrasts.csv")
seed_rows = sc[sc.seed != "mean"].copy(); seed_rows["seed"] = seed_rows.seed.astype(int)
raw = sm[sm.variant == "raw"][["seed", "model", "auc"]].rename(columns={"auc": "cell_auc"})
mg = seed_rows[seed_rows.contrast == "a_sizefree_split"].merge(raw, on=["seed", "model"])
check("size-matched cells vs contrasts: raw AUC agreement (max |diff|)",
      float((mg.cell_auc - mg.new_random_auc).abs().max()), 0.0, tol=1e-12)
check("contrasts: d_auc = new_random_auc - ref_auc in every seed row (max |dev|)",
      float((seed_rows.d_auc - (seed_rows.new_random_auc - seed_rows.ref_auc)).abs().max()), 0.0, tol=1e-12)
mean_rows = sc[sc.seed == "mean"].set_index(["contrast", "model"]).d_auc
recomp_means = seed_rows.groupby(["contrast", "model"]).d_auc.mean()
check("contrasts: printed mean rows agree with the seed rows (max |dev|)",
      float((recomp_means - mean_rows).abs().max()), 0.0, tol=1e-12)
a = seed_rows[seed_rows.contrast == "a_sizefree_split"].groupby("model").d_auc.mean()
check("size-matched gap-0 AUC contrast, seed means (LR / DeepFM / DCN-v2)",
      tuple(round(a[m], 4) for m in ("lr", "deepfm", "dcnv2")), (0.0140, 0.0150, 0.0154))
check("threat (4): matched-size contrast range across models (seed means)",
      (round(a.min(), 4), round(a.max(), 4)), (0.0140, 0.0154))
piv_a = seed_rows[seed_rows.contrast == "a_sizefree_split"].pivot(index=["model", "seed"], columns="contrast", values="d_auc")
piv_b = seed_rows[seed_rows.contrast == "b_pure_size"].pivot(index=["model", "seed"], columns="contrast", values="d_auc")
capped = (piv_a["a_sizefree_split"] - piv_b["b_pure_size"]).groupby("model").mean()
check("capped (4.00M) gap-0 AUC contrast, seed means (LR / DeepFM / DCN-v2)",
      tuple(round(capped[m], 4) for m in ("lr", "deepfm", "dcnv2")), (0.0137, 0.0151, 0.0152))
b = seed_rows[seed_rows.contrast == "b_pure_size"].groupby("model").d_auc.mean()
check("pure size effect on the random arm's AUC, seed means (LR / DeepFM / DCN-v2)",
      tuple(round(b[m], 4) for m in ("lr", "deepfm", "dcnv2")), (0.0002, -0.0001, 0.0002))
lr_ratio = seed_rows[(seed_rows.contrast == "a_sizefree_split") & (seed_rows.model == "lr")].sort_values("seed").ece_ratio
check("LR ECE ratio at matched size, per seed (42 / 43 / 44)",
      tuple(round(x, 3) for x in lr_ratio), (0.608, 0.639, 0.613))
check("LR ECE ratio at matched size: range as printed (0.61-0.64)",
      (round(lr_ratio.min(), 2), round(lr_ratio.max(), 2)), (0.61, 0.64))

# --------------------- v19: caliber pins + ECE-ratio / size-effect provenance
# v19 prints three distinct gap-0 calibers for the Tier-F draw (Sec. IV-A):
# the origin-1 gap-0 mean (+0.00015), the pooled-72 mean (+0.00015) and the
# pooled gap-0 mean (+0.00012). The Tier-F section above pins the two pooled
# values at print precision; these checks pin all three at full precision.
# (tierf/tierf_cells.csv is the Tier-F per-cell audit table, byte-identical to
# the project's results/tierf/audit_out/cells_with_deltas.csv.)
o1g0 = tf[(tf.origin == 1) & (tf.gap == 0)]
check("Tier-F control: origin-1 gap-0 cells (3 seeds x 3 models)", len(o1g0), 9)
check("Tier-F control: origin-1 gap-0 mean dAUC, full precision",
      round(float(o1g0.d_auc.mean()), 6), 0.000149)
check("Tier-F control: origin-1 gap-0 mean dAUC as printed",
      round(float(o1g0.d_auc.mean()), 5), 0.00015)
check("Tier-F control: pooled mean dAUC, full precision",
      round(float(tf.d_auc.mean()), 6), 0.000155)
check("Tier-F control: pooled gap-0 mean dAUC, full precision",
      round(float(tf[tf.gap == 0].d_auc.mean()), 6), 0.000117)
# v19 gives LR's gap-0 ECE ratio at BOTH training sizes: 0.61-0.64 at matched
# size (checked in the v18 section above) and 0.55-0.62 at the 4.00M cap,
# computed here as (random-arm raw ECE) / (origin-1 temporal gap-0 raw ECE)
# per seed from the per-seed metrics tables.
ra = pd.read_csv("seed_random_arm.csv")
ra = ra[(ra.model == "lr") & (ra.variant == "raw")].set_index("seed").ece
o1e = pd.read_csv("seed_o1_ece.csv")
o1e = o1e[(o1e.model == "lr") & (o1e.gap == 0) & (o1e.variant == "raw")].iloc[0]
cap_ratio = tuple(float(ra[s] / o1e[f"ece_s{s}"]) for s in (42, 43, 44))
check("LR ECE ratio at the cap (4.00M arm), per seed (42 / 43 / 44)",
      tuple(round(x, 4) for x in cap_ratio), (0.5519, 0.5864, 0.6185))
check("LR ECE ratio at the cap as printed per seed",
      tuple(round(x, 2) for x in cap_ratio), (0.55, 0.59, 0.62))
check("LR ECE ratio at the cap: range as printed in v19 (0.55-0.62)",
      (round(min(cap_ratio), 2), round(max(cap_ratio), 2)), (0.55, 0.62))
# Provenance of the printed pure size effects (+0.0002 / -0.0001 / +0.0002):
# they are the UNROUNDED seed means of contrast (b) (equal-size minus capped),
# which is why v19 declares them "from unrounded values" -- subtracting the
# rounded printed contrasts instead would give LR +0.0003 (final check).
check("pure size effect, unrounded seed means (LR / DeepFM / DCN-v2)",
      tuple(round(b[m], 6) for m in ("lr", "deepfm", "dcnv2")), (0.000227, -0.000118, 0.000196))
check("rounded-input subtraction would give LR +0.0003 (hence 'from unrounded values')",
      round(round(a["lr"], 4) - round(capped["lr"], 4), 4), 0.0003)

print(f"\n{bad} mismatch(es)")
sys.exit(bad)
