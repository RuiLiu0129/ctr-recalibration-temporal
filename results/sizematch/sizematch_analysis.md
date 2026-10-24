# Size-matched random arm — retraining RQ1's random split at 3,338,217 rows

_Run 2026-10-24 (chain `run_sizematch.sh`). Question: Paper 2's RQ1
random arm trains on 4,000,000 rows while origin 1's temporal arm
trains on 3,338,217 — a disclosed confound. The Tier-F same-target-day
control excluded the sampling-pattern explanation but never varied
training size. This run varies size inside the random-split protocol
itself: the project's own `src.run_experiments` random path, the same
configs that produced the existing 4M random runs, with the ONLY
change `--train-cap 3338217` (plus output locations and
`--save-predictions`, which only writes arrays). 3 models (LR, DeepFM,
DCN-v2) × 3 seeds (42, 43, 44). Scope was pre-specified; both contrasts
below were defined before the run._

## Run provenance

**Construction** (unchanged `src.data_pipeline._prepare_random`):
whole-file random-priority reservoir pool of ⌈cap/0.7⌉ rows seeded per
seed, seeded 70/10/20 permutation split, `VocabEncoder` refit per seed
on that seed's train split only (min_count 10, max 20,000/field).

**Exact counts, every seed** (asserted in code; all gates passed):
pool 4,768,882 → train **3,338,217** / val **476,888** / test
**953,777** — exactly the implementation's arithmetic and exactly the
pre-registered expectations. (At cap 4,000,000 the same code produced
pool 5,714,286 → 4,000,000 / 571,428 / 1,142,858.)

**Vocab sizes** (encoder refit per seed, as protocol): seed 42: 33,150;
seed 43: 33,090; seed 44: 33,080 — vs 34,314 / 34,222 / 34,304 for the
same seeds' 4M arms (fewer training rows → fewer values pass
min_count, the expected direction).

**Sanity gates** (`sizematch_gates.py`, report
`results/sizematch/gates_report.json`, zero failures): fingerprint
cap/seed correct per seed; split counts exact; labels binary on every
split; split CTRs 0.1693–0.1703 (Avazu overall 0.1698, tolerance
±0.005); all saved predictions finite and in [0, 1]; raw metrics
recomputed from the saved float32 test predictions + labels reproduce
each seed's `metrics_random.csv` to ≤ 9.9e-17 on AUC / log loss / ECE.

**Timings** (2 CPUs): prepare 961 s (seed 42) / 1,216 s (seed 43) per
fresh processed root; model fits per seed — LR 13–18 s, DeepFM
239–254 s, DCN-v2 640–708 s; train+eval ≈ 950–1,010 s per seed.

**Execution incidents (no protocol deviation):** two VM restarts
killed the chain mid-fit (seed 42's DeepFM on the first launch, seed
44's DCN-v2 on the second). The chain was relaunched per its resume
design: completed steps were skipped via `.done` markers, interrupted
steps re-ran from their banked prepares (fingerprint match, prepare
0.0 s). Where a killed run and its re-run both produced a number, they
agree exactly at log precision (seed 42 LR AUC 0.7469 both launches;
seed 44 LR 0.7464 and DeepFM 0.7580 both launches) — the re-runs are
the same deterministic computations, not new draws. Seed 44's
first-launch prepare duration was overwritten in its step log by the
resume; all other timings are as logged.

**One structural note on contrast (b):** the reservoir keys are
pool-size-independent, so each seed's 4,768,882-row pool is a strict
subset of its 5,714,286-row pool; but the 70/10/20 permutation is drawn
at the pool's own length, so train/val/test assignment is redrawn, and
the two arms' test sets are different (overlapping) samples. Contrast
(b) is therefore a protocol-level comparison (same seed, same code,
only cap changed), not a per-impression paired one — exactly the
comparison the pre-specification defined.

## (a) Size-free split contrast — new random (3,338,217) vs temporal gap-0, same seed (raw)

Temporal references: seed 42 `results/metrics_temporal.csv`; seed 43
`results/seeds/seed43/metrics_temporal.csv`; seed 44
`results/seeds/seed44/metrics_temporal.csv` (gap-0 bucket, raw variant;
temporal train = 3,338,217 rows at every seed).

| model | seed | new-random AUC | temporal gap-0 AUC | ΔAUC | Δ log loss | ΔECE | ECE ratio |
|---|---|---|---|---|---|---|---|
| LR | 42 | 0.74686 | 0.73205 | +0.01481 | −0.02135 | −0.00359 | 0.608 |
| LR | 43 | 0.74473 | 0.73202 | +0.01271 | −0.01991 | −0.00329 | 0.639 |
| LR | 44 | 0.74636 | 0.73200 | +0.01436 | −0.02196 | −0.00352 | 0.613 |
| LR | **mean** | | | **+0.01396** | **−0.02107** | **−0.00347** | **0.620** |
| DeepFM | 42 | 0.75840 | 0.74312 | +0.01528 | −0.02188 | −0.00334 | 0.518 |
| DeepFM | 43 | 0.75615 | 0.74313 | +0.01302 | −0.02038 | −0.00489 | 0.484 |
| DeepFM | 44 | 0.75795 | 0.74122 | +0.01673 | −0.02304 | −0.00040 | 0.929 |
| DeepFM | **mean** | | | **+0.01501** | **−0.02176** | **−0.00288** | **0.644** |
| DCN-v2 | 42 | 0.75725 | 0.74078 | +0.01647 | −0.02256 | −0.00213 | 0.695 |
| DCN-v2 | 43 | 0.75428 | 0.74140 | +0.01288 | −0.02041 | −0.00565 | 0.446 |
| DCN-v2 | 44 | 0.75611 | 0.73934 | +0.01677 | −0.02309 | −0.00036 | 0.956 |
| DCN-v2 | **mean** | | | **+0.01537** | **−0.02202** | **−0.00271** | **0.699** |

For reference, the original 4M-based gap-0 AUC contrasts (4M random
minus the same temporal references) are LR +0.01346 / +0.01426 /
+0.01348 (mean +0.01373), DeepFM +0.01390 / +0.01514 / +0.01634 (mean
+0.01513), DCN-v2 +0.01414 / +0.01521 / +0.01618 (mean +0.01518) — the
paper's +0.0135..+0.0163 range. The matched-size means above
(+0.01396 / +0.01501 / +0.01537) sit in the same range, per seed and on
average. The LR ECE ratio at matched size is 0.608 / 0.639 / 0.613 per
seed (4M-based: 0.552 / 0.586 / 0.618); the deep models' ECE ratios
remain seed-unstable at matched size (DeepFM 0.484–0.929, DCN-v2
0.446–0.956), the same qualitative behavior as at 4M.

## (b) Pure size effect — new random (3,338,217) minus existing random (4,000,000), same seed (raw)

4M references: seed 42 `results/metrics_random.csv`; seed 43
`results/random_seeds/seed43/metrics_random.csv`; seed 44
`results/random_seeds/seed44/metrics_random.csv` (test bucket, raw).

| model | seed | ΔAUC | Δ log loss | ΔECE |
|---|---|---|---|---|
| LR | 42 | +0.00135 | −0.00084 | +0.00051 |
| LR | 43 | −0.00155 | +0.00141 | +0.00048 |
| LR | 44 | +0.00088 | −0.00099 | −0.00005 |
| LR | **mean** | **+0.00023** | **−0.00014** | **+0.00031** |
| DeepFM | 42 | +0.00138 | −0.00115 | −0.00026 |
| DeepFM | 43 | −0.00212 | +0.00163 | −0.00120 |
| DeepFM | 44 | +0.00038 | −0.00083 | +0.00020 |
| DeepFM | **mean** | **−0.00012** | **−0.00012** | **−0.00042** |
| DCN-v2 | 42 | +0.00233 | −0.00159 | −0.00028 |
| DCN-v2 | 43 | −0.00233 | +0.00138 | −0.00608 |
| DCN-v2 | 44 | +0.00059 | −0.00072 | +0.00307 |
| DCN-v2 | **mean** | **+0.00020** | **−0.00031** | **−0.00110** |

Per-seed ΔAUC spans −0.00233..+0.00233 with mixed signs for every
model; no model shows a consistent direction across seeds. The largest
single ΔECE cell (DCN-v2 seed 43, −0.00608) is against that seed's 4M
test ECE of 0.010637 — the seed-unstable deep-model ECE cell already
known from the 4M runs; the matched-size run's own seed-43 DCN-v2 test
ECE is 0.00456.

## Bottom line (descriptive only)

At matched training size (3,338,217 rows on both arms), the
random-vs-temporal gap-0 AUC contrast, mean across seeds 42/43/44, is
**LR +0.0140, DeepFM +0.0150, DCN-v2 +0.0154** (per-seed range across
models +0.0127..+0.0168) — the same magnitude as the original
4,000,000-row contrast (+0.0135..+0.0163 across models and seeds;
per-model means +0.0137 / +0.0151 / +0.0152). The pure size effect of
reducing the random arm's training set from 4,000,000 to 3,338,217
rows (−16.5%) on its own test AUC is **+0.0002 (LR), −0.0001 (DeepFM),
+0.0002 (DCN-v2)** on average, with every per-seed value inside
±0.0024 and signs mixed across seeds for all three models — roughly
two orders of magnitude smaller than the split contrast it was
suspected of driving, and without a consistent sign.

## Artifacts

- This file; per-cell metrics (3 seeds × 3 models × 3 variants):
  `results/sizematch/sizematch_cells.csv`; contrasts:
  `results/sizematch/sizematch_contrasts.csv`; gates:
  `results/sizematch/gates_report.json`.
- Audit bundle `results/sizematch/preds/` (42 files): per seed ×
  model, test predictions (raw / Platt / isotonic, float32) + test
  labels, and val raw predictions + val labels (val is the
  recalibration slice, as in the existing random arm).
- Per-seed run outputs: `results/sizematch/seed{42,43,44}/`
  (metrics_random.csv, bidding_random.csv, reliability_*.csv, preds/).
- Prepared splits: `data/processed_sizematch_s{42,43,44}/random/`
  (manifests carry fingerprints, split counts, field dims).
- Chain: `run_sizematch.sh`; step logs `sizematch_*.log`, markers
  `sizematch_*.done` (all 0), status `results/sizematch/STATUS.txt`;
  code: `sizematch_gates.py`, `sizematch_collate.py` (no audited
  pipeline code was modified; the run used the stock
  `src.run_experiments` entry point).
