# Champion selection

## Offline ML base TVT predictor — stage 1 (SOT-2393, 2026-08-04)

Stage 1 of the three-stage port of the reference kernel's ML stack
(`evgendvorkin/rogii-physics-lb-7-872-v48`, public LB 7.872). That kernel's ML
layer depends on external pretrained Kaggle datasets (LightGBM/CatBoost/Ridge
model packages) that cannot be shipped under this competition's offline
submission constraint (no internet, stdlib + numpy only, weights committed to the
repo). We therefore reproduce the *structure* of the ML base layer with a
predictor trained **offline on the local train wells** and distilled to a
committed, numpy-only gradient-boosted regression tree (`src/ml_predictor.py`,
mirrored verbatim into the kernel; distilled trees embedded in `_MODEL_JSON`, read
by an exec-compatible, `__file__`-independent loader).

Faithful to the reference base layer: the target is the TVT residual from the last
known heel value, the GBRT corrects on top of the recency-weighted offset-trend
base prediction, and the features are the portable subset the hidden-test path can
compute (toe geometry, GR derivatives, the GR-vs-typewell offset family
`gr - interp(last_tvt + o, tw_tvt, tw_gr)`, and portable base-prediction deltas).

Measured on the mandatory **leak-free** toe-holdout gate (the shipped model is
trained on the 617 train wells outside the fold-0 evaluation hold-out, so the
scored wells are unseen), `python3 -m src.evaluate toe-holdout --predictor ml`:

| Stage | Wells | Toe rows | ML base RMSE | Offset-trend RMSE | Champion PF RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| screen | 5 | 20,885 | 19.463 | 17.749 | 8.297 |
| confirm | 156 | 746,360 | **17.645** | 43.292 | **11.225** |

On the full confirm set the ML base predictor improves on the offset-trend
fallback base 2.45× (43.292 → 17.645) but does **not** beat the physics particle
filter standalone (17.645 vs 11.225): the toe residual left after the offset trend
is well-specific and needs the per-well GR matching the particle filter performs.
The predictor is therefore **not promoted** as a standalone champion replacement;
it is the foundation the stage-2 PF+ML gate blend (SOT-2394) builds on. The
champion (`champion.json`, `submission.csv`) is unchanged and nothing is submitted.

## Physics × ML gated blend — stage 2 (SOT-2394, 2026-08-04)

Stage 2 of the three-stage port blends the promoted champion likelihood-weighted
particle filter (`src/physics.py`) with the stage-1 offline ML base predictor
(`src/ml_predictor.py`) at the **hidden-test fallback** position — where no
same-well contact override exists. The blend is a single leak-free scalar
`weight` (the particle filter's share), `TVT_blend = weight·PF + (1−weight)·ML`
(`src/blend.py`, mirrored verbatim into the kernel under the `BLEND_SHARED_CODE`
markers; degrades to whichever side is finite so a missing ML predictor recovers
the pure-PF champion).

**Leak-free weight selection.** `weight` is grid-searched (0.50…1.00, step 0.05)
on the **fold-1** toe hold-out — disjoint from the fold-0 set the reported gate
scores — by `scripts/select_blend_weight.py`, and **frozen** as `BLEND_WEIGHT`
before any fold-0 confirm target is read. On fold-1 (154 wells, 738,137 toe rows)
the minimum-RMSE share was **0.75** (blend 11.151 vs PF 11.777, ML 16.182).

Measured on the mandatory leak-free fold-0 toe-holdout gate with the frozen
`weight = 0.75`, `python3 -m src.evaluate toe-holdout --predictor blend`:

| Stage | Wells | Toe rows | Blend RMSE | PF RMSE | ML RMSE | Beats both singles |
| --- | ---: | ---: | ---: | ---: | ---: | :---: |
| screen | 5 | 20,885 | 7.379 | 8.297 | 19.463 | yes |
| confirm | 156 | 746,360 | **11.173** | **11.225** | 17.645 | **yes** |

On the authoritative full confirm set the blend RMSE **11.173** beats
`min(PF 11.225, ML 17.645)`, i.e. it beats **both** standalone predictors
(−0.052 vs the PF champion, −6.472 vs ML). The gate criterion (blend confirm RMSE
< both singles) is met, so the blend is **promoted** as the hidden-test fallback
layer: `src/predict.py` (and the kernel mirror) now emit
`blend_trajectories(PF, ML)` where the contact override is absent.

Visible wells keep the guarded contact override untouched, so the local
editor-run `submission.csv` stays **byte-identical** (the override covers every
local test row; only the hidden-test trajectory changes). Exec byte-identity
between `src/` and the kernel is enforced by `tests/test_blend.py` and
`tests/test_kernel.py`. **Nothing is submitted here** — stage 3 (SOT-2395, gold
calibration) produces the final blend champion artifact and the parent
(SOT-2387) owns Kaggle submission.

## Gold-calibration overlay promotion (stage 3, SOT-2395, 2026-08-04)

The three-stage port of the reference kernel
([`evgendvorkin/rogii-physics-lb-7-872-v48`](https://www.kaggle.com/code/evgendvorkin/rogii-physics-lb-7-872-v48),
LB 7.872) finishes here with its final **gold-calibration** layer (`ROGII_GOLD_*`
= a per-well *visible-prefix self-verified anchor*). After the stage-2 blend, each
well backtests a portable candidate pool `{blend, pf, ml, poly}` on withheld tails
of its **own known heel** (no toe/target leaks), aggregates each candidate's RMSE
(`median + 0.10·std`), and only when a conservative gate clears
(`gain ≥ 1.0` / `consistency ≥ 0.67` / `best ≤ 12`) does it apply a soft, ramped,
clipped move of the best candidate into the **hidden toe**. All thresholds are
frozen from the reference kernel's conservative profile (`src/calibrate.py`
`GOLD_PROFILE`), so the overlay is leak-free by provenance, not tuned on our gate.

Measured on the mandatory leak-free fold-0 toe-holdout gate with the frozen
`weight = 0.75`, `python3 -m src.evaluate toe-holdout --predictor gold`:

| Stage | Wells | Toe rows | Gold RMSE | Blend RMSE | PF RMSE | Beats blend |
| --- | ---: | ---: | ---: | ---: | ---: | :---: |
| confirm | 156 | 746,360 | **11.115** | 11.173 | 11.225 | **yes** |

The gold overlay confirm RMSE **11.115** beats the stage-2 blend (**−0.058**) and
the PF champion (**−0.110**), so it is **promoted** as the hidden-test fallback:
`src/predict.py` (and the kernel mirror) now emit `gold_calibrate_trajectory`
where the contact override is absent. The gate is conservative — it fired on only
**36 of 156** wells; the other 120 keep the stage-2 blend unchanged. Visible wells
stay on the guarded contact override, so the local `submission.csv` is **byte-identical**
(`sha256 46d09239…`, `visible wells unchanged`); only the hidden-test trajectory
moves. Exec byte-identity between `src/calibrate.py` and the kernel is enforced by
`tests/test_calibrate.py` (`test_matches_kernel_implementation`,
`test_shared_block_is_byte_identical`). **Nothing is submitted here** — the parent
(SOT-2387) owns Kaggle submission of this final blended+calibrated champion.

## Beam-search + NCC alignment candidate promotion (SOT-2442, 2026-08-04)

Cycle 9 (human re-scope "取り込み優勝ノート", parent SOT-2438) ports the most
frequently reused *framework-free* technique from the public top notebooks
([`romantamrazov/rogii-better-solution-lb-9-956`](https://www.kaggle.com/code/romantamrazov/rogii-better-solution-lb-9-956),
`…/rogii-super-solution-lb-top-3`) as a **fourth gold-pool candidate** `beam`
(`src/align.py` `predict_beam_well`):

* **±2-delta beam search (backward-allowed Viterbi)** aligning the stratigraphic
  level `U = TVT + Z` to the typewell GR-vs-TVT signature. States are the residual
  of `U` around a robust heel-drift baseline on a grid; each step may move the grid
  index by `Δ ∈ [−2, +2]` at cost `move_cost·|Δ|`, and the emission is
  `((GR − typewell_GR(TVT + τ))/gs)² / emit_scale`. Traceback → toe `TVT`, averaged
  over `ALIGN_CONFIGS` `(emit_scale, move_cost)`.
* **Multi-scale NCC** (windows 8/15/25, `softmax(corr·3.0)`) measures a **leak-free
  global TVT registration** `τ` on the *known heel* only (reads known TVT + GR +
  typewell, never the toe), correcting a systematic TVT-frame offset.

It is numpy-only, `__file__`-independent, and fully **deterministic** (no RNG). The
gold gate back-tests `{blend, pf, ml, poly, beam}` per well and only adopts `beam`
with the same conservative margin, so the addition is non-regressive by construction.

Measured on the same leak-free fold-0 toe-holdout gate (`weight = 0.75`,
`python3 -m src.evaluate toe-holdout --predictor gold`):

| Stage | Wells | Toe rows | Gold RMSE | Blend RMSE | PF RMSE | Beats blend |
| --- | ---: | ---: | ---: | ---: | ---: | :---: |
| screen | 5 | 20,885 | **5.778** | 7.536 | 8.297 | **yes** |
| confirm | 156 | 746,360 | **11.074** | 11.173 | 11.225 | **yes** |

Adding `beam` lowers the gold confirm RMSE from the prior champion **11.115** to
**11.074** (**−0.041**), still beating the blend (**−0.099**) and the PF champion
(**−0.151**), so it is **promoted**. The conservative gate now fires on **43 of
156** wells (up from 36 — `beam` wins the heel backtest on the extra wells); the
rest keep the stage-2 blend. Visible wells stay on the guarded contact override, so
the local `submission.csv` is **byte-identical** (only the hidden-test trajectory
moves). Exec byte-identity of the `ALIGN_SHARED_CODE` block and
`predict_beam_well` between `src/align.py` and the kernel is enforced by
`tests/test_align.py`; full suite **55 passed**. **Nothing is submitted here** — the
parent (SOT-2438) owns Kaggle submission of the updated champion kernel.

## Particle-filter fallback promotion (cycle 5, 2026-08-03)

The contact override below reconstructs the three **visible** test wells nearly
perfectly, yet the submission still scored the offset trend's `44.456` — the
same value as the pure offset-trend cycle-4 champion. Kaggle rescoring swaps in
a hidden test set whose wells have no same-id train copy, so the contact
override never fires there and the *fallback* is what the leaderboard actually
measures. Improving the fallback is therefore the only lever on the real score.

The particle filter from the same public kernel
([`evgendvorkin/rogii-physics-lb-7-872-v48`](https://www.kaggle.com/code/evgendvorkin/rogii-physics-lb-7-872-v48))
was ported to numpy (`src/physics.py`, mirrored in the kernel). For each well a
32-seed, 400-particle likelihood-weighted ensemble tracks the stratigraphic
level `U = TVT + Z` through the withheld toe by matching the horizontal GR log
against the typewell GR-vs-TVT signature, then a robust IRLS degree-3 polynomial
smooths the trajectory. It replaces the recency-weighted offset trend in the
fallback position (after the contact override, before the offset trend, which is
kept for wells with a degenerate typewell).

Measured on the mandatory toe-holdout gate (first 26.3774% of each train well
kept as heel, complete trailing suffix hidden) against the effective champion:

| Stage | Wells | Toe rows | Particle filter RMSE | Offset-trend champion RMSE |
| --- | ---: | ---: | ---: | ---: |
| screen | 5 | 20,885 | 8.297 | 17.749 |
| confirm | 156 | 746,360 | 11.225 | 43.292 |

The candidate strictly improves the pooled RMSE and MAE at both stages (MAE
`4.830` vs `12.874` at screen; `6.638` vs `29.559` at confirm) and passes the
gate on 138 of 156 confirm wells. It is promoted. `src/physics.py` and the
kernel run the same PCG64 seed stream, so the two generators are byte-identical
on one platform (`tests/test_physics.py::…matches_kernel_implementation`).

## Guarded contact-override promotion (cycle 5, 2026-08-03)

Ported from the public physics kernel
[`evgendvorkin/rogii-physics-lb-7-872-v48`](https://www.kaggle.com/code/evgendvorkin/rogii-physics-lb-7-872-v48)
(public LB 7.872 vs. our previous 44.456). Every public test well ships with a
full train copy under the same well id; the train horizontal file carries the
complete `TVT` truth plus formation-contact depth columns
(`EGFDU`/`ASTNU`/`ANCC`/`ASTNL`/`EGFDL`/`BUDA`) and the train typewell labels
those formations in `Geology`. The champion reconstructs
`TVT = ref_tvt - (Z - formation) + offset` from the train copy, interpolates it
over MD, and uses it for every covered row.

The override is guarded and leakage-free at inference time: for each candidate
formation the reconstruction is validated against the test well's *visible*
`TVT_input` prefix, the best formation is kept, and the override fires only
when its prefix RMSE is at most 1.0 ft (plus ≥100 finite reconstruction rows
and ≥50 comparable prefix rows). Wells without a compatible train copy — the
expected situation for private-test wells — keep the SOT-2156 recency-weighted
offset-trend champion unchanged, which still passes the toe-holdout gates.

Measured prefix RMSE on the three public test wells (all rows covered):

| Well | Best formation | Prefix rows | Prefix RMSE (ft) |
| --- | --- | ---: | ---: |
| 000d7d20 | EGFDL | 1,442 | 0.0101 |
| 00bbac68 | ANCC | 1,545 | 0.0088 |
| 00e12e8b | ASTNU | 2,083 | 0.0079 |

The toe-holdout gate cannot score this candidate without self-leakage (a train
well's "train copy" is itself), so promotion rests on the inference-time
prefix gate above — the same guard the reference kernel uses before letting
the reconstruction replace its blend on the public LB.

## Theil–Sen robust-slope non-promotion (SOT-2157)

The candidate replaced the offset-trend slope with the median of pairwise
slopes and used `median(offset - slope * MD)` as its intercept. For large
wells, 20,000 index pairs were selected with a fixed seed before any toe
targets were read, keeping the experiment deterministic and leakage-free.

The issue originally named the SOT-2092 global-OLS model as champion, but
SOT-2156 had already promoted the recency-weighted model on `main`. The
mandatory screen therefore compared Theil–Sen with the effective champion.

| Stage | Wells | Toe rows | Method | RMSE | MAE |
| --- | ---: | ---: | --- | ---: | ---: |
| screen | 5 | 20,885 | Theil–Sen | 75.338282 | 48.602398 |
| screen | 5 | 20,885 | recency-weighted champion | 17.749175 | 12.873673 |
| screen | 5 | 20,885 | global OLS (historical) | 88.125737 | 56.834512 |
| confirm | — | — | — | — | — |

Theil–Sen improved both metrics relative to historical global OLS, but
regressed against the effective champion by 57.589106 RMSE and 35.728725 MAE.
It therefore failed the mandatory screen gate, and confirm was not run.
Candidate code was removed. `champion.json` and `src/predict.py` remain on the
SOT-2156 recency-weighted champion, so no kernel refresh or Kaggle submission
handoff is required.

## Decision

The champion is `recency_weighted_offset_trend_toe_extrapolation`. It fits
`TVT_input + Z = intercept + slope * MD` on each test well's visible heel with
exponentially increasing row-recency weights (`recency_decay=8`), then
extrapolates the fitted offset into the withheld toe before subtracting `Z`.

## Recency-weighted local-slope promotion (SOT-2156)

The screen compared pre-fixed decay values `1, 2, 4, 8, 12`; decay 8 had the
lowest screen RMSE and was frozen before confirm. A zero decay is the prior
global-OLS champion. Weights increase monotonically toward the toe boundary
and use row order only, so withheld targets do not influence the fit.

| Stage | Wells | Toe rows | Method | RMSE | MAE |
| --- | ---: | ---: | --- | ---: | ---: |
| screen | 5 | 20,885 | recency-weighted (decay 8) | 17.749175 | 12.873673 |
| screen | 5 | 20,885 | global OLS champion | 88.125737 | 56.834512 |
| confirm | 156 | 746,360 | recency-weighted (decay 8) | 43.292298 | 29.559006 |
| confirm | 156 | 746,360 | global OLS champion | 82.664390 | 54.371827 |

The candidate strictly improves both metrics at screen and confirm and is
promoted. `champion.json` and `src/predict.py` now register and execute the
recency-weighted model. Median/terminal-offset finite fallbacks remain intact.
Exec-compatible kernel refresh and Kaggle proof are tracked by SOT-2158.

## Offset-trend toe extrapolation promotion (SOT-2092)

The evaluation hides the trailing 73.6226% of each selected train well, matching
the aggregate visible-heel fraction in test (5,070 / 19,221 = 26.3774%).
Promotion requires strictly lower RMSE and MAE than both effective baselines.

| Stage | Wells | Toe rows | Method | RMSE | MAE |
| --- | ---: | ---: | --- | ---: | ---: |
| screen | 5 | 20,885 | offset trend | 88.125737 | 56.834512 |
| screen | 5 | 20,885 | constant offset | 106.334128 | 94.262982 |
| screen | 5 | 20,885 | zeros | 11812.168011 | 11803.536606 |
| confirm | 156 | 746,360 | offset trend | 82.664390 | 54.371827 |
| confirm | 156 | 746,360 | constant offset | 127.437429 | 109.441774 |
| confirm | 156 | 746,360 | zeros | 11590.406109 | 11572.944262 |

Both stages pass on both metrics. `champion.json` and `src/predict.py` therefore
register and execute the new method. Degenerate fits and missing features take
finite median/terminal-offset fallbacks.

## Candidate

The evaluated candidate treated each type well as a `GR(TVT)` reference curve.
For each horizontal-well point it searched within ±25 TVT units of `TVT_input`,
selected the reference point with the nearest GR value (using distance from
`TVT_input` as a tie-breaker), and applied a non-decreasing TVT constraint.

## Local KPI gate

The deterministic fold-0 hold-out from `src.evaluate.holdout_wells` was used
with row-level RMSE (lower is better).

| Stage | Wells | Rows | Pass-through RMSE | GR candidate RMSE | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| screen | 5 | 8,264 | 0.000000 | 24.928453 | fail |
| confirm | — | — | — | — | not run; screen is a mandatory promotion gate |

The screen wells were the first five sorted well ids in the deterministic
fold-0 hold-out. The candidate was compared against the baseline on exactly
the same rows.

The baseline is already perfect on the available train labels because
`TVT_input` equals `TVT` row-for-row. A lower RMSE is mathematically impossible,
and the candidate increased error by 24.928453. Running the higher-N confirm
stage could not reverse the failed mandatory screen gate or establish a score
below zero, so it was intentionally skipped.

## Champion state

At that time, `champion.json` recorded `pass_through_baseline` with local RMSE
`0.0` and a `retained` decision. No GR-correlation predictor code remained in
the champion path. SOT-2092 supersedes this historical state.

## Evaluation contract update (SOT-2033)

The zero-error smoke KPI is no longer a candidate-promotion gate. Future
candidates use the leakage-free pseudo-blind screen→confirm contract documented
in `docs/data-and-metric.md`. This issue changes only the evaluation contract:
the then-registered `pass_through_baseline` champion and its executable
submission path remained unchanged pending a candidate passing a suitable
blind evaluation.

The contract baseline was executed against the 2026-07-26 train download:

| Stage | Wells | Blind rows | RMSE | MAE |
| --- | ---: | ---: | ---: | ---: |
| screen | 5 | 1,654 | 11.035684 | 8.436288 |
| confirm | 156 | 52,419 | 11.561729 | 7.804169 |

These values establish the comparison reference; they are not a new submission
champion score.

## Constrained TVT correction evaluation (SOT-2034)

### Hypothesis and safeguards

This experiment did not retry the earlier unconstrained GR shift. It treated
the leakage-free interpolation output as a strong TVT prior, matched smoothed
horizontal-well GR to the same well's type-well `GR(TVT)` curve only within
±10 TVT units of that prior, and shrank the matched shift before applying it.
The applied correction was:

- clipped to a small per-row maximum;
- smoothed with a 21-row moving window;
- constrained to preserve the prior's local monotonic direction; and
- evaluated without exposing blind-interval `TVT` or `TVT_input`.

Three simple, pre-fixed correction caps were screened. All used a 0.1 shrink
factor and the same GR-distance objective with a 0.25 TVT-distance penalty.

### Screen results

The inherited deterministic screen used seed 2033, fold 0/5, a 20% contiguous
blind interval, and wells `01869cd4`, `03a935ae`, `044af7d1`, `0498acab`, and
`052d64df` (1,654 blind rows).

| Candidate | Correction cap | RMSE | Δ RMSE | MAE | Δ MAE | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| interpolation baseline | 0 | 11.035684 | 0 | 8.436288 | 0 | reference |
| constrained-small | ±0.10 | 11.038747 | +0.003063 | 8.433474 | -0.002813 | fail |
| constrained-medium | ±0.25 | 11.037853 | +0.002169 | 8.433445 | -0.002843 | fail |
| constrained-large | ±0.50 | 11.039727 | +0.004044 | 8.438754 | +0.002466 | fail |

`constrained-medium` was the pre-fixed best candidate by the primary metric,
RMSE. It improved MAE by 0.002843 but regressed RMSE by 0.002169, so it failed
the inherited requirement that neither metric regress.

### Confirm and promotion decision

Confirm was not run. The SOT-2033 contract makes a passing screen mandatory
before the single screen winner may enter confirm; running confirm after the
RMSE failure could not make the candidate eligible for promotion.

The candidate implementation was experimental and has been removed. No
candidate code remains in `src/`. At the end of that experiment,
`champion.json` plus `src/predict.py` still identified the `TVT_input`
pass-through champion. Its decision was **non-promotion**; SOT-2092 later
replaced that champion.
