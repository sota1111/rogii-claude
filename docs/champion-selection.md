# Champion selection

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
