# Champion selection: GR correlation

## Decision

The existing `tvt = TVT_input` pass-through baseline remains the champion.
The type-well GR shift-matching candidate did not pass the screen gate, so its
implementation was reverted as required by the promotion policy.

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

`champion.json` records `pass_through_baseline` with local RMSE `0.0` and a
`retained` decision. No GR-correlation predictor code remains in the champion
path.

## Evaluation contract update (SOT-2033)

The zero-error smoke KPI is no longer a candidate-promotion gate. Future
candidates use the leakage-free pseudo-blind screen→confirm contract documented
in `docs/data-and-metric.md`. This issue changes only the evaluation contract:
the registered `pass_through_baseline` champion and its executable submission
path remain unchanged until a candidate passes both pseudo-blind stages and is
confirmed separately on Kaggle.

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
candidate code remains in `src/`, and `champion.json` plus `src/predict.py`
continue to identify and execute the `TVT_input` pass-through champion. The
decision is **non-promotion**.
