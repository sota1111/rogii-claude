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
