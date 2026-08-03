# ROGII data format and local KPI

## Fetching data

Install and authenticate the Kaggle CLI, accept the competition rules, then run:

```bash
scripts/fetch_data.sh
```

The command downloads `rogii-wellbore-geology-prediction` into `data/raw/`.
Raw CSVs and archives are ignored by Git.

The downloaded data observed on 2026-07-26 contains 773 train wells and 3 test
wells. Every well is represented by:

- `<well>__horizontal_well.csv`: coordinates, GR, `TVT_input`, and train-only
  `TVT` truth (plus additional official interpretation columns).
- `<well>__typewell.csv`: reference `TVT` and `GR`; train files also include
  optional `Geology`.

`sample_submission.csv` contains `id,tvt`; each id is
`<well>_<zero-based-horizontal-row-index>`. Not every horizontal row is
necessarily submitted, so the sample is the source of truth for membership and
row order.

## Submission generation

```bash
python3 -m src.evaluate submission --output submission.csv
```

The writer requires exactly the sample's ids and preserves sample order.
Missing, extra, duplicate, malformed, or out-of-range ids fail explicitly.
The format-smoke command above uses a configurable constant. The champion
submission applies the guarded contact override with the offset-trend fallback:

```bash
python3 -m src.predict \
  --test-dir data/raw/test \
  --sample data/raw/sample_submission.csv \
  --train-dir data/raw/train \
  --output submission.csv
```

Predictions resolve per well by the first applicable layer:

1. **Guarded contact override** (`src/contact.py`) — wells whose same-id train
   copy reconstructs the trajectory from a formation contact
   (`TVT = ref_tvt - (Z - formation) + offset`) within 1.0 ft prefix RMSE.
2. **Particle filter** (`src/physics.py`, requires numpy) — for wells with no
   compatible train copy, a likelihood-weighted particle-filter ensemble tracks
   `U = TVT + Z` against the typewell GR signature, then a robust IRLS
   polynomial smooths the trajectory. This is the layer the hidden-test
   leaderboard actually scores.
3. **Offset trend** (`src/data.py`) — the recency-weighted linear fit
   `offset = TVT_input + Z` over MD, predicting `TVT = offset(MD) - Z`;
   degenerate fits use median heel offset. Used whenever the particle filter
   cannot run.

Omitting `--train-dir` disables the contact override (wells then take the
particle filter / offset trend). Without numpy the particle filter is skipped
and only the contact override and offset trend run.

## Toe-holdout champion gate

The gate retains the first 26.3774% of each train well as heel, matching the
aggregate real-test heel ratio, and hides the complete trailing suffix:

```bash
python3 -m src.evaluate toe-holdout --stage screen
python3 -m src.evaluate toe-holdout --stage confirm
```

Screen uses the first five sorted fold-0 wells and confirm uses all fold-0
wells. The candidate must have strictly lower RMSE and MAE than both zero
predictions and median constant-offset extrapolation at both stages. SOT-2092's
measurements are recorded in `docs/champion-selection.md`.

## Leakage-free pseudo-blind evaluation contract

Candidate selection uses a two-stage, deterministic pseudo-blind gate:

```bash
python3 -m src.evaluate pseudo-blind --stage screen
python3 -m src.evaluate pseudo-blind --stage confirm
```

- Seed: `2033`; fold: SHA-256 well split `0/5`.
- Blind unit: one contiguous internal interval covering 20% of each selected
  horizontal well. The interval start is derived from SHA-256 of
  `<seed>:<well-id>`.
- Screen: the first 5 sorted wells in fold 0.
- Confirm: every train well in fold 0 (156 wells in the 2026-07-26 download).
- Metrics: row-level RMSE and MAE, both lower-is-better.
- Leakage-free baseline: linear interpolation between the visible
  `TVT_input` values immediately outside the blind interval.
- Promotion threshold: candidate RMSE and MAE must each be no worse than the
  baseline (`delta <= 0`) at screen and confirm. A screen failure skips confirm.

The predictor contract removes both `TVT` and `TVT_input` from feature rows and
replaces blind-interval observations with `None`. Truth is retained only by the
scorer. This makes accidental use of either truth-equivalent column fail in
tests instead of producing a misleading zero-error score. All candidate
comparisons must use `evaluate_gate` or the same constants and sanitized
`BlindWell` contract.

## Legacy smoke metric and hold-out

The exact competition evaluator is not published in the downloaded CSVs. We
use row-level root mean squared error:

`RMSE = sqrt(mean((predicted_tvt - true_tvt)^2))`

This explicit approximation is consistent with the observed all-zero public
score of about 11551.955 and the roughly 11,000-unit TVT scale; lower is
better. It does not model hidden per-well aggregation or clipping.

Wells are assigned deterministically to five folds using SHA-256 of the well
id. Fold 0 is the default hold-out:

```bash
python3 -m src.evaluate baseline
```

On the official data downloaded 2026-07-26, the default fold contains 156
wells / 262,059 rows. The `tvt = TVT_input` baseline produces RMSE **0.0**.

This perfect value is expected because supplied train `TVT_input` equals train
`TVT` row-for-row. It is a format/evaluator smoke baseline, not evidence of
generalization or a useful champion threshold. Candidate promotion must use the
pseudo-blind gate above, not this legacy smoke metric.
