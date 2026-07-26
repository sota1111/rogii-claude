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
This command uses zero as a format-smoke prediction (override with
`--constant`). Official test rows to be predicted have blank `TVT_input`, so
that field is not used as a test predictor; it is used only for the train smoke
KPI below.

## Local metric and hold-out

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
generalization or a useful champion threshold. Later predictors should use the
same fold membership and avoid held-out `TVT` or `TVT_input` leakage.
