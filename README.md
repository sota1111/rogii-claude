# rogii-claude

Utilities for the `rogii-wellbore-geology-prediction` Kaggle competition.

See [docs/data-and-metric.md](docs/data-and-metric.md) for data retrieval,
submission generation, and the local hold-out KPI.

## Generate the champion submission

The current champion passes through each available test row's `TVT_input` and
uses the sample's `0.0` baseline for the withheld suffix. Explicit paths keep
the entry point independent of the current directory and `__file__`:

```bash
python src/predict.py \
  --test-dir /absolute/path/to/data/raw/test \
  --sample /absolute/path/to/data/raw/sample_submission.csv \
  --output /absolute/path/to/submission.csv
```

Because the competition accepts notebook submissions only, `kaggle/` contains
the dependency-free Kaggle script and kernel metadata that produce the same
retained champion output as the CLI.
