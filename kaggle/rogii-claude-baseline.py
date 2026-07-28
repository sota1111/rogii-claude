"""Kaggle Notebook entry point for the cycle-2 retained zero-fallback champion."""
import csv
from pathlib import Path

samples = list(Path("/kaggle/input").rglob("sample_submission.csv"))
if len(samples) != 1:
    raise RuntimeError(f"Expected one sample_submission.csv, found {samples}")

output = Path("/kaggle/working/submission.csv")
with samples[0].open(newline="", encoding="utf-8-sig") as source:
    rows = list(csv.DictReader(source))
with output.open("w", newline="", encoding="utf-8") as target:
    writer = csv.writer(target, lineterminator="\n")
    writer.writerow(["id", "tvt"])
    writer.writerows((row["id"], f"{float(row['tvt']):.10f}") for row in rows)
print(f"Wrote {len(rows)} rows to {output}")
