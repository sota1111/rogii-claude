"""Kaggle Notebook entry point for the cycle-3 offset-trend toe-extrapolation champion.

Self-contained (Python standard library only, no internet). Reproduces the
``src.predict``/``src.data`` champion: for each horizontal well it fits the
vertical offset ``TVT_input + Z`` as a linear trend in MD over the heel rows
where ``TVT_input`` is known, then extrapolates the withheld toe as
``tvt = offset(MD) - Z``. Output is byte-identical to the repository's local
``src/predict.py`` generator, so the same file can be produced either way.

Kaggle runs this from ``/kaggle/input`` -> ``/kaggle/working``. The two paths may
be overridden with ``ROGII_INPUT_DIR`` / ``ROGII_OUTPUT`` for local byte-match
verification; the defaults are the Kaggle Notebook locations.
"""
from __future__ import annotations

import csv
import math
import os
import statistics
from dataclasses import dataclass
from pathlib import Path

HORIZONTAL_SUFFIX = "__horizontal_well.csv"

INPUT_DIR = Path(os.environ.get("ROGII_INPUT_DIR", "/kaggle/input"))
OUTPUT_PATH = Path(os.environ.get("ROGII_OUTPUT", "/kaggle/working/submission.csv"))


@dataclass(frozen=True)
class OffsetTrend:
    intercept: float
    slope: float
    fallback_offset: float
    terminal_offset: float


def fit_offset_trend(rows, recency_decay=0.0):
    if not math.isfinite(recency_decay) or recency_decay < 0.0:
        raise ValueError("recency_decay must be a finite non-negative number")
    points = []
    for row in rows:
        if not row.get("TVT_input") or not row.get("Z"):
            continue
        try:
            md = float(row["MD"])
            offset = float(row["TVT_input"]) + float(row["Z"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(md) and math.isfinite(offset):
            points.append((md, offset))
    if not points:
        raise ValueError("Cannot fit offset trend without finite heel observations")

    fallback = statistics.median(offset for _, offset in points)
    weights = [
        math.exp(recency_decay * (index - len(points) + 1) / max(1, len(points) - 1))
        for index in range(len(points))
    ]
    weight_sum = sum(weights)
    mean_md = sum(weight * md for weight, (md, _) in zip(weights, points)) / weight_sum
    mean_offset = (
        sum(weight * offset for weight, (_, offset) in zip(weights, points))
        / weight_sum
    )
    denominator = sum(
        weight * (md - mean_md) ** 2
        for weight, (md, _) in zip(weights, points)
    )
    if len(points) < 2 or denominator <= 1e-12:
        intercept, slope = fallback, 0.0
    else:
        slope = (
            sum(
                weight * (md - mean_md) * (offset - mean_offset)
                for weight, (md, offset) in zip(weights, points)
            )
            / denominator
        )
        intercept = mean_offset - slope * mean_md
    terminal_md = points[-1][0]
    terminal_offset = intercept + slope * terminal_md
    if not all(
        math.isfinite(value)
        for value in (intercept, slope, fallback, terminal_offset)
    ):
        intercept, slope, terminal_offset = fallback, 0.0, fallback
    return OffsetTrend(intercept, slope, fallback, terminal_offset)


def predict_offset_tvt(model, row):
    try:
        md = float(row["MD"]) if row.get("MD") else math.nan
    except (TypeError, ValueError):
        md = math.nan
    offset = (
        model.intercept + model.slope * md
        if math.isfinite(md)
        else model.terminal_offset
    )
    if not math.isfinite(offset):
        offset = model.fallback_offset
    try:
        z = float(row["Z"]) if row.get("Z") else math.nan
    except (TypeError, ValueError):
        z = math.nan
    prediction = offset - z if math.isfinite(z) else model.terminal_offset
    if not math.isfinite(prediction):
        prediction = model.fallback_offset
    return prediction


def _find_sample(input_dir):
    samples = list(input_dir.rglob("sample_submission.csv"))
    if len(samples) != 1:
        raise RuntimeError(f"Expected one sample_submission.csv, found {samples}")
    return samples[0]


def _index_horizontal_wells(input_dir):
    """Map each well to its horizontal file, preferring the competition test split.

    The same well id can appear under both ``train`` and ``test`` with different
    content (train exposes the target column). Submissions target the test split,
    so a path under a ``test`` directory always wins; ties break on sorted path.
    """
    index = {}
    for path in sorted(input_dir.rglob(f"*{HORIZONTAL_SUFFIX}")):
        well = path.name[: -len(HORIZONTAL_SUFFIX)]
        is_test = any(part.lower() == "test" for part in path.parts)
        if well not in index or (is_test and not index[well][0]):
            index[well] = (is_test, path)
    return {well: path for well, (_, path) in index.items()}


def main():
    sample_path = _find_sample(INPUT_DIR)
    horizontal_index = _index_horizontal_wells(INPUT_DIR)

    with sample_path.open(newline="", encoding="utf-8-sig") as handle:
        sample = csv.DictReader(handle)
        if sample.fieldnames != ["id", "tvt"]:
            raise ValueError(
                f"{sample_path} columns must be exactly ['id', 'tvt']; "
                f"got {sample.fieldnames}"
            )
        targets = list(sample)

    rows_by_well = {}
    models = {}
    output_rows = []
    for target in targets:
        target_id = target["id"]
        well, raw_index = target_id.rsplit("_", 1)
        index = int(raw_index)
        if well not in rows_by_well:
            horizontal_path = horizontal_index.get(well)
            if horizontal_path is None:
                raise FileNotFoundError(f"No horizontal well file for {well!r}")
            with horizontal_path.open(newline="", encoding="utf-8-sig") as handle:
                horizontal = csv.DictReader(handle)
                if "TVT_input" not in (horizontal.fieldnames or ()):
                    raise ValueError(f"{horizontal_path} is missing TVT_input")
                rows_by_well[well] = list(horizontal)
                models[well] = fit_offset_trend(
                    rows_by_well[well], recency_decay=8.0
                )
        rows = rows_by_well[well]
        if not 0 <= index < len(rows):
            raise IndexError(f"{target_id}: row {index} is outside {len(rows)} rows")
        prediction = predict_offset_tvt(models[well], rows[index])
        if not math.isfinite(prediction):
            raise ValueError(f"{target_id}: prediction is not finite")
        output_rows.append((target_id, f"{prediction:.10f}"))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["id", "tvt"])
        writer.writerows(output_rows)
    print(f"Wrote {len(output_rows)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
