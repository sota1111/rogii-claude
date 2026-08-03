"""Kaggle Notebook entry point for the cycle-5 guarded contact-override champion.

Self-contained (Python standard library only, no internet). Reproduces the
``src.predict`` champion: wells whose train copy (same well id, full ``TVT``
truth plus formation-contact columns) reconstructs the trajectory within
``PREFIX_RMSE_LIMIT`` of the visible ``TVT_input`` prefix are predicted by the
contact reconstruction ``TVT = ref_tvt - (Z - formation) + offset``; all other
rows fall back to the cycle-4 recency-weighted offset trend. Output is
byte-identical to the repository's local ``src/predict.py`` generator.

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
TYPEWELL_SUFFIX = "__typewell.csv"

REF_COLS = ("EGFDU", "ASTNU", "ANCC", "ASTNL", "EGFDL", "BUDA")
MIN_VALID_PHYS_ROWS = 100
MIN_KNOWN_PREFIX_ROWS = 50
PREFIX_RMSE_LIMIT = 1.0

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


@dataclass(frozen=True)
class ContactCurve:
    ref_col: str
    prefix_rmse: float
    mds: tuple
    tvts: tuple

    def covers(self, md):
        return math.isfinite(md) and self.mds[0] <= md <= self.mds[-1]

    def predict(self, md):
        return _interp(md, self.mds, self.tvts)


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _interp(x, xs, ys):
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    lo, hi = 0, len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    span = xs[hi] - xs[lo]
    if span <= 0.0:
        return ys[lo]
    return ys[lo] + (x - xs[lo]) / span * (ys[hi] - ys[lo])


def fit_contact_curve(train_horizontal, train_typewell, ref_col):
    ref_tvts = [
        _to_float(row.get("TVT"))
        for row in train_typewell
        if (row.get("Geology") or "").strip() == ref_col
    ]
    ref_tvts = [value for value in ref_tvts if math.isfinite(value)]
    if not ref_tvts:
        return None
    ref_tvt = min(ref_tvts)

    samples = []
    residuals = []
    for row in train_horizontal:
        md = _to_float(row.get("MD"))
        raw = ref_tvt - (_to_float(row.get("Z")) - _to_float(row.get(ref_col)))
        if math.isfinite(md) and math.isfinite(raw):
            samples.append((md, raw))
            truth = _to_float(row.get("TVT"))
            if math.isfinite(truth):
                residuals.append(truth - raw)
    if len(samples) < MIN_VALID_PHYS_ROWS or not residuals:
        return None
    offset = sum(residuals) / len(residuals)
    samples.sort()
    return (
        tuple(md for md, _ in samples),
        tuple(raw + offset for _, raw in samples),
    )


def best_contact_curve(test_horizontal, train_horizontal, train_typewell):
    known = []
    for row in test_horizontal:
        tvt = _to_float(row.get("TVT_input"))
        md = _to_float(row.get("MD"))
        if math.isfinite(tvt) and math.isfinite(md):
            known.append((md, tvt))
    best = None
    for ref_col in REF_COLS:
        curve = fit_contact_curve(train_horizontal, train_typewell, ref_col)
        if curve is None:
            continue
        mds, tvts = curve
        comparable = [(md, tvt) for md, tvt in known if mds[0] <= md <= mds[-1]]
        if len(comparable) < MIN_KNOWN_PREFIX_ROWS:
            continue
        squares = [(_interp(md, mds, tvts) - tvt) ** 2 for md, tvt in comparable]
        rmse = math.sqrt(sum(squares) / len(squares))
        if not math.isfinite(rmse):
            continue
        if best is None or rmse < best.prefix_rmse:
            best = ContactCurve(ref_col, rmse, mds, tvts)
    if best is None or best.prefix_rmse > PREFIX_RMSE_LIMIT:
        return None
    return best


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


def _index_train_wells(input_dir):
    """Map each train-split well to its (horizontal, typewell) file pair."""
    horizontal = {}
    typewells = {}
    for path in sorted(input_dir.rglob(f"*{HORIZONTAL_SUFFIX}")):
        if any(part.lower() == "train" for part in path.parts):
            horizontal.setdefault(path.name[: -len(HORIZONTAL_SUFFIX)], path)
    for path in sorted(input_dir.rglob(f"*{TYPEWELL_SUFFIX}")):
        if any(part.lower() == "train" for part in path.parts):
            typewells.setdefault(path.name[: -len(TYPEWELL_SUFFIX)], path)
    return {
        well: (horizontal[well], typewells[well])
        for well in horizontal.keys() & typewells.keys()
    }


def _read_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _load_contact_curve(train_index, well, test_rows):
    pair = train_index.get(well)
    if pair is None:
        return None
    try:
        return best_contact_curve(test_rows, _read_rows(pair[0]), _read_rows(pair[1]))
    except Exception as error:  # guarded: any train-copy defect keeps the fallback
        print(f"contact override skipped for {well}: {error}")
        return None


def main():
    sample_path = _find_sample(INPUT_DIR)
    horizontal_index = _index_horizontal_wells(INPUT_DIR)
    train_index = _index_train_wells(INPUT_DIR)

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
    curves = {}
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
                curves[well] = _load_contact_curve(
                    train_index, well, rows_by_well[well]
                )
                curve = curves[well]
                if curve is not None:
                    print(
                        f"contact override {well}: ref={curve.ref_col} "
                        f"prefix_rmse={curve.prefix_rmse:.4f}"
                    )
        rows = rows_by_well[well]
        if not 0 <= index < len(rows):
            raise IndexError(f"{target_id}: row {index} is outside {len(rows)} rows")
        curve = curves[well]
        try:
            md = float(rows[index]["MD"]) if rows[index].get("MD") else math.nan
        except (TypeError, ValueError):
            md = math.nan
        if curve is not None and curve.covers(md):
            prediction = curve.predict(md)
        else:
            prediction = predict_offset_tvt(models[well], rows[index])
        if not math.isfinite(prediction):
            raise ValueError(f"{target_id}: prediction is not finite")
        # Seven decimals keep the serialized artifact stable across Python
        # runtimes while remaining far below the competition metric precision.
        output_rows.append((target_id, f"{prediction:.7f}"))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["id", "tvt"])
        writer.writerows(output_rows)
    print(f"Wrote {len(output_rows)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
