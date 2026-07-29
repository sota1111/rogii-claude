from __future__ import annotations

import csv
import math
import statistics
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

HORIZONTAL_SUFFIX = "__horizontal_well.csv"
TYPEWELL_SUFFIX = "__typewell.csv"
HORIZONTAL_REQUIRED = {"MD", "X", "Y", "Z", "GR", "TVT_input"}
TYPEWELL_REQUIRED = {"TVT", "GR"}


@dataclass(frozen=True)
class SubmissionTarget:
    id: str
    well: str
    index: int


@dataclass(frozen=True)
class WellFiles:
    well: str
    horizontal: Path
    typewell: Path


@dataclass(frozen=True)
class OffsetTrend:
    """A per-well linear model for the TVT + Z vertical offset."""

    intercept: float
    slope: float
    fallback_offset: float
    terminal_offset: float


def fit_offset_trend(
    rows: Sequence[Mapping[str, str]], *, recency_decay: float = 0.0
) -> OffsetTrend:
    """Fit offset = intercept + slope * MD with optional exponential recency weights."""
    if not math.isfinite(recency_decay) or recency_decay < 0.0:
        raise ValueError("recency_decay must be a finite non-negative number")
    points: list[tuple[float, float]] = []
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


def predict_offset_tvt(model: OffsetTrend, row: Mapping[str, str]) -> float:
    """Extrapolate TVT from the offset trend, with finite missing-feature fallbacks."""
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


def _read_rows(path: Path, required: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        return list(reader)


def discover_wells(split_dir: str | Path) -> list[WellFiles]:
    root = Path(split_dir)
    horizontal = {
        path.name.removesuffix(HORIZONTAL_SUFFIX): path
        for path in root.glob(f"*{HORIZONTAL_SUFFIX}")
    }
    typewells = {
        path.name.removesuffix(TYPEWELL_SUFFIX): path
        for path in root.glob(f"*{TYPEWELL_SUFFIX}")
    }
    if horizontal.keys() != typewells.keys():
        missing_horizontal = sorted(typewells.keys() - horizontal.keys())
        missing_typewell = sorted(horizontal.keys() - typewells.keys())
        raise ValueError(
            "Unpaired well files: "
            f"missing horizontal={missing_horizontal}, missing typewell={missing_typewell}"
        )
    return [
        WellFiles(well, horizontal[well], typewells[well])
        for well in sorted(horizontal)
    ]


def load_horizontal(path: str | Path, require_target: bool = False) -> list[dict[str, str]]:
    required = HORIZONTAL_REQUIRED | ({"TVT"} if require_target else set())
    return _read_rows(Path(path), required)


def load_typewell(path: str | Path) -> list[dict[str, str]]:
    return _read_rows(Path(path), TYPEWELL_REQUIRED)


def load_submission_targets(path: str | Path) -> list[SubmissionTarget]:
    sample_path = Path(path)
    with sample_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["id", "tvt"]:
            raise ValueError(
                f"{sample_path} columns must be exactly ['id', 'tvt']; "
                f"got {reader.fieldnames}"
            )
        targets: list[SubmissionTarget] = []
        seen: set[str] = set()
        for row in reader:
            target_id = row["id"]
            try:
                well, raw_index = target_id.rsplit("_", 1)
                index = int(raw_index)
            except (ValueError, AttributeError) as error:
                raise ValueError(f"Invalid submission id: {target_id!r}") from error
            if target_id in seen:
                raise ValueError(f"Duplicate submission id: {target_id}")
            seen.add(target_id)
            targets.append(SubmissionTarget(target_id, well, index))
    return targets


def baseline_predictions(
    test_dir: str | Path, targets: Sequence[SubmissionTarget]
) -> dict[str, float]:
    rows_by_well: dict[str, list[dict[str, str]]] = {}
    predictions: dict[str, float] = {}
    root = Path(test_dir)
    for target in targets:
        if target.well not in rows_by_well:
            rows_by_well[target.well] = load_horizontal(
                root / f"{target.well}{HORIZONTAL_SUFFIX}"
            )
        rows = rows_by_well[target.well]
        if not 0 <= target.index < len(rows):
            raise IndexError(
                f"{target.id}: row {target.index} is outside {len(rows)} rows"
            )
        predictions[target.id] = float(rows[target.index]["TVT_input"])
    return predictions


def write_submission(
    sample_path: str | Path,
    predictions: Mapping[str, float],
    output_path: str | Path,
) -> Path:
    targets = load_submission_targets(sample_path)
    expected = {target.id for target in targets}
    supplied = set(predictions)
    if expected != supplied:
        raise ValueError(
            f"Prediction ids differ: missing={sorted(expected - supplied)[:5]}, "
            f"extra={sorted(supplied - expected)[:5]}"
        )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["id", "tvt"])
        writer.writerows((target.id, predictions[target.id]) for target in targets)
    return output


def iter_train_pairs(split_dir: str | Path, wells: set[str]) -> Iterator[tuple[float, float]]:
    for files in discover_wells(split_dir):
        if files.well not in wells:
            continue
        for row in load_horizontal(files.horizontal, require_target=True):
            if row["TVT"] and row["TVT_input"]:
                yield float(row["TVT"]), float(row["TVT_input"])
