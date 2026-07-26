from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

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
