"""Generate a ROGII submission with the guarded contact-override champion."""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[0]).resolve().parents[1]))
from src.contact import ContactCurve, best_contact_curve
from src.data import OffsetTrend, fit_offset_trend, predict_offset_tvt

HORIZONTAL_SUFFIX = "__horizontal_well.csv"
TYPEWELL_SUFFIX = "__typewell.csv"


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _load_contact_curve(
    train_dir: Path | None, well: str, test_rows: list[dict[str, str]]
) -> ContactCurve | None:
    """Guarded same-well contact reconstruction; None keeps the fallback."""
    if train_dir is None:
        return None
    horizontal_path = train_dir / f"{well}{HORIZONTAL_SUFFIX}"
    typewell_path = train_dir / f"{well}{TYPEWELL_SUFFIX}"
    if not horizontal_path.is_file() or not typewell_path.is_file():
        return None
    return best_contact_curve(
        test_rows, _read_rows(horizontal_path), _read_rows(typewell_path)
    )


def generate_submission(
    test_dir: Path,
    sample_path: Path,
    output_path: Path,
    train_dir: Path | None = None,
) -> int:
    """Predict each well via guarded contact override, offset trend otherwise."""
    with sample_path.open(newline="", encoding="utf-8-sig") as handle:
        sample = csv.DictReader(handle)
        if sample.fieldnames != ["id", "tvt"]:
            raise ValueError(
                f"{sample_path} columns must be exactly ['id', 'tvt']; got {sample.fieldnames}"
            )
        targets = list(sample)

    rows_by_well: dict[str, list[dict[str, str]]] = {}
    models: dict[str, OffsetTrend] = {}
    curves: dict[str, ContactCurve | None] = {}
    output_rows: list[tuple[str, str]] = []
    for target in targets:
        target_id = target["id"]
        try:
            well, raw_index = target_id.rsplit("_", 1)
            index = int(raw_index)
        except (AttributeError, ValueError) as error:
            raise ValueError(f"Invalid submission id: {target_id!r}") from error
        if well not in rows_by_well:
            horizontal_path = test_dir / f"{well}{HORIZONTAL_SUFFIX}"
            with horizontal_path.open(newline="", encoding="utf-8-sig") as handle:
                horizontal = csv.DictReader(handle)
                if "TVT_input" not in (horizontal.fieldnames or ()):
                    raise ValueError(f"{horizontal_path} is missing TVT_input")
                rows_by_well[well] = list(horizontal)
                models[well] = fit_offset_trend(
                    rows_by_well[well], recency_decay=8.0
                )
                curves[well] = _load_contact_curve(
                    train_dir, well, rows_by_well[well]
                )
        rows = rows_by_well[well]
        if not 0 <= index < len(rows):
            raise IndexError(f"{target_id}: row {index} is outside {len(rows)} rows")
        curve = curves[well]
        try:
            md = float(rows[index]["MD"])
        except (KeyError, TypeError, ValueError):
            md = math.nan
        if curve is not None and curve.covers(md):
            prediction = curve.predict(md)
        else:
            prediction = predict_offset_tvt(models[well], rows[index])
        if not math.isfinite(prediction):
            raise ValueError(f"{target_id}: prediction is not finite")
        # Seven decimals keep the serialized artifact stable across the local
        # Python 3.11 and Kaggle Python 3.12 math implementations.
        output_rows.append((target_id, f"{prediction:.7f}"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["id", "tvt"])
        writer.writerows(output_rows)
    return len(output_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate submission.csv with the guarded contact-override champion"
    )
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=None,
        help="train split with same-well copies; omit to disable the override",
    )
    args = parser.parse_args()
    count = generate_submission(
        args.test_dir, args.sample, args.output, train_dir=args.train_dir
    )
    print(f"Wrote {count} rows to {args.output}")


if __name__ == "__main__":
    main()
