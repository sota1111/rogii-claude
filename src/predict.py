"""Generate a ROGII submission with the registered offset-trend champion."""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[0]).resolve().parents[1]))
from src.data import OffsetTrend, fit_offset_trend, predict_offset_tvt

HORIZONTAL_SUFFIX = "__horizontal_well.csv"


def generate_submission(test_dir: Path, sample_path: Path, output_path: Path) -> int:
    """Fit each well's heel and extrapolate its withheld toe in sample order."""
    with sample_path.open(newline="", encoding="utf-8-sig") as handle:
        sample = csv.DictReader(handle)
        if sample.fieldnames != ["id", "tvt"]:
            raise ValueError(
                f"{sample_path} columns must be exactly ['id', 'tvt']; got {sample.fieldnames}"
            )
        targets = list(sample)

    rows_by_well: dict[str, list[dict[str, str]]] = {}
    models: dict[str, OffsetTrend] = {}
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
        rows = rows_by_well[well]
        if not 0 <= index < len(rows):
            raise IndexError(f"{target_id}: row {index} is outside {len(rows)} rows")
        prediction = predict_offset_tvt(models[well], rows[index])
        if not math.isfinite(prediction):
            raise ValueError(f"{target_id}: TVT_input is not finite")
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
        description="Generate submission.csv with the registered offset-trend champion"
    )
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    count = generate_submission(args.test_dir, args.sample, args.output)
    print(f"Wrote {count} rows to {args.output}")


if __name__ == "__main__":
    main()
