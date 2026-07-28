from __future__ import annotations

import csv
import math
from pathlib import Path

from src.predict import generate_submission


def test_generate_submission_matches_sample_contract(tmp_path: Path) -> None:
    sample_path = Path("data/raw/sample_submission.csv")
    output_path = tmp_path / "submission.csv"

    count = generate_submission(Path("data/raw/test"), sample_path, output_path)

    with sample_path.open(newline="", encoding="utf-8-sig") as handle:
        sample_rows = list(csv.DictReader(handle))
    with output_path.open(newline="", encoding="utf-8") as handle:
        submission = csv.DictReader(handle)
        assert submission.fieldnames == ["id", "tvt"]
        output_rows = list(submission)

    assert count == len(sample_rows) == len(output_rows)
    assert [row["id"] for row in output_rows] == [row["id"] for row in sample_rows]
    assert all(math.isfinite(float(row["tvt"])) for row in output_rows)


def test_predictions_differ_from_zero_for_withheld_toe(tmp_path: Path) -> None:
    output_path = tmp_path / "submission.csv"
    generate_submission(
        Path("data/raw/test"), Path("data/raw/sample_submission.csv"), output_path
    )
    with output_path.open(newline="", encoding="utf-8") as handle:
        values = [float(row["tvt"]) for row in csv.DictReader(handle)]
    assert values
    assert all(math.isfinite(value) for value in values)
    assert any(value != 0.0 for value in values)
