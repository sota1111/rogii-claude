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
