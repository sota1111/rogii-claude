from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from src.data import baseline_predictions, discover_wells
from src.data import load_submission_targets, write_submission
from src.evaluate import evaluate_baseline, rmse

HORIZONTAL_HEADER = ["MD", "X", "Y", "Z", "GR", "TVT_input", "TVT"]


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


class DataUtilitiesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def make_well(self, split: str, well: str, rows: list[list[object]]) -> None:
        directory = self.root / split
        write_csv(directory / f"{well}__horizontal_well.csv", HORIZONTAL_HEADER, rows)
        write_csv(
            directory / f"{well}__typewell.csv",
            ["TVT", "GR", "Geology"],
            [[100, 50, "sand"]],
        )

    def test_discovery_requires_complete_pairs(self) -> None:
        self.make_well("train", "well_a", [[1, 2, 3, 4, 5, 6, 7]])
        self.assertEqual([item.well for item in discover_wells(self.root / "train")], ["well_a"])
        (self.root / "train" / "well_b__horizontal_well.csv").touch()
        with self.assertRaisesRegex(ValueError, "Unpaired"):
            discover_wells(self.root / "train")

    def test_submission_preserves_sample_order_and_baseline_lookup(self) -> None:
        self.make_well(
            "test", "abc", [[1, 2, 3, 4, 5, 101, ""], [2, 2, 3, 4, 5, 102, ""]]
        )
        sample = self.root / "sample_submission.csv"
        write_csv(sample, ["id", "tvt"], [["abc_1", 0], ["abc_0", 0]])
        targets = load_submission_targets(sample)
        predictions = baseline_predictions(self.root / "test", targets)
        output = write_submission(sample, predictions, self.root / "submission.csv")
        self.assertEqual(output.read_text(), "id,tvt\nabc_1,102.0\nabc_0,101.0\n")

    def test_submission_rejects_wrong_ids(self) -> None:
        sample = self.root / "sample_submission.csv"
        write_csv(sample, ["id", "tvt"], [["abc_0", 0]])
        with self.assertRaisesRegex(ValueError, "Prediction ids differ"):
            write_submission(sample, {"wrong_0": 1.0}, self.root / "submission.csv")

    def test_rmse_and_deterministic_holdout_evaluation(self) -> None:
        score, count = rmse([(1, 2), (3, 3)])
        self.assertAlmostEqual(score, 2 ** -0.5)
        self.assertEqual(count, 2)
        for number in range(20):
            self.make_well("train", f"well_{number}", [[1, 2, 3, 4, 5, 10, 12]])
        result = evaluate_baseline(self.root / "train")
        self.assertEqual(result["metric"], "rmse")
        self.assertEqual(result["score"], 2.0)
        self.assertGreater(result["wells"], 0)


if __name__ == "__main__":
    unittest.main()
