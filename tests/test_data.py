from __future__ import annotations

import csv
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.data import (
    baseline_predictions,
    discover_wells,
    fit_offset_trend,
    load_submission_targets,
    predict_offset_tvt,
    write_submission,
)
from src.evaluate import evaluate_baseline, rmse
from src.predict import generate_submission

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

    def test_champion_submission_and_exec_runtime_compatibility(self) -> None:
        self.make_well(
            "test", "abc", [[1, 2, 3, 4, 5, 101, ""], [2, 2, 3, 4, 5, 102, ""]]
        )
        sample = self.root / "sample_submission.csv"
        output = self.root / "direct.csv"
        write_csv(sample, ["id", "tvt"], [["abc_1", 0], ["abc_0", 0]])
        self.assertEqual(generate_submission(self.root / "test", sample, output), 2)
        self.assertEqual(
            output.read_text(), "id,tvt\nabc_1,102.0000000\nabc_0,101.0000000\n"
        )

        script = Path(__file__).parents[1] / "src" / "predict.py"
        exec_output = self.root / "exec.csv"
        command = [
            sys.executable,
            "-c",
            (
                "import sys;"
                "source=open(sys.argv[1], encoding='utf-8').read();"
                "sys.argv=[sys.argv[1],*sys.argv[2:]];"
                "exec(compile(source, sys.argv[0], 'exec'), {'__name__':'__main__'})"
            ),
            str(script),
            "--test-dir",
            str(self.root / "test"),
            "--sample",
            str(sample),
            "--output",
            str(exec_output),
        ]
        with tempfile.TemporaryDirectory() as unrelated_cwd:
            result = subprocess.run(
                command,
                cwd=unrelated_cwd,
                env={"PATH": os.environ.get("PATH", "")},
                check=True,
                capture_output=True,
                text=True,
            )
        self.assertIn("Wrote 2 rows", result.stdout)
        self.assertEqual(exec_output.read_text(), output.read_text())

    def test_champion_extrapolates_withheld_tvt_input(self) -> None:
        self.make_well(
            "test",
            "abc",
            [
                [1, 2, 3, -10, 5, 110, ""],
                [2, 2, 3, -11, 5, 112, ""],
                [3, 2, 3, -12, 5, "", ""],
            ],
        )
        sample = self.root / "sample_submission.csv"
        output = self.root / "submission.csv"
        write_csv(sample, ["id", "tvt"], [["abc_2", 0.0]])
        generate_submission(self.root / "test", sample, output)
        self.assertEqual(output.read_text(), "id,tvt\nabc_2,114.0000000\n")

    def test_offset_trend_constant_and_missing_z_fallbacks_are_finite(self) -> None:
        model = fit_offset_trend(
            [
                {"MD": "5", "Z": "-10", "TVT_input": "110"},
                {"MD": "5", "Z": "-11", "TVT_input": "111"},
            ]
        )
        self.assertEqual(model.slope, 0.0)
        self.assertEqual(predict_offset_tvt(model, {"MD": "6", "Z": "-12"}), 112.0)
        self.assertTrue(
            math.isfinite(predict_offset_tvt(model, {"MD": "", "Z": ""}))
        )

    def test_offset_trend_recency_weighting_tracks_the_terminal_slope(self) -> None:
        rows = [
            {"MD": "0", "Z": "0", "TVT_input": "0"},
            {"MD": "1", "Z": "0", "TVT_input": "0"},
            {"MD": "2", "Z": "0", "TVT_input": "10"},
            {"MD": "3", "Z": "0", "TVT_input": "20"},
        ]
        global_model = fit_offset_trend(rows)
        local_model = fit_offset_trend(rows, recency_decay=8.0)
        self.assertGreater(local_model.slope, global_model.slope)
        self.assertTrue(math.isfinite(local_model.terminal_offset))
        with self.assertRaisesRegex(ValueError, "recency_decay"):
            fit_offset_trend(rows, recency_decay=-1.0)

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
