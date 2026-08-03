from __future__ import annotations

import csv
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.predict import generate_submission

REPO_ROOT = Path(__file__).resolve().parents[1]
KERNEL = REPO_ROOT / "kaggle" / "rogii-claude-baseline.py"
TEST_DIR = REPO_ROOT / "data" / "raw" / "test"
SAMPLE = REPO_ROOT / "data" / "raw" / "sample_submission.csv"

EXEC_RUNNER = (
    "import sys;"
    "source=open(sys.argv[1], encoding='utf-8').read();"
    "sys.argv=[sys.argv[1]];"
    "exec(compile(source, sys.argv[0], 'exec'), {'__name__':'__main__'})"
)


@unittest.skipUnless(SAMPLE.is_file() and TEST_DIR.is_dir(), "competition data absent")
class KernelExecCompatibilityTest(unittest.TestCase):
    def test_kernel_matches_local_generator_byte_for_byte(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            local = Path(workdir) / "local.csv"
            generate_submission(
                TEST_DIR,
                SAMPLE,
                local,
                train_dir=REPO_ROOT / "data" / "raw" / "train",
            )

            # Point the kernel at data/raw (train + test share well ids); it must
            # prefer the test split and reproduce the local file byte-for-byte,
            # run from an unrelated cwd with no __file__ defined (Kaggle-like).
            kernel_out = Path(workdir) / "kernel.csv"
            env = {
                "PATH": os.environ.get("PATH", ""),
                "ROGII_INPUT_DIR": str(REPO_ROOT / "data" / "raw"),
                "ROGII_OUTPUT": str(kernel_out),
            }
            result = subprocess.run(
                [sys.executable, "-c", EXEC_RUNNER, str(KERNEL)],
                cwd=workdir,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("Wrote", result.stdout)
            self.assertEqual(kernel_out.read_bytes(), local.read_bytes())

            with kernel_out.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, ["id", "tvt"])
                values = [float(row["tvt"]) for row in reader]
            self.assertTrue(values)
            self.assertTrue(all(math.isfinite(value) for value in values))
            self.assertTrue(any(value != 0.0 for value in values))
            with kernel_out.open(encoding="utf-8") as handle:
                next(handle)
                self.assertTrue(
                    all(len(line.rstrip().rsplit(".", 1)[-1]) == 7 for line in handle)
                )


if __name__ == "__main__":
    unittest.main()
