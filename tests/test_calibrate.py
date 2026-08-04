from __future__ import annotations

import importlib.util
import math
import re
import sys
import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is required for these tests
    np = None

from src.data import load_horizontal, load_typewell
from src.evaluate import TEST_HEEL_FRACTION, holdout_wells

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN = REPO_ROOT / "data" / "raw" / "train"
KERNEL = REPO_ROOT / "kaggle" / "rogii-claude-baseline.py"
SRC_MODULE = REPO_ROOT / "src" / "calibrate.py"

_SHARED_RE = re.compile(
    r"# === BEGIN GOLD_SHARED_CODE.*?# === END GOLD_SHARED_CODE ===",
    re.DOTALL,
)


def _shared_block(path: Path) -> str:
    match = _SHARED_RE.search(path.read_text(encoding="utf-8"))
    assert match is not None, f"GOLD_SHARED_CODE markers missing in {path}"
    return match.group(0)


def _screen_wells() -> list[str]:
    return sorted(holdout_wells(TRAIN, 5, 0))[:5]


def _mask_toe(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    heel = max(1, min(len(rows) - 1, round(len(rows) * TEST_HEEL_FRACTION)))
    return [
        dict(row, TVT_input=(row["TVT_input"] if i < heel else ""))
        for i, row in enumerate(rows)
    ]


class GoldSharedCodeTest(unittest.TestCase):
    """The gold-calibration layer must stay byte-identical between src and kernel."""

    def test_shared_block_is_byte_identical(self) -> None:
        self.assertEqual(
            _shared_block(SRC_MODULE),
            _shared_block(KERNEL),
            "src.calibrate and kernel GOLD_SHARED_CODE diverged",
        )

    def test_profile_thresholds_are_frozen(self) -> None:
        from src import calibrate

        # Frozen from the reference kernel's conservative profile (leak-free by
        # provenance): the gate must not silently loosen.
        self.assertEqual(calibrate.GOLD_PROFILE["min_gain"], 1.00)
        self.assertEqual(calibrate.GOLD_PROFILE["min_consistency"], 0.67)
        self.assertEqual(calibrate.GOLD_PROFILE["max_best"], 12.0)
        self.assertEqual(calibrate.GOLD_CUT_FRACS, (0.50, 0.65, 0.75))


@unittest.skipUnless(np is not None, "numpy absent")
class GoldMathTest(unittest.TestCase):
    def test_robust_poly_recovers_a_line(self) -> None:
        from src.calibrate import _gold_robust_poly_predict

        x = np.arange(50, dtype=float)
        y = 3.0 * x + 7.0
        out = _gold_robust_poly_predict(x[:40], y[:40], x, 2)
        self.assertTrue(np.allclose(out, y, atol=1e-6))

    def test_alpha_zero_when_gate_not_cleared(self) -> None:
        from src.calibrate import _gold_alpha

        # Below min_gain / min_consistency -> no move.
        weak = {"status": "ok", "gain": 0.1, "best_score": 3.0, "rank_margin": 0.5, "consistency": 1.0}
        self.assertEqual(_gold_alpha(weak, 1.0, 1.0), 0.0)
        inconsistent = {"status": "ok", "gain": 5.0, "best_score": 3.0, "rank_margin": 2.0, "consistency": 0.5}
        self.assertEqual(_gold_alpha(inconsistent, 1.0, 1.0), 0.0)

    def test_alpha_positive_and_capped_when_gate_cleared(self) -> None:
        from src.calibrate import GOLD_PROFILE, _gold_alpha

        strong = {"status": "ok", "gain": 4.0, "best_score": 2.0, "rank_margin": 2.0, "consistency": 1.0}
        alpha = _gold_alpha(strong, 1.0, 1.0)
        self.assertGreater(alpha, 0.0)
        self.assertLessEqual(alpha, GOLD_PROFILE["cap"])

    def test_apply_move_is_a_noop_when_candidate_equals_base(self) -> None:
        from src.calibrate import _gold_apply_move

        base = np.array([1.0, 2.0, 3.0, 4.0])
        report = {"status": "ok", "gain": 4.0, "best_score": 2.0, "rank_margin": 2.0, "consistency": 1.0}
        is_hidden = np.array([False, False, True, True])
        out = _gold_apply_move(base, base.copy(), report, is_hidden)
        self.assertTrue(np.array_equal(out, base))


@unittest.skipUnless(np is not None and TRAIN.is_dir(), "numpy or train data absent")
class GoldDataTest(unittest.TestCase):
    def test_matches_kernel_implementation(self) -> None:
        """kernel <-> src gold overlay must be byte-identical (exec gate)."""
        from src.calibrate import gold_calibrate_trajectory

        spec = importlib.util.spec_from_file_location("_rogii_kernel_gold", KERNEL)
        kernel = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = kernel
        spec.loader.exec_module(kernel)

        well = _screen_wells()[0]
        rows = load_horizontal(TRAIN / f"{well}__horizontal_well.csv", require_target=True)
        typewell = load_typewell(TRAIN / f"{well}__typewell.csv")
        masked = _mask_toe(rows)
        src_out = np.asarray(gold_calibrate_trajectory(masked, typewell))
        kernel_out = np.asarray(kernel.gold_calibrate_trajectory(masked, typewell))
        self.assertEqual(src_out.shape, kernel_out.shape)
        self.assertTrue(
            np.array_equal(np.nan_to_num(src_out), np.nan_to_num(kernel_out)),
            "src.calibrate and kernel gold overlay diverged",
        )

    def test_preserves_known_heel_rows(self) -> None:
        from src.calibrate import gold_calibrate_trajectory

        well = _screen_wells()[0]
        rows = load_horizontal(TRAIN / f"{well}__horizontal_well.csv", require_target=True)
        typewell = load_typewell(TRAIN / f"{well}__typewell.csv")
        masked = _mask_toe(rows)
        out = gold_calibrate_trajectory(masked, typewell)
        self.assertEqual(len(out), len(rows))
        self.assertTrue(all(math.isfinite(v) for v in out))
        for i, row in enumerate(masked):
            if row["TVT_input"]:
                self.assertAlmostEqual(float(out[i]), float(row["TVT_input"]), places=6)

    def test_no_regression_vs_blend_on_screen(self) -> None:
        """The gold overlay must never regress the stage-2 blend on the toe screen."""
        from src.evaluate import evaluate_gold_toe_holdout

        result = evaluate_gold_toe_holdout(TRAIN, stage="screen")
        self.assertTrue(math.isfinite(result["rmse"]))
        self.assertTrue(result["no_regression_vs_blend"], result)


if __name__ == "__main__":
    unittest.main()
