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
SRC_MODULE = REPO_ROOT / "src" / "blend.py"

_SHARED_RE = re.compile(
    r"# === BEGIN BLEND_SHARED_CODE.*?# === END BLEND_SHARED_CODE ===",
    re.DOTALL,
)


def _shared_block(path: Path) -> str:
    match = _SHARED_RE.search(path.read_text(encoding="utf-8"))
    assert match is not None, f"BLEND_SHARED_CODE markers missing in {path}"
    return match.group(0)


def _screen_wells() -> list[str]:
    return sorted(holdout_wells(TRAIN, 5, 0))[:5]


def _mask_toe(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    heel = max(1, min(len(rows) - 1, round(len(rows) * TEST_HEEL_FRACTION)))
    return [
        dict(row, TVT_input=(row["TVT_input"] if i < heel else ""))
        for i, row in enumerate(rows)
    ]


class BlendSharedCodeTest(unittest.TestCase):
    """The blend layer must stay byte-identical between src and the kernel."""

    def test_shared_block_is_byte_identical(self) -> None:
        self.assertEqual(
            _shared_block(SRC_MODULE),
            _shared_block(KERNEL),
            "src.blend and kernel BLEND_SHARED_CODE diverged",
        )

    def test_blend_weight_is_frozen_scalar(self) -> None:
        from src import blend

        self.assertIsInstance(blend.BLEND_WEIGHT, float)
        self.assertTrue(0.0 <= blend.BLEND_WEIGHT <= 1.0)


@unittest.skipUnless(np is not None, "numpy absent")
class BlendMathTest(unittest.TestCase):
    def test_linear_combination_where_both_finite(self) -> None:
        from src.blend import blend_trajectories

        pf = [10.0, 20.0]
        ml = [0.0, 0.0]
        out = blend_trajectories(pf, ml, 0.85)
        self.assertAlmostEqual(float(out[0]), 8.5)
        self.assertAlmostEqual(float(out[1]), 17.0)

    def test_weight_one_recovers_particle_filter(self) -> None:
        from src.blend import blend_trajectories

        pf = [3.0, 4.0, 5.0]
        ml = [99.0, 99.0, 99.0]
        out = blend_trajectories(pf, ml, 1.0)
        self.assertTrue(np.array_equal(np.asarray(out), np.asarray(pf, dtype=float)))

    def test_degrades_to_available_side(self) -> None:
        from src.blend import blend_trajectories

        pf = [math.nan, 4.0]
        ml = [7.0, math.nan]
        out = blend_trajectories(pf, ml, 0.85)
        # PF missing -> ML; ML missing -> PF (particle filter preferred).
        self.assertAlmostEqual(float(out[0]), 7.0)
        self.assertAlmostEqual(float(out[1]), 4.0)


@unittest.skipUnless(np is not None and TRAIN.is_dir(), "numpy or train data absent")
class BlendDataTest(unittest.TestCase):
    def test_matches_kernel_implementation(self) -> None:
        """kernel <-> src blend must be byte-identical (exec gate)."""
        from src.blend import blend_trajectories

        spec = importlib.util.spec_from_file_location("_rogii_kernel_blend", KERNEL)
        kernel = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = kernel
        spec.loader.exec_module(kernel)

        pf = np.array([10.0, 20.0, math.nan, 5.0])
        ml = np.array([0.0, 40.0, 3.0, math.nan])
        src_out = blend_trajectories(pf, ml)
        kernel_out = kernel.blend_trajectories(pf, ml)
        self.assertTrue(
            np.array_equal(
                np.nan_to_num(np.asarray(src_out)),
                np.nan_to_num(np.asarray(kernel_out)),
            ),
            "src.blend and kernel blend diverged",
        )

    def test_preserves_known_heel_rows(self) -> None:
        from src.blend import predict_blend_well

        well = _screen_wells()[0]
        rows = load_horizontal(TRAIN / f"{well}__horizontal_well.csv", require_target=True)
        typewell = load_typewell(TRAIN / f"{well}__typewell.csv")
        masked = _mask_toe(rows)
        pred = predict_blend_well(masked, typewell)
        self.assertEqual(len(pred), len(rows))
        self.assertTrue(all(math.isfinite(v) for v in pred))
        for i, row in enumerate(masked):
            if row["TVT_input"]:
                self.assertAlmostEqual(float(pred[i]), float(row["TVT_input"]), places=6)

    def test_blend_beats_both_singles_on_screen(self) -> None:
        """The gated blend must beat both PF and ML singles on the toe screen."""
        from src.evaluate import evaluate_blend_toe_holdout

        result = evaluate_blend_toe_holdout(TRAIN, stage="screen")
        self.assertTrue(math.isfinite(result["rmse"]))
        self.assertTrue(result["beats_pf"], result)
        self.assertTrue(result["beats_ml"], result)
        self.assertTrue(result["beats_both_singles"], result)


if __name__ == "__main__":
    unittest.main()
