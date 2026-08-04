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
SRC_MODULE = REPO_ROOT / "src" / "align.py"

_SHARED_RE = re.compile(
    r"# === BEGIN ALIGN_SHARED_CODE.*?# === END ALIGN_SHARED_CODE ===",
    re.DOTALL,
)


def _shared_block(path: Path) -> str:
    match = _SHARED_RE.search(path.read_text(encoding="utf-8"))
    assert match is not None, f"ALIGN_SHARED_CODE markers missing in {path}"
    return match.group(0)


def _screen_wells() -> list[str]:
    return sorted(holdout_wells(TRAIN, 5, 0))[:5]


def _mask_toe(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    heel = max(1, min(len(rows) - 1, round(len(rows) * TEST_HEEL_FRACTION)))
    return [
        dict(row, TVT_input=(row["TVT_input"] if i < heel else ""))
        for i, row in enumerate(rows)
    ]


class AlignSharedCodeTest(unittest.TestCase):
    """The beam+NCC alignment must stay byte-identical between src and kernel."""

    def test_shared_block_is_byte_identical(self) -> None:
        self.assertEqual(
            _shared_block(SRC_MODULE),
            _shared_block(KERNEL),
            "src.align and kernel ALIGN_SHARED_CODE diverged",
        )

    def test_kernel_defines_beam_before_gold_pool(self) -> None:
        text = KERNEL.read_text(encoding="utf-8")
        self.assertLess(
            text.index("def predict_beam_well"),
            text.index("def _gold_candidate_pool"),
            "predict_beam_well must be defined before the gold candidate pool",
        )


@unittest.skipUnless(np is not None, "numpy absent")
class AlignSyntheticTest(unittest.TestCase):
    """Beam alignment on a synthetic well whose GR follows the typewell exactly."""

    @staticmethod
    def _synthetic(n: int = 400, heel: int = 260):
        # Typewell: a smooth, monotone-in-shape GR-vs-TVT signature.
        tw_tvt = np.linspace(1000.0, 1400.0, 800)
        tw_gr = 80.0 + 40.0 * np.sin((tw_tvt - 1000.0) / 22.0)
        typewell = [
            {"TVT": f"{t:.4f}", "GR": f"{g:.4f}"} for t, g in zip(tw_tvt, tw_gr)
        ]
        md = 5000.0 + np.arange(n, dtype=float)
        # True stratigraphic level U drifts slowly; Z is a smooth ramp.
        z = 200.0 + 0.05 * np.arange(n, dtype=float)
        u = 1300.0 + 6.0 * np.sin(np.arange(n, dtype=float) / 130.0)
        true_tvt = u - z
        gr = 80.0 + 40.0 * np.sin((true_tvt - 1000.0) / 22.0)
        rows = []
        for i in range(n):
            rows.append(
                {
                    "MD": f"{md[i]:.4f}",
                    "Z": f"{z[i]:.6f}",
                    "GR": f"{gr[i]:.6f}",
                    "TVT_input": (f"{true_tvt[i]:.6f}" if i < heel else ""),
                    "TVT": f"{true_tvt[i]:.6f}",
                }
            )
        return rows, typewell, true_tvt, heel

    def test_preserves_known_heel_rows(self) -> None:
        from src.align import predict_beam_well

        rows, typewell, true_tvt, heel = self._synthetic()
        out = np.asarray(predict_beam_well(rows, typewell))
        self.assertEqual(len(out), len(rows))
        for i in range(heel):
            self.assertAlmostEqual(float(out[i]), float(true_tvt[i]), places=6)

    def test_toe_is_finite_and_bounded(self) -> None:
        from src.align import predict_beam_well

        rows, typewell, true_tvt, heel = self._synthetic()
        out = np.asarray(predict_beam_well(rows, typewell))
        toe = out[heel:]
        self.assertTrue(np.all(np.isfinite(toe)))
        # The beam should track the true toe far better than the flat last-known anchor.
        anchor = float(true_tvt[heel - 1])
        beam_rmse = float(np.sqrt(np.mean((toe - true_tvt[heel:]) ** 2)))
        flat_rmse = float(np.sqrt(np.mean((anchor - true_tvt[heel:]) ** 2)))
        self.assertLess(beam_rmse, flat_rmse)

    def test_is_deterministic(self) -> None:
        from src.align import predict_beam_well

        rows, typewell, _, _ = self._synthetic()
        a = np.asarray(predict_beam_well(rows, typewell))
        b = np.asarray(predict_beam_well(rows, typewell))
        self.assertTrue(np.array_equal(np.nan_to_num(a), np.nan_to_num(b)))

    def test_ncc_registration_recovers_a_lag(self) -> None:
        from src.align import _ncc_registration

        tw_tvt = np.linspace(1000.0, 1400.0, 800)
        tw_gr = 80.0 + 40.0 * np.sin((tw_tvt - 1000.0) / 22.0)
        n = 300
        z = np.zeros(n)
        true_tvt = np.linspace(1100.0, 1200.0, n)
        # Horizontal GR is the typewell shifted by a known +5.0 TVT lag.
        gr = 80.0 + 40.0 * np.sin(((true_tvt + 5.0) - 1000.0) / 22.0)
        tau = _ncc_registration(
            np.arange(n, dtype=float), z, gr, true_tvt, tw_tvt, tw_gr
        )
        self.assertTrue(math.isfinite(tau))
        self.assertLess(abs(tau - 5.0), 2.0)

    def test_degrades_to_finite_without_typewell(self) -> None:
        from src.align import predict_beam_well

        rows, _, true_tvt, heel = self._synthetic()
        out = np.asarray(predict_beam_well(rows, []))  # no typewell rows
        self.assertEqual(len(out), len(rows))
        self.assertTrue(np.all(np.isfinite(out)))
        for i in range(heel):
            self.assertAlmostEqual(float(out[i]), float(true_tvt[i]), places=6)


@unittest.skipUnless(np is not None and TRAIN.is_dir(), "numpy or train data absent")
class AlignDataTest(unittest.TestCase):
    def test_matches_kernel_implementation(self) -> None:
        """kernel <-> src predict_beam_well must be byte-identical (exec gate)."""
        from src.align import predict_beam_well

        spec = importlib.util.spec_from_file_location("_rogii_kernel_align", KERNEL)
        kernel = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = kernel
        spec.loader.exec_module(kernel)

        well = _screen_wells()[0]
        rows = load_horizontal(TRAIN / f"{well}__horizontal_well.csv", require_target=True)
        typewell = load_typewell(TRAIN / f"{well}__typewell.csv")
        masked = _mask_toe(rows)
        src_out = np.asarray(predict_beam_well(masked, typewell))
        kernel_out = np.asarray(kernel.predict_beam_well(masked, typewell))
        self.assertEqual(src_out.shape, kernel_out.shape)
        self.assertTrue(
            np.array_equal(np.nan_to_num(src_out), np.nan_to_num(kernel_out)),
            "src.align and kernel predict_beam_well diverged",
        )

    def test_beam_in_gold_pool(self) -> None:
        from src.calibrate import _gold_candidate_pool

        well = _screen_wells()[0]
        rows = load_horizontal(TRAIN / f"{well}__horizontal_well.csv", require_target=True)
        typewell = load_typewell(TRAIN / f"{well}__typewell.csv")
        masked = _mask_toe(rows)
        pool = _gold_candidate_pool(masked, typewell)
        self.assertIn("beam", pool)
        self.assertEqual(len(pool["beam"]), len(rows))
        # Known heel rows are preserved verbatim by the beam candidate.
        for i, row in enumerate(masked):
            if row["TVT_input"]:
                self.assertAlmostEqual(
                    float(pool["beam"][i]), float(row["TVT_input"]), places=6
                )


if __name__ == "__main__":
    unittest.main()
