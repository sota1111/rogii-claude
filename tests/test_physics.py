from __future__ import annotations

import importlib.util
import math
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


def _load_kernel():
    spec = importlib.util.spec_from_file_location("_rogii_kernel", KERNEL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclass field resolution needs this
    spec.loader.exec_module(module)
    return module


def _screen_wells() -> list[str]:
    return sorted(holdout_wells(TRAIN, 5, 0))[:5]


def _mask_toe(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[int]]:
    heel = max(1, min(len(rows) - 1, round(len(rows) * TEST_HEEL_FRACTION)))
    masked = [
        dict(row, TVT_input=(row["TVT_input"] if i < heel else ""))
        for i, row in enumerate(rows)
    ]
    toe = [i for i in range(heel, len(rows)) if rows[i].get("TVT")]
    return masked, toe


@unittest.skipUnless(np is not None and TRAIN.is_dir(), "numpy or train data absent")
class ParticleFilterTest(unittest.TestCase):
    def test_pf_preserves_known_rows_and_returns_finite(self) -> None:
        from src.physics import predict_pf_well

        well = _screen_wells()[0]
        rows = load_horizontal(TRAIN / f"{well}__horizontal_well.csv", require_target=True)
        typewell = load_typewell(TRAIN / f"{well}__typewell.csv")
        masked, _ = _mask_toe(rows)

        pred = predict_pf_well(masked, typewell)
        self.assertEqual(len(pred), len(rows))
        self.assertTrue(all(math.isfinite(v) for v in pred))
        for i, row in enumerate(masked):
            if row["TVT_input"]:
                self.assertAlmostEqual(float(pred[i]), float(row["TVT_input"]), places=6)

    def test_pf_beats_champion_on_pooled_screen(self) -> None:
        """The port must clear the toe-holdout screen the champion is gated on."""
        from src.data import fit_offset_trend, predict_offset_tvt
        from src.physics import predict_pf_well

        truth: list[float] = []
        pf_pred: list[float] = []
        champ_pred: list[float] = []
        for well in _screen_wells():
            rows = load_horizontal(
                TRAIN / f"{well}__horizontal_well.csv", require_target=True
            )
            typewell = load_typewell(TRAIN / f"{well}__typewell.csv")
            masked, toe = _mask_toe(rows)
            heel = toe[0]
            model = fit_offset_trend(masked[:heel], recency_decay=8.0)
            pf = predict_pf_well(masked, typewell)
            for i in toe:
                truth.append(float(rows[i]["TVT"]))
                pf_pred.append(float(pf[i]))
                champ_pred.append(predict_offset_tvt(model, rows[i]))
        truth_a = np.array(truth)
        pf_rmse = float(np.sqrt(np.mean((np.array(pf_pred) - truth_a) ** 2)))
        champ_rmse = float(np.sqrt(np.mean((np.array(champ_pred) - truth_a) ** 2)))
        self.assertLess(pf_rmse, champ_rmse)

    def test_pf_matches_kernel_implementation(self) -> None:
        from src.physics import predict_pf_well

        kernel = _load_kernel()
        well = _screen_wells()[0]
        rows = load_horizontal(TRAIN / f"{well}__horizontal_well.csv", require_target=True)
        typewell = load_typewell(TRAIN / f"{well}__typewell.csv")
        masked, _ = _mask_toe(rows)

        src_pred = predict_pf_well(masked, typewell)
        kernel_pred = kernel.predict_pf_well(masked, typewell)
        self.assertTrue(
            np.array_equal(np.asarray(src_pred), np.asarray(kernel_pred)),
            "src.physics and kernel particle filters diverged",
        )


if __name__ == "__main__":
    unittest.main()
