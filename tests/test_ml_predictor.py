from __future__ import annotations

import importlib.util
import math
import subprocess
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
SRC_MODULE = REPO_ROOT / "src" / "ml_predictor.py"


def _load_kernel():
    spec = importlib.util.spec_from_file_location("_rogii_kernel_ml", KERNEL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclass/global resolution needs this
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


class MlPredictorStructureTest(unittest.TestCase):
    """Structure checks that need no competition data."""

    def test_model_features_are_consistent(self) -> None:
        from src import ml_predictor as ml

        model = ml.load_model()
        self.assertEqual(model["n_features"], len(ml.FEATURE_NAMES))
        self.assertEqual(model["feature_names"], list(ml.FEATURE_NAMES))
        self.assertGreater(len(model["trees"]), 0)

    def test_loader_is_file_independent(self) -> None:
        """The exec-compatible loader must work with no __file__ (Kaggle-like)."""
        source = SRC_MODULE.read_text(encoding="utf-8")
        namespace: dict[str, object] = {"__name__": "_execd_ml"}
        # Compile and exec with no __file__ key in the namespace.
        exec(compile(source, "<execd-ml-predictor>", "exec"), namespace)
        self.assertNotIn("__file__", namespace)
        model = namespace["load_model"]()
        self.assertGreater(len(model["trees"]), 0)
        if np is not None:
            feats = np.zeros((3, model["n_features"]))
            out = namespace["predict_gbrt"](model, feats)
            self.assertEqual(out.shape, (3,))
            self.assertTrue(np.all(np.isfinite(out)))


@unittest.skipUnless(np is not None and TRAIN.is_dir(), "numpy or train data absent")
class MlPredictorDataTest(unittest.TestCase):
    def test_preserves_known_rows_and_returns_finite(self) -> None:
        from src.ml_predictor import predict_ml_well

        well = _screen_wells()[0]
        rows = load_horizontal(TRAIN / f"{well}__horizontal_well.csv", require_target=True)
        typewell = load_typewell(TRAIN / f"{well}__typewell.csv")
        masked, _ = _mask_toe(rows)

        pred = predict_ml_well(masked, typewell)
        self.assertEqual(len(pred), len(rows))
        self.assertTrue(all(math.isfinite(v) for v in pred))
        for i, row in enumerate(masked):
            if row["TVT_input"]:
                self.assertAlmostEqual(float(pred[i]), float(row["TVT_input"]), places=6)

    def test_matches_kernel_implementation(self) -> None:
        """kernel <-> src ML predictor must be byte-identical (exec gate)."""
        from src.ml_predictor import predict_ml_well

        kernel = _load_kernel()
        well = _screen_wells()[0]
        rows = load_horizontal(TRAIN / f"{well}__horizontal_well.csv", require_target=True)
        typewell = load_typewell(TRAIN / f"{well}__typewell.csv")
        masked, _ = _mask_toe(rows)

        src_pred = predict_ml_well(masked, typewell)
        kernel_pred = kernel.predict_ml_well(masked, typewell)
        self.assertTrue(
            np.array_equal(np.asarray(src_pred), np.asarray(kernel_pred)),
            "src.ml_predictor and kernel ML predictor diverged",
        )

    def test_beats_offset_trend_on_confirm(self) -> None:
        """The ML base predictor must improve on the offset-trend fallback base."""
        from src.evaluate import evaluate_ml_toe_holdout

        result = evaluate_ml_toe_holdout(TRAIN, stage="confirm")
        self.assertTrue(math.isfinite(result["rmse"]))
        self.assertTrue(result["beats_offset_trend"], result)
        # Recorded foundation relation to the champion particle filter.
        self.assertAlmostEqual(result["champion_pf_rmse"], 11.225)


if __name__ == "__main__":
    unittest.main()
