"""Leak-free selection of the physics × ML blend weight (SOT-2394).

The reported confirm gate scores the blend on the **fold-0** toe hold-out
(``src.evaluate.evaluate_blend_toe_holdout``). To keep the blend weight
selection leak-free, this script grid-searches the particle-filter share
``weight`` on a **disjoint** hold-out — the **fold-1** wells, which the fold-0
confirm gate never scores — and prints the ``weight`` that minimises the pooled
toe RMSE. That value is then frozen as ``BLEND_WEIGHT`` in ``src/blend.py`` (and
the verbatim mirror in ``kaggle/rogii-claude-baseline.py``) *before* any fold-0
test/confirm toe target is read, so the reported number is never used to tune the
weight.

Run:

    python3 scripts/select_blend_weight.py --train-dir data/raw/train

PF and ML trajectories are computed once per fold-1 well and reused across the
whole weight grid, so the sweep costs one PF+ML pass, not one per grid point.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.blend import blend_trajectories
from src.data import discover_wells, load_horizontal
from src.evaluate import (
    PSEUDO_BLIND_FOLDS,
    TEST_HEEL_FRACTION,
    _read_typewell,
    holdout_wells,
)
from src.ml_predictor import predict_ml_well
from src.physics import predict_pf_well

# fold-1 is disjoint from the fold-0 confirm/screen set used by the reported gate.
SELECTION_FOLD = 1
# Particle-filter share grid (1.0 == pure PF champion).
WEIGHT_GRID = tuple(round(0.50 + 0.05 * i, 2) for i in range(11))  # 0.50 .. 1.00


def _rmse(actual: list[float], predicted: list[float]) -> float:
    if not actual:
        raise ValueError("empty evaluation set")
    return math.sqrt(
        sum((p - a) ** 2 for a, p in zip(actual, predicted)) / len(actual)
    )


def select_weight(
    train_dir: str | Path,
    *,
    heel_fraction: float = TEST_HEEL_FRACTION,
    grid: tuple[float, ...] = WEIGHT_GRID,
) -> dict:
    selected = sorted(holdout_wells(train_dir, PSEUDO_BLIND_FOLDS, SELECTION_FOLD))
    root = Path(train_dir)
    actual: list[float] = []
    pf_pred: list[float] = []
    ml_pred: list[float] = []
    for files in discover_wells(train_dir):
        if files.well not in selected:
            continue
        rows = load_horizontal(files.horizontal, require_target=True)
        typewell = _read_typewell(root / f"{files.well}__typewell.csv")
        heel_count = max(1, min(len(rows) - 1, round(len(rows) * heel_fraction)))
        masked = [
            dict(row, TVT_input=(row["TVT_input"] if i < heel_count else ""))
            for i, row in enumerate(rows)
        ]
        pf = predict_pf_well(masked, typewell)
        ml = predict_ml_well(masked, typewell)
        for i in range(heel_count, len(rows)):
            if not rows[i].get("TVT"):
                continue
            if not (math.isfinite(pf[i]) and math.isfinite(ml[i])):
                continue
            actual.append(float(rows[i]["TVT"]))
            pf_pred.append(float(pf[i]))
            ml_pred.append(float(ml[i]))
    pf_arr = np.asarray(pf_pred)
    ml_arr = np.asarray(ml_pred)
    scores = {}
    for weight in grid:
        blended = blend_trajectories(pf_arr, ml_arr, weight).tolist()
        scores[weight] = _rmse(actual, blended)
    best_weight = min(scores, key=scores.get)
    return {
        "selection_fold": SELECTION_FOLD,
        "folds": PSEUDO_BLIND_FOLDS,
        "wells": len(selected),
        "rows": len(actual),
        "pf_rmse": _rmse(actual, pf_pred),
        "ml_rmse": _rmse(actual, ml_pred),
        "grid": list(grid),
        "rmse_by_weight": {str(w): scores[w] for w in grid},
        "best_weight": best_weight,
        "best_rmse": scores[best_weight],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=Path("data/raw/train"))
    args = parser.parse_args()
    print(json.dumps(select_weight(args.train_dir), indent=2))


if __name__ == "__main__":
    main()
