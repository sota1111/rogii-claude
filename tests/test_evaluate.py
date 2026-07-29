from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.evaluate import (
    build_blind_well,
    evaluate_gate,
    evaluate_pseudo_blind,
    evaluate_toe_gate,
    evaluate_toe_holdout,
    interpolation_baseline,
    pseudo_blind_interval,
)


def _write_well(root: Path, well: str, offset: float = 0.0) -> None:
    root.mkdir(parents=True, exist_ok=True)
    horizontal = root / f"{well}__horizontal_well.csv"
    with horizontal.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["MD", "X", "Y", "Z", "GR", "TVT_input", "TVT"]
        )
        writer.writeheader()
        for index in range(20):
            tvt = offset + index * 2.0
            writer.writerow(
                {
                    "MD": index,
                    "X": 0,
                    "Y": 0,
                    "Z": -index,
                    "GR": 50 + index,
                    "TVT_input": tvt,
                    "TVT": tvt,
                }
            )
    typewell = root / f"{well}__typewell.csv"
    typewell.write_text("TVT,GR\n0,50\n", encoding="utf-8")


def _selected_wells(root: Path, count: int = 8) -> list[str]:
    from src.evaluate import holdout_wells

    for index in range(100):
        _write_well(root, f"well-{index:03d}", float(index))
    selected = sorted(holdout_wells(root))
    assert len(selected) >= count
    return selected


def test_pseudo_blind_interval_is_deterministic_and_internal() -> None:
    first = pseudo_blind_interval("alpha", 100)
    second = pseudo_blind_interval("alpha", 100)
    assert first == second
    assert len(first) == 20
    assert first.start > 0
    assert first.stop < 100


def test_blind_predictor_cannot_read_truth_equivalent_columns() -> None:
    rows = [
        {
            "MD": str(index),
            "X": "0",
            "Y": "0",
            "Z": "0",
            "GR": "50",
            "TVT": str(index),
            "TVT_input": str(index),
        }
        for index in range(10)
    ]
    case, _ = build_blind_well("alpha", rows)
    blind_row = case.rows[case.blind_indices[0]]
    assert "TVT" not in blind_row
    assert "TVT_input" not in blind_row
    assert case.observed_tvt[case.blind_indices[0]] is None
    with pytest.raises(KeyError):
        _ = blind_row["TVT_input"]


def test_screen_and_confirm_are_reproducible(tmp_path: Path) -> None:
    selected = _selected_wells(tmp_path)
    screen_1 = evaluate_pseudo_blind(tmp_path, stage="screen")
    screen_2 = evaluate_pseudo_blind(tmp_path, stage="screen")
    confirm_1 = evaluate_pseudo_blind(tmp_path, stage="confirm")
    confirm_2 = evaluate_pseudo_blind(tmp_path, stage="confirm")
    assert screen_1 == screen_2
    assert confirm_1 == confirm_2
    assert screen_1["well_ids"] == selected[:5]
    assert confirm_1["well_ids"] == selected
    assert screen_1["rmse"] == pytest.approx(0.0)
    assert confirm_1["mae"] == pytest.approx(0.0)


def test_gate_stops_after_failed_screen(tmp_path: Path) -> None:
    _selected_wells(tmp_path)

    def biased(case):
        return [value + 1.0 for value in interpolation_baseline(case)]

    result = evaluate_gate(tmp_path, biased)
    assert result["promoted"] is False
    assert len(result["stages"]) == 1
    assert result["stages"][0]["passed"] is False
    assert result["stages"][0]["rmse_delta"] > 0


def test_toe_holdout_requires_strict_local_slope_improvement(tmp_path: Path) -> None:
    selected = _selected_wells(tmp_path)
    screen = evaluate_toe_holdout(tmp_path, stage="screen")
    confirm = evaluate_toe_holdout(tmp_path, stage="confirm")
    gate = evaluate_toe_gate(tmp_path)
    assert screen["well_ids"] == selected[:5]
    assert confirm["well_ids"] == selected
    assert screen["metrics"]["local_offset_trend"]["rmse"] == pytest.approx(0.0)
    assert screen["metrics"]["global_offset_trend"]["rmse"] == pytest.approx(0.0)
    assert screen["deltas"]["zeros"]["rmse"] < 0
    assert screen["deltas"]["const_offset"]["mae"] < 0
    assert screen["deltas"]["global_offset_trend"]["rmse"] == pytest.approx(0.0)
    assert screen["passed"] is False
    assert gate["promoted"] is False
