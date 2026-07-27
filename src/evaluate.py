from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .data import (
    discover_wells,
    iter_train_pairs,
    load_horizontal,
    load_submission_targets,
    write_submission,
)

PSEUDO_BLIND_SEED = 2033
PSEUDO_BLIND_FOLDS = 5
PSEUDO_BLIND_FOLD = 0
PSEUDO_BLIND_FRACTION = 0.20
SCREEN_WELLS = 5
MAX_RMSE_REGRESSION = 0.0
MAX_MAE_REGRESSION = 0.0


@dataclass(frozen=True)
class BlindWell:
    """Predictor input with target-equivalent values removed from the blind interval."""

    well: str
    rows: tuple[Mapping[str, str], ...]
    observed_tvt: tuple[float | None, ...]
    blind_indices: tuple[int, ...]


def rmse(pairs: Iterable[tuple[float, float]]) -> tuple[float, int]:
    squared_error = 0.0
    count = 0
    for actual, predicted in pairs:
        if not math.isfinite(actual) or not math.isfinite(predicted):
            raise ValueError("Metric inputs must be finite")
        squared_error += (predicted - actual) ** 2
        count += 1
    if count == 0:
        raise ValueError("Cannot score an empty evaluation set")
    return math.sqrt(squared_error / count), count


def _metrics(actual: Sequence[float], predicted: Sequence[float]) -> dict[str, float | int]:
    if len(actual) != len(predicted) or not actual:
        raise ValueError("Actual and predicted values must have the same non-zero length")
    pairs = list(zip(actual, predicted))
    score, count = rmse(pairs)
    mae = sum(abs(prediction - truth) for truth, prediction in pairs) / count
    return {"rmse": score, "mae": mae, "rows": count}


def holdout_wells(train_dir: str | Path, folds: int = 5, fold: int = 0) -> set[str]:
    if folds < 2 or not 0 <= fold < folds:
        raise ValueError("folds must be >= 2 and fold must be in [0, folds)")
    wells = {
        files.well
        for files in discover_wells(train_dir)
        if int(hashlib.sha256(files.well.encode()).hexdigest()[:8], 16) % folds == fold
    }
    if not wells:
        raise ValueError("The deterministic hold-out contains no wells")
    return wells


def evaluate_baseline(train_dir: str | Path, folds: int = 5, fold: int = 0) -> dict:
    selected = holdout_wells(train_dir, folds, fold)
    score, rows = rmse(iter_train_pairs(train_dir, selected))
    return {
        "metric": "rmse",
        "score": score,
        "rows": rows,
        "wells": len(selected),
        "fold": fold,
        "folds": folds,
        "predictor": "TVT_input",
    }


def pseudo_blind_interval(
    well: str,
    row_count: int,
    *,
    seed: int = PSEUDO_BLIND_SEED,
    fraction: float = PSEUDO_BLIND_FRACTION,
) -> range:
    """Choose one deterministic, internal, contiguous blind interval."""
    if row_count < 5:
        raise ValueError("Pseudo-blind evaluation requires at least five rows")
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be between zero and one")
    length = max(1, min(row_count - 2, round(row_count * fraction)))
    possible_starts = row_count - length - 1
    digest = hashlib.sha256(f"{seed}:{well}".encode()).digest()
    start = 1 + int.from_bytes(digest[:8], "big") % possible_starts
    return range(start, start + length)


def build_blind_well(
    well: str,
    rows: Sequence[Mapping[str, str]],
    *,
    seed: int = PSEUDO_BLIND_SEED,
    fraction: float = PSEUDO_BLIND_FRACTION,
) -> tuple[BlindWell, tuple[float, ...]]:
    visible_indices = [index for index, row in enumerate(rows) if row["TVT_input"]]
    if not visible_indices:
        raise ValueError(f"{well}: no visible TVT_input values")
    evaluable_rows = visible_indices[-1] + 1
    blind = pseudo_blind_interval(well, evaluable_rows, seed=seed, fraction=fraction)
    blind_set = set(blind)
    sanitized_rows: list[Mapping[str, str]] = []
    observed: list[float | None] = []
    truth: list[float] = []
    for index, row in enumerate(rows):
        clean = {key: value for key, value in row.items() if key not in {"TVT", "TVT_input"}}
        sanitized_rows.append(clean)
        if index in blind_set:
            observed.append(None)
            truth.append(float(row["TVT"]))
        else:
            observed.append(float(row["TVT_input"]) if row["TVT_input"] else None)
    return (
        BlindWell(well, tuple(sanitized_rows), tuple(observed), tuple(blind)),
        tuple(truth),
    )


def interpolation_baseline(case: BlindWell) -> list[float]:
    """Linearly interpolate an internal gap from its two visible endpoint values."""
    first = case.blind_indices[0]
    last = case.blind_indices[-1]
    left_index = next(
        (index for index in range(first - 1, -1, -1) if case.observed_tvt[index] is not None),
        None,
    )
    right_index = next(
        (
            index
            for index in range(last + 1, len(case.rows))
            if case.observed_tvt[index] is not None
        ),
        None,
    )
    if left_index is None or right_index is None:
        raise ValueError(f"{case.well}: blind interval has no visible endpoints")
    left = case.observed_tvt[left_index]
    right = case.observed_tvt[right_index]
    assert left is not None and right is not None
    left_md = float(case.rows[left_index]["MD"])
    right_md = float(case.rows[right_index]["MD"])
    if right_md == left_md:
        return [left] * len(case.blind_indices)
    return [
        left
        + (right - left)
        * (float(case.rows[index]["MD"]) - left_md)
        / (right_md - left_md)
        for index in case.blind_indices
    ]


Predictor = Callable[[BlindWell], Sequence[float]]


def evaluate_pseudo_blind(
    train_dir: str | Path,
    *,
    stage: str = "screen",
    predictor: Predictor = interpolation_baseline,
    seed: int = PSEUDO_BLIND_SEED,
    folds: int = PSEUDO_BLIND_FOLDS,
    fold: int = PSEUDO_BLIND_FOLD,
    fraction: float = PSEUDO_BLIND_FRACTION,
) -> dict:
    if stage not in {"screen", "confirm"}:
        raise ValueError("stage must be 'screen' or 'confirm'")
    selected = sorted(holdout_wells(train_dir, folds, fold))
    if stage == "screen":
        selected = selected[:SCREEN_WELLS]
    actual: list[float] = []
    predicted: list[float] = []
    for files in discover_wells(train_dir):
        if files.well not in selected:
            continue
        rows = load_horizontal(files.horizontal, require_target=True)
        case, truth = build_blind_well(files.well, rows, seed=seed, fraction=fraction)
        values = [float(value) for value in predictor(case)]
        if len(values) != len(case.blind_indices):
            raise ValueError(f"{files.well}: predictor returned the wrong number of rows")
        actual.extend(truth)
        predicted.extend(values)
    metrics = _metrics(actual, predicted)
    return {
        "stage": stage,
        "seed": seed,
        "fold": fold,
        "folds": folds,
        "blind_fraction": fraction,
        "wells": len(selected),
        "well_ids": selected,
        "baseline": "linear_interpolation_from_visible_TVT_input_endpoints",
        **metrics,
    }


def evaluate_gate(
    train_dir: str | Path,
    predictor: Predictor,
    *,
    baseline: Predictor = interpolation_baseline,
) -> dict:
    """Run screen first and confirm only when the candidate clears both thresholds."""
    stages: list[dict] = []
    for stage in ("screen", "confirm"):
        candidate_result = evaluate_pseudo_blind(train_dir, stage=stage, predictor=predictor)
        baseline_result = evaluate_pseudo_blind(train_dir, stage=stage, predictor=baseline)
        result = {
            **candidate_result,
            "baseline_rmse": baseline_result["rmse"],
            "baseline_mae": baseline_result["mae"],
            "rmse_delta": candidate_result["rmse"] - baseline_result["rmse"],
            "mae_delta": candidate_result["mae"] - baseline_result["mae"],
        }
        result["passed"] = (
            result["rmse_delta"] <= MAX_RMSE_REGRESSION
            and result["mae_delta"] <= MAX_MAE_REGRESSION
        )
        stages.append(result)
        if not result["passed"]:
            break
    return {"promoted": len(stages) == 2 and stages[-1]["passed"], "stages": stages}


def main() -> None:
    parser = argparse.ArgumentParser(description="ROGII local KPI and submission utility")
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate = subparsers.add_parser("baseline", help="score TVT_input on held-out train wells")
    evaluate.add_argument("--train-dir", type=Path, default=Path("data/raw/train"))
    evaluate.add_argument("--folds", type=int, default=5)
    evaluate.add_argument("--fold", type=int, default=0)
    blind = subparsers.add_parser(
        "pseudo-blind", help="score interpolation without blind-interval target leakage"
    )
    blind.add_argument("--train-dir", type=Path, default=Path("data/raw/train"))
    blind.add_argument("--stage", choices=("screen", "confirm"), default="screen")
    submission = subparsers.add_parser("submission", help="create a format-smoke submission")
    submission.add_argument("--sample", type=Path, default=Path("data/raw/sample_submission.csv"))
    submission.add_argument("--output", type=Path, default=Path("submission.csv"))
    submission.add_argument("--constant", type=float, default=0.0)
    args = parser.parse_args()
    if args.command == "baseline":
        print(json.dumps(evaluate_baseline(args.train_dir, args.folds, args.fold), indent=2))
    elif args.command == "pseudo-blind":
        print(json.dumps(evaluate_pseudo_blind(args.train_dir, stage=args.stage), indent=2))
    else:
        targets = load_submission_targets(args.sample)
        predictions = {target.id: args.constant for target in targets}
        output = write_submission(args.sample, predictions, args.output)
        print(f"Wrote {len(targets)} rows to {output}")


if __name__ == "__main__":
    main()
