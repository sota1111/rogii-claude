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
    fit_offset_trend,
    iter_train_pairs,
    load_horizontal,
    load_submission_targets,
    predict_offset_tvt,
    write_submission,
)

PSEUDO_BLIND_SEED = 2033
PSEUDO_BLIND_FOLDS = 5
PSEUDO_BLIND_FOLD = 0
PSEUDO_BLIND_FRACTION = 0.20
SCREEN_WELLS = 5
MAX_RMSE_REGRESSION = 0.0
MAX_MAE_REGRESSION = 0.0
TEST_HEEL_FRACTION = 5070 / 19221
LOCAL_OFFSET_RECENCY_DECAY = 8.0


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


def evaluate_toe_holdout(
    train_dir: str | Path,
    *,
    stage: str = "screen",
    heel_fraction: float = TEST_HEEL_FRACTION,
) -> dict:
    """Evaluate true suffix extrapolation using the observed test heel proportion."""
    if stage not in {"screen", "confirm"}:
        raise ValueError("stage must be 'screen' or 'confirm'")
    if not 0.0 < heel_fraction < 1.0:
        raise ValueError("heel_fraction must be between zero and one")
    selected = sorted(holdout_wells(train_dir, PSEUDO_BLIND_FOLDS, PSEUDO_BLIND_FOLD))
    if stage == "screen":
        selected = selected[:SCREEN_WELLS]
    actual: list[float] = []
    predictions: dict[str, list[float]] = {
        "local_offset_trend": [],
        "global_offset_trend": [],
        "const_offset": [],
        "zeros": [],
    }
    for files in discover_wells(train_dir):
        if files.well not in selected:
            continue
        rows = load_horizontal(files.horizontal, require_target=True)
        heel_count = max(1, min(len(rows) - 1, round(len(rows) * heel_fraction)))
        global_model = fit_offset_trend(rows[:heel_count])
        local_model = fit_offset_trend(
            rows[:heel_count], recency_decay=LOCAL_OFFSET_RECENCY_DECAY
        )
        for row in rows[heel_count:]:
            if not row.get("TVT"):
                continue
            truth = float(row["TVT"])
            actual.append(truth)
            predictions["local_offset_trend"].append(
                predict_offset_tvt(local_model, row)
            )
            predictions["global_offset_trend"].append(
                predict_offset_tvt(global_model, row)
            )
            try:
                z = float(row["Z"])
            except (KeyError, TypeError, ValueError):
                z = math.nan
            const_prediction = (
                global_model.fallback_offset - z
                if math.isfinite(z)
                else global_model.fallback_offset
            )
            predictions["const_offset"].append(const_prediction)
            predictions["zeros"].append(0.0)
    metrics = {name: _metrics(actual, values) for name, values in predictions.items()}
    candidate = metrics["local_offset_trend"]
    deltas = {
        baseline: {
            "rmse": candidate["rmse"] - metrics[baseline]["rmse"],
            "mae": candidate["mae"] - metrics[baseline]["mae"],
        }
        for baseline in ("global_offset_trend", "zeros", "const_offset")
    }
    passed = all(
        delta["rmse"] < 0 and delta["mae"] < 0 for delta in deltas.values()
    )
    return {
        "stage": stage,
        "fold": PSEUDO_BLIND_FOLD,
        "folds": PSEUDO_BLIND_FOLDS,
        "heel_fraction": heel_fraction,
        "wells": len(selected),
        "well_ids": selected,
        "metrics": metrics,
        "deltas": deltas,
        "passed": passed,
    }


def evaluate_toe_gate(train_dir: str | Path) -> dict:
    """Run the mandatory toe-holdout screen before the full confirmation."""
    stages = []
    for stage in ("screen", "confirm"):
        result = evaluate_toe_holdout(train_dir, stage=stage)
        stages.append(result)
        if not result["passed"]:
            break
    return {"promoted": len(stages) == 2 and stages[-1]["passed"], "stages": stages}


# Champion (PF fallback) toe-holdout RMSE, recorded in champion.json /
# docs/champion-selection.md; the ML base predictor is measured against it.
CHAMPION_PF_TOE_RMSE = {"screen": 8.297, "confirm": 11.225}


def evaluate_ml_toe_holdout(
    train_dir: str | Path,
    *,
    stage: str = "screen",
    heel_fraction: float = TEST_HEEL_FRACTION,
) -> dict:
    """Measure the offline ML base predictor on the leak-free toe hold-out.

    The withheld toe suffix of each fold-0 hold-out well is predicted by the
    distilled GBRT (``src.ml_predictor``). The predictor's pooled RMSE/MAE is
    reported against the champion particle-filter fallback (``CHAMPION_PF_TOE_RMSE``)
    and the recency-weighted offset-trend baseline it must also clear.
    """
    from .ml_predictor import predict_ml_well

    if stage not in {"screen", "confirm"}:
        raise ValueError("stage must be 'screen' or 'confirm'")
    if not 0.0 < heel_fraction < 1.0:
        raise ValueError("heel_fraction must be between zero and one")
    selected = sorted(holdout_wells(train_dir, PSEUDO_BLIND_FOLDS, PSEUDO_BLIND_FOLD))
    if stage == "screen":
        selected = selected[:SCREEN_WELLS]
    actual: list[float] = []
    ml_pred: list[float] = []
    offset_pred: list[float] = []
    root = Path(train_dir)
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
        model = fit_offset_trend(masked[:heel_count], recency_decay=LOCAL_OFFSET_RECENCY_DECAY)
        prediction = predict_ml_well(masked, typewell)
        for i in range(heel_count, len(rows)):
            if not rows[i].get("TVT"):
                continue
            actual.append(float(rows[i]["TVT"]))
            ml_pred.append(float(prediction[i]))
            offset_pred.append(predict_offset_tvt(model, rows[i]))
    ml_metrics = _metrics(actual, ml_pred)
    offset_metrics = _metrics(actual, offset_pred)
    champion = CHAMPION_PF_TOE_RMSE[stage]
    return {
        "stage": stage,
        "predictor": "ml_gbrt_base",
        "fold": PSEUDO_BLIND_FOLD,
        "folds": PSEUDO_BLIND_FOLDS,
        "heel_fraction": heel_fraction,
        "wells": len(selected),
        "well_ids": selected,
        "rmse": ml_metrics["rmse"],
        "mae": ml_metrics["mae"],
        "rows": ml_metrics["rows"],
        "offset_trend_rmse": offset_metrics["rmse"],
        "offset_trend_mae": offset_metrics["mae"],
        "champion_pf_rmse": champion,
        "rmse_vs_champion_pf": ml_metrics["rmse"] - champion,
        "beats_champion_pf": ml_metrics["rmse"] < champion,
        "beats_offset_trend": ml_metrics["rmse"] < offset_metrics["rmse"],
    }


def evaluate_blend_toe_holdout(
    train_dir: str | Path,
    *,
    stage: str = "screen",
    heel_fraction: float = TEST_HEEL_FRACTION,
    weight: float | None = None,
) -> dict:
    """Measure the physics × ML fallback blend on the leak-free toe hold-out (SOT-2394).

    For each fold-0 hold-out well the withheld toe suffix is predicted by the
    particle filter (``src.physics``), the offline ML base predictor
    (``src.ml_predictor``), and their gated blend ``weight*PF + (1-weight)*ML``
    (``src.blend``). Reports the pooled RMSE/MAE of each so the blend can be
    compared against ``min(PF champion, ML single)``. The blend **promotes** only
    when its confirm RMSE beats both singles.
    """
    from .blend import BLEND_WEIGHT, blend_trajectories
    from .ml_predictor import predict_ml_well
    from .physics import predict_pf_well

    if stage not in {"screen", "confirm"}:
        raise ValueError("stage must be 'screen' or 'confirm'")
    if not 0.0 < heel_fraction < 1.0:
        raise ValueError("heel_fraction must be between zero and one")
    if weight is None:
        weight = BLEND_WEIGHT
    selected = sorted(holdout_wells(train_dir, PSEUDO_BLIND_FOLDS, PSEUDO_BLIND_FOLD))
    if stage == "screen":
        selected = selected[:SCREEN_WELLS]
    actual: list[float] = []
    pf_pred: list[float] = []
    ml_pred: list[float] = []
    blend_pred: list[float] = []
    root = Path(train_dir)
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
        blended = blend_trajectories(pf, ml, weight)
        for i in range(heel_count, len(rows)):
            if not rows[i].get("TVT"):
                continue
            if not (math.isfinite(pf[i]) and math.isfinite(ml[i])):
                continue
            actual.append(float(rows[i]["TVT"]))
            pf_pred.append(float(pf[i]))
            ml_pred.append(float(ml[i]))
            blend_pred.append(float(blended[i]))
    pf_metrics = _metrics(actual, pf_pred)
    ml_metrics = _metrics(actual, ml_pred)
    blend_metrics = _metrics(actual, blend_pred)
    best_single = min(pf_metrics["rmse"], ml_metrics["rmse"])
    return {
        "stage": stage,
        "predictor": "pf_ml_blend",
        "weight": weight,
        "fold": PSEUDO_BLIND_FOLD,
        "folds": PSEUDO_BLIND_FOLDS,
        "heel_fraction": heel_fraction,
        "wells": len(selected),
        "well_ids": selected,
        "rmse": blend_metrics["rmse"],
        "mae": blend_metrics["mae"],
        "rows": blend_metrics["rows"],
        "pf_rmse": pf_metrics["rmse"],
        "ml_rmse": ml_metrics["rmse"],
        "best_single_rmse": best_single,
        "rmse_vs_pf": blend_metrics["rmse"] - pf_metrics["rmse"],
        "rmse_vs_ml": blend_metrics["rmse"] - ml_metrics["rmse"],
        "beats_pf": blend_metrics["rmse"] < pf_metrics["rmse"],
        "beats_ml": blend_metrics["rmse"] < ml_metrics["rmse"],
        "beats_both_singles": blend_metrics["rmse"] < best_single,
    }


def evaluate_gold_toe_holdout(
    train_dir: str | Path,
    *,
    stage: str = "screen",
    heel_fraction: float = TEST_HEEL_FRACTION,
    weight: float | None = None,
) -> dict:
    """Measure the gold-calibration overlay on the leak-free toe hold-out (SOT-2395).

    For each fold-0 hold-out well the withheld toe suffix is predicted by the
    stage-2 blend (``src.blend``) and by the gold-calibrated trajectory
    (``src.calibrate.gold_calibrate_trajectory``), which adds a per-well
    visible-prefix self-verified move on top of the blend. Reports the pooled
    RMSE/MAE of the pure particle filter, the blend, and the gold overlay so the
    overlay can be compared against the stage-2 blend it must not regress. Gold
    **promotes** only when its confirm RMSE beats the blend.
    """
    from .blend import BLEND_WEIGHT, blend_trajectories
    from .calibrate import gold_calibrate_trajectory
    from .ml_predictor import predict_ml_well
    from .physics import predict_pf_well

    if stage not in {"screen", "confirm"}:
        raise ValueError("stage must be 'screen' or 'confirm'")
    if not 0.0 < heel_fraction < 1.0:
        raise ValueError("heel_fraction must be between zero and one")
    if weight is None:
        weight = BLEND_WEIGHT
    selected = sorted(holdout_wells(train_dir, PSEUDO_BLIND_FOLDS, PSEUDO_BLIND_FOLD))
    if stage == "screen":
        selected = selected[:SCREEN_WELLS]
    actual: list[float] = []
    pf_pred: list[float] = []
    blend_pred: list[float] = []
    gold_pred: list[float] = []
    moved_wells = 0
    root = Path(train_dir)
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
        blended = blend_trajectories(pf, ml, weight)
        gold = gold_calibrate_trajectory(masked, typewell, weight)
        if not bool((gold != blended).any()):
            pass
        else:
            moved_wells += 1
        for i in range(heel_count, len(rows)):
            if not rows[i].get("TVT"):
                continue
            if not (math.isfinite(pf[i]) and math.isfinite(blended[i]) and math.isfinite(gold[i])):
                continue
            actual.append(float(rows[i]["TVT"]))
            pf_pred.append(float(pf[i]))
            blend_pred.append(float(blended[i]))
            gold_pred.append(float(gold[i]))
    pf_metrics = _metrics(actual, pf_pred)
    blend_metrics = _metrics(actual, blend_pred)
    gold_metrics = _metrics(actual, gold_pred)
    return {
        "stage": stage,
        "predictor": "gold_calibration",
        "weight": weight,
        "fold": PSEUDO_BLIND_FOLD,
        "folds": PSEUDO_BLIND_FOLDS,
        "heel_fraction": heel_fraction,
        "wells": len(selected),
        "moved_wells": moved_wells,
        "well_ids": selected,
        "rmse": gold_metrics["rmse"],
        "mae": gold_metrics["mae"],
        "rows": gold_metrics["rows"],
        "blend_rmse": blend_metrics["rmse"],
        "blend_mae": blend_metrics["mae"],
        "pf_rmse": pf_metrics["rmse"],
        "rmse_vs_blend": gold_metrics["rmse"] - blend_metrics["rmse"],
        "rmse_vs_pf": gold_metrics["rmse"] - pf_metrics["rmse"],
        "beats_blend": gold_metrics["rmse"] < blend_metrics["rmse"],
        "no_regression_vs_blend": gold_metrics["rmse"] <= blend_metrics["rmse"],
    }


def _read_typewell(path: str | Path) -> list[dict[str, str]]:
    import csv

    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


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
    toe = subparsers.add_parser(
        "toe-holdout", help="evaluate offset-trend suffix extrapolation"
    )
    toe.add_argument("--train-dir", type=Path, default=Path("data/raw/train"))
    toe.add_argument("--stage", choices=("screen", "confirm"), default="screen")
    toe.add_argument(
        "--predictor",
        choices=("offset-trend", "ml", "blend", "gold"),
        default="offset-trend",
        help="offset-trend (default), the ML base predictor, the PF×ML blend, or the gold overlay",
    )
    submission = subparsers.add_parser("submission", help="create a format-smoke submission")
    submission.add_argument("--sample", type=Path, default=Path("data/raw/sample_submission.csv"))
    submission.add_argument("--output", type=Path, default=Path("submission.csv"))
    submission.add_argument("--constant", type=float, default=0.0)
    args = parser.parse_args()
    if args.command == "baseline":
        print(json.dumps(evaluate_baseline(args.train_dir, args.folds, args.fold), indent=2))
    elif args.command == "pseudo-blind":
        print(json.dumps(evaluate_pseudo_blind(args.train_dir, stage=args.stage), indent=2))
    elif args.command == "toe-holdout":
        predictor = getattr(args, "predictor", "offset-trend")
        if predictor == "ml":
            print(json.dumps(evaluate_ml_toe_holdout(args.train_dir, stage=args.stage), indent=2))
        elif predictor == "blend":
            print(json.dumps(evaluate_blend_toe_holdout(args.train_dir, stage=args.stage), indent=2))
        elif predictor == "gold":
            print(json.dumps(evaluate_gold_toe_holdout(args.train_dir, stage=args.stage), indent=2))
        else:
            print(json.dumps(evaluate_toe_holdout(args.train_dir, stage=args.stage), indent=2))
    else:
        targets = load_submission_targets(args.sample)
        predictions = {target.id: args.constant for target in targets}
        output = write_submission(args.sample, predictions, args.output)
        print(f"Wrote {len(targets)} rows to {output}")


if __name__ == "__main__":
    main()
