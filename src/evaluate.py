from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

from .data import discover_wells, iter_train_pairs
from .data import load_submission_targets, write_submission


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


def main() -> None:
    parser = argparse.ArgumentParser(description="ROGII local KPI and submission utility")
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate = subparsers.add_parser("baseline", help="score TVT_input on held-out train wells")
    evaluate.add_argument("--train-dir", type=Path, default=Path("data/raw/train"))
    evaluate.add_argument("--folds", type=int, default=5)
    evaluate.add_argument("--fold", type=int, default=0)
    submission = subparsers.add_parser("submission", help="create a format-smoke submission")
    submission.add_argument("--sample", type=Path, default=Path("data/raw/sample_submission.csv"))
    submission.add_argument("--output", type=Path, default=Path("submission.csv"))
    submission.add_argument("--constant", type=float, default=0.0)
    args = parser.parse_args()
    if args.command == "baseline":
        print(json.dumps(evaluate_baseline(args.train_dir, args.folds, args.fold), indent=2))
    else:
        targets = load_submission_targets(args.sample)
        predictions = {target.id: args.constant for target in targets}
        output = write_submission(args.sample, predictions, args.output)
        print(f"Wrote {len(targets)} rows to {output}")


if __name__ == "__main__":
    main()
