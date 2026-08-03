"""Kaggle Notebook entry point for the cycle-5 contact-override + particle-filter champion.

Self-contained (numpy only, no internet). Reproduces the ``src.predict``
champion, resolving each well by the first applicable layer:

1. Guarded contact override — wells whose same-id train copy reconstructs the
   trajectory within ``PREFIX_RMSE_LIMIT`` of the visible ``TVT_input`` prefix
   are predicted by ``TVT = ref_tvt - (Z - formation) + offset``.
2. Likelihood-weighted particle filter — for wells with no compatible train
   copy (the hidden-test path), a numpy particle-filter ensemble tracks
   ``U = TVT + Z`` against the typewell GR signature, then a robust IRLS
   polynomial smooths the trajectory.
3. Recency-weighted offset trend — the cycle-4 fallback whenever the filter
   cannot run.

Output is byte-identical to the local ``src/predict.py`` generator on the same
platform. Both entry points share the numpy PCG64 seed stream, so the particle
filter is reproducible.

Kaggle runs this from ``/kaggle/input`` -> ``/kaggle/working``. The two paths may
be overridden with ``ROGII_INPUT_DIR`` / ``ROGII_OUTPUT`` for local byte-match
verification; the defaults are the Kaggle Notebook locations.
"""
from __future__ import annotations

import csv
import math
import os
import statistics
from dataclasses import dataclass
from pathlib import Path

try:
    import numpy as np
except ImportError:  # numpy absent: contact override + offset-trend only
    np = None

HORIZONTAL_SUFFIX = "__horizontal_well.csv"
TYPEWELL_SUFFIX = "__typewell.csv"

REF_COLS = ("EGFDU", "ASTNU", "ANCC", "ASTNL", "EGFDL", "BUDA")
MIN_VALID_PHYS_ROWS = 100
MIN_KNOWN_PREFIX_ROWS = 50
PREFIX_RMSE_LIMIT = 1.0

PF_N_PARTICLES = 400
PF_N_SEEDS = 32
PF_LIK_SCALE = 5.0
PF_PROJECTION_DEGREE = 3
PF_PROJECTION_BLEND_WEIGHT = 0.75

INPUT_DIR = Path(os.environ.get("ROGII_INPUT_DIR", "/kaggle/input"))
OUTPUT_PATH = Path(os.environ.get("ROGII_OUTPUT", "/kaggle/working/submission.csv"))


@dataclass(frozen=True)
class OffsetTrend:
    intercept: float
    slope: float
    fallback_offset: float
    terminal_offset: float


def fit_offset_trend(rows, recency_decay=0.0):
    if not math.isfinite(recency_decay) or recency_decay < 0.0:
        raise ValueError("recency_decay must be a finite non-negative number")
    points = []
    for row in rows:
        if not row.get("TVT_input") or not row.get("Z"):
            continue
        try:
            md = float(row["MD"])
            offset = float(row["TVT_input"]) + float(row["Z"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(md) and math.isfinite(offset):
            points.append((md, offset))
    if not points:
        raise ValueError("Cannot fit offset trend without finite heel observations")

    fallback = statistics.median(offset for _, offset in points)
    weights = [
        math.exp(recency_decay * (index - len(points) + 1) / max(1, len(points) - 1))
        for index in range(len(points))
    ]
    weight_sum = sum(weights)
    mean_md = sum(weight * md for weight, (md, _) in zip(weights, points)) / weight_sum
    mean_offset = (
        sum(weight * offset for weight, (_, offset) in zip(weights, points))
        / weight_sum
    )
    denominator = sum(
        weight * (md - mean_md) ** 2
        for weight, (md, _) in zip(weights, points)
    )
    if len(points) < 2 or denominator <= 1e-12:
        intercept, slope = fallback, 0.0
    else:
        slope = (
            sum(
                weight * (md - mean_md) * (offset - mean_offset)
                for weight, (md, offset) in zip(weights, points)
            )
            / denominator
        )
        intercept = mean_offset - slope * mean_md
    terminal_md = points[-1][0]
    terminal_offset = intercept + slope * terminal_md
    if not all(
        math.isfinite(value)
        for value in (intercept, slope, fallback, terminal_offset)
    ):
        intercept, slope, terminal_offset = fallback, 0.0, fallback
    return OffsetTrend(intercept, slope, fallback, terminal_offset)


def predict_offset_tvt(model, row):
    try:
        md = float(row["MD"]) if row.get("MD") else math.nan
    except (TypeError, ValueError):
        md = math.nan
    offset = (
        model.intercept + model.slope * md
        if math.isfinite(md)
        else model.terminal_offset
    )
    if not math.isfinite(offset):
        offset = model.fallback_offset
    try:
        z = float(row["Z"]) if row.get("Z") else math.nan
    except (TypeError, ValueError):
        z = math.nan
    prediction = offset - z if math.isfinite(z) else model.terminal_offset
    if not math.isfinite(prediction):
        prediction = model.fallback_offset
    return prediction


@dataclass(frozen=True)
class ContactCurve:
    ref_col: str
    prefix_rmse: float
    mds: tuple
    tvts: tuple

    def covers(self, md):
        return math.isfinite(md) and self.mds[0] <= md <= self.mds[-1]

    def predict(self, md):
        return _interp(md, self.mds, self.tvts)


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _interp(x, xs, ys):
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    lo, hi = 0, len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    span = xs[hi] - xs[lo]
    if span <= 0.0:
        return ys[lo]
    return ys[lo] + (x - xs[lo]) / span * (ys[hi] - ys[lo])


def fit_contact_curve(train_horizontal, train_typewell, ref_col):
    ref_tvts = [
        _to_float(row.get("TVT"))
        for row in train_typewell
        if (row.get("Geology") or "").strip() == ref_col
    ]
    ref_tvts = [value for value in ref_tvts if math.isfinite(value)]
    if not ref_tvts:
        return None
    ref_tvt = min(ref_tvts)

    samples = []
    residuals = []
    for row in train_horizontal:
        md = _to_float(row.get("MD"))
        raw = ref_tvt - (_to_float(row.get("Z")) - _to_float(row.get(ref_col)))
        if math.isfinite(md) and math.isfinite(raw):
            samples.append((md, raw))
            truth = _to_float(row.get("TVT"))
            if math.isfinite(truth):
                residuals.append(truth - raw)
    if len(samples) < MIN_VALID_PHYS_ROWS or not residuals:
        return None
    offset = sum(residuals) / len(residuals)
    samples.sort()
    return (
        tuple(md for md, _ in samples),
        tuple(raw + offset for _, raw in samples),
    )


def best_contact_curve(test_horizontal, train_horizontal, train_typewell):
    known = []
    for row in test_horizontal:
        tvt = _to_float(row.get("TVT_input"))
        md = _to_float(row.get("MD"))
        if math.isfinite(tvt) and math.isfinite(md):
            known.append((md, tvt))
    best = None
    for ref_col in REF_COLS:
        curve = fit_contact_curve(train_horizontal, train_typewell, ref_col)
        if curve is None:
            continue
        mds, tvts = curve
        comparable = [(md, tvt) for md, tvt in known if mds[0] <= md <= mds[-1]]
        if len(comparable) < MIN_KNOWN_PREFIX_ROWS:
            continue
        squares = [(_interp(md, mds, tvts) - tvt) ** 2 for md, tvt in comparable]
        rmse = math.sqrt(sum(squares) / len(squares))
        if not math.isfinite(rmse):
            continue
        if best is None or rmse < best.prefix_rmse:
            best = ContactCurve(ref_col, rmse, mds, tvts)
    if best is None or best.prefix_rmse > PREFIX_RMSE_LIMIT:
        return None
    return best


# --- Particle-filter physics (mirrors src/physics.py; requires numpy) ---


def _pf_column(rows, key):
    values = np.empty(len(rows))
    for i, row in enumerate(rows):
        raw = row.get(key)
        if raw is None or raw == "":
            values[i] = np.nan
        else:
            try:
                values[i] = float(raw)
            except (TypeError, ValueError):
                values[i] = np.nan
    return values


def _pf_interp_nan(values, fill):
    result = values.copy()
    finite = np.isfinite(result)
    if not finite.any():
        result[:] = fill
        return result
    indices = np.arange(len(result))
    result[~finite] = np.interp(indices[~finite], indices[finite], result[finite])
    return result


def _pf_run(md, z, gr, known_tvt, tw_tvt, tw_gr, n_particles, seed):
    known_mask = np.isfinite(known_tvt)
    eval_mask = ~known_mask
    out = known_tvt.copy()
    if not eval_mask.any():
        return out, 0.0
    if not known_mask.any():
        raise ValueError("particle filter needs at least one known TVT row")

    kn_idx = np.flatnonzero(known_mask)
    last = kn_idx[-1]
    last_tvt = float(known_tvt[last])
    last_z = float(z[last])
    last_md = float(md[last])

    tw_gr_filled = np.where(np.isfinite(tw_gr), tw_gr, np.nanmean(tw_gr))
    tw_at_known = np.interp(known_tvt[known_mask], tw_tvt, tw_gr_filled)
    gr_known = np.where(np.isfinite(gr[known_mask]), gr[known_mask], 0.0)
    gs = float(np.clip(np.nanstd(gr_known - tw_at_known), 10.0, 60.0))

    tail = kn_idx[-30:]
    dt = np.diff(known_tvt[tail])
    dz = np.diff(z[tail])
    dm = np.diff(md[tail])
    positive = dm > 0
    drift = float(np.median((dt + dz)[positive] / dm[positive])) if positive.sum() >= 3 else 0.0

    n = n_particles
    rng = np.random.default_rng(seed)
    pos = (last_tvt + last_z) + 4.5 * rng.standard_normal(n)
    rate = drift + 0.01 * rng.standard_normal(n)
    weights = np.full(n, 1.0 / n)

    momentum, vel_noise, pos_noise = 0.998, 0.002, 0.005
    resample_pos, resample_rate, resample_frac = 0.1, 0.001, 0.5

    ev_idx = np.flatnonzero(eval_mask)
    gr_filled = _pf_interp_nan(gr, float(np.nanmean(tw_gr_filled)))
    lower = tw_tvt[0] - 100.0
    upper = tw_tvt[-1] + 100.0

    prev_md = last_md
    log_lik = 0.0
    for i in ev_idx:
        dm_step = max(float(md[i]) - prev_md, 1.0)
        rate = momentum * rate + vel_noise * rng.standard_normal(n)
        pos = pos + rate * dm_step + pos_noise * rng.standard_normal(n)
        tvt_p = np.clip(pos - z[i], lower, upper)
        pos = tvt_p + z[i]

        expected_gr = np.interp(tvt_p, tw_tvt, tw_gr_filled)
        d = (gr_filled[i] - expected_gr) / gs
        lk = np.exp(-0.5 * np.minimum(d**2, 600.0))
        lk = np.maximum(lk, 1e-300)
        log_lik += math.log(max(float((weights * lk).sum()), 1e-300))
        weights = weights * lk
        total = weights.sum()
        weights = weights / total if total > 0 else np.full(n, 1.0 / n)

        n_eff = 1.0 / float((weights**2).sum())
        if n_eff < resample_frac * n:
            cum = np.cumsum(weights)
            u0 = rng.uniform(0, 1.0 / n)
            idx = np.clip(np.searchsorted(cum, u0 + np.arange(n) / n), 0, n - 1)
            pos = pos[idx] + resample_pos * rng.standard_normal(n)
            rate = rate[idx] + resample_rate * rng.standard_normal(n)
            weights = np.full(n, 1.0 / n)

        out[i] = float(np.dot(weights, pos - z[i]))
        prev_md = float(md[i])
    return out, log_lik


def _pf_ensemble(md, z, gr, known_tvt, tw_tvt, tw_gr, n_particles, n_seeds, scale):
    preds = []
    liks = []
    for seed in range(n_seeds):
        pred, log_lik = _pf_run(md, z, gr, known_tvt, tw_tvt, tw_gr, n_particles, seed)
        preds.append(pred)
        liks.append(log_lik)
    liks_arr = np.array(liks)
    weights = np.exp((liks_arr - liks_arr.max()) / scale)
    weights /= weights.sum()
    return (weights[:, None] * np.stack(preds, 0)).sum(0)


def _pf_projection(md, z, tvt, known_tvt, degree, blend):
    known_mask = np.isfinite(known_tvt)
    eval_mask = ~known_mask
    if not known_mask.any() or eval_mask.sum() < degree + 2:
        return tvt
    last = int(np.flatnonzero(known_mask)[-1])
    anchor = float(known_tvt[last]) + float(z[last])
    start_md = float(md[last])
    end_md = float(md[-1])
    s = (md[eval_mask] - start_md) / max(end_md - start_md, 1e-6)
    y = (tvt[eval_mask] + z[eval_mask]) - anchor

    coeffs = np.polyfit(s, y, degree)
    for _ in range(4):
        residual = y - np.polyval(coeffs, s)
        sigma = float(np.median(np.abs(residual))) * 1.4826 + 1e-6
        w = 1.0 / (1.0 + (residual / (2.0 * sigma)) ** 2)
        coeffs = np.polyfit(s, y, degree, w=w)
    fitted = (anchor + np.polyval(coeffs, s)) - z[eval_mask]
    smoothed_eval = (1.0 - blend) * tvt[eval_mask] + blend * fitted
    if not np.all(np.isfinite(smoothed_eval)):
        return tvt
    smoothed = tvt.copy()
    smoothed[eval_mask] = smoothed_eval
    return smoothed


def predict_pf_well(horizontal_rows, typewell_rows):
    md = _pf_column(horizontal_rows, "MD")
    z = _pf_column(horizontal_rows, "Z")
    gr = _pf_column(horizontal_rows, "GR")
    known_tvt = _pf_column(horizontal_rows, "TVT_input")
    tw_tvt_raw = _pf_column(typewell_rows, "TVT")
    tw_gr_raw = _pf_column(typewell_rows, "GR")
    finite = np.isfinite(tw_tvt_raw)
    if finite.sum() < 2:
        raise ValueError("typewell has fewer than two finite TVT rows")
    order = np.argsort(tw_tvt_raw[finite])
    tw_tvt = tw_tvt_raw[finite][order]
    tw_gr = tw_gr_raw[finite][order]
    if not np.isfinite(md).all():
        raise ValueError("horizontal MD contains non-finite values")
    tracked = _pf_ensemble(
        md, z, gr, known_tvt, tw_tvt, tw_gr,
        PF_N_PARTICLES, PF_N_SEEDS, PF_LIK_SCALE,
    )
    return _pf_projection(
        md, z, tracked, known_tvt, PF_PROJECTION_DEGREE, PF_PROJECTION_BLEND_WEIGHT
    )


def _load_pf_prediction(typewell_index, well, test_rows):
    """Full-well PF trajectory, or None so the caller uses the offset trend."""
    if np is None:
        return None
    typewell_path = typewell_index.get(well)
    if typewell_path is None:
        return None
    try:
        return predict_pf_well(test_rows, _read_rows(typewell_path))
    except Exception as error:
        print(f"particle filter skipped for {well}: {error}")
        return None


def _find_sample(input_dir):
    samples = list(input_dir.rglob("sample_submission.csv"))
    if len(samples) != 1:
        raise RuntimeError(f"Expected one sample_submission.csv, found {samples}")
    return samples[0]


def _index_horizontal_wells(input_dir):
    """Map each well to its horizontal file, preferring the competition test split.

    The same well id can appear under both ``train`` and ``test`` with different
    content (train exposes the target column). Submissions target the test split,
    so a path under a ``test`` directory always wins; ties break on sorted path.
    """
    index = {}
    for path in sorted(input_dir.rglob(f"*{HORIZONTAL_SUFFIX}")):
        well = path.name[: -len(HORIZONTAL_SUFFIX)]
        is_test = any(part.lower() == "test" for part in path.parts)
        if well not in index or (is_test and not index[well][0]):
            index[well] = (is_test, path)
    return {well: path for well, (_, path) in index.items()}


def _index_typewells(input_dir):
    """Map each well to its typewell file, preferring the competition test split."""
    index = {}
    for path in sorted(input_dir.rglob(f"*{TYPEWELL_SUFFIX}")):
        well = path.name[: -len(TYPEWELL_SUFFIX)]
        is_test = any(part.lower() == "test" for part in path.parts)
        if well not in index or (is_test and not index[well][0]):
            index[well] = (is_test, path)
    return {well: path for well, (_, path) in index.items()}


def _index_train_wells(input_dir):
    """Map each train-split well to its (horizontal, typewell) file pair."""
    horizontal = {}
    typewells = {}
    for path in sorted(input_dir.rglob(f"*{HORIZONTAL_SUFFIX}")):
        if any(part.lower() == "train" for part in path.parts):
            horizontal.setdefault(path.name[: -len(HORIZONTAL_SUFFIX)], path)
    for path in sorted(input_dir.rglob(f"*{TYPEWELL_SUFFIX}")):
        if any(part.lower() == "train" for part in path.parts):
            typewells.setdefault(path.name[: -len(TYPEWELL_SUFFIX)], path)
    return {
        well: (horizontal[well], typewells[well])
        for well in horizontal.keys() & typewells.keys()
    }


def _read_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _load_contact_curve(train_index, well, test_rows):
    pair = train_index.get(well)
    if pair is None:
        return None
    try:
        return best_contact_curve(test_rows, _read_rows(pair[0]), _read_rows(pair[1]))
    except Exception as error:  # guarded: any train-copy defect keeps the fallback
        print(f"contact override skipped for {well}: {error}")
        return None


def main():
    sample_path = _find_sample(INPUT_DIR)
    horizontal_index = _index_horizontal_wells(INPUT_DIR)
    typewell_index = _index_typewells(INPUT_DIR)
    train_index = _index_train_wells(INPUT_DIR)

    with sample_path.open(newline="", encoding="utf-8-sig") as handle:
        sample = csv.DictReader(handle)
        if sample.fieldnames != ["id", "tvt"]:
            raise ValueError(
                f"{sample_path} columns must be exactly ['id', 'tvt']; "
                f"got {sample.fieldnames}"
            )
        targets = list(sample)

    rows_by_well = {}
    models = {}
    curves = {}
    pf_by_well = {}
    output_rows = []
    for target in targets:
        target_id = target["id"]
        well, raw_index = target_id.rsplit("_", 1)
        index = int(raw_index)
        if well not in rows_by_well:
            horizontal_path = horizontal_index.get(well)
            if horizontal_path is None:
                raise FileNotFoundError(f"No horizontal well file for {well!r}")
            with horizontal_path.open(newline="", encoding="utf-8-sig") as handle:
                horizontal = csv.DictReader(handle)
                if "TVT_input" not in (horizontal.fieldnames or ()):
                    raise ValueError(f"{horizontal_path} is missing TVT_input")
                rows_by_well[well] = list(horizontal)
                models[well] = fit_offset_trend(
                    rows_by_well[well], recency_decay=8.0
                )
                curve = _load_contact_curve(train_index, well, rows_by_well[well])
                curves[well] = curve
                if curve is not None:
                    print(
                        f"contact override {well}: ref={curve.ref_col} "
                        f"prefix_rmse={curve.prefix_rmse:.4f}"
                    )
                # The contact override, when it fires, covers the whole well, so
                # run the particle filter only where the override is absent (the
                # hidden-test path, where no same-well train copy exists).
                pf_by_well[well] = (
                    None
                    if curve is not None
                    else _load_pf_prediction(
                        typewell_index, well, rows_by_well[well]
                    )
                )
        rows = rows_by_well[well]
        if not 0 <= index < len(rows):
            raise IndexError(f"{target_id}: row {index} is outside {len(rows)} rows")
        curve = curves[well]
        try:
            md = float(rows[index]["MD"]) if rows[index].get("MD") else math.nan
        except (TypeError, ValueError):
            md = math.nan
        pf = pf_by_well[well]
        if curve is not None and curve.covers(md):
            prediction = curve.predict(md)
        elif pf is not None and math.isfinite(pf[index]):
            prediction = float(pf[index])
        else:
            prediction = predict_offset_tvt(models[well], rows[index])
        if not math.isfinite(prediction):
            raise ValueError(f"{target_id}: prediction is not finite")
        # Seven decimals keep the serialized artifact stable across Python
        # runtimes while remaining far below the competition metric precision.
        output_rows.append((target_id, f"{prediction:.7f}"))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["id", "tvt"])
        writer.writerows(output_rows)
    print(f"Wrote {len(output_rows)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
