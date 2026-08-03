"""Particle-filter TVT tracking (ported from the public physics kernel).

Port of the likelihood-weighted particle-filter ensemble from
``evgendvorkin/rogii-physics-lb-7-872-v48``. For each well the filter tracks
the stratigraphic level ``U = TVT + Z`` through the evaluation zone, matching
the horizontal GR log against the typewell GR-vs-TVT signature. Seeds are
combined with likelihood weights, and the trajectory is smoothed with a robust
IRLS polynomial fit of ``U`` over normalized MD (the notebook's "projection").

Requires numpy (preinstalled on Kaggle; ``pip install numpy`` locally). The
caller is expected to fall back to the offset-trend champion when numpy is
unavailable or the filter cannot run (e.g. an empty typewell).
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

N_PARTICLES = 400
N_SEEDS = 32
LIK_SCALE = 5.0
PROJECTION_DEGREE = 3
PROJECTION_BLEND_WEIGHT = 0.75


def _column(rows: Sequence[Mapping[str, str]], key: str) -> np.ndarray:
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


def _interp_nan(values: np.ndarray, fill: float) -> np.ndarray:
    """Interpolate NaN gaps in both directions, like pandas interpolate."""
    result = values.copy()
    finite = np.isfinite(result)
    if not finite.any():
        result[:] = fill
        return result
    indices = np.arange(len(result))
    result[~finite] = np.interp(indices[~finite], indices[finite], result[finite])
    return result


def run_particle_filter(
    md: np.ndarray,
    z: np.ndarray,
    gr: np.ndarray,
    known_tvt: np.ndarray,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    *,
    n_particles: int = N_PARTICLES,
    seed: int = 42,
) -> tuple[np.ndarray, float]:
    """Track TVT through rows where ``known_tvt`` is NaN; returns (tvt, log_lik).

    ``md``/``z``/``gr``/``known_tvt`` are full-well arrays; ``tw_tvt`` must be
    sorted ascending with matching ``tw_gr``.
    """
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
    if positive.sum() >= 3:
        drift = float(np.median((dt + dz)[positive] / dm[positive]))
    else:
        drift = 0.0

    n = n_particles
    rng = np.random.default_rng(seed)
    pos = (last_tvt + last_z) + 4.5 * rng.standard_normal(n)
    rate = drift + 0.01 * rng.standard_normal(n)
    weights = np.full(n, 1.0 / n)

    momentum, vel_noise, pos_noise = 0.998, 0.002, 0.005
    resample_pos, resample_rate, resample_frac = 0.1, 0.001, 0.5

    ev_idx = np.flatnonzero(eval_mask)
    gr_filled = _interp_nan(gr, float(np.nanmean(tw_gr_filled)))
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


def pf_lik_ensemble(
    md: np.ndarray,
    z: np.ndarray,
    gr: np.ndarray,
    known_tvt: np.ndarray,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    *,
    n_particles: int = N_PARTICLES,
    n_seeds: int = N_SEEDS,
    scale: float = LIK_SCALE,
) -> np.ndarray:
    """Likelihood-weighted average of per-seed particle-filter trajectories."""
    preds = []
    liks = []
    for seed in range(n_seeds):
        pred, log_lik = run_particle_filter(
            md, z, gr, known_tvt, tw_tvt, tw_gr, n_particles=n_particles, seed=seed
        )
        preds.append(pred)
        liks.append(log_lik)
    liks_arr = np.array(liks)
    weights = np.exp((liks_arr - liks_arr.max()) / scale)
    weights /= weights.sum()
    return (weights[:, None] * np.stack(preds, 0)).sum(0)


def projection_smooth(
    md: np.ndarray,
    z: np.ndarray,
    tvt: np.ndarray,
    known_tvt: np.ndarray,
    *,
    degree: int = PROJECTION_DEGREE,
    blend: float = PROJECTION_BLEND_WEIGHT,
) -> np.ndarray:
    """Robust-polynomial smoothing of U = TVT + Z over the evaluation zone."""
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
        weights = 1.0 / (1.0 + (residual / (2.0 * sigma)) ** 2)
        coeffs = np.polyfit(s, y, degree, w=weights)
    fitted = (anchor + np.polyval(coeffs, s)) - z[eval_mask]
    smoothed_eval = (1.0 - blend) * tvt[eval_mask] + blend * fitted
    if not np.all(np.isfinite(smoothed_eval)):
        return tvt
    smoothed = tvt.copy()
    smoothed[eval_mask] = smoothed_eval
    return smoothed


def predict_pf_well(
    horizontal_rows: Sequence[Mapping[str, str]],
    typewell_rows: Sequence[Mapping[str, str]],
    *,
    n_particles: int = N_PARTICLES,
    n_seeds: int = N_SEEDS,
) -> np.ndarray:
    """Full-well TVT prediction: known rows pass through, eval rows are tracked."""
    md = _column(horizontal_rows, "MD")
    z = _column(horizontal_rows, "Z")
    gr = _column(horizontal_rows, "GR")
    known_tvt = _column(horizontal_rows, "TVT_input")
    tw_tvt_raw = _column(typewell_rows, "TVT")
    tw_gr_raw = _column(typewell_rows, "GR")
    finite = np.isfinite(tw_tvt_raw)
    if finite.sum() < 2:
        raise ValueError("typewell has fewer than two finite TVT rows")
    order = np.argsort(tw_tvt_raw[finite])
    tw_tvt = tw_tvt_raw[finite][order]
    tw_gr = tw_gr_raw[finite][order]
    if not np.isfinite(md).all():
        raise ValueError("horizontal MD contains non-finite values")

    tracked = pf_lik_ensemble(
        md, z, gr, known_tvt, tw_tvt, tw_gr, n_particles=n_particles, n_seeds=n_seeds
    )
    return projection_smooth(md, z, tracked, known_tvt)
