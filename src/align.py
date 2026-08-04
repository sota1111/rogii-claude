"""Beam-search + multi-scale NCC GR↔typewell alignment candidate (SOT-2442).

A fourth portable candidate for the gold-calibration pool (``src/calibrate.py``
``_gold_candidate_pool`` → ``{blend, pf, ml, poly, beam}``). It ports the most
frequently reused, framework-free technique from the public top notebooks
(``romantamrazov/rogii-better-solution-lb-9-956`` /
``…/rogii-super-solution-lb-top-3``): a **±2-delta beam search (backward
allowed)** that Viterbi-aligns the horizontal well's stratigraphic level
``U = TVT + Z`` to the typewell GR-vs-TVT signature, plus a **multi-scale NCC**
(windows 8/15/25, softmax weights) global registration measured on the *known*
heel and carried into the hidden toe.

Mechanism (numpy-only, ``__file__``-independent, fully deterministic — no RNG):

1. **NCC registration (known zone).** Smooth the horizontal GR and the typewell
   GR at three window scales (8/15/25). For each scale find the TVT lag ``τ``
   that maximises the normalised cross-correlation between the *known*-zone GR
   and the typewell GR sampled at ``known_TVT + τ`` (stride 3). Combine the
   per-scale best lags with ``softmax(corr·3.0)`` weights. ``τ`` is a leak-free
   global registration (it only reads known TVT + GR + typewell) that corrects a
   systematic TVT-frame offset between the horizontal and the typewell.
2. **Beam search DP (toe).** Anchor ``U`` at the last known row and extend a
   drift baseline ``U_base = U_anchor + drift·(MD − MD_last)`` (drift = robust
   heel slope of ``U`` over ``MD``). Discretise the residual ``r = U − U_base``
   on a grid and Viterbi-search the path minimising
   ``emission + move_cost·|Δ|`` where the per-step index move ``Δ ∈ [−2, +2]``
   (backward allowed) and the emission is
   ``((GR − typewell_GR(TVT + τ))/gs)² / emit_scale``. Traceback → ``TVT`` on
   the hidden toe. The reference notebook averages several ``(emit_scale,
   move_cost)`` configs, so we average the resulting trajectories over
   ``ALIGN_CONFIGS``.

Known heel rows always pass through unchanged. On any degeneracy (no typewell,
no known/eval rows) the candidate degrades to the finite drift baseline so it can
never inject non-finite values into the pool; where it cannot anchor at all it
returns the (all-NaN) eval rows so the gold backtest simply never selects it.

Because the gold gate back-tests every candidate on withheld tails of each well's
own visible heel and only adopts one with a conservative margin, adding ``beam``
is non-regressive by construction (worst case: the gate never picks it and the
champion blend is unchanged).

The block between the ``ALIGN_SHARED_CODE`` markers is copied verbatim into
``kaggle/rogii-claude-baseline.py`` (enforced by ``tests/test_align.py``), so the
kernel resolves the bare name ``predict_beam_well`` used inside the byte-identical
``GOLD_SHARED_CODE`` pool.
"""
from __future__ import annotations

import numpy as np

# === BEGIN ALIGN_SHARED_CODE (synced verbatim into kaggle/rogii-claude-baseline.py) ===
# Beam-search + multi-scale NCC GR↔typewell alignment (SOT-2442). numpy-only,
# __file__-independent, deterministic (no RNG). Ported from the public top
# notebooks' portable alignment core; averaged over ALIGN_CONFIGS.
ALIGN_CONFIGS = ((1.0, 0.8), (1.6, 0.4))   # (emit_scale, move_cost) beam configs, averaged
ALIGN_DU = 0.5                             # residual grid resolution (TVT units)
ALIGN_RADIUS = 18.0                        # +/- residual search radius around the drift baseline
ALIGN_MAX_DELTA = 2                        # per-step grid-index move (backward allowed)
ALIGN_NCC_WINDOWS = (8, 15, 25)            # multi-scale smoothing windows
ALIGN_NCC_STRIDE = 3                       # subsample stride for the known-zone NCC
ALIGN_NCC_SOFTMAX = 3.0                    # softmax temperature over per-scale NCC
ALIGN_NCC_LAG = 12.0                       # +/- TVT registration lag searched by NCC
ALIGN_NCC_LAG_STEP = 0.5                   # registration lag grid resolution


def _align_column(rows, key):
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


def _align_typewell(typewell_rows):
    """Sorted (tw_tvt, tw_gr) with NaN GR mean-filled; None if fewer than two rows."""
    tw_tvt = _align_column(typewell_rows, "TVT")
    tw_gr = _align_column(typewell_rows, "GR")
    finite = np.isfinite(tw_tvt)
    if int(finite.sum()) < 2:
        return None
    order = np.argsort(tw_tvt[finite])
    tvt = tw_tvt[finite][order]
    gr = tw_gr[finite][order]
    fill = float(np.nanmean(gr)) if np.isfinite(gr).any() else 0.0
    gr = np.where(np.isfinite(gr), gr, fill)
    return tvt, gr


def _align_smooth(values, window):
    """Odd-centred moving average; NaN-safe, preserves length."""
    values = np.asarray(values, dtype=float)
    if window <= 1 or len(values) == 0:
        return values.copy()
    filled = values.copy()
    finite = np.isfinite(filled)
    if not finite.all():
        if not finite.any():
            return filled
        idx = np.arange(len(filled))
        filled[~finite] = np.interp(idx[~finite], idx[finite], filled[finite])
    half = int(window) // 2
    kernel = np.ones(2 * half + 1, dtype=float) / float(2 * half + 1)
    padded = np.pad(filled, half, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _ncc_registration(md, z, gr, known_tvt, tw_tvt, tw_gr):
    """Leak-free global TVT registration lag from multi-scale known-zone NCC.

    Returns ``τ`` such that ``typewell_GR(TVT + τ)`` best matches the horizontal GR
    over the known heel, combined across scales with ``softmax(corr·k)`` weights.
    Reads only known TVT / GR / typewell, so it never touches the hidden toe.
    """
    known = np.isfinite(known_tvt)
    if int(known.sum()) < ALIGN_NCC_WINDOWS[0] + 2:
        return 0.0
    ki = np.flatnonzero(known)[:: max(1, int(ALIGN_NCC_STRIDE))]
    if len(ki) < 4:
        return 0.0
    ktvt = known_tvt[ki]
    kgr = gr[ki]
    lags = np.arange(-ALIGN_NCC_LAG, ALIGN_NCC_LAG + 1e-9, ALIGN_NCC_LAG_STEP)
    best_lags = []
    best_corrs = []
    for window in ALIGN_NCC_WINDOWS:
        smooth_gr = _align_smooth(kgr, window)
        sg = smooth_gr - np.mean(smooth_gr)
        sg_norm = float(np.sqrt(np.dot(sg, sg)))
        if sg_norm < 1e-6:
            continue
        best_corr = -2.0
        best_lag = 0.0
        for lag in lags:
            expected = np.interp(ktvt + lag, tw_tvt, tw_gr)
            se = _align_smooth(expected, window)
            se = se - np.mean(se)
            se_norm = float(np.sqrt(np.dot(se, se)))
            if se_norm < 1e-6:
                continue
            corr = float(np.dot(sg, se)) / (sg_norm * se_norm)
            if corr > best_corr:
                best_corr = corr
                best_lag = float(lag)
        if best_corr > -2.0:
            best_lags.append(best_lag)
            best_corrs.append(best_corr)
    if not best_lags:
        return 0.0
    corrs = np.asarray(best_corrs, dtype=float)
    weights = np.exp(ALIGN_NCC_SOFTMAX * (corrs - corrs.max()))
    weights = weights / weights.sum()
    return float(np.dot(weights, np.asarray(best_lags, dtype=float)))


def _beam_viterbi(md, z, gr, u_base, ev_idx, states, tw_tvt, tw_gr, tau, gs,
                  emit_scale, move_cost):
    """±ALIGN_MAX_DELTA beam (exact Viterbi) over the residual grid; returns eval TVT."""
    n_states = len(states)
    center = n_states // 2
    # Emission cost matrix over eval rows (rows x states).
    tvt_grid = (u_base[ev_idx][:, None] + states[None, :]) - z[ev_idx][:, None]
    expected = np.interp((tvt_grid + tau).ravel(), tw_tvt, tw_gr).reshape(tvt_grid.shape)
    gr_ev = gr[ev_idx][:, None]
    resid = (gr_ev - expected) / gs
    emission = (resid * resid) / float(emit_scale)
    emission = np.where(np.isfinite(emission), emission, 0.0)
    deltas = np.arange(-ALIGN_MAX_DELTA, ALIGN_MAX_DELTA + 1)
    # Start: prefer residual near zero (continuity with the known heel).
    dp = emission[0] + 0.05 * np.abs(states)
    back = np.empty((len(ev_idx), n_states), dtype=np.int32)
    back[0] = np.arange(n_states)
    for i in range(1, len(ev_idx)):
        best = np.full(n_states, np.inf)
        arg = np.zeros(n_states, dtype=np.int32)
        for delta in deltas:
            # transition from state a=b-delta into state b
            shifted = np.full(n_states, np.inf)
            cost = move_cost * abs(int(delta))
            if delta >= 0:
                if delta < n_states:
                    shifted[delta:] = dp[: n_states - delta] + cost
                src = np.arange(n_states) - delta
            else:
                d = -delta
                if d < n_states:
                    shifted[: n_states - d] = dp[d:] + cost
                src = np.arange(n_states) + d
            take = shifted < best
            best = np.where(take, shifted, best)
            arg = np.where(take, np.clip(src, 0, n_states - 1).astype(np.int32), arg)
        dp = best + emission[i]
        back[i] = arg
    # Traceback from the minimum-cost final state.
    s = int(np.argmin(dp))
    path = np.empty(len(ev_idx), dtype=np.int32)
    for i in range(len(ev_idx) - 1, -1, -1):
        path[i] = s
        s = int(back[i, s])
    chosen = states[path]
    return (u_base[ev_idx] + chosen) - z[ev_idx]


def _drift_baseline(md, z, known_tvt):
    """Finite U_base = U_anchor + drift·(MD − MD_last) from the known heel."""
    known = np.isfinite(known_tvt)
    kn = np.flatnonzero(known)
    last = int(kn[-1])
    u = known_tvt + z
    u_anchor = float(u[last])
    last_md = float(md[last])
    tail = kn[-30:]
    du = np.diff(u[tail])
    dm = np.diff(md[tail])
    positive = dm > 0
    drift = float(np.median(du[positive] / dm[positive])) if int(positive.sum()) >= 3 else 0.0
    if not np.isfinite(drift):
        drift = 0.0
    return u_anchor + drift * (md - last_md)


def predict_beam_well(horizontal_rows, typewell_rows):
    """Full-well beam+NCC TVT candidate: known heel passes through, toe is aligned.

    Returns a full-well trajectory the same length as ``horizontal_rows`` for the
    gold candidate pool. Known ``TVT_input`` rows are copied verbatim; the hidden
    toe is estimated by the beam Viterbi (averaged over ``ALIGN_CONFIGS``) with the
    NCC registration ``τ``. Degrades to the finite drift baseline on the toe when
    the typewell/GR signal is unusable, and to the raw known array when it cannot
    anchor at all.
    """
    md = _align_column(horizontal_rows, "MD")
    z = _align_column(horizontal_rows, "Z")
    gr = _align_column(horizontal_rows, "GR")
    known_tvt = _align_column(horizontal_rows, "TVT_input")
    out = known_tvt.copy()
    known = np.isfinite(known_tvt)
    ev = np.flatnonzero(~known)
    if not known.any() or ev.size == 0 or not np.isfinite(md).all():
        return out
    u_base = _drift_baseline(md, z, known_tvt)
    baseline_tvt = u_base - z            # finite fallback for the toe
    tw = _align_typewell(typewell_rows)
    if tw is None:
        out[ev] = baseline_tvt[ev]
        return out
    tw_tvt, tw_gr = tw
    tau = _ncc_registration(md, z, gr, known_tvt, tw_tvt, tw_gr)
    # GR emission scale from the known-zone residual (mirrors the particle filter's gs).
    kmask = known & np.isfinite(gr)
    if int(kmask.sum()) >= 3:
        tw_at_known = np.interp(known_tvt[kmask] + tau, tw_tvt, tw_gr)
        gs = float(np.clip(np.nanstd(gr[kmask] - tw_at_known), 10.0, 60.0))
    else:
        gs = 30.0
    k = int(round(ALIGN_RADIUS / ALIGN_DU))
    states = np.arange(-k, k + 1, dtype=float) * ALIGN_DU
    preds = []
    for emit_scale, move_cost in ALIGN_CONFIGS:
        try:
            preds.append(
                _beam_viterbi(md, z, gr, u_base, ev, states, tw_tvt, tw_gr,
                              tau, gs, emit_scale, move_cost)
            )
        except Exception:
            continue
    if not preds:
        out[ev] = baseline_tvt[ev]
        return out
    est = np.mean(np.stack(preds, axis=0), axis=0)
    est = np.where(np.isfinite(est), est, baseline_tvt[ev])
    out[ev] = est
    return out
# === END ALIGN_SHARED_CODE ===
