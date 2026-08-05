"""Constrained + stochastic (Gumbel) DTW GR↔typewell alignment candidate (SOT-2444).

A fifth portable candidate for the gold-calibration pool (``src/calibrate.py``
``_gold_candidate_pool`` → ``{blend, pf, ml, poly, beam, dtw}``). It ports the
distinctive technique of the public notebooks ``nihilisticneuralnet/9.251-DWT``
and ``rauffauzanrambe/9.538-training``: a **Sakoe-Chiba band-constrained DTW**
(radii 20/50/100/200, ensembled) that warps the horizontal well's GR onto the
typewell GR-vs-TVT signature, made **stochastic** with **Gumbel-perturbed
tracebacks** (K=12, temperature 3.0) so the K path realisations yield both a mean
TVT estimate and a per-row uncertainty (std).

Mechanism (numpy-only, ``__file__``-independent, fully deterministic — the Gumbel
noise is drawn from ``np.random.RandomState(DTW_SEED + …)`` with a fixed seed):

1. **Drift baseline (known heel).** Anchor ``U = TVT + Z`` at the last known row
   and extend a robust-slope drift baseline ``U_base`` over ``MD``; ``baseline_tvt
   = U_base − Z`` is the finite conservative fallback for the toe.
2. **Constrained DTW DP (toe).** Discretise the residual ``δ = TVT − baseline_tvt``
   on a grid within a **Sakoe-Chiba band** ``|δ| ≤ R`` and run a Viterbi/DTW DP
   whose emission is ``((GR − typewell_GR(TVT))/gs)² / emit_scale`` and whose
   per-step slope move is bounded (``DTW_MAX_DELTA``). Several band radii
   ``DTW_RADII`` are ensembled — a tight band stays close to the drift baseline
   (conservative), a wide band allows a large warp.
3. **Stochastic DTW (Gumbel).** For each radius run the forward DP once, then draw
   ``K`` Gumbel-perturbed tracebacks (perturb-and-sample: add
   ``−T·log(−log(U))`` to each backward transition cost and take the argmin). The
   ``K × |radii|`` path realisations give a per-row mean (the DTW estimate) and std
   (uncertainty).
4. **Uncertainty weighting.** Shrink the estimate toward the drift baseline where
   the paths disagree: ``tvt = baseline + w·(mean − baseline)`` with ``w =
   1/(1 + (std/DTW_STD_SCALE)²)``, so low-uncertainty (low-std) wells keep the full
   DTW move and ambiguous ones stay conservative.

Known heel rows always pass through unchanged. On any degeneracy (no typewell, no
known/eval rows) the candidate degrades to the finite drift baseline, so it can
never inject non-finite values into the pool.

Because the gold gate back-tests every candidate on withheld tails of each well's
own visible heel and only adopts one with a conservative margin, adding ``dtw`` is
non-regressive by construction (worst case: the gate never picks it and the
champion pool is unchanged).

The block between the ``DTW_SHARED_CODE`` markers is copied verbatim into
``kaggle/rogii-claude-baseline.py`` (enforced by ``tests/test_dtw_align.py``), so
the kernel resolves the bare name ``predict_dtw_well`` used inside the
byte-identical ``GOLD_SHARED_CODE`` pool.
"""
from __future__ import annotations

import numpy as np

# === BEGIN DTW_SHARED_CODE (synced verbatim into kaggle/rogii-claude-baseline.py) ===
# Constrained (Sakoe-Chiba band) + stochastic (Gumbel) DTW GR<->typewell alignment
# (SOT-2444). numpy-only, __file__-independent, deterministic (fixed RandomState
# seed). Ported from the public DWT/training notebooks' portable alignment core.
DTW_RADII = (20.0, 50.0, 100.0, 200.0)   # Sakoe-Chiba band radii (TVT units), ensembled
DTW_MAX_DELTA = 2                          # per-step grid-index slope move (backward allowed)
DTW_EMIT_SCALE = 1.2                       # emission softening (squared-GR-residual scale)
DTW_MOVE_COST = 0.6                        # per-step slope penalty per grid index
DTW_K = 12                                 # stochastic Gumbel traceback realisations
DTW_TEMPERATURE = 3.0                      # Gumbel temperature T for -T*log(-log(U))
DTW_SEED = 20442444                        # fixed seed => deterministic sampling
DTW_STD_SCALE = 6.0                        # uncertainty (std) scale for baseline shrinkage


def _dtw_column(rows, key):
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


def _dtw_typewell(typewell_rows):
    """Sorted (tw_tvt, tw_gr) with NaN GR mean-filled; None if fewer than two rows."""
    tw_tvt = _dtw_column(typewell_rows, "TVT")
    tw_gr = _dtw_column(typewell_rows, "GR")
    finite = np.isfinite(tw_tvt)
    if int(finite.sum()) < 2:
        return None
    order = np.argsort(tw_tvt[finite])
    tvt = tw_tvt[finite][order]
    gr = tw_gr[finite][order]
    fill = float(np.nanmean(gr)) if np.isfinite(gr).any() else 0.0
    gr = np.where(np.isfinite(gr), gr, fill)
    return tvt, gr


def _dtw_drift_baseline(md, z, known_tvt):
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


def _dtw_forward(emission, move_cost):
    """Band-constrained DTW forward pass; returns the accumulated-cost matrix (rows x states).

    Emission is (n_rows x n_states). Transitions allow a per-step grid-index move
    delta in [-DTW_MAX_DELTA, +DTW_MAX_DELTA] (backward allowed) with cost
    ``move_cost·|delta|`` — the DTW slope/continuity constraint inside the band.
    """
    n_rows, n_states = emission.shape
    dp = np.empty((n_rows, n_states), dtype=float)
    dp[0] = emission[0] + 0.05 * np.abs(np.arange(n_states) - n_states // 2)
    deltas = np.arange(-DTW_MAX_DELTA, DTW_MAX_DELTA + 1)
    prev = dp[0]
    for i in range(1, n_rows):
        best = np.full(n_states, np.inf)
        for delta in deltas:
            cost = move_cost * abs(int(delta))
            shifted = np.full(n_states, np.inf)
            if delta >= 0:
                if delta < n_states:
                    shifted[delta:] = prev[: n_states - delta] + cost
            else:
                d = -delta
                if d < n_states:
                    shifted[: n_states - d] = prev[d:] + cost
            best = np.minimum(best, shifted)
        prev = best + emission[i]
        dp[i] = prev
    return dp


def _dtw_gumbel_traceback(dp, move_cost, states, seed):
    """One Gumbel-perturbed backward path through the DTW DP; returns chosen residuals.

    Perturb-and-sample: at each backward step add Gumbel noise
    ``−T·log(−log(U))`` to the candidate transition costs and take the argmin,
    yielding a stochastic path that concentrates near the MAP path at temperature T.
    Deterministic for a given ``seed``.
    """
    n_rows, n_states = dp.shape
    rng = np.random.RandomState(int(seed) & 0x7FFFFFFF)
    deltas = np.arange(-DTW_MAX_DELTA, DTW_MAX_DELTA + 1)
    idx = np.arange(n_states)
    # Start state: argmin of final accumulated cost + Gumbel.
    u0 = rng.random_sample(n_states)
    g0 = -DTW_TEMPERATURE * np.log(-np.log(np.clip(u0, 1e-12, 1.0)))
    s = int(np.argmin(dp[n_rows - 1] + g0))
    path = np.empty(n_rows, dtype=np.int64)
    path[n_rows - 1] = s
    for i in range(n_rows - 1, 0, -1):
        # Candidate predecessors a = s - delta with transition cost move_cost*|delta|.
        u = rng.random_sample(len(deltas))
        gum = -DTW_TEMPERATURE * np.log(-np.log(np.clip(u, 1e-12, 1.0)))
        best_key = np.inf
        best_a = s
        for j, delta in enumerate(deltas):
            a = s - int(delta)
            if a < 0 or a >= n_states:
                continue
            key = dp[i - 1, a] + move_cost * abs(int(delta)) + gum[j]
            if key < best_key:
                best_key = key
                best_a = a
        s = best_a
        path[i - 1] = s
    return states[path[:n_rows]]


def predict_dtw_well(horizontal_rows, typewell_rows):
    """Full-well constrained+stochastic DTW TVT candidate: heel passes through, toe warped.

    Returns a full-well trajectory the same length as ``horizontal_rows`` for the
    gold candidate pool. Known ``TVT_input`` rows are copied verbatim; the hidden
    toe is estimated by band-constrained DTW ensembled over ``DTW_RADII`` with
    ``DTW_K`` Gumbel-perturbed tracebacks per radius, then shrunk toward the drift
    baseline by the per-row uncertainty (std). Degrades to the finite drift baseline
    on the toe when the typewell/GR signal is unusable, and to the raw known array
    when it cannot anchor at all.
    """
    md = _dtw_column(horizontal_rows, "MD")
    z = _dtw_column(horizontal_rows, "Z")
    gr = _dtw_column(horizontal_rows, "GR")
    known_tvt = _dtw_column(horizontal_rows, "TVT_input")
    out = known_tvt.copy()
    known = np.isfinite(known_tvt)
    ev = np.flatnonzero(~known)
    if not known.any() or ev.size == 0 or not np.isfinite(md).all():
        return out
    u_base = _dtw_drift_baseline(md, z, known_tvt)
    baseline_tvt = u_base - z            # finite fallback for the toe
    tw = _dtw_typewell(typewell_rows)
    if tw is None:
        out[ev] = baseline_tvt[ev]
        return out
    tw_tvt, tw_gr = tw
    # GR emission scale from the known-zone residual (mirrors the particle filter's gs).
    kmask = known & np.isfinite(gr)
    if int(kmask.sum()) >= 3:
        tw_at_known = np.interp(known_tvt[kmask], tw_tvt, tw_gr)
        gs = float(np.clip(np.nanstd(gr[kmask] - tw_at_known), 10.0, 60.0))
    else:
        gs = 30.0
    base_ev = baseline_tvt[ev]
    gr_ev = gr[ev][:, None]
    realisations = []
    for r_idx, radius in enumerate(DTW_RADII):
        du = max(0.5, float(radius) / 100.0)
        k = int(round(float(radius) / du))
        states = np.arange(-k, k + 1, dtype=float) * du
        # Emission over eval rows x band states.
        tvt_grid = base_ev[:, None] + states[None, :]
        expected = np.interp(tvt_grid.ravel(), tw_tvt, tw_gr).reshape(tvt_grid.shape)
        resid = (gr_ev - expected) / gs
        emission = (resid * resid) / DTW_EMIT_SCALE
        emission = np.where(np.isfinite(emission), emission, 0.0)
        try:
            dp = _dtw_forward(emission, DTW_MOVE_COST)
        except Exception:
            continue
        for kk in range(DTW_K):
            seed = DTW_SEED + 1000 * (r_idx + 1) + kk
            try:
                chosen = _dtw_gumbel_traceback(dp, DTW_MOVE_COST, states, seed)
            except Exception:
                continue
            realisations.append(base_ev + chosen)
    if not realisations:
        out[ev] = base_ev
        return out
    stack = np.stack(realisations, axis=0)
    mean = np.mean(stack, axis=0)
    std = np.std(stack, axis=0)
    weight = 1.0 / (1.0 + (std / DTW_STD_SCALE) ** 2)   # low std => trust DTW, high std => baseline
    est = base_ev + weight * (mean - base_ev)
    est = np.where(np.isfinite(est), est, base_ev)
    out[ev] = est
    return out
# === END DTW_SHARED_CODE ===
