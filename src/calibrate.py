"""Gold-calibration overlay — per-well visible-prefix self-verified anchor (SOT-2395).

Stage 3 (final) of the three-stage port of the reference kernel
``evgendvorkin/rogii-physics-lb-7-872-v48`` (public LB 7.872). Stage 1 (SOT-2393)
shipped the offline ML base predictor, stage 2 (SOT-2394) blended it with the
champion particle filter at the hidden-test fallback. This stage ports the
reference kernel's final **gold-calibration** layer (its ``ROGII_GOLD_*`` /
"visible-prefix calibration"): a per-well *self-verified anchor* that runs **after**
the blend and only makes a per-well move when the well's own visible prefix says a
different candidate beats the blended tracker.

Mechanism (faithful, portable, leak-free — mirrors ``_gold_calibrate_well`` /
``_gold_alpha`` / ``_gold_profile_output`` in the reference kernel):

1. Each well's **known heel** (visible ``TVT_input`` rows) is a self-verification
   set. Withhold the tail of the heel at several cut fractions, re-predict the
   withheld heel with a portable candidate pool ``{blend, pf, ml, poly}`` (the
   only-visible data is used, so no toe/target leaks), and score each candidate
   against the *known* withheld heel values.
2. Aggregate each candidate's score across cuts (``median + 0.10·std``), pick the
   best; the base/default is the stage-2 blend. Compute ``gain`` (blend − best),
   ``rank_margin`` (second − best) and ``consistency`` (fraction of cuts where the
   best beats the blend by ≥ 0.25).
3. A conservative gate (``GOLD_PROFILE``) turns the evidence into a soft weight
   ``alpha``; below the gain/consistency/best thresholds ``alpha = 0`` and the
   blend is kept unchanged. When it fires, the chosen candidate is blended into the
   base on the **hidden toe** with a ramped, clipped move
   ``base + clip(alpha·ramp·(candidate − base), ±max_move)``.

Because visible test wells are covered end-to-end by the guarded contact override,
this overlay never fires there — the local editor-run ``submission.csv`` stays
byte-identical; only the hidden-test fallback trajectory can change. The layer is
exec-compatible (numpy-only, no ``__file__``, no filesystem): the
``GOLD_SHARED_CODE`` block is copied verbatim into
``kaggle/rogii-claude-baseline.py`` (enforced by ``tests/test_calibrate.py``).
"""
from __future__ import annotations

import numpy as np

from src.align import predict_beam_well
from src.blend import BLEND_WEIGHT, blend_trajectories
from src.ml_predictor import predict_ml_well
from src.physics import predict_pf_well

# === BEGIN GOLD_SHARED_CODE (synced verbatim into kaggle/rogii-claude-baseline.py) ===
# Visible-prefix self-verified anchor (gold-calibration), ported from the reference
# kernel's ROGII_GOLD_* layer. All thresholds are frozen from the reference kernel's
# "conservative" profile (provenance = the public notebook, not tuned on our gate),
# so the overlay is a leak-free, self-limiting correction of the stage-2 blend.
GOLD_CUT_FRACS = (0.50, 0.65, 0.75)
GOLD_MIN_PREFIX = 140          # need a long enough known heel to self-verify
GOLD_MIN_HOLDOUT = 35          # rows withheld per cut for scoring
GOLD_MOVE_MARGIN = 0.25        # a cut "prefers" the best only if it beats blend by this
GOLD_PROFILE = {
    "min_gain": 1.00,          # best must beat the blend by >= 1.0 RMSE on the heel
    "max_best": 12.0,          # ...and be a trustworthy tracker (low absolute heel RMSE)
    "min_consistency": 0.67,   # ...on >= 2/3 of the cuts
    "min_margin": 0.0,         # best must lead the runner-up
    "base": 0.06,
    "gain_scale": 0.12,
    "margin_scale": 0.04,
    "quality_bonus": 0.02,
    "cap": 0.22,
    "clip_base": 8.0,
    "clip_gain": 3.0,
    "clip_max": 18.0,
    "delta_soft": 22.0,
    "p95_hard": 55.0,
}


def _gold_column(rows, key):
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


def _gold_rmse(pred, truth):
    pred = np.asarray(pred, dtype=float)
    truth = np.asarray(truth, dtype=float)
    mask = np.isfinite(pred) & np.isfinite(truth)
    if not mask.any():
        return float("nan")
    diff = pred[mask] - truth[mask]
    return float(np.sqrt(np.mean(diff * diff)))


def _gold_robust_poly_predict(x_known, y_known, x_all, deg):
    """IRLS robust-polynomial extrapolation of y over normalized x (reference port)."""
    x_known = np.asarray(x_known, dtype=float)
    y_known = np.asarray(y_known, dtype=float)
    x_all = np.asarray(x_all, dtype=float)
    m = np.isfinite(x_known) & np.isfinite(y_known)
    x_known = x_known[m]
    y_known = y_known[m]
    if len(x_known) < 3:
        fill = float(np.nanmedian(y_known)) if len(y_known) else 0.0
        return np.full_like(x_all, fill, dtype=float)
    deg = int(min(max(1, deg), len(x_known) - 1))
    x0 = float(x_known[0])
    xs = float(np.nanmax(x_known) - np.nanmin(x_known))
    if (not np.isfinite(xs)) or xs < 1e-6:
        xs = 1.0
    xk = (x_known - x0) / xs
    xa = (x_all - x0) / xs
    try:
        coef = np.polyfit(xk, y_known, deg)
        for _ in range(5):
            fit = np.polyval(coef, xk)
            res = y_known - fit
            sc = 1.4826 * float(np.nanmedian(np.abs(res - np.nanmedian(res)))) + 1e-6
            weights = 1.0 / (1.0 + (res / (2.5 * sc)) ** 2)
            coef = np.polyfit(xk, y_known, deg, w=weights)
        return np.polyval(coef, xa).astype(float)
    except Exception:
        return np.full_like(x_all, float(np.nanmedian(y_known)), dtype=float)


def _gold_candidate_pool(horizontal_rows, typewell_rows, weight=BLEND_WEIGHT):
    """Portable per-well candidate trajectories {blend, pf, ml, poly, beam}.

    ``blend`` (the stage-2 base) and its two singles come from the particle filter
    and the offline ML predictor; ``poly`` is a robust MD→TVT extrapolation from the
    known heel; ``beam`` is the beam-search + multi-scale NCC GR↔typewell alignment
    (``predict_beam_well``, SOT-2442). Known heel rows pass through unchanged for
    every candidate.
    """
    pf = predict_pf_well(horizontal_rows, typewell_rows)
    try:
        ml = predict_ml_well(horizontal_rows, typewell_rows)
    except Exception:
        ml = None
    ml_arr = pf if ml is None else np.asarray(ml, dtype=float)
    blend = blend_trajectories(pf, ml_arr, weight)
    md = _gold_column(horizontal_rows, "MD")
    ktvt = _gold_column(horizontal_rows, "TVT_input")
    known = np.isfinite(ktvt)
    poly = _gold_robust_poly_predict(md[known], ktvt[known], md, 2)
    poly = np.where(known, ktvt, poly)
    try:
        beam = np.asarray(predict_beam_well(horizontal_rows, typewell_rows), dtype=float)
    except Exception:
        beam = poly
    beam = np.where(np.isfinite(beam), beam, poly)
    return {
        "blend": np.asarray(blend, dtype=float),
        "pf": np.asarray(pf, dtype=float),
        "ml": ml_arr,
        "poly": poly,
        "beam": beam,
    }


def _gold_backtest_report(horizontal_rows, typewell_rows, weight=BLEND_WEIGHT):
    """Score the candidate pool on withheld tails of the well's own visible heel."""
    ktvt = _gold_column(horizontal_rows, "TVT_input")
    is_known = np.isfinite(ktvt)
    is_hidden = ~is_known
    if not bool(is_hidden.any()):
        return {"status": "no_hidden"}
    first_hidden = int(np.flatnonzero(is_hidden)[0])
    known_prefix = np.flatnonzero(is_known & (np.arange(len(ktvt)) < first_hidden))
    if len(known_prefix) < GOLD_MIN_PREFIX:
        return {"status": "skip_short_prefix"}
    cuts = []
    for frac in GOLD_CUT_FRACS:
        cut_pos = int(round(len(known_prefix) * float(frac)))
        cut_pos = max(50, min(cut_pos, len(known_prefix) - GOLD_MIN_HOLDOUT))
        if cut_pos <= 0 or cut_pos >= len(known_prefix):
            continue
        cutoff_idx = int(known_prefix[cut_pos - 1])
        hold_idx = known_prefix[cut_pos:]
        if len(hold_idx) >= GOLD_MIN_HOLDOUT:
            cuts.append((float(frac), cutoff_idx, hold_idx))
    if not cuts:
        return {"status": "skip_no_holdout"}
    scores = {}
    cut_rows = []
    for frac, cutoff_idx, hold_idx in cuts:
        masked = [
            dict(row, TVT_input=(row.get("TVT_input", "") if i <= cutoff_idx else ""))
            for i, row in enumerate(horizontal_rows)
        ]
        pool = _gold_candidate_pool(masked, typewell_rows, weight)
        y = ktvt[hold_idx]
        row = {"cut_frac": frac, "holdout_rows": int(len(hold_idx))}
        for name, pred in pool.items():
            err = _gold_rmse(np.asarray(pred)[hold_idx], y)
            if np.isfinite(err):
                scores.setdefault(name, []).append(err)
        row["blend_rmse"] = _gold_rmse(np.asarray(pool["blend"])[hold_idx], y)
        local = sorted(
            (_gold_rmse(np.asarray(p)[hold_idx], y), n) for n, p in pool.items()
        )
        row["best_name"] = local[0][1] if local else None
        row["best_rmse"] = float(local[0][0]) if local else float("nan")
        cut_rows.append(row)
    if not scores:
        return {"status": "skip_no_scores"}
    agg = {}
    for name, vals in scores.items():
        arr = np.asarray(vals, dtype=float)
        agg[name] = float(np.nanmedian(arr) + 0.10 * np.nanstd(arr))
    ordered = sorted((v, k) for k, v in agg.items() if np.isfinite(v))
    if not ordered:
        return {"status": "skip_nonfinite_scores"}
    best_score, best_name = ordered[0]
    second_score = ordered[1][0] if len(ordered) > 1 else best_score
    default_score = float(agg.get("blend", second_score))
    consistency = 0.0
    comparable = 0
    for row in cut_rows:
        if np.isfinite(row.get("blend_rmse", np.nan)):
            comparable += 1
            if row.get("best_rmse", float("inf")) <= row["blend_rmse"] - GOLD_MOVE_MARGIN:
                consistency += 1.0
    consistency = consistency / comparable if comparable else 0.0
    return {
        "status": "ok",
        "best_name": best_name,
        "best_score": float(best_score),
        "second_score": float(second_score),
        "default_score": float(default_score),
        "gain": float(default_score - best_score),
        "rank_margin": float(second_score - best_score),
        "consistency": float(consistency),
        "cuts": int(len(cut_rows)),
    }


def _gold_alpha(report, delta_rmse, delta_p95):
    """Turn the backtest evidence into a soft move weight (reference port)."""
    p = GOLD_PROFILE
    if report.get("status") != "ok":
        return 0.0
    gain = float(report.get("gain", 0.0))
    best = float(report.get("best_score", float("inf")))
    margin = float(report.get("rank_margin", 0.0))
    consistency = float(report.get("consistency", 0.0))
    if (
        (not np.isfinite(best))
        or best > p["max_best"]
        or gain < p["min_gain"]
        or consistency < p["min_consistency"]
        or margin < p["min_margin"]
    ):
        return 0.0
    alpha = p["base"]
    alpha += p["gain_scale"] * min(max(gain, 0.0), 5.0) / 5.0
    alpha += p["margin_scale"] * min(max(margin, 0.0), 3.0) / 3.0
    if best <= 5.0:
        alpha += p["quality_bonus"]
    if np.isfinite(delta_rmse) and delta_rmse > p["delta_soft"]:
        alpha *= max(0.20, p["delta_soft"] / max(delta_rmse, 1e-6))
    if np.isfinite(delta_p95) and delta_p95 > p["p95_hard"]:
        return 0.0
    return float(min(p["cap"], max(0.0, alpha * 1.75)))


def _gold_apply_move(base, candidate, report, is_hidden):
    """Soft, ramped, clipped blend of ``candidate`` into ``base`` on the hidden toe."""
    p = GOLD_PROFILE
    base = np.asarray(base, dtype=float).copy()
    candidate = np.asarray(candidate, dtype=float)
    idx = np.flatnonzero(np.asarray(is_hidden, dtype=bool))
    if idx.size == 0:
        return base
    cand = candidate[idx]
    cur = base[idx]
    ok = np.isfinite(cand) & np.isfinite(cur)
    if int(ok.sum()) != len(cur):
        return base
    diff = cand - cur
    delta_rmse = float(np.sqrt(np.mean(diff * diff))) if len(diff) else float("nan")
    delta_p95 = float(np.quantile(np.abs(diff), 0.95)) if len(diff) else float("nan")
    alpha = _gold_alpha(report, delta_rmse, delta_p95)
    if alpha <= 0.0:
        return base
    gain = max(0.0, float(report.get("gain", 0.0)))
    max_move = min(p["clip_max"], p["clip_base"] + p["clip_gain"] * np.sqrt(gain + 1e-9))
    ramp = 1.0 - np.exp(-np.arange(len(diff), dtype=float) / max(80.0, 0.12 * max(1, len(diff))))
    move = np.clip(alpha * ramp * diff, -max_move, max_move)
    base[idx] = cur + move
    return base


def gold_calibrate_trajectory(horizontal_rows, typewell_rows, weight=BLEND_WEIGHT):
    """Full-well gold-calibrated trajectory: stage-2 blend + per-well self-verified move.

    Returns the stage-2 ``blend_trajectories(PF, ML)`` unchanged unless the well's own
    visible-prefix backtest clears the conservative gate, in which case the best
    self-verified candidate is softly blended into the hidden toe. Known heel rows are
    untouched, so a well with no hidden rows returns the base blend.
    """
    pool = _gold_candidate_pool(horizontal_rows, typewell_rows, weight)
    base = pool["blend"]
    report = _gold_backtest_report(horizontal_rows, typewell_rows, weight)
    if report.get("status") != "ok":
        return base
    best_name = report.get("best_name")
    if best_name is None or best_name == "blend":
        return base
    ktvt = _gold_column(horizontal_rows, "TVT_input")
    is_hidden = ~np.isfinite(ktvt)
    return _gold_apply_move(base, pool[best_name], report, is_hidden)
# === END GOLD_SHARED_CODE ===
