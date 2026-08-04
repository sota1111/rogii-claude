"""Physics × ML gated blend at the hidden-test fallback (SOT-2394).

Stage 2 of the three-stage port of the reference kernel
``evgendvorkin/rogii-physics-lb-7-872-v48``. Stage 1 (SOT-2393) shipped the
offline ML base predictor (``src.ml_predictor``) as a foundation that, on its
own, does not beat the champion likelihood-weighted particle filter
(``src.physics``). This stage **blends** the two at the hidden-test fallback
position — where no same-well contact override exists — reproducing the
reference kernel's physics × ML blend layer.

The blend is a single leak-free scalar ``weight`` (the particle filter's share):

    TVT_blend = weight * PF + (1 - weight) * ML

``weight`` is selected on a leak-free toe-holdout **disjoint** from the fold-0
confirm set (``scripts/select_blend_weight.py`` grid-searches it on fold-1 wells,
which the reported confirm gate never scores), then frozen as the ``BLEND_WEIGHT``
constant below before any test/confirm toe target is read.

Visible wells keep the guarded contact override untouched; only the hidden-test
fallback trajectory changes, so the local editor-run artifact stays byte-identical
(the contact override covers every local test row). The blend is exec-compatible
(numpy-only, no ``__file__``, no filesystem): the shared ``BLEND_SHARED_CODE``
block is copied verbatim into ``kaggle/rogii-claude-baseline.py`` so the two entry
points stay byte-identical (enforced by ``tests/test_blend.py``).
"""
from __future__ import annotations

import numpy as np

# === BEGIN BLEND_SHARED_CODE (synced verbatim into kaggle/rogii-claude-baseline.py) ===
# Particle-filter share of the fallback blend ``weight*PF + (1-weight)*ML``.
# Selected on a leak-free fold-1 toe-holdout subset (disjoint from the fold-0
# confirm set) by ``scripts/select_blend_weight.py`` and frozen here; 1.0 would
# recover the pure particle-filter champion. See docs/champion-selection.md.
BLEND_WEIGHT = 0.75


def blend_trajectories(pf, ml, weight=BLEND_WEIGHT):
    """Blend two full-well TVT trajectories elementwise.

    Returns ``weight * pf + (1 - weight) * ml`` where both are finite; where only
    one is finite it degrades to that one (particle filter preferred), so a
    missing/failed ML predictor recovers the pure-PF champion behaviour.
    """
    pf = np.asarray(pf, dtype=float)
    ml = np.asarray(ml, dtype=float)
    both = np.isfinite(pf) & np.isfinite(ml)
    out = np.where(np.isfinite(pf), pf, ml)
    return np.where(both, weight * pf + (1.0 - weight) * ml, out)
# === END BLEND_SHARED_CODE ===


def predict_blend_well(
    horizontal_rows,
    typewell_rows,
    *,
    weight=BLEND_WEIGHT,
    pf=None,
    ml=None,
):
    """Full-well blended TVT: known heel rows pass through, toe rows are blended.

    ``pf`` / ``ml`` may be precomputed full-well trajectories (avoids recomputing
    the expensive particle filter); otherwise they are computed here. Known heel
    rows are identical in both inputs, so blending them is a no-op.
    """
    if pf is None:
        from src.physics import predict_pf_well

        pf = predict_pf_well(horizontal_rows, typewell_rows)
    if ml is None:
        from src.ml_predictor import predict_ml_well

        ml = predict_ml_well(horizontal_rows, typewell_rows)
    return blend_trajectories(pf, ml, weight)
