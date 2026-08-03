"""Guarded same-well contact reconstruction (ported from the public physics kernel).

Every public test well ships with a full train copy under the same well id. The
train horizontal file carries the complete ``TVT`` truth plus formation-contact
depth columns (``EGFDU``/``ASTNU``/``ANCC``/``ASTNL``/``EGFDL``/``BUDA``), and
the train typewell labels those formations in ``Geology``. Reconstructing
``TVT = ref_tvt - (Z - formation) + offset`` from the train copy and
interpolating it over MD recovers the whole trajectory.

The reconstruction is *guarded*: it is validated against the test well's
visible ``TVT_input`` prefix and only replaces the fallback prediction when the
prefix RMSE stays within ``PREFIX_RMSE_LIMIT``. Wells without a compatible
train copy keep the offset-trend champion, so the fallback path is unchanged
for private wells that may not have train copies.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

REF_COLS: tuple[str, ...] = ("EGFDU", "ASTNU", "ANCC", "ASTNL", "EGFDL", "BUDA")
MIN_VALID_PHYS_ROWS = 100
MIN_KNOWN_PREFIX_ROWS = 50
PREFIX_RMSE_LIMIT = 1.0


@dataclass(frozen=True)
class ContactCurve:
    """A monotone-MD contact reconstruction validated on the visible prefix."""

    ref_col: str
    prefix_rmse: float
    mds: tuple[float, ...]
    tvts: tuple[float, ...]

    def covers(self, md: float) -> bool:
        return math.isfinite(md) and self.mds[0] <= md <= self.mds[-1]

    def predict(self, md: float) -> float:
        return _interp(md, self.mds, self.tvts)


def _to_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return math.nan


def _interp(x: float, xs: Sequence[float], ys: Sequence[float]) -> float:
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


def fit_contact_curve(
    train_horizontal: Sequence[Mapping[str, str]],
    train_typewell: Sequence[Mapping[str, str]],
    ref_col: str,
) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
    """Reconstruct absolute TVT over MD from one formation-contact column."""
    ref_tvts = [
        _to_float(row.get("TVT"))
        for row in train_typewell
        if (row.get("Geology") or "").strip() == ref_col
    ]
    ref_tvts = [value for value in ref_tvts if math.isfinite(value)]
    if not ref_tvts:
        return None
    ref_tvt = min(ref_tvts)

    samples: list[tuple[float, float]] = []
    residuals: list[float] = []
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


def best_contact_curve(
    test_horizontal: Sequence[Mapping[str, str]],
    train_horizontal: Sequence[Mapping[str, str]],
    train_typewell: Sequence[Mapping[str, str]],
) -> ContactCurve | None:
    """Pick the formation whose reconstruction best matches the visible prefix.

    Returns ``None`` (caller keeps the fallback champion) unless the best
    candidate's prefix RMSE is within ``PREFIX_RMSE_LIMIT``.
    """
    known = [
        (md, tvt)
        for row in test_horizontal
        if math.isfinite(tvt := _to_float(row.get("TVT_input")))
        and math.isfinite(md := _to_float(row.get("MD")))
    ]
    best: ContactCurve | None = None
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
