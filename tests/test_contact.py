from __future__ import annotations

import math

from src.contact import (
    MIN_KNOWN_PREFIX_ROWS,
    MIN_VALID_PHYS_ROWS,
    best_contact_curve,
    fit_contact_curve,
)

REF_TVT = 11200.0


def _train_typewell() -> list[dict[str, str]]:
    rows = [{"TVT": "11100.0", "GR": "80.0", "Geology": ""}]
    rows.append({"TVT": str(REF_TVT), "GR": "90.0", "Geology": "EGFDU"})
    rows.append({"TVT": str(REF_TVT + 5.0), "GR": "95.0", "Geology": "EGFDU"})
    return rows


def _train_horizontal(n: int = 200) -> list[dict[str, str]]:
    """Synthetic well whose contact reconstruction is exact: TVT = 11000 + MD/100."""
    rows = []
    for i in range(n):
        md = 1000.0 + 10.0 * i
        tvt = 11000.0 + md / 100.0
        z = -9000.0 - 0.5 * i
        # formation depth column such that ref_tvt - (Z - col) == tvt - 3.0
        col = tvt - 3.0 - REF_TVT + z
        rows.append(
            {
                "MD": str(md),
                "Z": str(z),
                "EGFDU": str(col),
                "TVT": str(tvt),
            }
        )
    return rows


def _test_horizontal(n: int = 200, known: int = 100) -> list[dict[str, str]]:
    rows = []
    for i in range(n):
        md = 1000.0 + 10.0 * i
        tvt = 11000.0 + md / 100.0
        rows.append(
            {
                "MD": str(md),
                "Z": str(-9000.0 - 0.5 * i),
                "TVT_input": str(tvt) if i < known else "",
            }
        )
    return rows


def test_fit_contact_curve_recovers_absolute_tvt() -> None:
    curve = fit_contact_curve(_train_horizontal(), _train_typewell(), "EGFDU")
    assert curve is not None
    mds, tvts = curve
    assert len(mds) == 200
    for md, tvt in zip(mds, tvts):
        assert math.isclose(tvt, 11000.0 + md / 100.0, abs_tol=1e-9)


def test_fit_contact_curve_requires_geology_and_rows() -> None:
    typewell_no_ref = [{"TVT": "11100.0", "GR": "80.0", "Geology": "OTHER"}]
    assert fit_contact_curve(_train_horizontal(), typewell_no_ref, "EGFDU") is None
    short = _train_horizontal(MIN_VALID_PHYS_ROWS - 1)
    assert fit_contact_curve(short, _train_typewell(), "EGFDU") is None


def test_best_contact_curve_promotes_within_prefix_gate() -> None:
    curve = best_contact_curve(
        _test_horizontal(), _train_horizontal(), _train_typewell()
    )
    assert curve is not None
    assert curve.ref_col == "EGFDU"
    assert curve.prefix_rmse < 1e-6
    # toe rows (beyond the known prefix) are reconstructed, not extrapolated
    assert math.isclose(curve.predict(2990.0), 11000.0 + 29.9, abs_tol=1e-9)
    assert curve.covers(2990.0)
    assert not curve.covers(99999.0)


def test_best_contact_curve_rejects_prefix_mismatch() -> None:
    shifted = [
        {**row, "TVT_input": str(float(row["TVT_input"]) + 5.0) if row["TVT_input"] else ""}
        for row in _test_horizontal()
    ]
    assert best_contact_curve(shifted, _train_horizontal(), _train_typewell()) is None


def test_best_contact_curve_requires_known_prefix_rows() -> None:
    sparse = _test_horizontal(known=MIN_KNOWN_PREFIX_ROWS - 1)
    assert best_contact_curve(sparse, _train_horizontal(), _train_typewell()) is None
