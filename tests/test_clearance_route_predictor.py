"""
Tests for Phase 434 — clearance_route_predictor.py
~22 tests
"""

import pytest

from omega_pbpk.prediction.clearance_route_predictor import (
    ClearanceRouteResult,
    predict_clearance_route,
    screen_clearance_routes,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def default_predict(
    drug_name="TestDrug",
    mw_Da=300.0,
    logp=2.0,
    pka=7.4,
    ionization_type="neutral",
    fu_plasma=0.2,
):
    return predict_clearance_route(
        drug_name=drug_name,
        mw_Da=mw_Da,
        logp=logp,
        pka=pka,
        ionization_type=ionization_type,
        fu_plasma=fu_plasma,
    )


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type():
    r = default_predict()
    assert isinstance(r, ClearanceRouteResult)


def test_drug_name_preserved():
    r = default_predict(drug_name="Metformin")
    assert r.drug_name == "Metformin"


# ---------------------------------------------------------------------------
# Fraction properties
# ---------------------------------------------------------------------------


def test_all_fractions_sum_to_one():
    r = default_predict()
    total = (
        r.hepatic_cl_fraction + r.renal_cl_fraction + r.biliary_cl_fraction + r.other_cl_fraction
    )
    assert total == pytest.approx(1.0, abs=0.01)


def test_hepatic_fraction_in_0_1():
    r = default_predict()
    assert 0.0 <= r.hepatic_cl_fraction <= 1.0


def test_renal_fraction_in_0_1():
    r = default_predict()
    assert 0.0 <= r.renal_cl_fraction <= 1.0


def test_biliary_fraction_in_0_1():
    r = default_predict()
    assert 0.0 <= r.biliary_cl_fraction <= 1.0


def test_other_fraction_in_0_1():
    r = default_predict()
    assert 0.0 <= r.other_cl_fraction <= 1.0


# ---------------------------------------------------------------------------
# Physical relationships
# ---------------------------------------------------------------------------


def test_low_logp_high_renal_fraction():
    """Low logP (< 1) neutral drug should have high renal fraction."""
    r = predict_clearance_route(
        drug_name="LowLogP",
        mw_Da=200.0,
        logp=0.0,
        pka=7.0,
        ionization_type="neutral",
        fu_plasma=0.5,
    )
    assert r.renal_cl_fraction > 0.4


def test_high_logp_high_hepatic_fraction():
    """High logP (> 2) drug should have high hepatic fraction."""
    r = predict_clearance_route(
        drug_name="HighLogP",
        mw_Da=350.0,
        logp=4.0,
        pka=9.0,
        ionization_type="neutral",
        fu_plasma=0.1,
    )
    assert r.hepatic_cl_fraction > 0.4


def test_high_mw_low_renal_fraction():
    """High MW (> 500) drug is not renally filtered."""
    r = predict_clearance_route(
        drug_name="LargeMol",
        mw_Da=650.0,
        logp=0.5,
        pka=7.0,
        ionization_type="neutral",
        fu_plasma=0.3,
    )
    assert r.renal_cl_fraction <= 0.1


def test_primary_route_is_highest_fraction():
    r = default_predict()
    fractions = {
        "hepatic": r.hepatic_cl_fraction,
        "renal": r.renal_cl_fraction,
        "biliary": r.biliary_cl_fraction,
        "other": r.other_cl_fraction,
    }
    best = max(fractions, key=lambda k: fractions[k])
    assert r.primary_route == best


# ---------------------------------------------------------------------------
# Clinical flags
# ---------------------------------------------------------------------------


def test_renal_sensitivity_high_when_fraction_above_0p5():
    r = predict_clearance_route(
        drug_name="RenalDrug",
        mw_Da=200.0,
        logp=0.0,
        pka=7.0,
        ionization_type="neutral",
        fu_plasma=0.9,
    )
    if r.renal_cl_fraction > 0.5:
        assert r.renal_sensitivity == "high"


def test_renal_sensitivity_low_when_fraction_below_0p5():
    r = predict_clearance_route(
        drug_name="HepaticDrug",
        mw_Da=350.0,
        logp=4.0,
        pka=9.0,
        ionization_type="neutral",
        fu_plasma=0.1,
    )
    if r.renal_cl_fraction <= 0.5:
        assert r.renal_sensitivity == "low"


def test_ckd_adjustment_required_when_renal_high():
    r = predict_clearance_route(
        drug_name="RenalDrug",
        mw_Da=200.0,
        logp=0.0,
        pka=7.0,
        ionization_type="neutral",
        fu_plasma=0.9,
    )
    if r.renal_cl_fraction > 0.3:
        assert r.dose_adjustment_ckd == "required"


def test_predicted_total_cl_positive():
    r = default_predict()
    assert r.predicted_total_cl_L_per_h > 0.0


def test_hepatic_cl_fraction_of_total():
    r = default_predict()
    expected = r.predicted_total_cl_L_per_h * r.hepatic_cl_fraction
    assert r.predicted_hepatic_cl_L_per_h == pytest.approx(expected, rel=1e-6)


def test_renal_cl_fraction_of_total():
    r = default_predict()
    expected = r.predicted_total_cl_L_per_h * r.renal_cl_fraction
    assert r.predicted_renal_cl_L_per_h == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# screen_clearance_routes
# ---------------------------------------------------------------------------


def test_screen_returns_correct_length():
    compounds = [
        {
            "drug_name": "A",
            "mw_Da": 200.0,
            "logp": 0.5,
            "pka": 7.0,
            "ionization_type": "neutral",
            "fu_plasma": 0.5,
        },
        {
            "drug_name": "B",
            "mw_Da": 350.0,
            "logp": 3.0,
            "pka": 8.0,
            "ionization_type": "base",
            "fu_plasma": 0.1,
        },
        {
            "drug_name": "C",
            "mw_Da": 600.0,
            "logp": 1.5,
            "pka": 5.0,
            "ionization_type": "acid",
            "fu_plasma": 0.3,
        },
    ]
    results = screen_clearance_routes(compounds)
    assert len(results) == 3


def test_screen_sorted_descending():
    compounds = [
        {
            "drug_name": "A",
            "mw_Da": 200.0,
            "logp": 0.5,
            "pka": 7.0,
            "ionization_type": "neutral",
            "fu_plasma": 0.5,
        },
        {
            "drug_name": "B",
            "mw_Da": 350.0,
            "logp": 3.0,
            "pka": 8.0,
            "ionization_type": "base",
            "fu_plasma": 0.9,
        },
    ]
    results = screen_clearance_routes(compounds)
    assert results[0].predicted_total_cl_L_per_h >= results[1].predicted_total_cl_L_per_h


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_mw_zero():
    with pytest.raises(ValueError, match="mw_Da"):
        predict_clearance_route("Drug", 0.0, 2.0, 7.0, "neutral", 0.2)


def test_validation_mw_negative():
    with pytest.raises(ValueError, match="mw_Da"):
        predict_clearance_route("Drug", -100.0, 2.0, 7.0, "neutral", 0.2)


def test_validation_fu_plasma_zero():
    with pytest.raises(ValueError, match="fu_plasma"):
        predict_clearance_route("Drug", 300.0, 2.0, 7.0, "neutral", 0.0)


def test_validation_fu_plasma_above_one():
    with pytest.raises(ValueError, match="fu_plasma"):
        predict_clearance_route("Drug", 300.0, 2.0, 7.0, "neutral", 1.5)


def test_validation_bad_ionization_type():
    with pytest.raises(ValueError, match="ionization_type"):
        predict_clearance_route("Drug", 300.0, 2.0, 7.0, "zwitterion", 0.2)
