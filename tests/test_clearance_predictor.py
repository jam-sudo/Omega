"""Tests for Phase 783 — clearance_predictor.py"""

import pytest

from omega_pbpk.prediction.clearance_predictor import (
    ClearancePredictionResult,
    predict_clearance,
    screen_clearance,
)

# ---------------------------------------------------------------------------
# Basic return type and structure
# ---------------------------------------------------------------------------


def test_returns_correct_type():
    result = predict_clearance("DrugA", logp=2.0, mw_da=300.0, fup=0.1)
    assert isinstance(result, ClearancePredictionResult)


def test_result_fields_present():
    result = predict_clearance("DrugA", logp=2.0, mw_da=300.0, fup=0.1)
    assert hasattr(result, "compound_name")
    assert hasattr(result, "predicted_cl_total_L_per_h")
    assert hasattr(result, "predicted_t_half_h")
    assert hasattr(result, "predicted_vd_L")
    assert hasattr(result, "cl_class")
    assert hasattr(result, "elimination_route")
    assert hasattr(result, "notes")


# ---------------------------------------------------------------------------
# Positivity checks
# ---------------------------------------------------------------------------


def test_cl_total_positive():
    result = predict_clearance("DrugA", logp=2.0, mw_da=300.0, fup=0.1)
    assert result.predicted_cl_total_L_per_h > 0.0


def test_t_half_positive():
    result = predict_clearance("DrugA", logp=2.0, mw_da=300.0, fup=0.1)
    assert result.predicted_t_half_h > 0.0


def test_vd_positive():
    result = predict_clearance("DrugA", logp=2.0, mw_da=300.0, fup=0.1)
    assert result.predicted_vd_L > 0.0


def test_cl_hepatic_non_negative():
    result = predict_clearance("DrugA", logp=2.0, mw_da=300.0, fup=0.1)
    assert result.predicted_cl_hepatic_L_per_h >= 0.0


def test_cl_renal_non_negative():
    result = predict_clearance("DrugA", logp=2.0, mw_da=300.0, fup=0.1)
    assert result.predicted_cl_renal_L_per_h >= 0.0


# ---------------------------------------------------------------------------
# CLint effect on hepatic CL
# ---------------------------------------------------------------------------


def test_higher_cl_int_gives_higher_hepatic_cl():
    low_cl_int = predict_clearance(
        "DrugA", logp=2.0, mw_da=300.0, fup=0.3, cl_int_scaled_L_per_h=5.0
    )
    high_cl_int = predict_clearance(
        "DrugA", logp=2.0, mw_da=300.0, fup=0.3, cl_int_scaled_L_per_h=50.0
    )
    assert high_cl_int.predicted_cl_hepatic_L_per_h > low_cl_int.predicted_cl_hepatic_L_per_h


def test_cl_int_well_stirred_bounded_by_qh():
    """CLhep from well-stirred model should never exceed Qh=97 L/h."""
    result = predict_clearance(
        "DrugA", logp=2.0, mw_da=300.0, fup=1.0, cl_int_scaled_L_per_h=1000.0
    )
    assert result.predicted_cl_hepatic_L_per_h < 97.0


# ---------------------------------------------------------------------------
# Renal clearance: low logP → higher renal fraction
# ---------------------------------------------------------------------------


def test_low_logp_higher_renal_fraction():
    """Low logP compound should have a higher renal CL fraction than high logP."""
    low_lp = predict_clearance("LowLogP", logp=-1.0, mw_da=200.0, fup=1.0)
    high_lp = predict_clearance("HighLogP", logp=5.0, mw_da=200.0, fup=1.0)
    low_ren_frac = low_lp.predicted_cl_renal_L_per_h / low_lp.predicted_cl_total_L_per_h
    high_ren_frac = high_lp.predicted_cl_renal_L_per_h / high_lp.predicted_cl_total_L_per_h
    assert low_ren_frac > high_ren_frac


def test_very_high_logp_renal_cl_near_zero():
    """Very high logP → renal factor near zero → minimal renal CL."""
    result = predict_clearance("LipophilicDrug", logp=25.0, mw_da=400.0, fup=0.1)
    assert result.predicted_cl_renal_L_per_h == 0.0


# ---------------------------------------------------------------------------
# cl_class values
# ---------------------------------------------------------------------------


def test_cl_class_in_valid_set():
    result = predict_clearance("DrugA", logp=2.0, mw_da=300.0, fup=0.1)
    assert result.cl_class in {"low", "moderate", "high"}


def test_cl_class_low_for_small_cl():
    """Small fup and high MW → low CL."""
    result = predict_clearance("LowCL", logp=0.0, mw_da=500.0, fup=0.01)
    assert result.cl_class == "low"


def test_cl_class_high_for_large_cl():
    """High fup, moderate MW, high CLint → high CL."""
    result = predict_clearance(
        "HighCL", logp=1.0, mw_da=200.0, fup=1.0, cl_int_scaled_L_per_h=500.0
    )
    assert result.cl_class == "high"


# ---------------------------------------------------------------------------
# elimination_route
# ---------------------------------------------------------------------------


def test_elimination_route_valid():
    result = predict_clearance("DrugA", logp=2.0, mw_da=300.0, fup=0.1)
    assert result.elimination_route in {"hepatic", "renal", "mixed"}


def test_elimination_route_hepatic_for_high_logp():
    """Very high logP (>=20) makes renal factor=0 → hepatic route."""
    # renal_factor = max(0, 1 - 0.05*20) = 0, so cl_ren = 0 → hepatic
    result = predict_clearance("FatSoluble", logp=20.0, mw_da=350.0, fup=0.5)
    assert result.elimination_route == "hepatic"


# ---------------------------------------------------------------------------
# Vd bounds
# ---------------------------------------------------------------------------


def test_vd_clamped_minimum():
    """Even for tiny MW and negative logP, Vd >= 3."""
    result = predict_clearance("Tiny", logp=-5.0, mw_da=10.0, fup=0.5)
    assert result.predicted_vd_L >= 3.0


def test_vd_clamped_maximum():
    """Very high MW and logP should not exceed 500 L."""
    result = predict_clearance("Giant", logp=10.0, mw_da=5000.0, fup=0.5)
    assert result.predicted_vd_L <= 500.0


# ---------------------------------------------------------------------------
# Input validation errors
# ---------------------------------------------------------------------------


def test_invalid_fup_zero_raises():
    with pytest.raises(ValueError):
        predict_clearance("X", logp=2.0, mw_da=300.0, fup=0.0)


def test_invalid_fup_negative_raises():
    with pytest.raises(ValueError):
        predict_clearance("X", logp=2.0, mw_da=300.0, fup=-0.1)


def test_invalid_fup_above_one_raises():
    with pytest.raises(ValueError):
        predict_clearance("X", logp=2.0, mw_da=300.0, fup=1.5)


def test_invalid_mw_zero_raises():
    with pytest.raises(ValueError):
        predict_clearance("X", logp=2.0, mw_da=0.0, fup=0.5)


def test_invalid_mw_negative_raises():
    with pytest.raises(ValueError):
        predict_clearance("X", logp=2.0, mw_da=-100.0, fup=0.5)


# ---------------------------------------------------------------------------
# screen_clearance
# ---------------------------------------------------------------------------


def test_screen_returns_list():
    compounds = [
        {"compound_name": "A", "logp": 2.0, "mw_da": 300.0, "fup": 0.1},
        {"compound_name": "B", "logp": 1.0, "mw_da": 200.0, "fup": 0.5},
    ]
    results = screen_clearance(compounds)
    assert isinstance(results, list)
    assert len(results) == 2


def test_screen_sorted_ascending_cl():
    compounds = [
        {
            "compound_name": "HighCL",
            "logp": 2.0,
            "mw_da": 200.0,
            "fup": 1.0,
            "cl_int_scaled_L_per_h": 500.0,
        },
        {"compound_name": "LowCL", "logp": 0.0, "mw_da": 500.0, "fup": 0.01},
    ]
    results = screen_clearance(compounds)
    # First result should have lowest CL
    assert results[0].predicted_cl_total_L_per_h <= results[1].predicted_cl_total_L_per_h


def test_screen_returns_correct_type_each_element():
    compounds = [
        {"compound_name": "A", "logp": 2.0, "mw_da": 300.0, "fup": 0.1},
    ]
    results = screen_clearance(compounds)
    assert isinstance(results[0], ClearancePredictionResult)
