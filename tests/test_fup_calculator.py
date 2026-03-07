"""Tests for prediction/fup_calculator.py (Phase 741)."""

import pytest

from omega_pbpk.prediction.fup_calculator import (
    FupCalculatorResult,
    calculate_fup,
    screen_fup,
)

# ---------------------------------------------------------------------------
# Basic return type and field validation
# ---------------------------------------------------------------------------


def test_returns_fup_calculator_result():
    result = calculate_fup("TestDrug")
    assert isinstance(result, FupCalculatorResult)


def test_fup_human_in_valid_range():
    result = calculate_fup("TestDrug", logp=2.0, mw_da=300.0, psa=60.0)
    assert 0.001 <= result.fup_human <= 0.999


def test_fup_rat_in_valid_range():
    result = calculate_fup("TestDrug", logp=2.0, mw_da=300.0, psa=60.0)
    assert 0.001 <= result.fup_rat <= 0.999


def test_ppb_human_pct_formula():
    result = calculate_fup("TestDrug", logp=2.0, mw_da=300.0, psa=60.0)
    expected_ppb = 100.0 * (1.0 - result.fup_human)
    assert abs(result.ppb_human_pct - expected_ppb) < 1e-9


def test_compound_name_preserved():
    result = calculate_fup("Warfarin")
    assert result.compound_name == "Warfarin"


def test_notes_nonempty():
    result = calculate_fup("TestDrug")
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# logP effect: high logP → lower fup (more protein bound)
# ---------------------------------------------------------------------------


def test_high_logp_lower_fup():
    low_logp = calculate_fup("Low", logp=0.0, mw_da=300.0, psa=60.0)
    high_logp = calculate_fup("High", logp=5.0, mw_da=300.0, psa=60.0)
    assert high_logp.fup_human < low_logp.fup_human


def test_very_high_logp_low_fup():
    result = calculate_fup("Lipophilic", logp=8.0, mw_da=400.0, psa=30.0)
    assert result.fup_human < 0.1


def test_low_logp_higher_fup():
    result = calculate_fup("Hydrophilic", logp=-2.0, mw_da=200.0, psa=120.0)
    assert result.fup_human > 0.3


# ---------------------------------------------------------------------------
# pKa effects
# ---------------------------------------------------------------------------


def test_strong_acid_lower_fup():
    neutral = calculate_fup("Neutral", logp=2.0, mw_da=300.0, psa=60.0)
    acid = calculate_fup("StrongAcid", logp=2.0, mw_da=300.0, psa=60.0, pka_acid=3.0)
    assert acid.fup_human < neutral.fup_human


def test_basic_drug_higher_fup():
    neutral = calculate_fup("Neutral", logp=2.0, mw_da=300.0, psa=60.0)
    basic = calculate_fup("Basic", logp=2.0, mw_da=300.0, psa=60.0, pka_basic=9.0)
    assert basic.fup_human > neutral.fup_human


def test_weak_acid_no_albumin_adjustment():
    """pKa_acid >= 5 should not trigger the adjustment."""
    neutral = calculate_fup("Neutral", logp=2.0, mw_da=300.0, psa=60.0)
    weak_acid = calculate_fup("WeakAcid", logp=2.0, mw_da=300.0, psa=60.0, pka_acid=6.0)
    assert abs(neutral.fup_human - weak_acid.fup_human) < 1e-9


def test_weak_base_no_adjustment():
    """pKa_basic <= 7.4 should not trigger the basic adjustment."""
    neutral = calculate_fup("Neutral", logp=2.0, mw_da=300.0, psa=60.0)
    weak_base = calculate_fup("WeakBase", logp=2.0, mw_da=300.0, psa=60.0, pka_basic=7.0)
    assert abs(neutral.fup_human - weak_base.fup_human) < 1e-9


# ---------------------------------------------------------------------------
# Binding category
# ---------------------------------------------------------------------------


def test_binding_category_low():
    result = calculate_fup("HydroLow", logp=-3.0, mw_da=150.0, psa=150.0)
    assert result.binding_category == "low"
    assert result.fup_human > 0.5


def test_binding_category_valid_values():
    result = calculate_fup("TestDrug", logp=2.0, mw_da=300.0, psa=60.0)
    assert result.binding_category in ("low", "moderate", "high")


def test_high_binding_category():
    result = calculate_fup("HighBind", logp=6.0, mw_da=500.0, psa=20.0)
    assert result.binding_category == "high"
    assert result.fup_human < 0.1


def test_cl_blood_correction_field():
    result = calculate_fup("TestDrug")
    assert result.cl_blood_correction in ("minimal", "moderate", "significant")


# ---------------------------------------------------------------------------
# screen_fup
# ---------------------------------------------------------------------------


def test_screen_fup_returns_list():
    compounds = [
        {"compound_name": "A", "logp": 1.0, "mw_da": 300.0, "psa": 60.0},
        {"compound_name": "B", "logp": 3.0, "mw_da": 300.0, "psa": 60.0},
        {"compound_name": "C", "logp": 0.5, "mw_da": 250.0, "psa": 80.0},
    ]
    results = screen_fup(compounds)
    assert isinstance(results, list)
    assert len(results) == 3
    assert all(isinstance(r, FupCalculatorResult) for r in results)


def test_screen_fup_sorted_descending():
    compounds = [
        {"compound_name": "HighBind", "logp": 6.0, "mw_da": 400.0, "psa": 20.0},
        {"compound_name": "LowBind", "logp": 0.0, "mw_da": 200.0, "psa": 120.0},
        {"compound_name": "Mid", "logp": 2.0, "mw_da": 300.0, "psa": 60.0},
    ]
    results = screen_fup(compounds)
    fups = [r.fup_human for r in results]
    assert fups == sorted(fups, reverse=True)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validation_mw_zero_raises():
    with pytest.raises(ValueError):
        calculate_fup("X", mw_da=0.0)


def test_validation_mw_negative_raises():
    with pytest.raises(ValueError):
        calculate_fup("X", mw_da=-100.0)


def test_validation_psa_negative_raises():
    with pytest.raises(ValueError):
        calculate_fup("X", psa=-1.0)
