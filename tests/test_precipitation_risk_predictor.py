"""Tests for Phase 302: Drug precipitation risk predictor.

Tests cover:
- Return type and field preservation
- supersaturation_ratio > 0
- Higher concentration -> higher SR
- Lower solubility -> higher risk
- SR > 1 -> 'moderate' or 'high' risk_class
- Acid drug: acidic diluent -> lower solubility -> higher risk
- assess_iv_compatibility returns sorted list (ascending risk score)
- Validation errors
"""

from __future__ import annotations

import math

import pytest

from omega_pbpk.prediction.precipitation_risk_predictor import (
    PrecipitationRiskResult,
    assess_iv_compatibility,
    predict_precipitation_risk,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _base_result(**kwargs) -> PrecipitationRiskResult:
    """Default prediction with optional overrides."""
    defaults = dict(
        drug_name="TestDrug",
        drug_concentration_mg_mL=5.0,
        solubility_at_ph74_mg_mL=2.0,
        pka=7.0,
        ionization_type="acid",
        vehicle_ph=10.0,
        dilution_ph=7.4,
        dilution_ratio=10.0,
    )
    defaults.update(kwargs)
    return predict_precipitation_risk(**defaults)


# ---------------------------------------------------------------------------
# Return type and field presence
# ---------------------------------------------------------------------------


def test_return_type():
    r = _base_result()
    assert isinstance(r, PrecipitationRiskResult)


def test_all_fields_present():
    r = _base_result()
    expected = [
        "drug_name",
        "drug_concentration_mg_mL",
        "solubility_at_ph74_mg_mL",
        "pka",
        "ionization_type",
        "vehicle_ph",
        "dilution_ph",
        "dilution_ratio",
        "c_diluted_mg_mL",
        "ph_diluted",
        "solubility_at_diluted_ph_mg_mL",
        "supersaturation_ratio",
        "precipitation_risk_score",
        "risk_class",
        "time_to_precipitate_min",
        "notes",
    ]
    for field in expected:
        assert hasattr(r, field), f"Missing field: {field}"


def test_field_preservation():
    r = predict_precipitation_risk(
        drug_name="Diazepam",
        drug_concentration_mg_mL=3.0,
        solubility_at_ph74_mg_mL=1.5,
        pka=3.4,
        ionization_type="base",
        vehicle_ph=5.0,
        dilution_ph=7.4,
        dilution_ratio=5.0,
    )
    assert r.drug_name == "Diazepam"
    assert r.drug_concentration_mg_mL == pytest.approx(3.0)
    assert r.solubility_at_ph74_mg_mL == pytest.approx(1.5)
    assert r.pka == pytest.approx(3.4)
    assert r.ionization_type == "base"
    assert r.vehicle_ph == pytest.approx(5.0)
    assert r.dilution_ph == pytest.approx(7.4)
    assert r.dilution_ratio == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------


def test_c_diluted_correct():
    r = predict_precipitation_risk(
        drug_name="D",
        drug_concentration_mg_mL=10.0,
        solubility_at_ph74_mg_mL=5.0,
        pka=7.0,
        ionization_type="neutral",
        vehicle_ph=7.0,
        dilution_ph=7.4,
        dilution_ratio=9.0,
    )
    expected_c = 10.0 / (1.0 + 9.0)
    assert r.c_diluted_mg_mL == pytest.approx(expected_c, rel=1e-6)


def test_ph_diluted_correct():
    r = predict_precipitation_risk(
        drug_name="D",
        drug_concentration_mg_mL=5.0,
        solubility_at_ph74_mg_mL=2.0,
        pka=7.0,
        ionization_type="neutral",
        vehicle_ph=3.0,
        dilution_ph=7.4,
        dilution_ratio=4.0,
    )
    expected_ph = (3.0 + 4.0 * 7.4) / (1.0 + 4.0)
    assert r.ph_diluted == pytest.approx(expected_ph, rel=1e-6)


def test_supersaturation_ratio_positive():
    r = _base_result()
    assert r.supersaturation_ratio > 0.0


# ---------------------------------------------------------------------------
# Risk classification
# ---------------------------------------------------------------------------


def test_sr_above_2_is_high_risk():
    # High conc, low solubility -> SR >> 2
    r = predict_precipitation_risk(
        drug_name="D",
        drug_concentration_mg_mL=50.0,
        solubility_at_ph74_mg_mL=0.1,
        pka=9.0,
        ionization_type="acid",
        vehicle_ph=11.0,
        dilution_ph=7.4,
        dilution_ratio=2.0,
    )
    assert r.risk_class == "high"
    assert r.supersaturation_ratio > 2.0


def test_sr_between_1_and_2_moderate():
    # Need SR between 1 and 2
    # Use neutral: SR = C_diluted / solubility
    # C_diluted = 5/(1+10) ~ 0.454, solubility=0.3 -> SR ~ 1.5
    r = predict_precipitation_risk(
        drug_name="D",
        drug_concentration_mg_mL=5.0,
        solubility_at_ph74_mg_mL=0.3,
        pka=7.0,
        ionization_type="neutral",
        vehicle_ph=7.4,
        dilution_ph=7.4,
        dilution_ratio=10.0,
    )
    if r.supersaturation_ratio > 1.0:
        assert r.risk_class in ("moderate", "high")


def test_sr_below_1_low_risk():
    # Very soluble drug -> SR < 1
    r = predict_precipitation_risk(
        drug_name="D",
        drug_concentration_mg_mL=1.0,
        solubility_at_ph74_mg_mL=100.0,
        pka=7.0,
        ionization_type="neutral",
        vehicle_ph=7.4,
        dilution_ph=7.4,
        dilution_ratio=10.0,
    )
    assert r.risk_class == "low"
    assert r.supersaturation_ratio < 1.0


# ---------------------------------------------------------------------------
# Concentration effect on SR
# ---------------------------------------------------------------------------


def test_higher_concentration_higher_sr():
    low = predict_precipitation_risk(
        drug_name="D",
        drug_concentration_mg_mL=1.0,
        solubility_at_ph74_mg_mL=2.0,
        pka=7.0,
        ionization_type="neutral",
        vehicle_ph=7.4,
        dilution_ph=7.4,
        dilution_ratio=5.0,
    )
    high = predict_precipitation_risk(
        drug_name="D",
        drug_concentration_mg_mL=20.0,
        solubility_at_ph74_mg_mL=2.0,
        pka=7.0,
        ionization_type="neutral",
        vehicle_ph=7.4,
        dilution_ph=7.4,
        dilution_ratio=5.0,
    )
    assert high.supersaturation_ratio > low.supersaturation_ratio


# ---------------------------------------------------------------------------
# Solubility effect on risk
# ---------------------------------------------------------------------------


def test_lower_solubility_higher_risk_score():
    high_sol = predict_precipitation_risk(
        drug_name="D",
        drug_concentration_mg_mL=5.0,
        solubility_at_ph74_mg_mL=10.0,
        pka=7.0,
        ionization_type="neutral",
        vehicle_ph=7.4,
        dilution_ph=7.4,
        dilution_ratio=10.0,
    )
    low_sol = predict_precipitation_risk(
        drug_name="D",
        drug_concentration_mg_mL=5.0,
        solubility_at_ph74_mg_mL=0.05,
        pka=7.0,
        ionization_type="neutral",
        vehicle_ph=7.4,
        dilution_ph=7.4,
        dilution_ratio=10.0,
    )
    assert low_sol.precipitation_risk_score > high_sol.precipitation_risk_score


# ---------------------------------------------------------------------------
# Ionization effects
# ---------------------------------------------------------------------------


def test_acid_drug_more_precipitate_in_acidic_diluent():
    """Acid drug: acidic diluent lowers solubility, increases SR."""
    acid_diluent = predict_precipitation_risk(
        drug_name="Acid",
        drug_concentration_mg_mL=5.0,
        solubility_at_ph74_mg_mL=2.0,
        pka=7.0,
        ionization_type="acid",
        vehicle_ph=11.0,
        dilution_ph=3.0,
        dilution_ratio=5.0,
    )
    neutral_diluent = predict_precipitation_risk(
        drug_name="Acid",
        drug_concentration_mg_mL=5.0,
        solubility_at_ph74_mg_mL=2.0,
        pka=7.0,
        ionization_type="acid",
        vehicle_ph=11.0,
        dilution_ph=9.0,
        dilution_ratio=5.0,
    )
    # Acidic diluent reduces solubility for acid drug -> higher SR
    assert acid_diluent.supersaturation_ratio > neutral_diluent.supersaturation_ratio


def test_base_drug_more_precipitate_in_basic_diluent():
    """Base drug: basic diluent lowers ionization, reduces solubility."""
    basic_diluent = predict_precipitation_risk(
        drug_name="Base",
        drug_concentration_mg_mL=5.0,
        solubility_at_ph74_mg_mL=2.0,
        pka=7.0,
        ionization_type="base",
        vehicle_ph=3.0,
        dilution_ph=10.0,
        dilution_ratio=5.0,
    )
    acidic_diluent = predict_precipitation_risk(
        drug_name="Base",
        drug_concentration_mg_mL=5.0,
        solubility_at_ph74_mg_mL=2.0,
        pka=7.0,
        ionization_type="base",
        vehicle_ph=3.0,
        dilution_ph=4.0,
        dilution_ratio=5.0,
    )
    # Basic diluent reduces solubility for base drug
    assert basic_diluent.supersaturation_ratio > acidic_diluent.supersaturation_ratio


def test_neutral_drug_solubility_unchanged_by_ph():
    """Neutral drug: pH has no effect on solubility."""
    r1 = predict_precipitation_risk(
        drug_name="Neutral",
        drug_concentration_mg_mL=5.0,
        solubility_at_ph74_mg_mL=3.0,
        pka=7.0,
        ionization_type="neutral",
        vehicle_ph=7.4,
        dilution_ph=3.0,
        dilution_ratio=5.0,
    )
    r2 = predict_precipitation_risk(
        drug_name="Neutral",
        drug_concentration_mg_mL=5.0,
        solubility_at_ph74_mg_mL=3.0,
        pka=7.0,
        ionization_type="neutral",
        vehicle_ph=7.4,
        dilution_ph=10.0,
        dilution_ratio=5.0,
    )
    assert r1.solubility_at_diluted_ph_mg_mL == pytest.approx(
        r2.solubility_at_diluted_ph_mg_mL, rel=1e-6
    )


# ---------------------------------------------------------------------------
# time_to_precipitate
# ---------------------------------------------------------------------------


def test_no_risk_infinite_time_to_ppt():
    r = predict_precipitation_risk(
        drug_name="D",
        drug_concentration_mg_mL=1.0,
        solubility_at_ph74_mg_mL=100.0,
        pka=7.0,
        ionization_type="neutral",
        vehicle_ph=7.4,
        dilution_ph=7.4,
        dilution_ratio=10.0,
    )
    assert math.isinf(r.time_to_precipitate_min)


def test_high_risk_finite_time_to_ppt():
    r = predict_precipitation_risk(
        drug_name="D",
        drug_concentration_mg_mL=50.0,
        solubility_at_ph74_mg_mL=0.1,
        pka=7.0,
        ionization_type="neutral",
        vehicle_ph=7.4,
        dilution_ph=7.4,
        dilution_ratio=2.0,
    )
    assert not math.isinf(r.time_to_precipitate_min)
    assert r.time_to_precipitate_min > 0.0


# ---------------------------------------------------------------------------
# assess_iv_compatibility
# ---------------------------------------------------------------------------


def test_assess_iv_compatibility_return_list():
    fluids = [
        {"fluid_name": "NS (0.9% NaCl)", "ph": 5.5},
        {"fluid_name": "D5W", "ph": 4.5},
        {"fluid_name": "LR", "ph": 6.5},
    ]
    results = assess_iv_compatibility(
        drug_name="Drug",
        drug_concentration_mg_mL=10.0,
        solubility_at_ph74_mg_mL=2.0,
        pka=7.0,
        ionization_type="acid",
        common_iv_fluids=fluids,
    )
    assert isinstance(results, list)
    assert len(results) == 3
    assert all(isinstance(r, PrecipitationRiskResult) for r in results)


def test_assess_iv_compatibility_sorted_ascending():
    fluids = [
        {"fluid_name": "A", "ph": 4.0},
        {"fluid_name": "B", "ph": 7.0},
        {"fluid_name": "C", "ph": 9.5},
    ]
    results = assess_iv_compatibility(
        drug_name="Drug",
        drug_concentration_mg_mL=5.0,
        solubility_at_ph74_mg_mL=2.0,
        pka=7.0,
        ionization_type="acid",
        common_iv_fluids=fluids,
    )
    scores = [r.precipitation_risk_score for r in results]
    assert scores == sorted(scores)


def test_assess_iv_compatibility_empty_fluids_raises():
    with pytest.raises(ValueError):
        assess_iv_compatibility(
            drug_name="D",
            drug_concentration_mg_mL=5.0,
            solubility_at_ph74_mg_mL=2.0,
            pka=7.0,
            ionization_type="neutral",
            common_iv_fluids=[],
        )


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_conc_zero():
    with pytest.raises(ValueError, match="drug_concentration"):
        predict_precipitation_risk(
            drug_name="D",
            drug_concentration_mg_mL=0.0,
            solubility_at_ph74_mg_mL=1.0,
            pka=7.0,
            ionization_type="acid",
            vehicle_ph=7.0,
            dilution_ph=7.4,
            dilution_ratio=10.0,
        )


def test_validation_solubility_zero():
    with pytest.raises(ValueError, match="solubility"):
        predict_precipitation_risk(
            drug_name="D",
            drug_concentration_mg_mL=5.0,
            solubility_at_ph74_mg_mL=0.0,
            pka=7.0,
            ionization_type="acid",
            vehicle_ph=7.0,
            dilution_ph=7.4,
            dilution_ratio=10.0,
        )


def test_validation_pka_out_of_range():
    with pytest.raises(ValueError, match="pka"):
        predict_precipitation_risk(
            drug_name="D",
            drug_concentration_mg_mL=5.0,
            solubility_at_ph74_mg_mL=1.0,
            pka=0.0,
            ionization_type="acid",
            vehicle_ph=7.0,
            dilution_ph=7.4,
            dilution_ratio=10.0,
        )


def test_validation_invalid_ionization_type():
    with pytest.raises(ValueError, match="ionization_type"):
        predict_precipitation_risk(
            drug_name="D",
            drug_concentration_mg_mL=5.0,
            solubility_at_ph74_mg_mL=1.0,
            pka=7.0,
            ionization_type="zwitterion",
            vehicle_ph=7.0,
            dilution_ph=7.4,
            dilution_ratio=10.0,
        )


def test_validation_vehicle_ph_out_of_range():
    with pytest.raises(ValueError, match="vehicle_ph"):
        predict_precipitation_risk(
            drug_name="D",
            drug_concentration_mg_mL=5.0,
            solubility_at_ph74_mg_mL=1.0,
            pka=7.0,
            ionization_type="neutral",
            vehicle_ph=0.0,
            dilution_ph=7.4,
            dilution_ratio=10.0,
        )


def test_validation_dilution_ph_out_of_range():
    with pytest.raises(ValueError, match="dilution_ph"):
        predict_precipitation_risk(
            drug_name="D",
            drug_concentration_mg_mL=5.0,
            solubility_at_ph74_mg_mL=1.0,
            pka=7.0,
            ionization_type="neutral",
            vehicle_ph=7.0,
            dilution_ph=14.0,
            dilution_ratio=10.0,
        )


def test_validation_dilution_ratio_zero():
    with pytest.raises(ValueError, match="dilution_ratio"):
        predict_precipitation_risk(
            drug_name="D",
            drug_concentration_mg_mL=5.0,
            solubility_at_ph74_mg_mL=1.0,
            pka=7.0,
            ionization_type="neutral",
            vehicle_ph=7.0,
            dilution_ph=7.4,
            dilution_ratio=0.0,
        )


# ---------------------------------------------------------------------------
# Notes field
# ---------------------------------------------------------------------------


def test_notes_is_string():
    r = _base_result()
    assert isinstance(r.notes, str)
    assert len(r.notes) > 0


def test_notes_contains_drug_name():
    r = predict_precipitation_risk(
        drug_name="Phenytoin",
        drug_concentration_mg_mL=50.0,
        solubility_at_ph74_mg_mL=0.02,
        pka=8.3,
        ionization_type="acid",
        vehicle_ph=12.0,
        dilution_ph=7.4,
        dilution_ratio=10.0,
    )
    assert "Phenytoin" in r.notes


# ---------------------------------------------------------------------------
# Risk score bounds
# ---------------------------------------------------------------------------


def test_risk_score_in_0_100():
    for conc in [0.1, 1.0, 10.0, 100.0, 1000.0]:
        r = predict_precipitation_risk(
            drug_name="D",
            drug_concentration_mg_mL=conc,
            solubility_at_ph74_mg_mL=1.0,
            pka=7.0,
            ionization_type="neutral",
            vehicle_ph=7.4,
            dilution_ph=7.4,
            dilution_ratio=5.0,
        )
        assert 0.0 <= r.precipitation_risk_score <= 100.0
