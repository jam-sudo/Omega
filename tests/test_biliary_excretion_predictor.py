"""Tests for the biliary excretion predictor (Phase 838)."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.biliary_excretion_predictor import (
    BiliaryExcretionResult,
    predict_biliary_excretion,
    screen_biliary_risk,
)

# ---------------------------------------------------------------------------
# Basic return-type tests
# ---------------------------------------------------------------------------


def test_returns_correct_type():
    result = predict_biliary_excretion("DrugA", mw_da=600.0, logp=2.0)
    assert isinstance(result, BiliaryExcretionResult)


def test_result_is_frozen():
    result = predict_biliary_excretion("DrugA", mw_da=600.0, logp=2.0)
    with pytest.raises((AttributeError, TypeError)):
        result.drug_name = "Other"  # type: ignore[misc]


def test_drug_name_preserved():
    result = predict_biliary_excretion("Compound99", mw_da=700.0, logp=1.5)
    assert result.drug_name == "Compound99"


def test_mw_threshold_is_500():
    result = predict_biliary_excretion("X", mw_da=400.0, logp=2.0)
    assert result.molecular_weight_threshold == 500.0


# ---------------------------------------------------------------------------
# f_biliary range
# ---------------------------------------------------------------------------


def test_f_biliary_in_valid_range():
    for mw in [100.0, 300.0, 500.0, 800.0, 1000.0]:
        r = predict_biliary_excretion("X", mw_da=mw, logp=2.0)
        assert 0.01 <= r.f_biliary <= 0.95, f"f_biliary={r.f_biliary} out of range for mw={mw}"


def test_f_biliary_clamped_low():
    """Highly unfavorable compound should hit lower bound 0.01."""
    result = predict_biliary_excretion(
        "LowBiliary",
        mw_da=150.0,
        logp=5.0,
        ionization_type="base",
        fu_plasma=0.1,  # high binding → -0.1
    )
    assert result.f_biliary >= 0.01


def test_f_biliary_clamped_high():
    """Very favorable compound should not exceed 0.95."""
    result = predict_biliary_excretion(
        "HighBiliary",
        mw_da=900.0,
        logp=2.0,
        ionization_type="acid",
        fu_plasma=0.9,
    )
    assert result.f_biliary <= 0.95


# ---------------------------------------------------------------------------
# MW effect on f_biliary
# ---------------------------------------------------------------------------


def test_high_mw_gives_higher_f_biliary():
    """MW > 500 compound should have higher f_biliary than MW < 350."""
    low_mw = predict_biliary_excretion("Small", mw_da=200.0, logp=2.0)
    high_mw = predict_biliary_excretion("Large", mw_da=700.0, logp=2.0)
    assert high_mw.f_biliary > low_mw.f_biliary


def test_mw_above_750_gives_extra_bonus():
    """MW > 750 should score higher than 500 < MW < 750 (same other params)."""
    mid = predict_biliary_excretion("Mid", mw_da=600.0, logp=2.0)
    high = predict_biliary_excretion("High", mw_da=800.0, logp=2.0)
    assert high.f_biliary > mid.f_biliary


# ---------------------------------------------------------------------------
# ionization_favorable
# ---------------------------------------------------------------------------


def test_ionization_favorable_acid():
    result = predict_biliary_excretion("X", mw_da=400.0, logp=2.0, ionization_type="acid")
    assert result.ionization_favorable is True


def test_ionization_favorable_zwitterion():
    result = predict_biliary_excretion("X", mw_da=400.0, logp=2.0, ionization_type="zwitterion")
    assert result.ionization_favorable is True


def test_ionization_favorable_base_false():
    result = predict_biliary_excretion("X", mw_da=400.0, logp=2.0, ionization_type="base")
    assert result.ionization_favorable is False


def test_ionization_favorable_neutral_false():
    result = predict_biliary_excretion("X", mw_da=400.0, logp=2.0, ionization_type="neutral")
    assert result.ionization_favorable is False


# ---------------------------------------------------------------------------
# EHC potential valid values
# ---------------------------------------------------------------------------


def test_ehc_potential_valid_values():
    valid = {"high", "moderate", "low", "none"}
    for mw in [100.0, 400.0, 600.0, 900.0]:
        r = predict_biliary_excretion("X", mw_da=mw, logp=1.5, ionization_type="acid")
        assert r.ehc_potential in valid, f"Unexpected ehc_potential={r.ehc_potential}"


def test_high_f_biliary_glucuronide_gives_high_ehc():
    """MW >500 + acid + optimal logP → high f_biliary + glucuronide → 'high' EHC."""
    result = predict_biliary_excretion(
        "BigAcid",
        mw_da=800.0,
        logp=1.5,
        ionization_type="acid",
        fu_plasma=0.5,
    )
    assert result.ehc_potential == "high"


# ---------------------------------------------------------------------------
# secondary_peak_likelihood matches ehc_potential
# ---------------------------------------------------------------------------


def test_secondary_peak_matches_ehc_high():
    result = predict_biliary_excretion(
        "BigAcid", mw_da=800.0, logp=1.5, ionization_type="acid", fu_plasma=0.5
    )
    assert result.ehc_potential == "high"
    assert result.secondary_peak_likelihood == "likely"


def test_secondary_peak_unlikely_when_ehc_low_or_none():
    """Small, neutral compound → low/no EHC → secondary peak unlikely."""
    result = predict_biliary_excretion(
        "Small", mw_da=200.0, logp=3.0, ionization_type="neutral", fu_plasma=0.9
    )
    assert result.secondary_peak_likelihood == "unlikely"


def test_secondary_peak_possible_when_ehc_moderate():
    """Find a compound that gets 'moderate' EHC and 'possible' secondary peak."""
    # f_biliary needs to be in (0.3, 0.5] and not (>0.5 + glucuronide)
    # MW=520 (>500 but ≤750): +0.3; logP=2 (in [0,4]): +0.1; base: -0.05; base score: 0.1
    # total = 0.10 + 0.30 + 0.10 - 0.05 = 0.45 → f_biliary=0.45 → "moderate" EHC
    result = predict_biliary_excretion(
        "Moderate", mw_da=520.0, logp=2.0, ionization_type="base", fu_plasma=0.9
    )
    assert result.ehc_potential == "moderate"
    assert result.secondary_peak_likelihood == "possible"


# ---------------------------------------------------------------------------
# dose_adjustment_hepatic
# ---------------------------------------------------------------------------


def test_dose_adjustment_required_high_f_biliary():
    result = predict_biliary_excretion(
        "BigAcid", mw_da=800.0, logp=1.5, ionization_type="acid", fu_plasma=0.9
    )
    assert result.f_biliary > 0.4
    assert result.dose_adjustment_hepatic == "required"


def test_dose_adjustment_not_required_low_f_biliary():
    result = predict_biliary_excretion(
        "Small", mw_da=200.0, logp=3.0, ionization_type="neutral", fu_plasma=0.9
    )
    assert result.f_biliary <= 0.4
    assert result.dose_adjustment_hepatic == "not_required"


# ---------------------------------------------------------------------------
# bil_cl relationship
# ---------------------------------------------------------------------------


def test_bil_cl_proportional_to_hepatic_cl():
    r1 = predict_biliary_excretion("X", mw_da=600.0, logp=2.0, hepatic_cl_mL_per_min_per_kg=10.0)
    r2 = predict_biliary_excretion("X", mw_da=600.0, logp=2.0, hepatic_cl_mL_per_min_per_kg=20.0)
    assert abs(r2.bil_cl_mL_per_min_per_kg - 2 * r1.bil_cl_mL_per_min_per_kg) < 1e-9


def test_bil_cl_equals_hepatic_cl_times_f_biliary():
    hep_cl = 15.0
    result = predict_biliary_excretion(
        "X", mw_da=600.0, logp=2.0, hepatic_cl_mL_per_min_per_kg=hep_cl
    )
    expected = hep_cl * result.f_biliary
    assert abs(result.bil_cl_mL_per_min_per_kg - expected) < 1e-9


# ---------------------------------------------------------------------------
# predicted_ehc_fraction
# ---------------------------------------------------------------------------


def test_predicted_ehc_fraction_is_half_f_biliary():
    result = predict_biliary_excretion("X", mw_da=600.0, logp=2.0)
    assert abs(result.predicted_ehc_fraction - result.f_biliary * 0.5) < 1e-9


# ---------------------------------------------------------------------------
# screen_biliary_risk
# ---------------------------------------------------------------------------


def test_screen_returns_correct_type():
    candidates = [
        {"drug_name": "A", "mw_da": 300.0, "logp": 2.0},
        {"drug_name": "B", "mw_da": 700.0, "logp": 1.5},
    ]
    results = screen_biliary_risk(candidates)
    assert all(isinstance(r, BiliaryExcretionResult) for r in results)


def test_screen_sorted_by_f_biliary_descending():
    candidates = [
        {"drug_name": "Small", "mw_da": 200.0, "logp": 5.0},
        {"drug_name": "Large", "mw_da": 900.0, "logp": 1.5, "ionization_type": "acid"},
        {"drug_name": "Mid", "mw_da": 520.0, "logp": 2.0},
    ]
    results = screen_biliary_risk(candidates)
    f_values = [r.f_biliary for r in results]
    assert f_values == sorted(f_values, reverse=True)


def test_screen_returns_all_candidates():
    candidates = [{"drug_name": f"Drug{i}", "mw_da": 300.0 + i * 50, "logp": 2.0} for i in range(5)]
    results = screen_biliary_risk(candidates)
    assert len(results) == 5


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_raises_on_zero_mw():
    with pytest.raises(ValueError, match="mw_da"):
        predict_biliary_excretion("X", mw_da=0.0, logp=2.0)


def test_raises_on_negative_mw():
    with pytest.raises(ValueError):
        predict_biliary_excretion("X", mw_da=-100.0, logp=2.0)


def test_raises_on_fu_plasma_zero():
    with pytest.raises(ValueError, match="fu_plasma"):
        predict_biliary_excretion("X", mw_da=400.0, logp=2.0, fu_plasma=0.0)


def test_raises_on_fu_plasma_above_one():
    with pytest.raises(ValueError, match="fu_plasma"):
        predict_biliary_excretion("X", mw_da=400.0, logp=2.0, fu_plasma=1.5)


def test_raises_on_negative_hepatic_cl():
    with pytest.raises(ValueError, match="hepatic_cl"):
        predict_biliary_excretion("X", mw_da=400.0, logp=2.0, hepatic_cl_mL_per_min_per_kg=-5.0)
