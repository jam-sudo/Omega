"""Tests for environmental_fate_risk module (Phase 714)."""

import math

import pytest

from omega_pbpk.risk.environmental_fate_risk import (
    EnvironmentalFateResult,
    assess_environmental_fate,
    estimate_aquatic_toxicity_lc50,
    screen_environmental_risk,
)

# ---------------------------------------------------------------------------
# assess_environmental_fate — return type and basic structure
# ---------------------------------------------------------------------------


def test_returns_environmental_fate_result():
    result = assess_environmental_fate("TestCompound", logP=2.0, mw=300.0)
    assert isinstance(result, EnvironmentalFateResult)


def test_high_logp_gives_high_bcf():
    result = assess_environmental_fate("HighLogP", logP=5.0, mw=400.0)
    assert result.bcf > 2000.0


def test_low_logp_gives_low_bcf():
    result = assess_environmental_fate("LowLogP", logP=-1.0, mw=200.0)
    assert result.bcf < 100.0


def test_high_logp_is_bioaccumulative():
    result = assess_environmental_fate("BioAcc", logP=5.0, mw=400.0)
    assert result.is_bioaccumulative is True


def test_low_logp_not_bioaccumulative():
    result = assess_environmental_fate("NotBioAcc", logP=1.0, mw=150.0)
    assert result.is_bioaccumulative is False


def test_is_pbt_true_when_all_three_flags():
    # logP=6 → persistent (t½=180d > 60d), bioaccumulative (BCF>2000), toxic (logP>5)
    result = assess_environmental_fate("PBT", logP=6.0, mw=600.0)
    assert result.is_persistent is True
    assert result.is_bioaccumulative is True
    assert result.is_pbt is True


def test_is_pbt_false_when_not_all_three():
    # logP=1 → not persistent, not bioaccumulative, not toxic
    result = assess_environmental_fate("Safe", logP=1.0, mw=200.0)
    assert result.is_pbt is False


def test_environmental_risk_class_valid_values():
    result = assess_environmental_fate("Test", logP=2.0, mw=300.0)
    assert result.environmental_risk_class in {"low", "medium", "high"}


def test_high_logp_high_risk_class():
    # logP=6 → BCF huge → ERQ very small → "high" risk
    result = assess_environmental_fate("Risky", logP=6.0, mw=400.0)
    assert result.environmental_risk_class == "high"


def test_low_logp_low_risk_class():
    result = assess_environmental_fate("Safe", logP=-1.0, mw=100.0)
    assert result.environmental_risk_class == "low"


def test_water_solubility_estimated_when_not_provided():
    result = assess_environmental_fate("Test", logP=2.0, mw=300.0)
    assert result.water_solubility_mg_per_L > 0.0


def test_water_solubility_uses_provided_value():
    result = assess_environmental_fate("Test", logP=2.0, mw=300.0, water_solubility_mg_per_L=500.0)
    assert math.isclose(result.water_solubility_mg_per_L, 500.0)


def test_biodegradation_t_half_estimated_when_not_provided():
    result = assess_environmental_fate("Test", logP=2.0, mw=300.0)
    assert result.biodegradation_t_half_days > 0.0


def test_biodegradation_t_half_uses_provided_value():
    result = assess_environmental_fate(
        "Test", logP=2.0, mw=300.0, biodegradation_half_life_days=45.0
    )
    assert math.isclose(result.biodegradation_t_half_days, 45.0)


def test_low_logp_low_mw_readily_biodegradable():
    result = assess_environmental_fate("Easy", logP=1.0, mw=150.0)
    assert math.isclose(result.biodegradation_t_half_days, 2.0)


def test_medium_logp_slowly_biodegradable():
    result = assess_environmental_fate("Medium", logP=3.0, mw=300.0)
    assert math.isclose(result.biodegradation_t_half_days, 30.0)


def test_high_logp_persistent():
    result = assess_environmental_fate("Persistent", logP=5.0, mw=300.0)
    assert math.isclose(result.biodegradation_t_half_days, 180.0)
    assert result.is_persistent is True


def test_notes_nonempty():
    result = assess_environmental_fate("Test", logP=2.0, mw=300.0)
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


def test_validation_mw_zero_raises():
    with pytest.raises(ValueError, match="mw must be > 0"):
        assess_environmental_fate("Bad", logP=1.0, mw=0.0)


def test_validation_mw_negative_raises():
    with pytest.raises(ValueError, match="mw must be > 0"):
        assess_environmental_fate("Bad", logP=1.0, mw=-100.0)


# ---------------------------------------------------------------------------
# screen_environmental_risk
# ---------------------------------------------------------------------------


def test_screen_environmental_risk_sorted_by_bcf_descending():
    compounds = [
        {"compound_name": "LowLogP", "logP": 1.0, "mw": 200.0},
        {"compound_name": "HighLogP", "logP": 5.0, "mw": 400.0},
        {"compound_name": "MedLogP", "logP": 3.0, "mw": 300.0},
    ]
    results = screen_environmental_risk(compounds)
    assert len(results) == 3
    assert results[0].bcf >= results[1].bcf >= results[2].bcf


def test_screen_environmental_risk_returns_list():
    compounds = [{"compound_name": "A", "logP": 2.0, "mw": 250.0}]
    results = screen_environmental_risk(compounds)
    assert isinstance(results, list)
    assert isinstance(results[0], EnvironmentalFateResult)


# ---------------------------------------------------------------------------
# estimate_aquatic_toxicity_lc50
# ---------------------------------------------------------------------------


def test_high_logp_low_lc50_more_toxic():
    lc50_high = estimate_aquatic_toxicity_lc50(logP=5.0, mw=400.0)
    lc50_low = estimate_aquatic_toxicity_lc50(logP=1.0, mw=200.0)
    assert lc50_high < lc50_low


def test_low_logp_high_lc50_less_toxic():
    lc50 = estimate_aquatic_toxicity_lc50(logP=0.0, mw=100.0)
    # log_LC50 = -0.87*0 + 2.0 = 2.0 → LC50 = 100 mg/L
    assert math.isclose(lc50, 100.0, rel_tol=1e-6)


def test_lc50_always_positive():
    for logp in [-2.0, 0.0, 3.0, 7.0]:
        lc50 = estimate_aquatic_toxicity_lc50(logP=logp, mw=300.0)
        assert lc50 > 0.0


def test_lc50_minimum_clamped():
    # Very high logP should clamp to 0.001
    lc50 = estimate_aquatic_toxicity_lc50(logP=100.0, mw=300.0)
    assert lc50 >= 0.001
