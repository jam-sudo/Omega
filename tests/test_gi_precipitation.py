"""Tests for Phase 488 — Drug Precipitation in GI Tract."""

import math

import pytest

from omega_pbpk.prediction.gi_precipitation import (
    GIPrecipitationResult,
    predict_gi_precipitation,
    screen_dose_range,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _call(**kwargs):
    defaults = dict(
        drug_name="test_drug",
        dose_mg=100.0,
        pKa=7.0,
        ionization_type="neutral",
        intrinsic_solubility_mg_L=500.0,
    )
    defaults.update(kwargs)
    return predict_gi_precipitation(**defaults)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = _call()
    assert isinstance(result, GIPrecipitationResult)


def test_result_is_frozen():
    result = _call()
    with pytest.raises((AttributeError, TypeError)):
        result.dose_mg = 999.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# High-solubility drug → no precipitation
# ---------------------------------------------------------------------------


def test_high_solubility_no_risk():
    """Very soluble drug: all dose dissolves -> no precipitation risk."""
    result = _call(
        dose_mg=10.0,
        intrinsic_solubility_mg_L=100_000.0,
        ionization_type="neutral",
    )
    assert result.precipitation_risk == "none"
    assert math.isclose(result.predicted_fa, 1.0, rel_tol=1e-6)
    assert result.precipitated_total_mg < 0.01


def test_high_solubility_total_dissolved_equals_dose():
    result = _call(dose_mg=5.0, intrinsic_solubility_mg_L=1_000_000.0, ionization_type="neutral")
    assert math.isclose(result.total_dissolved_mg, 5.0, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# Low-solubility base: more dissolved in stomach
# ---------------------------------------------------------------------------


def test_base_more_dissolved_in_stomach():
    """Basic drug: higher solubility at low pH (stomach) -> more dissolved there."""
    result = _call(
        dose_mg=200.0,
        pKa=8.0,
        ionization_type="base",
        intrinsic_solubility_mg_L=0.1,
    )
    assert result.dissolved_stomach_mg >= result.dissolved_jejunum_mg


def test_acid_more_dissolved_in_ileum():
    """Acidic drug: higher solubility at high pH -> more dissolved in ileum than stomach."""
    result = _call(
        dose_mg=200.0,
        pKa=4.0,
        ionization_type="acid",
        intrinsic_solubility_mg_L=0.05,
    )
    assert result.dissolved_ileum_mg >= result.dissolved_stomach_mg


# ---------------------------------------------------------------------------
# predicted_fa in [0, 1]
# ---------------------------------------------------------------------------


def test_predicted_fa_in_range_low_solubility():
    result = _call(dose_mg=1000.0, intrinsic_solubility_mg_L=0.01, ionization_type="neutral")
    assert 0.0 <= result.predicted_fa <= 1.0


def test_predicted_fa_in_range_high_solubility():
    result = _call(dose_mg=1.0, intrinsic_solubility_mg_L=100_000.0)
    assert 0.0 <= result.predicted_fa <= 1.0


# ---------------------------------------------------------------------------
# Mass balance: total_dissolved + precipitated_total == dose_mg
# ---------------------------------------------------------------------------


def test_mass_balance_neutral():
    result = _call(dose_mg=100.0, intrinsic_solubility_mg_L=50.0, ionization_type="neutral")
    total = result.total_dissolved_mg + result.precipitated_total_mg
    assert math.isclose(total, 100.0, rel_tol=1e-9)


def test_mass_balance_base():
    result = _call(dose_mg=300.0, pKa=6.0, ionization_type="base", intrinsic_solubility_mg_L=1.0)
    total = result.total_dissolved_mg + result.precipitated_total_mg
    assert math.isclose(total, 300.0, rel_tol=1e-9)


def test_mass_balance_acid():
    result = _call(dose_mg=50.0, pKa=3.0, ionization_type="acid", intrinsic_solubility_mg_L=0.5)
    total = result.total_dissolved_mg + result.precipitated_total_mg
    assert math.isclose(total, 50.0, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# dose_normalized_dissolution == total_dissolved / dose_mg
# ---------------------------------------------------------------------------


def test_dose_normalized_dissolution_consistent():
    result = _call()
    expected = result.total_dissolved_mg / result.dose_mg
    assert math.isclose(result.dose_normalized_dissolution, expected, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# screen_dose_range sorted by predicted_fa descending
# ---------------------------------------------------------------------------


def test_screen_sorted_descending():
    results = screen_dose_range("drug_X", 7.0, "neutral", 10.0, [10.0, 100.0, 500.0, 1000.0])
    fas = [r.predicted_fa for r in results]
    assert fas == sorted(fas, reverse=True)


def test_screen_returns_all_doses():
    doses = [50.0, 200.0, 500.0]
    results = screen_dose_range("drug_Y", 5.0, "acid", 20.0, doses)
    assert len(results) == len(doses)


def test_screen_all_are_gi_precipitation_results():
    results = screen_dose_range("drug_Z", 8.0, "base", 5.0, [100.0, 400.0])
    for r in results:
        assert isinstance(r, GIPrecipitationResult)


# ---------------------------------------------------------------------------
# Higher dose -> more precipitation (lower fa)
# ---------------------------------------------------------------------------


def test_higher_dose_lower_fa():
    """With identical solubility, larger dose yields lower predicted_fa."""
    low = predict_gi_precipitation("d", 10.0, 7.0, "neutral", 2.0)
    high = predict_gi_precipitation("d", 1000.0, 7.0, "neutral", 2.0)
    assert low.predicted_fa >= high.predicted_fa


def test_higher_dose_higher_precipitation_pct():
    low = predict_gi_precipitation("d", 10.0, 7.0, "neutral", 1.0)
    high = predict_gi_precipitation("d", 500.0, 7.0, "neutral", 1.0)
    low_pct = low.precipitated_total_mg / low.dose_mg
    high_pct = high.precipitated_total_mg / high.dose_mg
    assert high_pct >= low_pct


# ---------------------------------------------------------------------------
# Precipitation risk classification
# ---------------------------------------------------------------------------


def test_precipitation_risk_none():
    result = _call(dose_mg=1.0, intrinsic_solubility_mg_L=1_000_000.0)
    assert result.precipitation_risk == "none"


def test_precipitation_risk_high():
    result = _call(dose_mg=10_000.0, intrinsic_solubility_mg_L=0.001, ionization_type="neutral")
    assert result.precipitation_risk == "high"


def test_precipitation_risk_valid_values():
    result = _call()
    assert result.precipitation_risk in ("none", "low", "moderate", "high")


# ---------------------------------------------------------------------------
# Supersaturation index
# ---------------------------------------------------------------------------


def test_si_jejunum_positive():
    result = _call(dose_mg=500.0, intrinsic_solubility_mg_L=0.01, ionization_type="neutral")
    assert result.supersaturation_index_jejunum >= 0.0


def test_si_jejunum_low_when_high_solubility():
    """Highly soluble drug: SI <= 1 (not supersaturated in segment)."""
    result = _call(dose_mg=1.0, intrinsic_solubility_mg_L=100_000.0, ionization_type="neutral")
    assert result.supersaturation_index_jejunum <= 1.0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        _call(dose_mg=0.0)


def test_invalid_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        _call(dose_mg=-10.0)


def test_invalid_solubility_zero():
    with pytest.raises(ValueError, match="intrinsic_solubility"):
        _call(intrinsic_solubility_mg_L=0.0)


def test_invalid_solubility_negative():
    with pytest.raises(ValueError, match="intrinsic_solubility"):
        _call(intrinsic_solubility_mg_L=-1.0)


def test_invalid_ionization_type():
    with pytest.raises(ValueError, match="ionization_type"):
        _call(ionization_type="zwitterion")
