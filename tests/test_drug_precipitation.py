"""Tests for biopharmaceutics.drug_precipitation (Phase 192)."""
from __future__ import annotations

import math

import pytest

from omega_pbpk.biopharmaceutics.drug_precipitation import (
    PrecipitationResult,
    simulate_precipitation,
    supersaturation_index,
)


# ---------------------------------------------------------------------------
# supersaturation_index
# ---------------------------------------------------------------------------

def test_supersaturation_index_saturated():
    assert supersaturation_index(1.0, 1.0) == pytest.approx(1.0)


def test_supersaturation_index_unsaturated():
    assert supersaturation_index(0.5, 1.0) == pytest.approx(0.5)


def test_supersaturation_index_supersaturated():
    assert supersaturation_index(10.0, 2.0) == pytest.approx(5.0)


def test_supersaturation_index_invalid_solubility():
    with pytest.raises(ValueError):
        supersaturation_index(1.0, 0.0)


# ---------------------------------------------------------------------------
# No supersaturation (C <= Cs)
# ---------------------------------------------------------------------------

def test_no_supersaturation_risk_none():
    result = simulate_precipitation("Drug", 0.5, 1.0)
    assert result.risk_category == "none"


def test_no_supersaturation_absorbed_fraction_positive():
    result = simulate_precipitation("Drug", 0.5, 1.0)
    assert result.absorbed_fraction_before_precip > 0.0


def test_no_supersaturation_precip_rate_zero():
    result = simulate_precipitation("Drug", 0.5, 1.0)
    assert result.precipitation_rate_mg_per_mL_h == pytest.approx(0.0)


def test_exact_saturation_risk_none():
    """C == Cs should give risk = 'none' (ss_ratio = 1.0, not > 1)."""
    result = simulate_precipitation("Drug", 1.0, 1.0)
    assert result.risk_category == "none"


# ---------------------------------------------------------------------------
# Low supersaturation
# ---------------------------------------------------------------------------

def test_low_supersaturation_risk():
    result = simulate_precipitation("Drug", 1.5, 1.0)
    assert result.risk_category == "low"


# ---------------------------------------------------------------------------
# Moderate supersaturation
# ---------------------------------------------------------------------------

def test_moderate_supersaturation_risk():
    result = simulate_precipitation("Drug", 3.0, 1.0)
    assert result.risk_category == "moderate"


# ---------------------------------------------------------------------------
# High supersaturation
# ---------------------------------------------------------------------------

def test_high_supersaturation_risk():
    result = simulate_precipitation("Drug", 10.0, 1.0)
    assert result.risk_category == "high"


def test_high_supersaturation_formulation_strategy():
    result = simulate_precipitation("Drug", 10.0, 1.0)
    assert any("amorphous" in s.lower() for s in result.formulation_strategy)


# ---------------------------------------------------------------------------
# Effect of k_precip on absorbed fraction
# ---------------------------------------------------------------------------

def test_higher_kprecip_lower_absorbed_fraction():
    """Higher precipitation rate → less drug absorbed."""
    r_low = simulate_precipitation("Drug", 5.0, 1.0, k_precip_per_h=0.1)
    r_high = simulate_precipitation("Drug", 5.0, 1.0, k_precip_per_h=2.0)
    assert r_high.absorbed_fraction_before_precip < r_low.absorbed_fraction_before_precip


# ---------------------------------------------------------------------------
# Supersaturation ratio stored correctly
# ---------------------------------------------------------------------------

def test_supersaturation_ratio_stored():
    result = simulate_precipitation("Drug", 5.0, 1.0)
    assert result.supersaturation_ratio == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

def test_invalid_initial_conc_negative():
    with pytest.raises(ValueError):
        simulate_precipitation("Drug", -1.0, 1.0)


def test_invalid_equilibrium_solubility_zero():
    with pytest.raises(ValueError):
        simulate_precipitation("Drug", 1.0, 0.0)


def test_invalid_kprecip_negative():
    with pytest.raises(ValueError):
        simulate_precipitation("Drug", 1.0, 1.0, k_precip_per_h=-0.1)


def test_invalid_kabs_zero():
    with pytest.raises(ValueError):
        simulate_precipitation("Drug", 1.0, 1.0, k_abs_per_h=0.0)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

def test_returns_precipitation_result():
    result = simulate_precipitation("TestDrug", 2.0, 1.0)
    assert isinstance(result, PrecipitationResult)
    assert result.drug_name == "TestDrug"
