"""
Tests for Phase 745 — Drug Precipitation in GI Tract.
"""

import math

import pytest

from omega_pbpk.core.gi_precipitation_pk import (
    GIPrecipitationResult,
    assess_precipitation_risk,
    simulate_gi_precipitation_pk,
)

# ---------------------------------------------------------------------------
# Basic result structure
# ---------------------------------------------------------------------------


def test_returns_gi_precipitation_result():
    result = simulate_gi_precipitation_pk("TestDrug", dose_mg=10.0)
    assert isinstance(result, GIPrecipitationResult)


def test_cmax_positive():
    result = simulate_gi_precipitation_pk("TestDrug", dose_mg=50.0)
    assert result.cmax_mg_L > 0.0


def test_auc_positive():
    result = simulate_gi_precipitation_pk("TestDrug", dose_mg=50.0)
    assert result.auc_mg_h_per_L > 0.0


def test_drug_name_preserved():
    result = simulate_gi_precipitation_pk("Ibuprofen", dose_mg=100.0)
    assert result.drug_name == "Ibuprofen"


def test_dose_preserved():
    result = simulate_gi_precipitation_pk("DrugA", dose_mg=75.0)
    assert result.dose_mg == 75.0


def test_tmax_within_simulation_time():
    result = simulate_gi_precipitation_pk("TestDrug", dose_mg=20.0, t_end_h=12.0)
    assert 0.0 <= result.tmax_h <= 12.0


# ---------------------------------------------------------------------------
# Low dose — below saturation — minimal precipitation
# ---------------------------------------------------------------------------


def test_low_dose_minimal_precipitation():
    """Dose well below C_sat × V_GI should produce very little precipitation."""
    # C_sat * V_GI = 5.0 * 0.25 = 1.25 mg; dose 0.5 mg is well below
    result = simulate_gi_precipitation_pk("LowSolDrug", dose_mg=0.5, c_sat_mg_L=5.0, v_gi_L=0.25)
    # Fraction precipitated should be small (< 5%)
    assert result.fraction_precipitated < 0.05


def test_high_dose_causes_precipitation():
    """Dose far above saturation should cause significant precipitation."""
    # C_sat * V_GI = 5.0 * 0.25 = 1.25 mg; dose 50 mg is ~40x above saturation
    result = simulate_gi_precipitation_pk(
        "PoorSolDrug", dose_mg=50.0, c_sat_mg_L=5.0, v_gi_L=0.25, k_precip_per_h=2.0
    )
    assert result.fraction_precipitated > 0.0
    assert result.peak_precipitate_mg > 0.0


# ---------------------------------------------------------------------------
# k_precip effect: higher rate → more precipitation → lower AUC
# ---------------------------------------------------------------------------


def test_higher_k_precip_increases_precipitation():
    base = simulate_gi_precipitation_pk("Drug", dose_mg=20.0, c_sat_mg_L=5.0, k_precip_per_h=0.5)
    fast = simulate_gi_precipitation_pk("Drug", dose_mg=20.0, c_sat_mg_L=5.0, k_precip_per_h=5.0)
    assert fast.peak_precipitate_mg >= base.peak_precipitate_mg


def test_higher_k_precip_reduces_auc():
    low = simulate_gi_precipitation_pk("Drug", dose_mg=20.0, c_sat_mg_L=5.0, k_precip_per_h=0.1)
    high = simulate_gi_precipitation_pk("Drug", dose_mg=20.0, c_sat_mg_L=5.0, k_precip_per_h=5.0)
    assert high.auc_mg_h_per_L < low.auc_mg_h_per_L


# ---------------------------------------------------------------------------
# fraction_precipitated and fraction_absorbed bounds
# ---------------------------------------------------------------------------


def test_fraction_precipitated_in_bounds():
    result = simulate_gi_precipitation_pk("DrugB", dose_mg=100.0)
    assert 0.0 <= result.fraction_precipitated <= 1.0


def test_fraction_absorbed_in_bounds():
    result = simulate_gi_precipitation_pk("DrugB", dose_mg=100.0)
    assert 0.0 <= result.fraction_absorbed <= 1.0


# ---------------------------------------------------------------------------
# t_half reasonable
# ---------------------------------------------------------------------------


def test_t_half_positive():
    result = simulate_gi_precipitation_pk("Drug", dose_mg=50.0, cl_L_per_h=5.0, vd_L=50.0)
    assert result.t_half_h > 0.0


def test_t_half_matches_ke():
    """t½ = ln(2) / ke = ln(2) × Vd / CL."""
    result = simulate_gi_precipitation_pk("Drug", dose_mg=50.0, cl_L_per_h=10.0, vd_L=50.0)
    expected_t_half = math.log(2.0) * 50.0 / 10.0
    assert abs(result.t_half_h - expected_t_half) < 1e-9


# ---------------------------------------------------------------------------
# assess_precipitation_risk
# ---------------------------------------------------------------------------


def test_assess_risk_low():
    risk = assess_precipitation_risk(dose_mg=0.5, c_sat_mg_L=5.0, v_gi_L=0.25)
    assert risk["risk_level"] == "low"
    assert risk["dose_number"] < 1.0


def test_assess_risk_moderate():
    # Do = 2.5 / (5.0 * 0.25) = 2 → moderate
    risk = assess_precipitation_risk(dose_mg=2.5, c_sat_mg_L=5.0, v_gi_L=0.25)
    assert risk["risk_level"] == "moderate"


def test_assess_risk_high():
    # Do = 50 / (5.0 * 0.25) = 40 → high
    risk = assess_precipitation_risk(dose_mg=50.0, c_sat_mg_L=5.0, v_gi_L=0.25)
    assert risk["risk_level"] == "high"
    assert risk["dose_number"] > 5.0


def test_assess_risk_returns_dict():
    risk = assess_precipitation_risk(dose_mg=10.0, c_sat_mg_L=5.0)
    assert isinstance(risk, dict)
    assert "dose_number" in risk
    assert "risk_level" in risk


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validate_dose_zero_raises():
    with pytest.raises(ValueError):
        simulate_gi_precipitation_pk("Drug", dose_mg=0.0)


def test_validate_dose_negative_raises():
    with pytest.raises(ValueError):
        simulate_gi_precipitation_pk("Drug", dose_mg=-10.0)


def test_validate_c_sat_zero_raises():
    with pytest.raises(ValueError):
        simulate_gi_precipitation_pk("Drug", dose_mg=10.0, c_sat_mg_L=0.0)


def test_validate_cl_zero_raises():
    with pytest.raises(ValueError):
        simulate_gi_precipitation_pk("Drug", dose_mg=10.0, cl_L_per_h=0.0)


def test_validate_vd_zero_raises():
    with pytest.raises(ValueError):
        simulate_gi_precipitation_pk("Drug", dose_mg=10.0, vd_L=0.0)


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


def test_notes_nonempty():
    result = simulate_gi_precipitation_pk("DrugX", dose_mg=10.0)
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0
