"""Tests for Phase 189: renal tubular secretion and reabsorption model."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.core.renal_tubular import (
    TubularResult,
    renal_clearance_components,
    simulate_tubular_handling,
)

# --- Basic smoke test ---

def test_simulate_returns_tubular_result():
    result = simulate_tubular_handling(
        drug_name="TestDrug",
        plasma_conc_mg_L=0.5,
    )
    assert isinstance(result, TubularResult)
    assert result.drug_name == "TestDrug"


# --- Filtration clearance ---

def test_filtration_cl_equals_gfr_times_fu():
    result = simulate_tubular_handling(
        drug_name="Drug",
        plasma_conc_mg_L=1.0,
        gfr_mL_per_min=100.0,
        fu_plasma=0.2,
        vmax_secretion_mg_per_min=0.5,
        km_secretion_mg_L=1.0,
        f_reabsorption=0.0,
    )
    assert math.isclose(result.cl_filtration_mL_per_min, 100.0 * 0.2, rel_tol=1e-9)


# --- Secretion saturation ---

def test_low_conc_gives_low_saturation():
    result = simulate_tubular_handling(
        drug_name="Drug",
        plasma_conc_mg_L=0.01,
        km_secretion_mg_L=1.0,
    )
    # At 1% of Km, saturation should be ~1%
    assert result.secretion_saturation_pct < 5.0


def test_high_conc_gives_high_saturation():
    result = simulate_tubular_handling(
        drug_name="Drug",
        plasma_conc_mg_L=100.0,
        km_secretion_mg_L=1.0,
    )
    # At 100x Km, saturation should be >98%
    assert result.secretion_saturation_pct > 98.0


def test_saturation_at_km_is_50_percent():
    result = simulate_tubular_handling(
        drug_name="Drug",
        plasma_conc_mg_L=2.0,
        km_secretion_mg_L=2.0,
    )
    assert math.isclose(result.secretion_saturation_pct, 50.0, rel_tol=1e-9)


# --- Reabsorption ---

def test_no_reabsorption_when_f_reabs_zero():
    result = simulate_tubular_handling(
        drug_name="Drug",
        plasma_conc_mg_L=1.0,
        f_reabsorption=0.0,
    )
    assert result.cl_reabsorption_mL_per_min == 0.0


def test_full_reabsorption_reduces_net_cl():
    result_no_reabs = simulate_tubular_handling(
        drug_name="Drug",
        plasma_conc_mg_L=1.0,
        f_reabsorption=0.0,
    )
    result_with_reabs = simulate_tubular_handling(
        drug_name="Drug",
        plasma_conc_mg_L=1.0,
        f_reabsorption=0.8,
    )
    assert result_with_reabs.cl_renal_total_mL_per_min < result_no_reabs.cl_renal_total_mL_per_min


# --- fe_urine ---

def test_fe_urine_between_0_and_1():
    for conc in [0.01, 0.5, 1.0, 10.0]:
        result = simulate_tubular_handling(
            drug_name="Drug",
            plasma_conc_mg_L=conc,
        )
        assert 0.0 <= result.fe_urine <= 1.0


def test_fe_urine_increases_with_lower_hepatic_cl():
    result_low_hep = simulate_tubular_handling(
        drug_name="Drug",
        plasma_conc_mg_L=1.0,
        cl_hepatic_L_per_h=0.1,
    )
    result_high_hep = simulate_tubular_handling(
        drug_name="Drug",
        plasma_conc_mg_L=1.0,
        cl_hepatic_L_per_h=50.0,
    )
    assert result_low_hep.fe_urine > result_high_hep.fe_urine


# --- Net renal CL is non-negative ---

def test_cl_renal_total_non_negative():
    result = simulate_tubular_handling(
        drug_name="Drug",
        plasma_conc_mg_L=1.0,
        f_reabsorption=0.99,
        gfr_mL_per_min=5.0,
        vmax_secretion_mg_per_min=0.01,
    )
    assert result.cl_renal_total_mL_per_min >= 0.0


# --- Secretion CL in linear range ---

def test_secretion_cl_linear_at_low_conc():
    """At very low concentration, cl_sec ≈ Vmax/Km (linear limit)."""
    vmax = 0.5
    km = 2.0
    result = simulate_tubular_handling(
        drug_name="Drug",
        plasma_conc_mg_L=1e-6,
        vmax_secretion_mg_per_min=vmax,
        km_secretion_mg_L=km,
        f_reabsorption=0.0,
        gfr_mL_per_min=0.0,
        fu_plasma=1.0,
    )
    expected_cl_sec = vmax / km  # mL/min
    assert math.isclose(result.cl_secretion_mL_per_min, expected_cl_sec, rel_tol=1e-3)


# --- renal_clearance_components ---

def test_renal_clearance_components_returns_correct_length():
    concs = [0.1, 0.5, 1.0, 5.0, 10.0]
    results = renal_clearance_components(
        drug_name="Drug",
        plasma_conc_range_mg_L=concs,
    )
    assert len(results) == len(concs)


def test_renal_clearance_components_all_tubular_results():
    results = renal_clearance_components(
        drug_name="Drug",
        plasma_conc_range_mg_L=[0.1, 1.0, 10.0],
    )
    for r in results:
        assert isinstance(r, TubularResult)


def test_renal_clearance_components_saturation_increases():
    """Higher concentration → higher secretion saturation."""
    concs = [0.01, 1.0, 10.0, 100.0]
    results = renal_clearance_components(
        drug_name="Drug",
        plasma_conc_range_mg_L=concs,
        km_secretion_mg_L=1.0,
    )
    saturations = [r.secretion_saturation_pct for r in results]
    for i in range(len(saturations) - 1):
        assert saturations[i] < saturations[i + 1]


def test_renal_clearance_components_empty_list():
    results = renal_clearance_components(
        drug_name="Drug",
        plasma_conc_range_mg_L=[],
    )
    assert results == []


# --- Validation errors ---

def test_negative_gfr_raises():
    with pytest.raises(ValueError, match="gfr_mL_per_min"):
        simulate_tubular_handling(
            drug_name="Drug",
            plasma_conc_mg_L=1.0,
            gfr_mL_per_min=-1.0,
        )


def test_negative_plasma_conc_raises():
    with pytest.raises(ValueError, match="plasma_conc_mg_L"):
        simulate_tubular_handling(
            drug_name="Drug",
            plasma_conc_mg_L=-0.1,
        )


def test_invalid_fu_plasma_raises():
    with pytest.raises(ValueError, match="fu_plasma"):
        simulate_tubular_handling(
            drug_name="Drug",
            plasma_conc_mg_L=1.0,
            fu_plasma=0.0,
        )


def test_zero_vmax_raises():
    with pytest.raises(ValueError, match="vmax_secretion_mg_per_min"):
        simulate_tubular_handling(
            drug_name="Drug",
            plasma_conc_mg_L=1.0,
            vmax_secretion_mg_per_min=0.0,
        )


def test_zero_km_raises():
    with pytest.raises(ValueError, match="km_secretion_mg_L"):
        simulate_tubular_handling(
            drug_name="Drug",
            plasma_conc_mg_L=1.0,
            km_secretion_mg_L=0.0,
        )


# --- Transporter stored ---

def test_transporter_stored_in_result():
    result = simulate_tubular_handling(
        drug_name="Drug",
        plasma_conc_mg_L=1.0,
        transporter="OCT2",
    )
    assert result.transporter == "OCT2"
