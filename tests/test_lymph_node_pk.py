"""Tests for lymph node drug accumulation module (Phase 163)."""

from __future__ import annotations

import pytest

from omega_pbpk.core.lymph_node_pk import (
    LymphNodePKResult,
    compare_formulation_lymph,
    simulate_lymph_node_pk,
)

# Default parameters for convenience
DEFAULTS = dict(
    drug_name="TestDrug",
    dose_mg=100.0,
    cl_L_per_h=1.0,
    vd_systemic_L=10.0,
)


def _run(**overrides):
    kw = {**DEFAULTS, **overrides}
    return simulate_lymph_node_pk(**kw)


# --- 1. Basic simulation ---


def test_basic_simulation_returns_result():
    result = _run()
    assert isinstance(result, LymphNodePKResult)


def test_result_fields_populated():
    result = _run()
    assert result.drug_name == "TestDrug"
    assert result.dose_mg == 100.0
    assert result.times_h is not None
    assert result.c_plasma_mg_L is not None
    assert result.c_lymph_mg_L is not None
    assert result.cmax_plasma is not None
    assert result.cmax_lymph is not None
    assert result.auc_plasma is not None
    assert result.auc_lymph is not None
    assert result.lymph_to_plasma_ratio is not None
    assert result.lymph_enrichment_factor is not None
    assert result.t_half_lymph_h is not None
    assert result.notes is not None


def test_times_length_matches_concentrations():
    result = _run()
    assert len(result.times_h) == len(result.c_plasma_mg_L)
    assert len(result.times_h) == len(result.c_lymph_mg_L)


# --- Validation ---


def test_negative_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        _run(dose_mg=-10.0)


def test_zero_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        _run(dose_mg=0.0)


def test_negative_clearance_raises():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        _run(cl_L_per_h=-1.0)


def test_negative_vd_raises():
    with pytest.raises(ValueError, match="vd_systemic_L"):
        _run(vd_systemic_L=-5.0)


def test_f_lymph_uptake_out_of_range_raises():
    with pytest.raises(ValueError, match="f_lymph_uptake"):
        _run(f_lymph_uptake=1.5)
    with pytest.raises(ValueError, match="f_lymph_uptake"):
        _run(f_lymph_uptake=-0.1)


def test_negative_lymph_volume_raises():
    with pytest.raises(ValueError, match="lymph_volume_L"):
        _run(lymph_volume_L=-1.0)


# --- PK behaviour ---


def test_higher_k_lymph_in_increases_lymph_cmax():
    low = _run(k_lymph_in_per_h=0.01)
    high = _run(k_lymph_in_per_h=0.5)
    assert high.cmax_lymph > low.cmax_lymph


def test_lower_k_lymph_out_increases_lymph_retention():
    fast_out = _run(k_lymph_out_per_h=0.5)
    slow_out = _run(k_lymph_out_per_h=0.005)
    assert slow_out.t_half_lymph_h > fast_out.t_half_lymph_h


def test_lymph_equilibrates_slower_than_plasma():
    result = _run(f_lymph_uptake=0.0)
    # Lymph peak should occur after time 0 (plasma peaks at t=0 for IV bolus)
    c_lymph = result.c_lymph_mg_L
    idx_max = c_lymph.index(max(c_lymph))
    assert idx_max > 0


def test_linear_pk_dose_proportionality():
    r1 = _run(dose_mg=50.0)
    r2 = _run(dose_mg=100.0)
    ratio = r2.auc_plasma / r1.auc_plasma
    assert abs(ratio - 2.0) < 0.1


def test_mass_balance_initial():
    dose = 100.0
    f = 0.1
    result = _run(dose_mg=dose, f_lymph_uptake=f)
    # At t=0: plasma amount + lymph amount = dose
    a_plasma_0 = result.c_plasma_mg_L[0] * 10.0  # vd_systemic_L=10
    a_lymph_0 = result.c_lymph_mg_L[0] * 2.0  # lymph_volume_L=2
    assert abs(a_plasma_0 + a_lymph_0 - dose) < 1e-6


def test_plasma_concentration_starts_positive():
    result = _run()
    # With default f_lymph_uptake=0.1, 90% goes to plasma
    assert result.c_plasma_mg_L[0] > 0


# --- Derived metrics ---


def test_lymph_to_plasma_ratio_positive():
    result = _run()
    assert result.lymph_to_plasma_ratio > 0


def test_enrichment_factor_computed():
    result = _run()
    assert result.lymph_enrichment_factor > 0


def test_t_half_lymph_positive():
    result = _run()
    assert result.t_half_lymph_h > 0


def test_high_accumulation_note():
    # Very high k_lymph_in and low k_lymph_out should give high ratio
    result = _run(
        k_lymph_in_per_h=2.0,
        k_lymph_out_per_h=0.001,
        t_end_h=200.0,
    )
    assert any("High lymph node accumulation" in n for n in result.notes)


def test_auc_plasma_positive():
    result = _run()
    assert result.auc_plasma > 0


def test_auc_lymph_positive():
    result = _run()
    assert result.auc_lymph > 0


def test_concentrations_non_negative():
    result = _run()
    assert all(c >= 0 for c in result.c_plasma_mg_L)
    assert all(c >= 0 for c in result.c_lymph_mg_L)


# --- compare_formulation_lymph ---


def test_compare_formulation_returns_list():
    formulations = [
        {"name": "FormA", "k_lymph_in_per_h": 0.05},
        {"name": "FormB", "k_lymph_in_per_h": 0.2},
    ]
    results = compare_formulation_lymph("TestDrug", 100.0, formulations)
    assert isinstance(results, list)
    assert len(results) == 2


def test_compare_formulation_empty_raises():
    with pytest.raises(ValueError, match="formulations"):
        compare_formulation_lymph("TestDrug", 100.0, [])


def test_compare_formulation_different_results():
    formulations = [
        {"name": "FormA", "k_lymph_in_per_h": 0.01},
        {"name": "FormB", "k_lymph_in_per_h": 0.5},
    ]
    results = compare_formulation_lymph("TestDrug", 100.0, formulations)
    assert results[0].cmax_lymph != results[1].cmax_lymph
