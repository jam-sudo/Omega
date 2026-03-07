"""Tests for Phase 1029 — Drug Synovial Fluid PK."""

import pytest

from omega_pbpk.core.synovial_fluid_pk_p1029 import (
    SynovialFluidPKResult,
    simulate_synovial_fluid_pk,
)

# --- Return type ---


def test_returns_synovial_fluid_pk_result():
    result = simulate_synovial_fluid_pk("TestDrug", dose_mg=100.0)
    assert isinstance(result, SynovialFluidPKResult)


# --- Systemic route ---


def test_systemic_plasma_nonzero_at_t0():
    result = simulate_synovial_fluid_pk("DrugA", dose_mg=100.0, route="systemic")
    assert result.c_plasma_mg_L[0] > 0.0


def test_systemic_synovial_zero_at_t0():
    result = simulate_synovial_fluid_pk("DrugA", dose_mg=100.0, route="systemic")
    assert result.c_synovial_mg_L[0] == 0.0


# --- Intraarticular route ---


def test_ia_synovial_nonzero_at_t0():
    result = simulate_synovial_fluid_pk("DrugB", dose_mg=50.0, route="intraarticular")
    assert result.c_synovial_mg_L[0] > 0.0


def test_ia_plasma_zero_at_t0():
    result = simulate_synovial_fluid_pk("DrugB", dose_mg=50.0, route="intraarticular")
    assert result.c_plasma_mg_L[0] == 0.0


# --- AUC positivity ---


def test_auc_plasma_positive_systemic():
    result = simulate_synovial_fluid_pk("DrugA", dose_mg=100.0, route="systemic")
    assert result.auc_plasma > 0.0


def test_auc_synovial_positive_systemic():
    result = simulate_synovial_fluid_pk("DrugA", dose_mg=100.0, route="systemic")
    assert result.auc_synovial > 0.0


def test_auc_plasma_positive_ia():
    result = simulate_synovial_fluid_pk("DrugB", dose_mg=50.0, route="intraarticular")
    assert result.auc_plasma > 0.0


def test_auc_synovial_positive_ia():
    result = simulate_synovial_fluid_pk("DrugB", dose_mg=50.0, route="intraarticular")
    assert result.auc_synovial > 0.0


# --- Cmax / Tmax ---


def test_cmax_synovial_positive():
    result = simulate_synovial_fluid_pk("DrugA", dose_mg=100.0, route="systemic")
    assert result.cmax_synovial_mg_L > 0.0


def test_tmax_synovial_nonneg():
    result = simulate_synovial_fluid_pk("DrugA", dose_mg=100.0, route="systemic")
    assert result.tmax_synovial_h >= 0.0


# --- IA gives higher synovial AUC ---


def test_ia_higher_synovial_auc_than_systemic():
    sys_result = simulate_synovial_fluid_pk("DrugC", dose_mg=100.0, route="systemic")
    ia_result = simulate_synovial_fluid_pk("DrugC", dose_mg=100.0, route="intraarticular")
    assert ia_result.auc_synovial > sys_result.auc_synovial


# --- Synovial-to-plasma AUC ratio ---


def test_synovial_to_plasma_auc_ratio_positive():
    result = simulate_synovial_fluid_pk("DrugA", dose_mg=100.0, route="systemic")
    assert result.synovial_to_plasma_auc_ratio > 0.0


# --- Half-life ---


def test_t_half_synovial_positive():
    result = simulate_synovial_fluid_pk("DrugA", dose_mg=100.0, route="systemic")
    assert result.t_half_synovial_h > 0.0


# --- Higher MW → lower synovial exposure for systemic route ---


def test_higher_mw_lower_synovial_cmax_systemic():
    """Higher MW reduces membrane permeability, yielding lower peak synovial concentration."""
    low_mw = simulate_synovial_fluid_pk("SmallMol", dose_mg=100.0, route="systemic", mw_Da=200.0)
    high_mw = simulate_synovial_fluid_pk("LargeMol", dose_mg=100.0, route="systemic", mw_Da=5000.0)
    # Lower permeability factor (high MW) leads to lower peak synovial concentration
    assert low_mw.cmax_synovial_mg_L > high_mw.cmax_synovial_mg_L


# --- Notes nonempty ---


def test_notes_nonempty():
    result = simulate_synovial_fluid_pk("DrugA", dose_mg=100.0)
    assert len(result.notes) > 0


# --- Validation errors ---


def test_zero_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_synovial_fluid_pk("DrugA", dose_mg=0.0)


def test_zero_cl_raises():
    with pytest.raises(ValueError, match="cl_sys_L_per_h"):
        simulate_synovial_fluid_pk("DrugA", dose_mg=100.0, cl_sys_L_per_h=0.0)


def test_invalid_route_raises():
    with pytest.raises(ValueError, match="route"):
        simulate_synovial_fluid_pk("DrugA", dose_mg=100.0, route="oral")


def test_zero_kp_raises():
    with pytest.raises(ValueError, match="kp_synovial"):
        simulate_synovial_fluid_pk("DrugA", dose_mg=100.0, kp_synovial=0.0)


# --- Additional structural checks ---


def test_times_h_starts_at_zero():
    result = simulate_synovial_fluid_pk("DrugA", dose_mg=100.0)
    assert result.times_h[0] == 0.0


def test_lists_same_length():
    result = simulate_synovial_fluid_pk("DrugA", dose_mg=100.0)
    assert len(result.times_h) == len(result.c_plasma_mg_L) == len(result.c_synovial_mg_L)
