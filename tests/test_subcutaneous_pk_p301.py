"""Tests for Phase 301: Subcutaneous absorption PK model.

Tests cover:
- Return type and field consistency
- Basic PK metrics (cmax > 0, auc > 0)
- Depot depletion (starts near 100%, ends near 0%)
- Effect of ka on tmax (higher ka -> earlier tmax)
- Effect of mw_kDa on auto lymph_fraction
- compare_formulations sorted descending by AUC
- Validation errors
"""

from __future__ import annotations

import math

import pytest

from omega_pbpk.core.subcutaneous_pk_p301 import (
    SubcutaneousPKResult,
    compare_formulations,
    simulate_sc_pk,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _base_result(**kwargs) -> SubcutaneousPKResult:
    """Return a default simulation result with optional overrides."""
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        ka_per_h=0.5,
        f_sc=0.8,
        cl_L_per_h=2.0,
        vd_L=30.0,
        t_end_h=24.0,
        dt_h=0.1,
        mw_kDa=0.5,
    )
    defaults.update(kwargs)
    return simulate_sc_pk(**defaults)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type():
    r = _base_result()
    assert isinstance(r, SubcutaneousPKResult)


def test_all_fields_present():
    r = _base_result()
    expected_fields = [
        "drug_name",
        "dose_mg",
        "ka_per_h",
        "f_sc",
        "mw_kDa",
        "lymph_fraction",
        "times_h",
        "c_plasma_mg_L",
        "depot_remaining_pct",
        "auc_plasma_mg_h_per_L",
        "cmax_plasma_mg_L",
        "tmax_plasma_h",
        "t_half_absorption_h",
        "lymphatic_contribution_pct",
        "notes",
    ]
    for field in expected_fields:
        assert hasattr(r, field), f"Missing field: {field}"


# ---------------------------------------------------------------------------
# Time array consistency
# ---------------------------------------------------------------------------


def test_times_array_length_consistent():
    r = _base_result()
    assert len(r.times_h) == len(r.c_plasma_mg_L)
    assert len(r.times_h) == len(r.depot_remaining_pct)


def test_times_start_at_zero():
    r = _base_result()
    assert r.times_h[0] == pytest.approx(0.0)


def test_times_end_at_t_end():
    r = _base_result(t_end_h=20.0)
    assert r.times_h[-1] == pytest.approx(20.0, rel=1e-3)


# ---------------------------------------------------------------------------
# Basic PK metrics
# ---------------------------------------------------------------------------


def test_cmax_positive():
    r = _base_result()
    assert r.cmax_plasma_mg_L > 0.0


def test_auc_positive():
    r = _base_result()
    assert r.auc_plasma_mg_h_per_L > 0.0


def test_tmax_within_range():
    r = _base_result(t_end_h=24.0)
    assert 0.0 <= r.tmax_plasma_h <= 24.0


# ---------------------------------------------------------------------------
# Depot depletion
# ---------------------------------------------------------------------------


def test_depot_starts_near_100():
    r = _base_result()
    assert r.depot_remaining_pct[0] == pytest.approx(80.0, abs=1.0)
    # Note: initial depot = dose * f_sc = 80% of dose


def test_depot_decreases_over_time():
    r = _base_result(t_end_h=48.0)
    # Depot should be mostly depleted by end
    assert r.depot_remaining_pct[-1] < 5.0


def test_depot_monotonically_decreasing():
    r = _base_result(t_end_h=24.0)
    for i in range(1, len(r.depot_remaining_pct)):
        assert r.depot_remaining_pct[i] <= r.depot_remaining_pct[i - 1] + 1e-9


# ---------------------------------------------------------------------------
# Effect of ka on tmax
# ---------------------------------------------------------------------------


def test_higher_ka_earlier_tmax():
    r_slow = simulate_sc_pk(
        drug_name="D",
        dose_mg=100,
        ka_per_h=0.2,
        f_sc=0.8,
        cl_L_per_h=1.0,
        vd_L=20.0,
        t_end_h=48.0,
        dt_h=0.1,
    )
    r_fast = simulate_sc_pk(
        drug_name="D",
        dose_mg=100,
        ka_per_h=2.0,
        f_sc=0.8,
        cl_L_per_h=1.0,
        vd_L=20.0,
        t_end_h=48.0,
        dt_h=0.1,
    )
    assert r_fast.tmax_plasma_h < r_slow.tmax_plasma_h


# ---------------------------------------------------------------------------
# f_sc effect on AUC
# ---------------------------------------------------------------------------


def test_higher_f_sc_higher_auc():
    r_low = simulate_sc_pk(
        drug_name="D",
        dose_mg=100,
        ka_per_h=0.5,
        f_sc=0.3,
        cl_L_per_h=2.0,
        vd_L=30.0,
        t_end_h=48.0,
    )
    r_high = simulate_sc_pk(
        drug_name="D",
        dose_mg=100,
        ka_per_h=0.5,
        f_sc=0.9,
        cl_L_per_h=2.0,
        vd_L=30.0,
        t_end_h=48.0,
    )
    assert r_high.auc_plasma_mg_h_per_L > r_low.auc_plasma_mg_h_per_L


# ---------------------------------------------------------------------------
# Molecular weight -> lymph fraction
# ---------------------------------------------------------------------------


def test_small_mol_zero_lymph_fraction():
    r = simulate_sc_pk(
        drug_name="Small",
        dose_mg=50,
        ka_per_h=1.0,
        f_sc=0.9,
        cl_L_per_h=2.0,
        vd_L=30.0,
        mw_kDa=0.3,
    )
    assert r.lymph_fraction == pytest.approx(0.0)


def test_large_mol_higher_lymph_fraction():
    r_small = simulate_sc_pk(
        drug_name="Small",
        dose_mg=50,
        ka_per_h=0.1,
        f_sc=0.8,
        cl_L_per_h=1.0,
        vd_L=50.0,
        mw_kDa=1.0,
    )
    r_large = simulate_sc_pk(
        drug_name="Large",
        dose_mg=50,
        ka_per_h=0.1,
        f_sc=0.8,
        cl_L_per_h=1.0,
        vd_L=50.0,
        mw_kDa=30.0,
    )
    assert r_large.lymph_fraction > r_small.lymph_fraction


def test_mw_20_kDa_auto_lymph_fraction():
    r = simulate_sc_pk(
        drug_name="Biologic",
        dose_mg=10,
        ka_per_h=0.05,
        f_sc=0.7,
        cl_L_per_h=0.5,
        vd_L=10.0,
        mw_kDa=20.0,
    )
    expected_lf = min(0.8, 20.0 / 30.0)
    assert r.lymph_fraction == pytest.approx(expected_lf, rel=1e-6)


def test_lymphatic_contribution_pct_matches():
    r = _base_result(mw_kDa=15.0)
    assert r.lymphatic_contribution_pct == pytest.approx(r.lymph_fraction * 100.0)


def test_explicit_lymph_fraction_override():
    r = simulate_sc_pk(
        drug_name="D",
        dose_mg=100,
        ka_per_h=0.5,
        f_sc=0.8,
        cl_L_per_h=2.0,
        vd_L=30.0,
        mw_kDa=100.0,  # would give 0.8 auto
        lymph_fraction=0.2,
    )
    assert r.lymph_fraction == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Half-life calculation
# ---------------------------------------------------------------------------


def test_t_half_absorption():
    r = simulate_sc_pk(
        drug_name="D",
        dose_mg=100,
        ka_per_h=1.0,
        f_sc=0.8,
        cl_L_per_h=2.0,
        vd_L=30.0,
    )
    expected = math.log(2.0) / 1.0
    assert r.t_half_absorption_h == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# compare_formulations
# ---------------------------------------------------------------------------


def test_compare_formulations_return_type():
    results = compare_formulations(
        drug_name="D",
        dose_mg=100.0,
        ka_list=[0.3, 0.8, 1.5],
        f_sc_list=[0.6, 0.8, 0.9],
        cl_L_per_h=2.0,
        vd_L=30.0,
    )
    assert isinstance(results, list)
    assert len(results) == 3
    assert all(isinstance(r, SubcutaneousPKResult) for r in results)


def test_compare_formulations_sorted_descending():
    results = compare_formulations(
        drug_name="D",
        dose_mg=100.0,
        ka_list=[0.2, 0.5, 1.0],
        f_sc_list=[0.5, 0.8, 0.9],
        cl_L_per_h=2.0,
        vd_L=30.0,
        t_end_h=48.0,
    )
    aucs = [r.auc_plasma_mg_h_per_L for r in results]
    assert aucs == sorted(aucs, reverse=True)


def test_compare_formulations_single_entry():
    results = compare_formulations(
        drug_name="D",
        dose_mg=50.0,
        ka_list=[0.5],
        f_sc_list=[0.8],
        cl_L_per_h=1.0,
        vd_L=20.0,
    )
    assert len(results) == 1


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_sc_pk("D", dose_mg=0.0, ka_per_h=0.5, f_sc=0.8, cl_L_per_h=2.0, vd_L=30.0)


def test_validation_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_sc_pk("D", dose_mg=-10.0, ka_per_h=0.5, f_sc=0.8, cl_L_per_h=2.0, vd_L=30.0)


def test_validation_ka_zero():
    with pytest.raises(ValueError, match="ka_per_h"):
        simulate_sc_pk("D", dose_mg=100.0, ka_per_h=0.0, f_sc=0.8, cl_L_per_h=2.0, vd_L=30.0)


def test_validation_f_sc_zero():
    with pytest.raises(ValueError, match="f_sc"):
        simulate_sc_pk("D", dose_mg=100.0, ka_per_h=0.5, f_sc=0.0, cl_L_per_h=2.0, vd_L=30.0)


def test_validation_f_sc_above_one():
    with pytest.raises(ValueError, match="f_sc"):
        simulate_sc_pk("D", dose_mg=100.0, ka_per_h=0.5, f_sc=1.5, cl_L_per_h=2.0, vd_L=30.0)


def test_validation_cl_zero():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_sc_pk("D", dose_mg=100.0, ka_per_h=0.5, f_sc=0.8, cl_L_per_h=0.0, vd_L=30.0)


def test_validation_vd_negative():
    with pytest.raises(ValueError, match="vd_L"):
        simulate_sc_pk("D", dose_mg=100.0, ka_per_h=0.5, f_sc=0.8, cl_L_per_h=2.0, vd_L=-5.0)


def test_validation_mw_zero():
    with pytest.raises(ValueError, match="mw_kDa"):
        simulate_sc_pk(
            "D", dose_mg=100.0, ka_per_h=0.5, f_sc=0.8, cl_L_per_h=2.0, vd_L=30.0, mw_kDa=0.0
        )


def test_validation_lymph_fraction_out_of_range():
    with pytest.raises(ValueError, match="lymph_fraction"):
        simulate_sc_pk(
            "D",
            dose_mg=100.0,
            ka_per_h=0.5,
            f_sc=0.8,
            cl_L_per_h=2.0,
            vd_L=30.0,
            lymph_fraction=1.5,
        )


def test_validation_compare_formulations_mismatched_lists():
    with pytest.raises(ValueError):
        compare_formulations(
            drug_name="D",
            dose_mg=100.0,
            ka_list=[0.5, 1.0],
            f_sc_list=[0.8],
            cl_L_per_h=2.0,
            vd_L=30.0,
        )


def test_validation_compare_formulations_empty_lists():
    with pytest.raises(ValueError):
        compare_formulations(
            drug_name="D",
            dose_mg=100.0,
            ka_list=[],
            f_sc_list=[],
            cl_L_per_h=2.0,
            vd_L=30.0,
        )


# ---------------------------------------------------------------------------
# Notes field
# ---------------------------------------------------------------------------


def test_notes_is_string():
    r = _base_result()
    assert isinstance(r.notes, str)
    assert len(r.notes) > 0


def test_notes_contains_drug_name():
    r = simulate_sc_pk(
        drug_name="Adalimumab",
        dose_mg=40.0,
        ka_per_h=0.05,
        f_sc=0.64,
        cl_L_per_h=0.01,
        vd_L=8.0,
        mw_kDa=148.0,
    )
    assert "Adalimumab" in r.notes
