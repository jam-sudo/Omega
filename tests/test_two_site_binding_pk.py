"""Tests for Phase 950 — Two-Site Receptor Binding PK Model."""

import pytest
from omega_pbpk.core.two_site_binding_pk import (
    TwoSiteBindingResult,
    simulate_two_site_binding,
    optimize_selectivity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _sim(**kwargs):
    defaults = dict(drug_name="TestDrug", dose_mg=10.0, mw_Da=300.0, dt_h=0.05)
    defaults.update(kwargs)
    return simulate_two_site_binding(**defaults)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------
def test_return_type_is_two_site_binding_result():
    result = _sim()
    assert isinstance(result, TwoSiteBindingResult)


# ---------------------------------------------------------------------------
# Initial conditions and lists
# ---------------------------------------------------------------------------
def test_c_free_starts_positive():
    result = _sim()
    assert result.c_free_nM[0] > 0


def test_peak_free_nM_positive():
    result = _sim()
    assert result.peak_free_nM > 0


def test_peak_r1_occupancy_range():
    result = _sim()
    assert 0.0 <= result.peak_r1_occupancy <= 1.0


def test_peak_r2_occupancy_range():
    result = _sim()
    assert 0.0 <= result.peak_r2_occupancy <= 1.0


def test_selectivity_index_positive():
    result = _sim()
    assert result.selectivity_index > 0


def test_auc_free_positive():
    result = _sim()
    assert result.auc_free_nM_h > 0


def test_time_above_kd1_non_negative():
    result = _sim()
    assert result.time_above_kd1_h >= 0


def test_notes_non_empty():
    result = _sim()
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# List length consistency
# ---------------------------------------------------------------------------
def test_times_and_c_free_same_length():
    result = _sim()
    assert len(result.times_h) == len(result.c_free_nM)


def test_c_bound_r1_same_length():
    result = _sim()
    assert len(result.c_bound_r1_nM) == len(result.c_free_nM)


def test_c_bound_r2_same_length():
    result = _sim()
    assert len(result.c_bound_r2_nM) == len(result.c_free_nM)


# ---------------------------------------------------------------------------
# Physics / model direction
# ---------------------------------------------------------------------------
def test_weaker_site1_gives_lower_r1_occupancy():
    """Higher Kd1 (weaker affinity) should reduce peak R1 occupancy."""
    tight = simulate_two_site_binding("drug", dose_mg=10.0, kd1_nM=1.0, dt_h=0.05)
    weak = simulate_two_site_binding("drug", dose_mg=10.0, kd1_nM=50.0, dt_h=0.05)
    assert tight.peak_r1_occupancy >= weak.peak_r1_occupancy


def test_very_high_dose_both_sites_occupied():
    """Extremely high dose should occupy both sites substantially."""
    result = simulate_two_site_binding("drug", dose_mg=10000.0, dt_h=0.05)
    assert result.peak_r1_occupancy > 0.5
    assert result.peak_r2_occupancy > 0.1


def test_low_dose_higher_selectivity():
    """Low dose should preferentially engage high-affinity site (R1), giving higher selectivity."""
    low = simulate_two_site_binding("drug", dose_mg=1.0, dt_h=0.05)
    high = simulate_two_site_binding("drug", dose_mg=500.0, dt_h=0.05)
    assert low.selectivity_index >= high.selectivity_index


def test_high_dose_lower_selectivity():
    """High dose should saturate off-target, reducing selectivity."""
    low = simulate_two_site_binding("drug", dose_mg=1.0, dt_h=0.05)
    high = simulate_two_site_binding("drug", dose_mg=500.0, dt_h=0.05)
    assert high.selectivity_index <= low.selectivity_index


# ---------------------------------------------------------------------------
# optimize_selectivity
# ---------------------------------------------------------------------------
def test_optimize_selectivity_returns_five_by_default():
    results = optimize_selectivity("drug", mw_Da=300.0, dt_h=0.05)
    assert len(results) == 5


def test_optimize_selectivity_sorted_descending():
    results = optimize_selectivity("drug", mw_Da=300.0, dt_h=0.05)
    si = [r.selectivity_index for r in results]
    assert si == sorted(si, reverse=True)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------
def test_validation_dose_mg_zero_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_two_site_binding("drug", dose_mg=0.0)


def test_validation_dose_mg_negative_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_two_site_binding("drug", dose_mg=-1.0)


def test_validation_mw_da_zero_raises():
    with pytest.raises(ValueError, match="mw_Da"):
        simulate_two_site_binding("drug", dose_mg=10.0, mw_Da=0.0)


def test_validation_mw_da_negative_raises():
    with pytest.raises(ValueError, match="mw_Da"):
        simulate_two_site_binding("drug", dose_mg=10.0, mw_Da=-100.0)


def test_validation_kd1_zero_raises():
    with pytest.raises(ValueError, match="kd1_nM"):
        simulate_two_site_binding("drug", dose_mg=10.0, kd1_nM=0.0)


def test_validation_kd2_zero_raises():
    with pytest.raises(ValueError, match="kd2_nM"):
        simulate_two_site_binding("drug", dose_mg=10.0, kd2_nM=0.0)


def test_validation_cl_zero_raises():
    with pytest.raises(ValueError, match="cl_free_L_per_h"):
        simulate_two_site_binding("drug", dose_mg=10.0, cl_free_L_per_h=0.0)


def test_validation_vd_zero_raises():
    with pytest.raises(ValueError, match="vd_L"):
        simulate_two_site_binding("drug", dose_mg=10.0, vd_L=0.0)
