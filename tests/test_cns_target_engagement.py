"""Tests for CNS target engagement model."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.cns_target_engagement import (
    CNSTargetEngagementResult,
    occupancy_dose_response,
    simulate_cns_target_engagement,
)

# ---- Helpers ----


def _default_result(**kwargs) -> CNSTargetEngagementResult:
    defaults = dict(drug_name="TestDrug", dose_mg=10.0)
    defaults.update(kwargs)
    return simulate_cns_target_engagement(**defaults)


# ---- Tests ----


def test_basic_simulation_returns_result():
    r = _default_result()
    assert isinstance(r, CNSTargetEngagementResult)


def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_cns_target_engagement("X", dose_mg=0)


def test_validation_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_cns_target_engagement("X", dose_mg=-1)


def test_validation_cl_zero():
    with pytest.raises(ValueError, match="cl_plasma"):
        simulate_cns_target_engagement("X", 10.0, cl_plasma_L_per_h=0)


def test_validation_bmax_zero():
    with pytest.raises(ValueError, match="bmax"):
        simulate_cns_target_engagement("X", 10.0, bmax_nM=0)


def test_validation_kon_zero():
    with pytest.raises(ValueError, match="kon"):
        simulate_cns_target_engagement("X", 10.0, kon_per_nM_per_h=0)


def test_validation_koff_zero():
    with pytest.raises(ValueError, match="koff"):
        simulate_cns_target_engagement("X", 10.0, koff_per_h=0)


def test_peak_occupancy_in_range():
    r = _default_result()
    assert 0.0 <= r.peak_occupancy_pct <= 100.0


def test_occupancy_values_in_range():
    r = _default_result()
    assert all(0.0 <= v <= 100.0 for v in r.receptor_occupancy)


def test_higher_efflux_lower_brain_ratio():
    """Higher BBB efflux should reduce brain/plasma ratio."""
    r_low_eff = _default_result(k_bbb_out_per_h=0.1)
    r_high_eff = _default_result(k_bbb_out_per_h=5.0)
    assert r_high_eff.brain_plasma_ratio < r_low_eff.brain_plasma_ratio


def test_brain_plasma_ratio_positive():
    r = _default_result()
    assert r.brain_plasma_ratio > 0


def test_time_above_50pct_non_negative():
    r = _default_result()
    assert r.time_above_50pct_h >= 0.0


def test_kd_equals_koff_over_kon():
    r = _default_result(koff_per_h=2.0, kon_per_nM_per_h=0.05)
    assert abs(r.kd_nM - 2.0 / 0.05) < 1e-6


def test_higher_dose_higher_occupancy():
    """Use high Kd and low doses so occupancy doesn't saturate."""
    params = dict(
        kon_per_nM_per_h=0.001, koff_per_h=10.0, bmax_nM=1000.0, vbrain_L=1.0, k_bbb_in_per_h=0.01
    )
    r_low = _default_result(dose_mg=0.01, **params)
    r_high = _default_result(dose_mg=10.0, **params)
    assert r_high.peak_occupancy_pct > r_low.peak_occupancy_pct


def test_occupancy_drops_after_elimination():
    """After drug is eliminated, occupancy should decline toward zero."""
    r = _default_result(t_end_h=72.0, dose_mg=10.0)
    # Last value should be less than peak
    assert r.receptor_occupancy[-1] < r.peak_occupancy_pct


def test_auc_occupancy_positive():
    r = _default_result()
    assert r.auc_occupancy > 0


def test_rc_returns_to_zero_long_sim():
    """After long simulation, receptor complex should be near zero."""
    r = _default_result(t_end_h=200.0, dose_mg=5.0)
    assert r.rc_nM[-1] < 1.0  # near zero


def test_dose_response_length():
    doses = [1.0, 5.0, 10.0, 50.0]
    results = occupancy_dose_response("TestDrug", doses)
    assert len(results) == len(doses)


def test_dose_response_monotonic():
    doses = [1.0, 10.0, 100.0]
    results = occupancy_dose_response("TestDrug", doses)
    peaks = [r.peak_occupancy_pct for r in results]
    assert peaks == sorted(peaks)


def test_oral_peak_delayed_vs_iv():
    r_iv = _default_result(route="iv")
    r_oral = _default_result(route="oral", ka_per_h=1.0)
    assert r_oral.tmax_occupancy_h >= r_iv.tmax_occupancy_h


def test_tmax_occupancy_in_range():
    r = _default_result()
    assert 0.0 <= r.tmax_occupancy_h <= 24.0


def test_peak_brain_positive():
    r = _default_result()
    assert max(r.c_brain_nM) > 0


def test_free_receptor_at_t0_approx_bmax():
    r = _default_result(route="iv")
    # At t=0, no binding has occurred yet
    assert abs(r.free_receptor_nM[0] - 100.0) < 1e-6


def test_no_bbb_penetration_no_occupancy():
    r = _default_result(k_bbb_in_per_h=0.0)
    assert r.peak_occupancy_pct < 0.1


def test_high_koff_low_occupancy():
    """High koff with low brain conc → low occupancy."""
    r = _default_result(
        koff_per_h=100.0,
        kon_per_nM_per_h=0.001,
        dose_mg=0.001,
        vbrain_L=1.0,
        k_bbb_in_per_h=0.01,
    )
    assert r.peak_occupancy_pct < 50.0


def test_notes_contain_drug_name():
    r = _default_result(drug_name="Donepezil")
    assert "Donepezil" in r.notes


def test_times_length_greater_than_one():
    r = _default_result()
    assert len(r.times_h) > 1
