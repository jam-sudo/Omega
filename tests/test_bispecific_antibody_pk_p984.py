"""Tests for Phase 984 — Bispecific antibody PK model (new API)."""

import pytest

from omega_pbpk.prediction.bispecific_antibody_pk import (
    BispecificAntibodyPKResult,
    simulate_bispecific_ab_pk,
)


def _default(**kwargs):
    defaults = dict(drug_name="TestBsAb", dose_mg=100.0)
    defaults.update(kwargs)
    return simulate_bispecific_ab_pk(**defaults)


def test_returns_bispecific_result():
    result = _default()
    assert isinstance(result, BispecificAntibodyPKResult)


def test_drug_name_stored():
    result = _default(drug_name="MyBsAb")
    assert result.drug_name == "MyBsAb"


def test_cmax_free_positive():
    result = _default()
    assert result.cmax_free_ug_mL > 0


def test_auc_positive():
    result = _default()
    assert result.auc_free_ug_h_per_mL > 0


def test_target1_occupancy_in_range():
    result = _default()
    assert 0.0 <= result.target1_occupancy_at_cmax <= 1.0


def test_target2_occupancy_in_range():
    result = _default()
    assert 0.0 <= result.target2_occupancy_at_cmax <= 1.0


def test_t_half_positive():
    result = _default()
    assert result.t_half_h > 0


def test_lower_kd1_higher_occupancy():
    """Tighter binding (lower Kd) should give higher occupancy at Cmax."""
    low_kd = _default(kd1_nM=0.1)
    high_kd = _default(kd1_nM=10.0)
    assert low_kd.target1_occupancy_at_cmax > high_kd.target1_occupancy_at_cmax


def test_higher_dose_higher_cmax():
    result_low = _default(dose_mg=10.0)
    result_high = _default(dose_mg=200.0)
    assert result_high.cmax_free_ug_mL > result_low.cmax_free_ug_mL


def test_iv_initial_concentration_positive():
    result = _default(route="iv")
    assert result.c_free_ug_mL[0] > 0.0


def test_sc_initial_concentration_zero():
    result = _default(route="sc")
    assert result.c_free_ug_mL[0] == pytest.approx(0.0)


def test_tmax_nonnegative():
    result = _default()
    assert result.tmax_h >= 0.0


def test_times_length_matches_concentrations():
    result = _default()
    assert len(result.times_h) == len(result.c_free_ug_mL)
    assert len(result.times_h) == len(result.c_target1_bound_ug_mL)
    assert len(result.times_h) == len(result.c_target2_bound_ug_mL)


def test_invalid_dose_raises():
    with pytest.raises(ValueError):
        _default(dose_mg=-5.0)


def test_zero_dose_raises():
    with pytest.raises(ValueError):
        _default(dose_mg=0.0)


def test_invalid_cl_raises():
    with pytest.raises(ValueError):
        _default(cl_L_per_day=-1.0)


def test_zero_cl_raises():
    with pytest.raises(ValueError):
        _default(cl_L_per_day=0.0)


def test_invalid_route_raises():
    with pytest.raises(ValueError):
        _default(route="oral")


def test_target_names_stored():
    result = _default(target1_name="HER2", target2_name="PD-L1")
    assert result.target1_name == "HER2"
    assert result.target2_name == "PD-L1"
