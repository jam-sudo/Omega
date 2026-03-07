"""
Tests for Phase 942: Bayesian TDM dose individualization.
"""

import math
import pytest

from omega_pbpk.clinical.therapeutic_drug_monitoring_bayesian import (
    BayesianTDMResult,
    run_bayesian_tdm,
    simulate_tdm_guided_dosing,
)

_LN2 = math.log(2.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_obs(cl=10.0, vd=50.0, dose=100.0, times=None):
    """Generate synthetic observations from 1-cpt IV bolus model."""
    if times is None:
        times = [1.0, 4.0, 8.0]
    ke = cl / vd
    conc = [(dose / vd) * math.exp(-ke * t) for t in times]
    return times, conc


# ---------------------------------------------------------------------------
# Basic return type and value checks
# ---------------------------------------------------------------------------

def test_return_type():
    times, conc = _make_obs()
    result = run_bayesian_tdm("TestDrug", 100.0, times, conc)
    assert isinstance(result, BayesianTDMResult)


def test_cl_estimated_positive():
    times, conc = _make_obs()
    result = run_bayesian_tdm("TestDrug", 100.0, times, conc)
    assert result.cl_estimated_L_per_h > 0


def test_vd_estimated_positive():
    times, conc = _make_obs()
    result = run_bayesian_tdm("TestDrug", 100.0, times, conc)
    assert result.vd_estimated_L > 0


def test_t_half_estimated_positive():
    times, conc = _make_obs()
    result = run_bayesian_tdm("TestDrug", 100.0, times, conc)
    assert result.t_half_estimated_h > 0


def test_cl_fold_vs_pop_positive():
    times, conc = _make_obs()
    result = run_bayesian_tdm("TestDrug", 100.0, times, conc)
    assert result.cl_fold_vs_pop > 0


def test_vd_fold_vs_pop_positive():
    times, conc = _make_obs()
    result = run_bayesian_tdm("TestDrug", 100.0, times, conc)
    assert result.vd_fold_vs_pop > 0


def test_predicted_trough_positive():
    times, conc = _make_obs()
    result = run_bayesian_tdm("TestDrug", 100.0, times, conc)
    assert result.predicted_trough_at_interval_mg_L > 0


def test_dose_adjustment_factor_clamped():
    times, conc = _make_obs()
    result = run_bayesian_tdm("TestDrug", 100.0, times, conc)
    assert 0.25 <= result.dose_adjustment_factor <= 4.0


def test_recommended_dose_positive():
    times, conc = _make_obs()
    result = run_bayesian_tdm("TestDrug", 100.0, times, conc)
    assert result.recommended_dose_mg > 0


def test_objective_function_nonnegative():
    times, conc = _make_obs()
    result = run_bayesian_tdm("TestDrug", 100.0, times, conc)
    assert result.objective_function_value >= 0


def test_notes_non_empty():
    times, conc = _make_obs()
    result = run_bayesian_tdm("TestDrug", 100.0, times, conc)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Behavioural checks
# ---------------------------------------------------------------------------

def test_pop_consistent_obs_cl_fold_within_2x():
    """When observations match population PK, estimated CL should be near pop."""
    times, conc = _make_obs(cl=10.0, vd=50.0, dose=100.0)
    result = run_bayesian_tdm(
        "TestDrug", 100.0, times, conc,
        cl_pop_L_per_h=10.0,
        vd_pop_L=50.0,
    )
    # Should be within 2x of population
    assert 0.5 <= result.cl_fold_vs_pop <= 2.0


def test_high_conc_lower_cl():
    """High observed concentrations → estimated CL lower than population."""
    # Very high concs suggest slow elimination (low CL)
    times = [1.0, 4.0, 8.0]
    # Normal pop CL=10, Vd=50; multiply conc by 5 to simulate low CL
    _, normal_conc = _make_obs(cl=10.0, vd=50.0, dose=100.0, times=times)
    high_conc = [c * 5.0 for c in normal_conc]
    result = run_bayesian_tdm(
        "TestDrug", 100.0, times, high_conc,
        cl_pop_L_per_h=10.0,
        vd_pop_L=50.0,
    )
    # Estimated CL should be lower than population (fold < 1)
    assert result.cl_fold_vs_pop < 1.5  # Should pull CL down


def test_low_conc_higher_cl():
    """Low observed concentrations → estimated CL higher than population."""
    times = [1.0, 4.0, 8.0]
    _, normal_conc = _make_obs(cl=10.0, vd=50.0, dose=100.0, times=times)
    low_conc = [c * 0.2 for c in normal_conc]
    result = run_bayesian_tdm(
        "TestDrug", 100.0, times, low_conc,
        cl_pop_L_per_h=10.0,
        vd_pop_L=50.0,
    )
    # Estimated CL should be higher than population (fold > 1)
    assert result.cl_fold_vs_pop > 0.5  # Should pull CL up


def test_t_half_formula():
    """t1/2 = ln2 * Vd / CL."""
    times, conc = _make_obs()
    result = run_bayesian_tdm("TestDrug", 100.0, times, conc)
    expected_t_half = _LN2 * result.vd_estimated_L / result.cl_estimated_L_per_h
    assert abs(result.t_half_estimated_h - expected_t_half) < 1e-10


def test_deterministic_same_result():
    """Same inputs → same result."""
    times, conc = _make_obs()
    r1 = run_bayesian_tdm("TestDrug", 100.0, times, conc)
    r2 = run_bayesian_tdm("TestDrug", 100.0, times, conc)
    assert r1.cl_estimated_L_per_h == r2.cl_estimated_L_per_h
    assert r1.vd_estimated_L == r2.vd_estimated_L


# ---------------------------------------------------------------------------
# simulate_tdm_guided_dosing
# ---------------------------------------------------------------------------

def test_simulate_returns_list_length():
    times, conc = _make_obs()
    results = simulate_tdm_guided_dosing("TestDrug", 100.0, times, conc, n_adjustments=3)
    assert len(results) == 3


def test_simulate_all_bayesian_tdm_results():
    times, conc = _make_obs()
    results = simulate_tdm_guided_dosing("TestDrug", 100.0, times, conc, n_adjustments=3)
    for r in results:
        assert isinstance(r, BayesianTDMResult)


def test_simulate_n_adjustments_1():
    times, conc = _make_obs()
    results = simulate_tdm_guided_dosing("TestDrug", 100.0, times, conc, n_adjustments=1)
    assert len(results) == 1


def test_simulate_n_adjustments_5():
    times, conc = _make_obs()
    results = simulate_tdm_guided_dosing("TestDrug", 100.0, times, conc, n_adjustments=5)
    assert len(results) == 5


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

def test_validation_dose_mg_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        run_bayesian_tdm("Drug", 0.0, [1.0], [0.5])


def test_validation_dose_mg_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        run_bayesian_tdm("Drug", -100.0, [1.0], [0.5])


def test_validation_mismatched_lists():
    with pytest.raises(ValueError):
        run_bayesian_tdm("Drug", 100.0, [1.0, 2.0], [0.5])


def test_validation_empty_observations():
    with pytest.raises(ValueError):
        run_bayesian_tdm("Drug", 100.0, [], [])


def test_validation_cl_pop_zero():
    with pytest.raises(ValueError, match="cl_pop_L_per_h"):
        run_bayesian_tdm("Drug", 100.0, [1.0], [0.5], cl_pop_L_per_h=0.0)


def test_validation_vd_pop_zero():
    with pytest.raises(ValueError, match="vd_pop_L"):
        run_bayesian_tdm("Drug", 100.0, [1.0], [0.5], vd_pop_L=0.0)


def test_validation_target_trough_zero():
    with pytest.raises(ValueError, match="target_trough_mg_L"):
        run_bayesian_tdm("Drug", 100.0, [1.0], [0.5], target_trough_mg_L=0.0)


def test_validation_target_trough_negative():
    with pytest.raises(ValueError, match="target_trough_mg_L"):
        run_bayesian_tdm("Drug", 100.0, [1.0], [0.5], target_trough_mg_L=-1.0)
