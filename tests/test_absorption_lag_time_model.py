"""
Tests for Phase 408 — Drug Absorption Lag Time Model
"""

import pytest

from omega_pbpk.core.absorption_lag_time_model import (
    AbsorptionLagResult,
    compare_lag_times,
    simulate_absorption_lag,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_sim(t_lag_h=0.0, dose_mg=500.0, ka_per_h=1.5,
             cl_L_per_h=5.0, vd_L=20.0, f_oral=1.0, t_end_h=24.0):
    return simulate_absorption_lag(
        drug_name="TestDrug",
        dose_mg=dose_mg,
        t_lag_h=t_lag_h,
        ka_per_h=ka_per_h,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
        f_oral=f_oral,
        t_end_h=t_end_h,
    )


# ---------------------------------------------------------------------------
# 1. Return type
# ---------------------------------------------------------------------------

def test_return_type():
    result = make_sim()
    assert isinstance(result, AbsorptionLagResult)


def test_result_is_dataclass():
    """AbsorptionLagResult should be a plain dataclass (mutable)."""
    result = make_sim()
    # Should be able to set attribute (plain dataclass, not frozen)
    result.drug_name = "Other"
    assert result.drug_name == "Other"


# ---------------------------------------------------------------------------
# 2. Basic fields
# ---------------------------------------------------------------------------

def test_times_and_concentrations_match_length():
    result = make_sim()
    assert len(result.times_h) == len(result.c_plasma_mg_L)


def test_times_start_at_zero():
    result = make_sim()
    assert result.times_h[0] == pytest.approx(0.0, abs=1e-6)


def test_concentrations_nonnegative():
    result = make_sim()
    assert all(c >= 0.0 for c in result.c_plasma_mg_L)


def test_cmax_positive():
    result = make_sim()
    assert result.cmax_mg_L > 0.0


def test_auc_positive():
    result = make_sim()
    assert result.auc_mg_h_per_L > 0.0


def test_t_half_positive():
    result = make_sim()
    assert result.t_half_apparent_h > 0.0


# ---------------------------------------------------------------------------
# 3. Tmax vs lag time
# ---------------------------------------------------------------------------

def test_tmax_no_lag_less_than_with_lag():
    """tmax with lag > tmax without lag."""
    no_lag = make_sim(t_lag_h=0.0)
    with_lag = make_sim(t_lag_h=2.0)
    assert with_lag.tmax_h > no_lag.tmax_h


def test_tmax_at_least_lag():
    """tmax >= t_lag."""
    for lag in [0.5, 1.0, 2.0, 4.0]:
        result = make_sim(t_lag_h=lag)
        assert result.tmax_h >= lag - 1e-6


def test_tmax_increases_with_lag():
    """Larger lag → later tmax."""
    lags = [0.0, 1.0, 2.0, 4.0]
    tmaxes = [make_sim(t_lag_h=lag_val).tmax_h for lag_val in lags]
    assert all(tmaxes[i] <= tmaxes[i + 1] for i in range(len(tmaxes) - 1))


# ---------------------------------------------------------------------------
# 4. Cmax effect of lag
# ---------------------------------------------------------------------------

def test_cmax_reduced_by_lag():
    """Cmax with lag <= Cmax without lag."""
    no_lag = make_sim(t_lag_h=0.0)
    with_lag = make_sim(t_lag_h=2.0)
    # Cmax can be slightly lower or equal; lag shifts absorption window
    # For typical 1-cpt, lag reduces apparent Cmax since elimination
    # window is different; allow for small numerical tolerance
    assert with_lag.cmax_mg_L <= no_lag.cmax_mg_L + 1e-3


def test_lag_time_effect_zero_for_no_lag():
    """lag_time_effect_pct = 0 when t_lag = 0."""
    result = make_sim(t_lag_h=0.0)
    assert result.lag_time_effect_pct == pytest.approx(0.0, abs=1e-6)


def test_lag_time_effect_increases_with_lag():
    """Larger lag → larger lag_time_effect_pct."""
    lags = [0.5, 1.0, 2.0, 4.0]
    effects = [make_sim(t_lag_h=lag_val).lag_time_effect_pct for lag_val in lags]
    # Should be non-decreasing
    for i in range(len(effects) - 1):
        assert effects[i] <= effects[i + 1] + 1.0  # allow 1% tolerance


def test_lag_time_effect_nonnegative():
    """lag_time_effect_pct >= 0."""
    for lag in [0.0, 0.5, 1.0, 2.0]:
        result = make_sim(t_lag_h=lag)
        assert result.lag_time_effect_pct >= 0.0


# ---------------------------------------------------------------------------
# 5. AUC approximately conserved
# ---------------------------------------------------------------------------

def test_auc_approximately_conserved_with_lag():
    """
    Total absorbed drug is the same regardless of lag (same dose, ka, F).
    AUC should be similar (within 5% for t_end >> t_half).
    """
    no_lag = make_sim(t_lag_h=0.0, t_end_h=48.0)
    with_lag = make_sim(t_lag_h=2.0, t_end_h=48.0)
    # AUC should be close since same total amount absorbed
    ratio = with_lag.auc_mg_h_per_L / no_lag.auc_mg_h_per_L
    assert 0.90 <= ratio <= 1.10


# ---------------------------------------------------------------------------
# 6. Absorption complete
# ---------------------------------------------------------------------------

def test_absorption_complete_after_lag():
    """absorption_complete_h > t_lag."""
    result = make_sim(t_lag_h=1.0)
    assert result.absorption_complete_h > 1.0


def test_absorption_complete_before_end():
    """absorption_complete_h <= t_end_h."""
    result = make_sim(t_lag_h=0.5, t_end_h=24.0)
    assert result.absorption_complete_h <= 24.0


# ---------------------------------------------------------------------------
# 7. compare_lag_times
# ---------------------------------------------------------------------------

def test_compare_returns_list():
    results = compare_lag_times("Drug", 500.0, 1.5, 5.0, 20.0)
    assert isinstance(results, list)


def test_compare_default_5_results():
    results = compare_lag_times("Drug", 500.0, 1.5, 5.0, 20.0)
    assert len(results) == 5


def test_compare_sorted_by_tmax():
    results = compare_lag_times("Drug", 500.0, 1.5, 5.0, 20.0)
    tmaxes = [r.tmax_h for r in results]
    assert tmaxes == sorted(tmaxes)


def test_compare_custom_lag_times():
    results = compare_lag_times("Drug", 500.0, 1.5, 5.0, 20.0,
                                lag_times_h=[0.0, 1.0, 3.0])
    assert len(results) == 3


def test_compare_all_are_lag_results():
    results = compare_lag_times("Drug", 500.0, 1.5, 5.0, 20.0)
    for r in results:
        assert isinstance(r, AbsorptionLagResult)


# ---------------------------------------------------------------------------
# 8. Validation errors
# ---------------------------------------------------------------------------

def test_invalid_dose():
    with pytest.raises(ValueError, match="dose_mg"):
        make_sim(dose_mg=-1.0)


def test_invalid_negative_lag():
    with pytest.raises(ValueError, match="t_lag"):
        make_sim(t_lag_h=-0.5)


def test_invalid_ka_zero():
    with pytest.raises(ValueError, match="ka_per_h"):
        make_sim(ka_per_h=0.0)


def test_invalid_cl():
    with pytest.raises(ValueError, match="cl"):
        make_sim(cl_L_per_h=0.0)


def test_invalid_vd():
    with pytest.raises(ValueError, match="vd"):
        make_sim(vd_L=-5.0)


def test_invalid_f_oral_zero():
    with pytest.raises(ValueError, match="f_oral"):
        make_sim(f_oral=0.0)


def test_invalid_f_oral_above_one():
    with pytest.raises(ValueError, match="f_oral"):
        make_sim(f_oral=1.5)
