"""Tests for nca_toolkit — Phase 484."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.analysis.nca_toolkit import (
    NCAResult,
    compute_accumulation_ratio,
    compute_aumc,
    compute_nca_full,
    compute_terminal_slope,
    superposition_steady_state,
)

# ---------------------------------------------------------------------------
# Helpers to build test profiles
# ---------------------------------------------------------------------------


def mono_exp_iv(ke: float, c0: float, times: list[float]) -> list[float]:
    """Monoexponential IV decline C(t) = C0 * exp(-ke*t)."""
    return [c0 * math.exp(-ke * t) for t in times]


TIMES_H = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0]
KE = 0.15  # hr^-1  → t_half = ln2/0.15 ≈ 4.62 h
C0 = 10.0  # mg/L
CONCS = mono_exp_iv(KE, C0, TIMES_H)


# ---------------------------------------------------------------------------
# NCAResult dataclass
# ---------------------------------------------------------------------------


def test_result_is_dataclass():
    r = compute_nca_full(TIMES_H, CONCS, dose_mg=100.0)
    assert isinstance(r, NCAResult)
    assert r.route == "iv"


# ---------------------------------------------------------------------------
# compute_terminal_slope
# ---------------------------------------------------------------------------


def test_terminal_slope_perfect_exponential():
    """Perfect monoexponential → lambda_z ≈ ke and r_squared ≈ 1."""
    lz, rsq = compute_terminal_slope(TIMES_H, CONCS, n_terminal=4)
    assert abs(lz - KE) / KE < 0.02  # within 2%
    assert rsq > 0.999


def test_terminal_slope_r_squared_one_mono_exp():
    lz, rsq = compute_terminal_slope(TIMES_H, CONCS, n_terminal=3)
    assert abs(rsq - 1.0) < 1e-6


def test_terminal_slope_invalid_n_terminal():
    with pytest.raises(ValueError, match="n_terminal"):
        compute_terminal_slope(TIMES_H, CONCS, n_terminal=1)


def test_terminal_slope_not_enough_points():
    with pytest.raises(ValueError):
        compute_terminal_slope([1.0, 2.0], [1.0, 0.5], n_terminal=3)


def test_terminal_slope_positive():
    lz, _ = compute_terminal_slope(TIMES_H, CONCS)
    assert lz > 0


# ---------------------------------------------------------------------------
# compute_aumc
# ---------------------------------------------------------------------------


def test_aumc_positive():
    aumc = compute_aumc(TIMES_H, CONCS)
    assert aumc > 0.0


def test_aumc_increases_with_longer_profile():
    aumc_short = compute_aumc(TIMES_H[:5], CONCS[:5])
    aumc_long = compute_aumc(TIMES_H, CONCS)
    assert aumc_long > aumc_short


def test_aumc_validation_mismatched_length():
    with pytest.raises(ValueError):
        compute_aumc([1.0, 2.0, 3.0], [1.0, 0.5])


def test_aumc_flat_profile():
    t = [0.0, 1.0, 2.0]
    c = [5.0, 5.0, 5.0]
    aumc = compute_aumc(t, c)
    # integral of t*5 from 0 to 2 = 5 * [t^2/2]_0^2 = 10
    assert abs(aumc - 10.0) < 0.01


# ---------------------------------------------------------------------------
# compute_nca_full — basic correctness
# ---------------------------------------------------------------------------


def test_t_half_matches_ke():
    r = compute_nca_full(TIMES_H, CONCS, dose_mg=100.0)
    expected_t_half = math.log(2.0) / KE
    assert abs(r.t_half_h - expected_t_half) / expected_t_half < 0.01  # within 1%


def test_auc_inf_greater_than_auc_last():
    r = compute_nca_full(TIMES_H, CONCS, dose_mg=100.0)
    assert r.auc_inf_mg_h_L > r.auc_last_mg_h_L


def test_mrt_equals_aumc_over_auc():
    r = compute_nca_full(TIMES_H, CONCS, dose_mg=100.0)
    ratio = r.aumc_mg_h2_L / r.auc_inf_mg_h_L
    assert abs(r.mrt_h - ratio) < 1e-6


def test_pct_extrapolated_low_when_long_profile():
    # Long profile → small extrapolation
    r = compute_nca_full(TIMES_H, CONCS, dose_mg=100.0)
    assert r.pct_auc_extrapolated < 20.0


def test_pct_extrapolated_higher_when_short_profile():
    # Short profile (ends at 4h, t_half ~4.6h) → larger extrapolation
    r = compute_nca_full(TIMES_H[:5], CONCS[:5], dose_mg=100.0)
    assert r.pct_auc_extrapolated > 0.0


def test_cmax_is_max_concentration():
    r = compute_nca_full(TIMES_H, CONCS, dose_mg=100.0)
    assert abs(r.cmax_mg_L - max(CONCS)) < 1e-9


def test_tmax_matches_cmax_time():
    r = compute_nca_full(TIMES_H, CONCS, dose_mg=100.0)
    assert abs(r.tmax_h - TIMES_H[CONCS.index(max(CONCS))]) < 1e-9


def test_cl_obs_positive():
    r = compute_nca_full(TIMES_H, CONCS, dose_mg=100.0)
    assert r.cl_obs_L_per_h > 0


def test_vd_obs_positive():
    r = compute_nca_full(TIMES_H, CONCS, dose_mg=100.0)
    assert r.vd_obs_L > 0


def test_oral_route_accepted():
    r = compute_nca_full(TIMES_H, CONCS, dose_mg=100.0, route="oral")
    assert r.route == "oral"


def test_risk_score_in_range():
    """risk_score → pct_auc_extrapolated should be in [0, 100]"""
    r = compute_nca_full(TIMES_H, CONCS, dose_mg=100.0)
    assert 0.0 <= r.pct_auc_extrapolated <= 100.0


# ---------------------------------------------------------------------------
# Input validation — compute_nca_full
# ---------------------------------------------------------------------------


def test_invalid_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        compute_nca_full(TIMES_H, CONCS, dose_mg=0.0)


def test_invalid_route_raises():
    with pytest.raises(ValueError, match="route"):
        compute_nca_full(TIMES_H, CONCS, dose_mg=100.0, route="sublingual")


def test_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        compute_nca_full(TIMES_H, CONCS[:5], dose_mg=100.0)


def test_negative_conc_raises():
    bad_concs = CONCS[:]
    bad_concs[3] = -1.0
    with pytest.raises(ValueError):
        compute_nca_full(TIMES_H, bad_concs, dose_mg=100.0)


# ---------------------------------------------------------------------------
# compute_accumulation_ratio
# ---------------------------------------------------------------------------


def test_accumulation_ratio_definition():
    ratio = compute_accumulation_ratio(auc_ss=200.0, auc_single=100.0)
    assert abs(ratio - 2.0) < 1e-9


def test_accumulation_ratio_no_accumulation():
    ratio = compute_accumulation_ratio(auc_ss=50.0, auc_single=50.0)
    assert abs(ratio - 1.0) < 1e-9


def test_accumulation_ratio_invalid_single():
    with pytest.raises(ValueError):
        compute_accumulation_ratio(100.0, 0.0)


# ---------------------------------------------------------------------------
# superposition_steady_state
# ---------------------------------------------------------------------------


def test_cmax_ss_greater_than_single_cmax():
    result = superposition_steady_state(TIMES_H, CONCS, dose_interval_h=12.0)
    single_cmax = max(CONCS)
    assert result["cmax_ss"] > single_cmax


def test_ss_returns_required_keys():
    result = superposition_steady_state(TIMES_H, CONCS, dose_interval_h=12.0)
    for key in ("cmax_ss", "cmin_ss", "auc_ss", "accumulation_ratio"):
        assert key in result


def test_accumulation_ratio_from_ss_geq_one():
    result = superposition_steady_state(TIMES_H, CONCS, dose_interval_h=12.0)
    # Allow slight numerical error from trapezoidal vs continuous integration
    assert result["accumulation_ratio"] >= 0.99


def test_auc_ss_positive():
    result = superposition_steady_state(TIMES_H, CONCS, dose_interval_h=12.0)
    assert result["auc_ss"] > 0.0


def test_ss_invalid_interval_raises():
    with pytest.raises(ValueError, match="dose_interval_h"):
        superposition_steady_state(TIMES_H, CONCS, dose_interval_h=0.0)


def test_ss_long_interval_less_accumulation():
    short_tau = superposition_steady_state(TIMES_H, CONCS, dose_interval_h=4.0)
    long_tau = superposition_steady_state(TIMES_H, CONCS, dose_interval_h=24.0)
    # Shorter interval → more accumulation
    assert short_tau["accumulation_ratio"] > long_tau["accumulation_ratio"]
