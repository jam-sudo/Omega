"""Tests for nonlinear_ppb module (Phase 280)."""

from __future__ import annotations

import pytest

from omega_pbpk.core.nonlinear_ppb import (
    fu_concentration_profile,
    nonlinear_fu,
    simulate_nonlinear_ppb_pk,
)

# ---------------------------------------------------------------------------
# nonlinear_fu — validation
# ---------------------------------------------------------------------------


def test_fu_invalid_bmax_zero():
    with pytest.raises(ValueError, match="bmax"):
        nonlinear_fu(1.0, bmax_mg_L=0.0, kd_mg_L=1.0)


def test_fu_invalid_bmax_negative():
    with pytest.raises(ValueError, match="bmax"):
        nonlinear_fu(1.0, bmax_mg_L=-1.0, kd_mg_L=1.0)


def test_fu_invalid_kd_zero():
    with pytest.raises(ValueError, match="kd"):
        nonlinear_fu(1.0, bmax_mg_L=1.0, kd_mg_L=0.0)


def test_fu_invalid_negative_concentration():
    with pytest.raises(ValueError, match="c_total"):
        nonlinear_fu(-1.0, bmax_mg_L=1.0, kd_mg_L=1.0)


def test_fu_invalid_n_sites_zero():
    with pytest.raises(ValueError, match="n_sites"):
        nonlinear_fu(1.0, bmax_mg_L=1.0, kd_mg_L=1.0, n_sites=0.0)


# ---------------------------------------------------------------------------
# nonlinear_fu — fu bounds and monotonicity
# ---------------------------------------------------------------------------


def test_fu_in_range():
    fu = nonlinear_fu(1.0, bmax_mg_L=5.0, kd_mg_L=2.0)
    assert 0.0 <= fu <= 1.0


def test_fu_zero_concentration_returns_linear_limit():
    # fu at c=0: 1/(1 + bmax/kd)
    bmax, kd = 4.0, 2.0
    expected = 1.0 / (1.0 + bmax / kd)
    fu = nonlinear_fu(0.0, bmax_mg_L=bmax, kd_mg_L=kd)
    assert abs(fu - expected) < 1e-6


def test_fu_increases_with_concentration():
    """At higher concentrations, more sites are saturated, so fu increases."""
    bmax, kd = 5.0, 1.0
    fu_low = nonlinear_fu(0.01, bmax_mg_L=bmax, kd_mg_L=kd)
    fu_mid = nonlinear_fu(5.0, bmax_mg_L=bmax, kd_mg_L=kd)
    fu_high = nonlinear_fu(50.0, bmax_mg_L=bmax, kd_mg_L=kd)
    assert fu_low < fu_mid < fu_high


def test_fu_approaches_one_at_high_concentration():
    bmax, kd = 1.0, 0.1
    fu = nonlinear_fu(1000.0, bmax_mg_L=bmax, kd_mg_L=kd)
    # At very high c, bound ~ bmax (fixed), so fu ~ (c - bmax) / c -> 1
    assert fu > 0.99


def test_fu_low_concentration_limit():
    """At very low concentration, fu should match the analytical linear limit."""
    bmax, kd = 3.0, 1.0
    fu_expected = 1.0 / (1.0 + bmax / kd)
    fu = nonlinear_fu(1e-6, bmax_mg_L=bmax, kd_mg_L=kd)
    assert abs(fu - fu_expected) < 1e-4


def test_fu_high_bmax_low_fu_at_low_conc():
    """High bmax => strong binding => low fu at low concentrations."""
    fu = nonlinear_fu(0.1, bmax_mg_L=100.0, kd_mg_L=1.0)
    assert fu < 0.1


def test_fu_mass_balance():
    """Verify bound + free = total."""
    c_total = 5.0
    bmax, kd = 3.0, 1.0
    fu = nonlinear_fu(c_total, bmax_mg_L=bmax, kd_mg_L=kd)
    c_free = fu * c_total
    bound = c_total - c_free
    # Check Langmuir: bound = bmax * c_free / (kd + c_free)
    langmuir_bound = bmax * c_free / (kd + c_free)
    assert abs(bound - langmuir_bound) < 1e-6


# ---------------------------------------------------------------------------
# fu_concentration_profile
# ---------------------------------------------------------------------------


def test_profile_result_frozen():
    result = fu_concentration_profile([1.0, 2.0], bmax_mg_L=2.0, kd_mg_L=1.0)
    with pytest.raises(Exception):
        result.fu_low_conc = 0.5  # type: ignore[misc]


def test_profile_lengths_match():
    c_range = [0.1, 1.0, 10.0, 100.0]
    result = fu_concentration_profile(c_range, bmax_mg_L=2.0, kd_mg_L=1.0)
    assert len(result.c_range) == len(c_range)
    assert len(result.fu_values) == len(c_range)


def test_profile_fu_increasing():
    c_range = [0.01, 0.1, 1.0, 10.0, 100.0]
    result = fu_concentration_profile(c_range, bmax_mg_L=2.0, kd_mg_L=1.0)
    fu_vals = list(result.fu_values)
    assert all(fu_vals[i] <= fu_vals[i + 1] for i in range(len(fu_vals) - 1))


def test_profile_fu_low_conc_matches_nonlinear_fu():
    bmax, kd = 2.0, 1.0
    result = fu_concentration_profile([1.0, 5.0], bmax_mg_L=bmax, kd_mg_L=kd)
    expected = nonlinear_fu(1e-6, bmax_mg_L=bmax, kd_mg_L=kd)
    assert abs(result.fu_low_conc - expected) < 1e-6


def test_profile_empty_raises():
    with pytest.raises(ValueError):
        fu_concentration_profile([], bmax_mg_L=1.0, kd_mg_L=1.0)


# ---------------------------------------------------------------------------
# simulate_nonlinear_ppb_pk — validation
# ---------------------------------------------------------------------------


def test_pk_invalid_dose():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_nonlinear_ppb_pk(
            dose_mg=0.0, vd_L=50.0, cl_L_per_h=5.0, bmax_mg_L=2.0, kd_mg_L=1.0
        )


def test_pk_invalid_vd():
    with pytest.raises(ValueError, match="vd_L"):
        simulate_nonlinear_ppb_pk(
            dose_mg=100.0, vd_L=0.0, cl_L_per_h=5.0, bmax_mg_L=2.0, kd_mg_L=1.0
        )


def test_pk_invalid_cl():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_nonlinear_ppb_pk(
            dose_mg=100.0, vd_L=50.0, cl_L_per_h=-1.0, bmax_mg_L=2.0, kd_mg_L=1.0
        )


# ---------------------------------------------------------------------------
# simulate_nonlinear_ppb_pk — results
# ---------------------------------------------------------------------------


def _run_pk(**kwargs):
    defaults = dict(
        dose_mg=100.0,
        vd_L=50.0,
        cl_L_per_h=5.0,
        bmax_mg_L=2.0,
        kd_mg_L=1.0,
        ka_per_h=1.5,
        t_end_h=24.0,
        dt_h=0.1,
    )
    defaults.update(kwargs)
    return simulate_nonlinear_ppb_pk(**defaults)


def test_pk_result_frozen():
    result = _run_pk()
    with pytest.raises(Exception):
        result.cmax = 0.0  # type: ignore[misc]


def test_pk_cmax_positive():
    result = _run_pk()
    assert result.cmax > 0.0


def test_pk_tmax_in_range():
    result = _run_pk()
    assert 0.0 <= result.tmax_h <= 24.0


def test_pk_auc_positive():
    result = _run_pk()
    assert result.auc > 0.0


def test_pk_times_length():
    result = _run_pk(t_end_h=12.0, dt_h=0.5)
    # n_steps = 12/0.5 = 24, so 25 points
    assert len(result.times_h) == 25
    assert len(result.c_plasma) == 25
    assert len(result.fu_profile) == 25


def test_pk_concentrations_nonnegative():
    result = _run_pk()
    assert all(c >= 0.0 for c in result.c_plasma)


def test_pk_fu_profile_in_range():
    result = _run_pk()
    assert all(0.0 <= fu <= 1.0 for fu in result.fu_profile)


def test_pk_higher_dose_higher_cmax():
    r1 = _run_pk(dose_mg=50.0)
    r2 = _run_pk(dose_mg=200.0)
    assert r2.cmax > r1.cmax


def test_pk_fu_increases_at_cmax():
    """At Cmax, fu should be higher than at near-zero concentration."""
    result = _run_pk(dose_mg=500.0, bmax_mg_L=2.0, kd_mg_L=1.0)
    tmax_idx = result.times_h.index(result.tmax_h)
    fu_at_cmax = result.fu_profile[tmax_idx]
    fu_linear = nonlinear_fu(1e-6, bmax_mg_L=2.0, kd_mg_L=1.0)
    assert fu_at_cmax >= fu_linear - 1e-6


def test_pk_notes_contains_cmax():
    result = _run_pk()
    assert "Cmax" in result.notes


def test_pk_start_concentration_zero():
    result = _run_pk()
    assert result.c_plasma[0] == 0.0
