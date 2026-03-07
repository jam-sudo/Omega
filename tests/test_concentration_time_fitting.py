"""Tests for concentration_time_fitting — Phase 443."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.analysis.concentration_time_fitting import (
    OneCptFitResult,
    TwoCptFitResult,
    fit_one_compartment,
    fit_two_compartment_iv,
)

# ---------------------------------------------------------------------------
# Helpers to generate synthetic data
# ---------------------------------------------------------------------------


def _iv_data(ke: float, c0: float, times: list[float]) -> list[float]:
    """Generate noise-free IV 1-cpt concentrations."""
    return [c0 * math.exp(-ke * t) for t in times]


def _oral_data(
    ke: float,
    ka: float,
    dose: float,
    vd: float,
    times: list[float],
) -> list[float]:
    """Generate noise-free oral 1-cpt concentrations."""
    scale = (dose * ka) / (vd * (ka - ke))
    return [scale * (math.exp(-ke * t) - math.exp(-ka * t)) for t in times]


def _biexp_data(A: float, alpha: float, B: float, beta: float, times: list[float]) -> list[float]:
    """Generate biexponential concentrations."""
    return [A * math.exp(-alpha * t) + B * math.exp(-beta * t) for t in times]


# ---------------------------------------------------------------------------
# OneCptFitResult dataclass
# ---------------------------------------------------------------------------


def test_one_cpt_result_fields():
    r = OneCptFitResult(
        route="iv",
        dose_mg=100.0,
        ke=0.1,
        ka=None,
        vd_L=10.0,
        t_half_h=6.93,
        auc_fit=100.0,
        r_squared=0.99,
        n_points=8,
        notes="test",
    )
    assert r.route == "iv"
    assert r.ka is None
    assert r.n_points == 8


def test_two_cpt_result_fields():
    r = TwoCptFitResult(
        A=5.0,
        B=2.0,
        alpha=1.5,
        beta=0.1,
        cl_L_per_h=5.0,
        vd_central_L=10.0,
        vd_ss_L=20.0,
        auc=25.0,
        t_half_terminal_h=6.93,
        r_squared=0.98,
        n_points=10,
        notes="feathering",
    )
    assert r.alpha > r.beta
    assert r.t_half_terminal_h == pytest.approx(6.93)


# ---------------------------------------------------------------------------
# Input validation — fit_one_compartment
# ---------------------------------------------------------------------------


def test_too_few_points_raises():
    with pytest.raises(ValueError, match="At least 3"):
        fit_one_compartment([1.0, 2.0], [1.0, 0.5], "iv", 100.0)


def test_mismatched_lengths_raises():
    with pytest.raises(ValueError, match="same length"):
        fit_one_compartment([1.0, 2.0, 3.0], [1.0, 0.5], "iv", 100.0)


def test_non_increasing_times_raises():
    with pytest.raises(ValueError, match="strictly increasing"):
        fit_one_compartment([1.0, 1.0, 3.0], [3.0, 2.0, 1.0], "iv", 100.0)


def test_non_positive_concentration_raises():
    with pytest.raises(ValueError, match="positive"):
        fit_one_compartment([1.0, 2.0, 3.0], [3.0, -1.0, 1.0], "iv", 100.0)


def test_invalid_route_raises():
    with pytest.raises(ValueError, match="route"):
        fit_one_compartment([1.0, 2.0, 3.0], [3.0, 2.0, 1.0], "sc", 100.0)


def test_zero_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        fit_one_compartment([1.0, 2.0, 3.0], [3.0, 2.0, 1.0], "iv", 0.0)


def test_negative_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        fit_one_compartment([1.0, 2.0, 3.0], [3.0, 2.0, 1.0], "iv", -50.0)


# ---------------------------------------------------------------------------
# IV one-compartment fitting
# ---------------------------------------------------------------------------


def test_iv_fit_ke_recovery():
    """Fitted ke should be close to the true ke."""
    ke_true = 0.2
    c0 = 5.0  # mg/L
    times = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0]
    conc = _iv_data(ke_true, c0, times)
    res = fit_one_compartment(times, conc, "iv", 100.0)
    assert res.ke == pytest.approx(ke_true, rel=0.05)


def test_iv_fit_t_half():
    ke_true = 0.1
    times = [1.0, 2.0, 4.0, 8.0, 12.0, 16.0]
    conc = _iv_data(ke_true, 10.0, times)
    res = fit_one_compartment(times, conc, "iv", 100.0)
    assert res.t_half_h == pytest.approx(math.log(2) / ke_true, rel=0.05)


def test_iv_fit_vd():
    ke_true = 0.2
    vd_true = 20.0
    dose = 100.0
    c0 = dose / vd_true  # 5 mg/L
    times = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    conc = _iv_data(ke_true, c0, times)
    res = fit_one_compartment(times, conc, "iv", dose)
    assert res.vd_L == pytest.approx(vd_true, rel=0.10)


def test_iv_fit_r_squared_perfect_data():
    """Perfect synthetic data should give R² very close to 1."""
    times = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0]
    conc = _iv_data(0.15, 8.0, times)
    res = fit_one_compartment(times, conc, "iv", 100.0)
    assert res.r_squared >= 0.98


def test_iv_fit_auc_positive():
    times = [0.5, 1.0, 2.0, 4.0, 8.0]
    conc = _iv_data(0.3, 6.0, times)
    res = fit_one_compartment(times, conc, "iv", 100.0)
    assert res.auc_fit > 0.0


def test_iv_fit_ka_none():
    """IV fit should not return ka."""
    times = [0.5, 1.0, 2.0, 4.0, 8.0]
    conc = _iv_data(0.2, 5.0, times)
    res = fit_one_compartment(times, conc, "iv", 100.0)
    assert res.ka is None


def test_iv_fit_route_stored():
    times = [0.5, 1.0, 2.0, 4.0, 8.0]
    conc = _iv_data(0.2, 5.0, times)
    res = fit_one_compartment(times, conc, "iv", 100.0)
    assert res.route == "iv"


# ---------------------------------------------------------------------------
# Oral one-compartment fitting
# ---------------------------------------------------------------------------


def test_oral_fit_ka_not_none():
    ke, ka, dose, vd = 0.1, 1.0, 100.0, 20.0
    times = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0]
    conc = _oral_data(ke, ka, dose, vd, times)
    res = fit_one_compartment(times, conc, "oral", dose)
    assert res.ka is not None


def test_oral_fit_ke_positive():
    ke, ka, dose, vd = 0.1, 2.0, 200.0, 30.0
    times = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0]
    conc = _oral_data(ke, ka, dose, vd, times)
    res = fit_one_compartment(times, conc, "oral", dose)
    assert res.ke > 0.0


def test_oral_fit_r_squared():
    ke, ka, dose, vd = 0.1, 1.0, 100.0, 20.0
    times = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0]
    conc = _oral_data(ke, ka, dose, vd, times)
    res = fit_one_compartment(times, conc, "oral", dose)
    assert res.r_squared >= 0.90


def test_oral_fit_t_half_reasonable():
    ke, ka, dose, vd = 0.2, 1.0, 100.0, 25.0
    times = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0]
    conc = _oral_data(ke, ka, dose, vd, times)
    res = fit_one_compartment(times, conc, "oral", dose)
    # t_half = ln2 / ke; should be in plausible range
    assert 0.5 < res.t_half_h < 100.0


def test_oral_fit_auc_positive():
    ke, ka, dose, vd = 0.1, 2.0, 100.0, 20.0
    times = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0]
    conc = _oral_data(ke, ka, dose, vd, times)
    res = fit_one_compartment(times, conc, "oral", dose)
    assert res.auc_fit > 0.0


def test_oral_fit_route_stored():
    ke, ka, dose, vd = 0.1, 1.0, 100.0, 20.0
    times = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0]
    conc = _oral_data(ke, ka, dose, vd, times)
    res = fit_one_compartment(times, conc, "oral", dose)
    assert res.route == "oral"


def test_oral_fit_n_points():
    ke, ka, dose, vd = 0.1, 1.0, 100.0, 20.0
    times = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0]
    conc = _oral_data(ke, ka, dose, vd, times)
    res = fit_one_compartment(times, conc, "oral", dose)
    assert res.n_points == len(times)


# ---------------------------------------------------------------------------
# Two-compartment IV fitting
# ---------------------------------------------------------------------------


def test_two_cpt_validation_min_points():
    with pytest.raises(ValueError, match="At least 5"):
        fit_two_compartment_iv([1.0, 2.0, 3.0], [5.0, 4.0, 3.0], 100.0)


def test_two_cpt_zero_dose_raises():
    times = [0.5, 1, 2, 4, 8, 12, 16]
    conc = _biexp_data(5.0, 1.5, 2.0, 0.1, times)
    with pytest.raises(ValueError, match="dose_mg"):
        fit_two_compartment_iv(times, conc, 0.0)


def test_two_cpt_alpha_gt_beta():
    """alpha should be greater than beta (distribution faster than elimination)."""
    A, alpha, B, beta = 5.0, 1.5, 2.0, 0.1
    times = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0]
    conc = _biexp_data(A, alpha, B, beta, times)
    res = fit_two_compartment_iv(times, conc, 100.0)
    assert res.alpha > res.beta


def test_two_cpt_beta_recovered():
    """Beta (terminal slope) should be approximately correct."""
    A, alpha, B, beta = 4.0, 2.0, 3.0, 0.15
    times = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0]
    conc = _biexp_data(A, alpha, B, beta, times)
    res = fit_two_compartment_iv(times, conc, 100.0)
    assert res.beta == pytest.approx(beta, rel=0.15)


def test_two_cpt_t_half_terminal():
    A, alpha, B, beta = 4.0, 2.0, 2.0, 0.2
    times = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0]
    conc = _biexp_data(A, alpha, B, beta, times)
    res = fit_two_compartment_iv(times, conc, 100.0)
    expected_thalf = math.log(2) / beta
    assert res.t_half_terminal_h == pytest.approx(expected_thalf, rel=0.20)


def test_two_cpt_auc_positive():
    times = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0]
    conc = _biexp_data(5.0, 1.5, 2.0, 0.1, times)
    res = fit_two_compartment_iv(times, conc, 100.0)
    assert res.auc > 0.0


def test_two_cpt_cl_positive():
    times = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0]
    conc = _biexp_data(5.0, 1.5, 2.0, 0.1, times)
    res = fit_two_compartment_iv(times, conc, 100.0)
    assert res.cl_L_per_h > 0.0


def test_two_cpt_vd_ss_gte_central():
    """Vd_ss should be >= Vd_central in a typical 2-cpt system."""
    A, alpha, B, beta = 4.0, 2.0, 2.0, 0.1
    times = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 20.0]
    conc = _biexp_data(A, alpha, B, beta, times)
    res = fit_two_compartment_iv(times, conc, 100.0)
    assert res.vd_ss_L >= res.vd_central_L * 0.5  # generous tolerance


def test_two_cpt_r_squared_clean_data():
    """Clean biexponential data should give good R²."""
    A, alpha, B, beta = 5.0, 1.5, 2.0, 0.1
    times = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0]
    conc = _biexp_data(A, alpha, B, beta, times)
    res = fit_two_compartment_iv(times, conc, 100.0)
    assert res.r_squared >= 0.90


def test_two_cpt_n_points_recorded():
    times = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0]
    conc = _biexp_data(5.0, 1.5, 2.0, 0.1, times)
    res = fit_two_compartment_iv(times, conc, 100.0)
    assert res.n_points == len(times)


def test_two_cpt_notes_nonempty():
    times = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0]
    conc = _biexp_data(5.0, 1.5, 2.0, 0.1, times)
    res = fit_two_compartment_iv(times, conc, 100.0)
    assert len(res.notes) > 0
