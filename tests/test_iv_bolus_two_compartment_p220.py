"""Tests for Phase 220 — 2-Compartment IV Bolus Analytical Model."""

import math

import pytest

from omega_pbpk.core.iv_bolus_two_compartment_p220 import (
    IVTwoCompResult,
    estimate_two_comp_params,
    simulate_iv_two_compartment,
)

# Reference parameters used across tests
DRUG = "TestDrug"
DOSE = 100.0
ALPHA = 2.0  # h⁻¹
BETA = 0.2  # h⁻¹
A = 8.0  # mg/L
B = 2.0  # mg/L


def make_result(**kwargs):
    params = dict(
        drug_name=DRUG,
        dose_mg=DOSE,
        alpha=ALPHA,
        beta=BETA,
        A_coeff=A,
        B_coeff=B,
    )
    params.update(kwargs)
    return simulate_iv_two_compartment(**params)


# ---------------------------------------------------------------------------
# Return type and field preservation
# ---------------------------------------------------------------------------


def test_return_type():
    result = make_result()
    assert isinstance(result, IVTwoCompResult)


def test_field_drug_name():
    result = make_result()
    assert result.drug_name == DRUG


def test_field_dose_mg():
    result = make_result()
    assert result.dose_mg == DOSE


def test_field_alpha():
    result = make_result()
    assert result.alpha_per_h == ALPHA


def test_field_beta():
    result = make_result()
    assert result.beta_per_h == BETA


# ---------------------------------------------------------------------------
# Analytical identities
# ---------------------------------------------------------------------------


def test_cmax_equals_a_plus_b():
    result = make_result()
    assert abs(result.cmax_mg_L - (A + B)) < 1e-9


def test_c_at_time_zero_equals_a_plus_b():
    result = make_result()
    assert abs(result.c_plasma_mg_L[0] - (A + B)) < 1e-9


def test_auc_analytical():
    result = make_result()
    expected_auc = A / ALPHA + B / BETA
    assert abs(result.auc_mg_h_per_L - expected_auc) < 1e-6


def test_cl_equals_dose_over_auc():
    result = make_result()
    expected_cl = DOSE / result.auc_mg_h_per_L
    assert abs(result.cl_L_per_h - expected_cl) < 1e-6


def test_vd_central_equals_dose_over_cmax():
    result = make_result()
    expected_vd = DOSE / (A + B)
    assert abs(result.vd_central_L - expected_vd) < 1e-9


def test_t_half_alpha_less_than_t_half_beta():
    result = make_result()
    assert result.t_half_alpha_h < result.t_half_beta_h


def test_t_half_alpha_consistent():
    result = make_result()
    expected = math.log(2) / ALPHA
    assert abs(result.t_half_alpha_h - expected) < 1e-9


def test_t_half_beta_consistent():
    result = make_result()
    expected = math.log(2) / BETA
    assert abs(result.t_half_beta_h - expected) < 1e-9


# ---------------------------------------------------------------------------
# Simulated profile
# ---------------------------------------------------------------------------


def test_times_list_not_empty():
    result = make_result(t_end_h=12.0, dt=0.5)
    assert len(result.times_h) > 0


def test_c_plasma_list_length_matches_times():
    result = make_result(t_end_h=12.0, dt=0.5)
    assert len(result.c_plasma_mg_L) == len(result.times_h)


def test_concentration_decreases_monotonically():
    """For a well-conditioned biexponential, concentration should generally decrease."""
    result = make_result(t_end_h=24.0, dt=0.5)
    # Allow one non-decrease at most (numerical noise)
    increases = sum(
        1
        for i in range(len(result.c_plasma_mg_L) - 1)
        if result.c_plasma_mg_L[i + 1] > result.c_plasma_mg_L[i]
    )
    assert increases == 0


def test_concentration_at_end_less_than_at_start():
    result = make_result(t_end_h=24.0)
    assert result.c_plasma_mg_L[-1] < result.c_plasma_mg_L[0]


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_alpha_not_greater_than_beta_raises():
    with pytest.raises(ValueError, match="alpha"):
        simulate_iv_two_compartment(DRUG, DOSE, alpha=0.1, beta=0.2, A_coeff=A, B_coeff=B)


def test_beta_not_positive_raises():
    with pytest.raises(ValueError, match="beta"):
        simulate_iv_two_compartment(DRUG, DOSE, alpha=2.0, beta=0.0, A_coeff=A, B_coeff=B)


def test_dose_not_positive_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_iv_two_compartment(DRUG, 0.0, alpha=ALPHA, beta=BETA, A_coeff=A, B_coeff=B)


# ---------------------------------------------------------------------------
# estimate_two_comp_params
# ---------------------------------------------------------------------------


def _synthetic_data(alpha=2.0, beta=0.2, A=8.0, B=2.0, n=12, t_end=10.0):
    """Generate noise-free biexponential data."""
    times = [t_end * i / (n - 1) for i in range(n)]
    concs = [A * math.exp(-alpha * t) + B * math.exp(-beta * t) for t in times]
    return times, concs


def test_estimate_returns_valid_result():
    ts, cs = _synthetic_data()
    result = estimate_two_comp_params(DRUG, ts, cs)
    assert isinstance(result, IVTwoCompResult)


def test_estimate_alpha_greater_than_beta():
    ts, cs = _synthetic_data()
    result = estimate_two_comp_params(DRUG, ts, cs)
    assert result.alpha_per_h > result.beta_per_h


def test_estimate_r_squared_high():
    """R² annotation should indicate a good fit for noise-free synthetic data."""
    ts, cs = _synthetic_data(n=20)
    result = estimate_two_comp_params(DRUG, ts, cs)
    # Extract R² from notes string
    import re

    match = re.search(r"R²=([0-9]+(?:\.[0-9]+)?)", result.notes)
    assert match is not None, f"R² not found in notes: {result.notes}"
    r2 = float(match.group(1))
    assert r2 > 0.9


def test_estimate_auc_positive():
    ts, cs = _synthetic_data()
    result = estimate_two_comp_params(DRUG, ts, cs)
    assert result.auc_mg_h_per_L > 0


def test_estimate_cl_positive():
    ts, cs = _synthetic_data()
    result = estimate_two_comp_params(DRUG, ts, cs)
    assert result.cl_L_per_h > 0


def test_estimate_fewer_than_4_points_raises():
    with pytest.raises(ValueError, match="4 data points"):
        estimate_two_comp_params(DRUG, [0.0, 1.0, 2.0], [10.0, 5.0, 2.5])


def test_estimate_mismatched_lengths_raises():
    with pytest.raises(ValueError, match="same length"):
        estimate_two_comp_params(DRUG, [0.0, 1.0, 2.0, 3.0], [10.0, 5.0])


def test_estimate_recovers_approximate_beta():
    """Beta should be in the right ballpark for noise-free data."""
    ts, cs = _synthetic_data(alpha=3.0, beta=0.3, A=6.0, B=4.0, n=16)
    result = estimate_two_comp_params(DRUG, ts, cs)
    # Beta within 50% of true value
    assert abs(result.beta_per_h - 0.3) / 0.3 < 0.5


def test_estimate_notes_contains_method():
    ts, cs = _synthetic_data()
    result = estimate_two_comp_params(DRUG, ts, cs)
    assert "residual" in result.notes.lower() or "feather" in result.notes.lower()
