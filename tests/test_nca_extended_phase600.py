"""Tests for Phase 600 extended NCA API (compute_nca, compute_partial_auc, estimate_lambda_z)."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.analysis.nca_extended import (
    NCAResult,
    compute_nca,
    compute_partial_auc,
    estimate_lambda_z,
)

# ---------------------------------------------------------------------------
# Helper: 1-compartment oral concentration profile
# ---------------------------------------------------------------------------


def _make_oral_profile(
    dose: float = 100.0,
    cl: float = 5.0,
    vd: float = 20.0,
    ka: float = 1.5,
    f: float = 1.0,
    t_end: float = 24.0,
    dt: float = 0.1,
):
    """Generate 1-cpt oral concentration profile."""
    ke = cl / vd
    times = [round(i * dt, 6) for i in range(int(t_end / dt) + 1)]
    concs = []
    for t in times:
        if ka != ke:
            c = f * dose * ka / (vd * (ka - ke)) * (math.exp(-ke * t) - math.exp(-ka * t))
        else:
            c = f * dose * ke / vd * t * math.exp(-ke * t)
        concs.append(max(0.0, c))
    return times, concs


def _make_iv_profile(
    dose: float = 100.0,
    cl: float = 5.0,
    vd: float = 20.0,
    t_end: float = 24.0,
    dt: float = 0.1,
):
    """Generate 1-cpt IV concentration profile."""
    ke = cl / vd
    times = [round(i * dt, 6) for i in range(int(t_end / dt) + 1)]
    # Start at t=0.1 to avoid log(0) issues
    times = [max(t, 1e-6) for t in times]
    concs = [dose / vd * math.exp(-ke * t) for t in times]
    return times, concs


# Shared profiles
ORAL_TIMES, ORAL_CONCS = _make_oral_profile()
IV_TIMES, IV_CONCS = _make_iv_profile(cl=5.0, vd=20.0)


# ---------------------------------------------------------------------------
# Return type tests
# ---------------------------------------------------------------------------


def test_returns_result():
    result = compute_nca("drug", ORAL_TIMES, ORAL_CONCS, dose_mg=100.0, route="oral")
    assert isinstance(result, NCAResult)


def test_cmax_positive():
    result = compute_nca("drug", ORAL_TIMES, ORAL_CONCS, dose_mg=100.0, route="oral")
    assert result.cmax_mg_L > 0


def test_tmax_positive():
    result = compute_nca("drug", ORAL_TIMES, ORAL_CONCS, dose_mg=100.0, route="oral")
    assert result.tmax_h > 0


def test_auc_0t_positive():
    result = compute_nca("drug", ORAL_TIMES, ORAL_CONCS, dose_mg=100.0, route="oral")
    assert result.auc_0_t_mg_h_L > 0


def test_auc_0inf_gt_auc_0t():
    result = compute_nca("drug", ORAL_TIMES, ORAL_CONCS, dose_mg=100.0, route="oral")
    assert result.auc_0_inf_mg_h_L > result.auc_0_t_mg_h_L


def test_mrt_positive():
    result = compute_nca("drug", ORAL_TIMES, ORAL_CONCS, dose_mg=100.0, route="oral")
    assert result.mrt_h > 0


def test_cl_nca_positive():
    result = compute_nca("drug", ORAL_TIMES, ORAL_CONCS, dose_mg=100.0, route="oral")
    assert result.cl_nca_L_per_h > 0


def test_vss_positive():
    result = compute_nca("drug", ORAL_TIMES, ORAL_CONCS, dose_mg=100.0, route="oral")
    assert result.vss_nca_L > 0


def test_t_half_positive():
    result = compute_nca("drug", ORAL_TIMES, ORAL_CONCS, dose_mg=100.0, route="oral")
    assert result.t_half_terminal_h > 0


def test_lambda_z_positive():
    result = compute_nca("drug", ORAL_TIMES, ORAL_CONCS, dose_mg=100.0, route="oral")
    assert result.lambda_z_per_h > 0


def test_r2_lambda_z_in_0_1():
    result = compute_nca("drug", ORAL_TIMES, ORAL_CONCS, dose_mg=100.0, route="oral")
    assert 0.0 <= result.r2_lambda_z <= 1.0


# ---------------------------------------------------------------------------
# Partial AUC tests
# ---------------------------------------------------------------------------


def test_partial_auc_has_default_keys():
    result = compute_nca("drug", ORAL_TIMES, ORAL_CONCS, dose_mg=100.0, route="oral")
    assert "0-4h" in result.partial_auc
    assert "0-8h" in result.partial_auc
    assert "0-24h" in result.partial_auc


def test_partial_auc_0_24h_le_auc_0t():
    """Partial AUC [0,24h] should be <= AUC[0,t] when t=24."""
    result = compute_nca("drug", ORAL_TIMES, ORAL_CONCS, dose_mg=100.0, route="oral")
    assert result.partial_auc["0-24h"] <= result.auc_0_t_mg_h_L + 1e-6


def test_partial_auc_increasing():
    """0-4h < 0-8h < 0-24h for typical oral profile."""
    result = compute_nca("drug", ORAL_TIMES, ORAL_CONCS, dose_mg=100.0, route="oral")
    assert result.partial_auc["0-4h"] < result.partial_auc["0-8h"]
    assert result.partial_auc["0-8h"] < result.partial_auc["0-24h"]


# ---------------------------------------------------------------------------
# Flip-flop test
# ---------------------------------------------------------------------------


def test_flip_flop_false_for_normal():
    """Normal oral profile with fast absorption should not show flip-flop."""
    result = compute_nca("drug", ORAL_TIMES, ORAL_CONCS, dose_mg=100.0, route="oral")
    # ka=1.5 >> ke=0.25, so t_half_terminal << MRT — no flip-flop
    assert result.flip_flop is False


# ---------------------------------------------------------------------------
# CL accuracy test (IV)
# ---------------------------------------------------------------------------


def test_cl_nca_close_to_true():
    """For IV profile, CL NCA should be close to true CL (±20%)."""
    true_cl = 5.0
    result = compute_nca("drug", IV_TIMES, IV_CONCS, dose_mg=100.0, route="iv")
    assert result.cl_nca_L_per_h == pytest.approx(true_cl, rel=0.20)


def test_vss_gt_0():
    result = compute_nca("drug", IV_TIMES, IV_CONCS, dose_mg=100.0, route="iv")
    assert result.vss_nca_L > 0


# ---------------------------------------------------------------------------
# estimate_lambda_z tests
# ---------------------------------------------------------------------------


def test_estimate_lambda_z_positive():
    lz, r2 = estimate_lambda_z(ORAL_TIMES, ORAL_CONCS, n_points=4)
    assert lz > 0


def test_estimate_lambda_z_r2_positive():
    lz, r2 = estimate_lambda_z(ORAL_TIMES, ORAL_CONCS, n_points=4)
    assert 0.0 <= r2 <= 1.0


# ---------------------------------------------------------------------------
# compute_partial_auc custom intervals
# ---------------------------------------------------------------------------


def test_compute_partial_auc_custom_intervals():
    result = compute_partial_auc(
        ORAL_TIMES,
        ORAL_CONCS,
        intervals=[(0.0, 2.0), (0.0, 12.0)],
    )
    assert "0-2.0h" in result or "0-2h" in result or any("2" in k for k in result)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


def test_invalid_mismatched_lengths_raises():
    with pytest.raises(ValueError, match="same length"):
        compute_nca("drug", [0.0, 1.0, 2.0, 3.0], [1.0, 0.8], dose_mg=100.0)


def test_invalid_too_few_points_raises():
    with pytest.raises(ValueError, match="4 time points"):
        compute_nca("drug", [0.0, 1.0, 2.0], [1.0, 0.8, 0.6], dose_mg=100.0)


def test_invalid_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        compute_nca("drug", ORAL_TIMES, ORAL_CONCS, dose_mg=0.0)


def test_invalid_route_raises():
    with pytest.raises(ValueError, match="route"):
        compute_nca("drug", ORAL_TIMES, ORAL_CONCS, dose_mg=100.0, route="im")


def test_notes_nonempty():
    result = compute_nca("drug", ORAL_TIMES, ORAL_CONCS, dose_mg=100.0, route="oral")
    assert len(result.notes) > 0
