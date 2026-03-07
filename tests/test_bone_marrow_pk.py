"""Tests for Phase 916 — Drug Bone Marrow Distribution."""

import pytest

from omega_pbpk.core.bone_marrow_pk import (
    BoneMarrowPKResult,
    compare_dosing_regimens,
    simulate_bone_marrow_pk,
)

# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_dose_raises():
    with pytest.raises(ValueError, match="dose_mg must be > 0"):
        simulate_bone_marrow_pk("Drug", dose_mg=0.0)


def test_invalid_cl_raises():
    with pytest.raises(ValueError, match="cl_L_per_h must be > 0"):
        simulate_bone_marrow_pk("Drug", dose_mg=100.0, cl_L_per_h=0.0)


def test_invalid_v_marrow_raises():
    with pytest.raises(ValueError, match="v_marrow_L must be > 0"):
        simulate_bone_marrow_pk("Drug", dose_mg=100.0, v_marrow_L=0.0)


# ---------------------------------------------------------------------------
# Basic simulation structure
# ---------------------------------------------------------------------------


def test_returns_dataclass():
    r = simulate_bone_marrow_pk("Drug", dose_mg=100.0)
    assert isinstance(r, BoneMarrowPKResult)


def test_time_series_length_consistent():
    r = simulate_bone_marrow_pk("Drug", dose_mg=100.0, t_end_h=10.0, dt_h=0.1)
    assert len(r.times_h) == len(r.c_plasma_mg_L) == len(r.c_bone_marrow_mg_L)


def test_time_series_starts_at_zero():
    r = simulate_bone_marrow_pk("Drug", dose_mg=100.0)
    assert r.times_h[0] == 0.0


def test_time_series_ends_at_t_end():
    r = simulate_bone_marrow_pk("Drug", dose_mg=100.0, t_end_h=24.0, dt_h=0.1)
    assert abs(r.times_h[-1] - 24.0) < 0.01


def test_concentrations_non_negative():
    r = simulate_bone_marrow_pk("Drug", dose_mg=100.0)
    assert all(c >= 0 for c in r.c_plasma_mg_L)
    assert all(c >= 0 for c in r.c_bone_marrow_mg_L)


# ---------------------------------------------------------------------------
# PK metrics
# ---------------------------------------------------------------------------


def test_cmax_plasma_positive():
    r = simulate_bone_marrow_pk("Drug", dose_mg=100.0)
    assert r.cmax_plasma > 0


def test_cmax_marrow_positive():
    r = simulate_bone_marrow_pk("Drug", dose_mg=100.0)
    assert r.cmax_marrow > 0


def test_auc_plasma_positive():
    r = simulate_bone_marrow_pk("Drug", dose_mg=100.0)
    assert r.auc_plasma > 0


def test_auc_marrow_positive():
    r = simulate_bone_marrow_pk("Drug", dose_mg=100.0)
    assert r.auc_marrow > 0


def test_marrow_plasma_kp_is_ratio():
    r = simulate_bone_marrow_pk("Drug", dose_mg=100.0)
    expected = r.auc_marrow / r.auc_plasma
    assert abs(r.marrow_plasma_kp - expected) < 1e-6


def test_higher_uptake_increases_marrow_kp():
    r_low = simulate_bone_marrow_pk("Drug", dose_mg=100.0, k_marrow_uptake_per_h=0.1)
    r_high = simulate_bone_marrow_pk("Drug", dose_mg=100.0, k_marrow_uptake_per_h=2.0)
    assert r_high.marrow_plasma_kp > r_low.marrow_plasma_kp


def test_higher_dose_scales_cmax():
    r1 = simulate_bone_marrow_pk("Drug", dose_mg=100.0)
    r2 = simulate_bone_marrow_pk("Drug", dose_mg=200.0)
    assert abs(r2.cmax_plasma / r1.cmax_plasma - 2.0) < 0.05


# ---------------------------------------------------------------------------
# Myelosuppression scoring
# ---------------------------------------------------------------------------


def test_myelosuppression_score_range():
    r = simulate_bone_marrow_pk("Drug", dose_mg=100.0)
    assert 0 <= r.myelosuppression_score <= 100


def test_myelosuppression_risk_levels():
    valid = {"low", "moderate", "high"}
    r = simulate_bone_marrow_pk("Drug", dose_mg=100.0)
    assert r.myelosuppression_risk in valid


def test_high_uptake_gives_high_risk():
    r = simulate_bone_marrow_pk(
        "Cyclophosphamide",
        dose_mg=5000.0,
        k_marrow_uptake_per_h=5.0,
        k_marrow_release_per_h=0.01,
        cl_L_per_h=1.0,
        vd_L=10.0,
        v_marrow_L=1.0,
    )
    assert r.myelosuppression_risk == "high"


# ---------------------------------------------------------------------------
# Nadir time
# ---------------------------------------------------------------------------


def test_nadir_time_after_tmax_marrow():
    r = simulate_bone_marrow_pk("Drug", dose_mg=100.0, t_end_h=72.0, dt_h=0.1)
    tmax_marrow = r.times_h[list(r.c_bone_marrow_mg_L).index(r.cmax_marrow)]
    assert r.estimated_nadir_time_h >= tmax_marrow


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


def test_notes_high_risk():
    r = simulate_bone_marrow_pk(
        "Cyclophosphamide",
        dose_mg=5000.0,
        k_marrow_uptake_per_h=5.0,
        k_marrow_release_per_h=0.01,
        cl_L_per_h=1.0,
        vd_L=10.0,
        v_marrow_L=1.0,
    )
    assert "G-CSF" in r.notes


def test_notes_low_risk():
    r = simulate_bone_marrow_pk(
        "SafeDrug",
        dose_mg=10.0,
        cl_L_per_h=100.0,
        vd_L=200.0,
        k_marrow_uptake_per_h=0.01,
        k_marrow_release_per_h=5.0,
    )
    assert "Low myelosuppression" in r.notes


# ---------------------------------------------------------------------------
# compare_dosing_regimens
# ---------------------------------------------------------------------------


def test_compare_dosing_regimens_sorted():
    results = compare_dosing_regimens("Drug", dose_mg=100.0, k_abs_values=[0.1, 1.0, 5.0])
    assert len(results) == 3
    scores = [r.myelosuppression_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_compare_dosing_regimens_empty():
    results = compare_dosing_regimens("Drug", dose_mg=100.0, k_abs_values=[])
    assert results == []


def test_result_is_frozen():
    r = simulate_bone_marrow_pk("Drug", dose_mg=100.0)
    with pytest.raises((AttributeError, TypeError)):
        r.dose_mg = 999.0
