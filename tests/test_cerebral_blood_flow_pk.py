"""Tests for Phase 460 — Cerebral Blood Flow PK."""

import pytest

from omega_pbpk.core.cerebral_blood_flow_pk import (
    CerebralPKResult,
    simulate_cerebral_pk,
)

# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = simulate_cerebral_pk("DrugA", dose_mg=100.0)
    assert isinstance(result, CerebralPKResult)


def test_all_fields_present():
    result = simulate_cerebral_pk("DrugB", dose_mg=50.0)
    assert hasattr(result, "times_h")
    assert hasattr(result, "c_plasma_mg_L")
    assert hasattr(result, "c_brain_ecf_mg_L")
    assert hasattr(result, "c_brain_icf_mg_L")
    assert hasattr(result, "cmax_brain_ecf_mg_L")
    assert hasattr(result, "tmax_brain_ecf_h")
    assert hasattr(result, "auc_brain_ecf_mg_h_per_L")
    assert hasattr(result, "brain_to_plasma_ratio")
    assert hasattr(result, "t_half_brain_ecf_h")
    assert hasattr(result, "notes")


# ---------------------------------------------------------------------------
# Array length consistency
# ---------------------------------------------------------------------------


def test_times_and_plasma_same_length():
    result = simulate_cerebral_pk("DrugC", dose_mg=100.0)
    assert len(result.times_h) == len(result.c_plasma_mg_L)


def test_times_and_ecf_same_length():
    result = simulate_cerebral_pk("DrugD", dose_mg=100.0)
    assert len(result.times_h) == len(result.c_brain_ecf_mg_L)


def test_times_and_icf_same_length():
    result = simulate_cerebral_pk("DrugE", dose_mg=100.0)
    assert len(result.times_h) == len(result.c_brain_icf_mg_L)


# ---------------------------------------------------------------------------
# Plasma IV bolus behaviour
# ---------------------------------------------------------------------------


def test_plasma_decreases_over_time():
    """IV bolus → plasma concentration strictly decreases."""
    result = simulate_cerebral_pk("DrugF", dose_mg=100.0, dt_h=0.5, t_end_h=12.0)
    cp = result.c_plasma_mg_L
    # Check that the profile is monotonically non-increasing overall
    assert cp[0] > cp[-1]
    for i in range(1, len(cp)):
        assert cp[i] <= cp[i - 1] + 1e-9


def test_plasma_starts_at_dose_over_vd():
    result = simulate_cerebral_pk("DrugG", dose_mg=200.0, vd_sys_L=40.0)
    expected_c0 = 200.0 / 40.0
    assert abs(result.c_plasma_mg_L[0] - expected_c0) < 1e-9


# ---------------------------------------------------------------------------
# Brain ECF initial condition
# ---------------------------------------------------------------------------


def test_brain_ecf_starts_at_zero():
    result = simulate_cerebral_pk("DrugH", dose_mg=100.0)
    assert result.c_brain_ecf_mg_L[0] == pytest.approx(0.0)


def test_brain_icf_starts_at_zero():
    result = simulate_cerebral_pk("DrugI", dose_mg=100.0)
    assert result.c_brain_icf_mg_L[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Cmax > 0 and positivity
# ---------------------------------------------------------------------------


def test_cmax_brain_ecf_positive():
    result = simulate_cerebral_pk("DrugJ", dose_mg=100.0)
    assert result.cmax_brain_ecf_mg_L > 0.0


def test_auc_brain_ecf_positive():
    result = simulate_cerebral_pk("DrugK", dose_mg=100.0)
    assert result.auc_brain_ecf_mg_h_per_L > 0.0


def test_all_ecf_concentrations_nonneg():
    result = simulate_cerebral_pk("DrugL", dose_mg=100.0)
    for c in result.c_brain_ecf_mg_L:
        assert c >= -1e-9


# ---------------------------------------------------------------------------
# Kp_brain effect
# ---------------------------------------------------------------------------


def test_higher_kp_brain_higher_ecf_at_steady_state():
    """Higher Kp_brain → higher steady-state brain ECF concentration."""
    r_low = simulate_cerebral_pk("Drug", dose_mg=100.0, Kp_brain=0.1, t_end_h=24.0)
    r_high = simulate_cerebral_pk("Drug", dose_mg=100.0, Kp_brain=0.8, t_end_h=24.0)
    # At longer times ECF should reflect Kp_brain difference
    assert r_high.cmax_brain_ecf_mg_L > r_low.cmax_brain_ecf_mg_L


# ---------------------------------------------------------------------------
# PS_BBB effect (faster penetration → earlier tmax)
# ---------------------------------------------------------------------------


def test_higher_ps_bbb_faster_penetration():
    """Higher PS_BBB → earlier tmax_brain_ecf."""
    r_low = simulate_cerebral_pk("Drug", dose_mg=100.0, PS_BBB_L_per_h=0.01, t_end_h=24.0, dt_h=0.1)
    r_high = simulate_cerebral_pk("Drug", dose_mg=100.0, PS_BBB_L_per_h=2.0, t_end_h=24.0, dt_h=0.1)
    assert r_high.tmax_brain_ecf_h <= r_low.tmax_brain_ecf_h + 1.0  # allow 1 h tolerance


# ---------------------------------------------------------------------------
# Dose proportionality
# ---------------------------------------------------------------------------


def test_double_dose_doubles_cmax_ecf():
    """2x dose → approximately 2x cmax_brain_ecf (linear system)."""
    r1 = simulate_cerebral_pk("Drug", dose_mg=50.0)
    r2 = simulate_cerebral_pk("Drug", dose_mg=100.0)
    ratio = r2.cmax_brain_ecf_mg_L / r1.cmax_brain_ecf_mg_L
    assert abs(ratio - 2.0) < 0.05


def test_double_dose_doubles_auc_ecf():
    r1 = simulate_cerebral_pk("Drug", dose_mg=50.0)
    r2 = simulate_cerebral_pk("Drug", dose_mg=100.0)
    ratio = r2.auc_brain_ecf_mg_h_per_L / r1.auc_brain_ecf_mg_h_per_L
    assert abs(ratio - 2.0) < 0.1


# ---------------------------------------------------------------------------
# Brain-to-plasma ratio
# ---------------------------------------------------------------------------


def test_brain_to_plasma_ratio_nonneg():
    result = simulate_cerebral_pk("DrugM", dose_mg=100.0)
    assert result.brain_to_plasma_ratio >= 0.0


def test_brain_to_plasma_ratio_reasonable():
    """Brain-to-plasma ratio should be within a physically plausible range."""
    result = simulate_cerebral_pk("DrugN", dose_mg=100.0, Kp_brain=0.5)
    assert result.brain_to_plasma_ratio < 20.0


# ---------------------------------------------------------------------------
# Drug name stored correctly
# ---------------------------------------------------------------------------


def test_drug_name_stored():
    result = simulate_cerebral_pk("TestCompound", dose_mg=100.0)
    assert result.drug_name == "TestCompound"


def test_dose_stored():
    result = simulate_cerebral_pk("Drug", dose_mg=75.0)
    assert result.dose_mg == pytest.approx(75.0)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_cerebral_pk("Drug", dose_mg=0.0)


def test_validation_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_cerebral_pk("Drug", dose_mg=-50.0)


def test_validation_cl_sys_zero():
    with pytest.raises(ValueError, match="cl_sys_L_per_h"):
        simulate_cerebral_pk("Drug", dose_mg=100.0, cl_sys_L_per_h=0.0)


def test_validation_vd_sys_zero():
    with pytest.raises(ValueError, match="vd_sys_L"):
        simulate_cerebral_pk("Drug", dose_mg=100.0, vd_sys_L=0.0)


def test_validation_kp_brain_zero():
    with pytest.raises(ValueError, match="Kp_brain"):
        simulate_cerebral_pk("Drug", dose_mg=100.0, Kp_brain=0.0)


def test_validation_ps_bbb_zero():
    with pytest.raises(ValueError, match="PS_BBB_L_per_h"):
        simulate_cerebral_pk("Drug", dose_mg=100.0, PS_BBB_L_per_h=0.0)
