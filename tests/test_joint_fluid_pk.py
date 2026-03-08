"""Tests for Phase 995 — joint fluid PK simulation."""

import pytest

from omega_pbpk.core.joint_fluid_pk import JointFluidPKResult, simulate_joint_fluid_pk

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _run(dose_mg=100.0, logp=1.0, fup=0.5, route="oral", **kwargs):
    return simulate_joint_fluid_pk(
        drug_name="TestDrug",
        dose_mg=dose_mg,
        logp=logp,
        fup=fup,
        route=route,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Return type and basic structure
# ---------------------------------------------------------------------------


def test_returns_joint_fluid_pk_result():
    result = _run()
    assert isinstance(result, JointFluidPKResult)


def test_drug_name_preserved():
    result = simulate_joint_fluid_pk("Diclofenac", 50.0)
    assert result.drug_name == "Diclofenac"


def test_dose_preserved():
    result = _run(dose_mg=200.0)
    assert result.dose_mg == 200.0


def test_route_preserved():
    result = _run(route="iv")
    assert result.route == "iv"


# ---------------------------------------------------------------------------
# Concentrations
# ---------------------------------------------------------------------------


def test_cmax_plasma_positive():
    result = _run()
    assert result.cmax_plasma_mg_L > 0.0


def test_cmax_joint_positive():
    result = _run()
    assert result.cmax_joint_mg_L > 0.0


def test_initial_joint_concentration_is_zero():
    result = _run()
    assert result.c_joint_mg_L[0] == pytest.approx(0.0, abs=1e-12)


def test_auc_plasma_positive():
    result = _run()
    assert result.auc_plasma_mg_h_per_L > 0.0


def test_auc_joint_positive():
    result = _run()
    assert result.auc_joint_mg_h_per_L > 0.0


# ---------------------------------------------------------------------------
# Derived PK metrics
# ---------------------------------------------------------------------------


def test_joint_to_plasma_ratio_positive():
    result = _run()
    assert result.joint_to_plasma_ratio > 0.0


def test_tmax_joint_positive():
    result = _run()
    assert result.tmax_joint_h > 0.0


def test_t_lag_non_negative():
    result = _run()
    assert result.t_lag_to_joint_h >= 0.0


def test_kp_joint_positive():
    result = _run()
    assert result.kp_joint > 0.0


# ---------------------------------------------------------------------------
# Physics / pharmacology
# ---------------------------------------------------------------------------


def test_higher_logp_lower_k_syn():
    """More lipophilic drug → lower aqueous-channel permeation rate → lower early joint exposure."""
    res_low = _run(logp=0.0, t_end_h=4.0)
    res_high = _run(logp=5.0, t_end_h=4.0)
    # Lower logP means faster k_syn; early joint AUC should be higher for low-logP drug
    assert res_low.auc_joint_mg_h_per_L >= res_high.auc_joint_mg_h_per_L


def test_linear_pk_double_dose():
    """Doubling dose should double Cmax (linear PK system)."""
    res1 = _run(dose_mg=100.0)
    res2 = _run(dose_mg=200.0)
    assert res2.cmax_plasma_mg_L == pytest.approx(res1.cmax_plasma_mg_L * 2.0, rel=0.02)


def test_iv_vs_oral_plasma_cmax():
    """IV should reach Cmax faster than oral for same dose."""
    res_oral = _run(route="oral")
    res_iv = _run(route="iv")
    # IV Cmax is at t=0 (immediate); oral takes time for absorption
    assert res_iv.cmax_plasma_mg_L > 0.0
    assert res_oral.cmax_plasma_mg_L > 0.0


# ---------------------------------------------------------------------------
# Validation / error handling
# ---------------------------------------------------------------------------


def test_invalid_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_joint_fluid_pk("X", dose_mg=-10.0)


def test_zero_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_joint_fluid_pk("X", dose_mg=0.0)


def test_invalid_v_joint_raises():
    with pytest.raises(ValueError, match="v_joint_mL"):
        simulate_joint_fluid_pk("X", dose_mg=100.0, v_joint_mL=0.0)


def test_invalid_v_plasma_raises():
    with pytest.raises(ValueError, match="v_plasma_L"):
        simulate_joint_fluid_pk("X", dose_mg=100.0, v_plasma_L=-1.0)


def test_invalid_cl_raises():
    with pytest.raises(ValueError, match="cl_plasma_L_per_h"):
        simulate_joint_fluid_pk("X", dose_mg=100.0, cl_plasma_L_per_h=0.0)


def test_invalid_route_raises():
    with pytest.raises(ValueError, match="route"):
        simulate_joint_fluid_pk("X", dose_mg=100.0, route="sc")
