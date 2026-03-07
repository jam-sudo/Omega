"""Tests for Phase 753 — Drug Partitioning into Adipose Tissue."""

import pytest

from omega_pbpk.core.adipose_pk import (
    AdiposePKResult,
    _calc_kp_adipose,
    compare_obesity_effect,
    simulate_adipose_pk,
)

# ---------------------------------------------------------------------------
# Basic return type and structure
# ---------------------------------------------------------------------------


def test_returns_adipose_pk_result():
    result = simulate_adipose_pk("TestDrug", dose_mg=100.0, logp=3.0)
    assert isinstance(result, AdiposePKResult)


def test_cmax_plasma_positive():
    result = simulate_adipose_pk("TestDrug", dose_mg=100.0, logp=2.0)
    assert result.cmax_plasma > 0


def test_auc_plasma_positive():
    result = simulate_adipose_pk("TestDrug", dose_mg=100.0, logp=2.0)
    assert result.auc_plasma > 0


def test_auc_adipose_positive():
    result = simulate_adipose_pk("TestDrug", dose_mg=100.0, logp=2.0)
    assert result.auc_adipose > 0


def test_kp_adipose_positive():
    result = simulate_adipose_pk("TestDrug", dose_mg=100.0, logp=2.0)
    assert result.kp_adipose > 0


# ---------------------------------------------------------------------------
# Kp_adipose calculation
# ---------------------------------------------------------------------------


def test_high_logp_higher_kp():
    kp_low = _calc_kp_adipose(1.0)
    kp_high = _calc_kp_adipose(5.0)
    assert kp_high > kp_low


def test_negative_logp_gives_min_kp():
    kp = _calc_kp_adipose(-10.0)
    assert kp == pytest.approx(0.1)


def test_kp_formula_correctness():
    logp = 3.0
    expected = 10.0 ** (0.7 * logp - 0.5)
    assert _calc_kp_adipose(logp) == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Adipose accumulation relationship to logP
# ---------------------------------------------------------------------------


def test_high_logp_higher_adipose_auc_ratio():
    r_low = simulate_adipose_pk("Drug", dose_mg=100.0, logp=1.0, t_end_h=48.0)
    r_high = simulate_adipose_pk("Drug", dose_mg=100.0, logp=5.0, t_end_h=48.0)
    ratio_low = r_low.auc_adipose / r_low.auc_plasma
    ratio_high = r_high.auc_adipose / r_high.auc_plasma
    assert ratio_high > ratio_low


# ---------------------------------------------------------------------------
# Depot effect
# ---------------------------------------------------------------------------


def test_depot_effect_positive_for_lipophilic():
    result = simulate_adipose_pk("LipoDrug", dose_mg=100.0, logp=4.0, t_end_h=72.0)
    assert result.depot_effect_h > 0


# ---------------------------------------------------------------------------
# Effective Vd
# ---------------------------------------------------------------------------


def test_effective_vd_greater_than_vd_plasma():
    result = simulate_adipose_pk("TestDrug", dose_mg=100.0, logp=2.0)
    assert (
        result.effective_vd_L > result.vd_plasma_L
        if hasattr(result, "vd_plasma_L")
        else result.effective_vd_L > 5.0
    )


def test_effective_vd_formula():
    logp = 3.0
    vd_pl = 5.0
    vd_ad = 15.0
    kp = _calc_kp_adipose(logp)
    expected_eff_vd = vd_pl + kp * vd_ad
    result = simulate_adipose_pk(
        "X", dose_mg=100.0, logp=logp, vd_plasma_L=vd_pl, vd_adipose_L=vd_ad
    )
    assert result.effective_vd_L == pytest.approx(expected_eff_vd, rel=1e-4)


# ---------------------------------------------------------------------------
# Obesity comparison
# ---------------------------------------------------------------------------


def test_compare_obesity_effect_returns_dict():
    out = compare_obesity_effect("TestDrug", dose_mg=100.0, logp=3.0)
    assert isinstance(out, dict)


def test_compare_obesity_effect_keys():
    out = compare_obesity_effect("TestDrug", dose_mg=100.0, logp=3.0)
    assert "normal_result" in out
    assert "obese_result" in out
    assert "auc_ratio" in out
    assert "t_half_ratio" in out
    assert "notes" in out


def test_obese_effective_vd_larger():
    """Obesity (larger adipose volume) always produces a larger effective Vd.

    For an IV bolus model: effective_vd = Vd_plasma + Kp_adipose * Vd_adipose.
    Obese Vd_adipose > normal Vd_adipose → obese effective_vd > normal effective_vd.
    """
    out = compare_obesity_effect(
        "LipoDrug",
        dose_mg=100.0,
        logp=4.0,
        vd_adipose_normal=15.0,
        vd_adipose_obese=50.0,
    )
    assert out["obese_result"].effective_vd_L > out["normal_result"].effective_vd_L


def test_obese_higher_effective_vd():
    out = compare_obesity_effect("Drug", dose_mg=100.0, logp=3.0, vd_adipose_obese=50.0)
    assert out["obese_result"].effective_vd_L > out["normal_result"].effective_vd_L


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_invalid_dose_raises():
    with pytest.raises(ValueError):
        simulate_adipose_pk("Bad", dose_mg=0.0)


def test_negative_dose_raises():
    with pytest.raises(ValueError):
        simulate_adipose_pk("Bad", dose_mg=-10.0)


def test_invalid_vd_plasma_raises():
    with pytest.raises(ValueError):
        simulate_adipose_pk("Bad", dose_mg=100.0, vd_plasma_L=0.0)


def test_invalid_vd_adipose_raises():
    with pytest.raises(ValueError):
        simulate_adipose_pk("Bad", dose_mg=100.0, vd_adipose_L=0.0)


def test_invalid_cl_raises():
    with pytest.raises(ValueError):
        simulate_adipose_pk("Bad", dose_mg=100.0, cl_L_per_h=0.0)


# ---------------------------------------------------------------------------
# Notes non-empty
# ---------------------------------------------------------------------------


def test_notes_nonempty():
    result = simulate_adipose_pk("TestDrug", dose_mg=100.0)
    assert result.notes and len(result.notes) > 0


# ---------------------------------------------------------------------------
# Linear dose scaling
# ---------------------------------------------------------------------------


def test_linear_dose_scaling():
    r1 = simulate_adipose_pk("Drug", dose_mg=100.0, logp=2.0, t_end_h=48.0)
    r2 = simulate_adipose_pk("Drug", dose_mg=200.0, logp=2.0, t_end_h=48.0)
    assert r2.cmax_plasma == pytest.approx(r1.cmax_plasma * 2.0, rel=1e-3)
    assert r2.auc_plasma == pytest.approx(r1.auc_plasma * 2.0, rel=1e-3)


# ---------------------------------------------------------------------------
# Adipose-to-plasma ratio at steady state
# ---------------------------------------------------------------------------


def test_adipose_to_plasma_ratio_positive():
    result = simulate_adipose_pk("Drug", dose_mg=100.0, logp=3.0, t_end_h=72.0)
    assert result.adipose_to_plasma_ratio_at_steady >= 0.0
