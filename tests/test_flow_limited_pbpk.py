"""Tests for Phase 246 — Physiological Flow-Limited PBPK.

Covers:
  - Input validation errors
  - vd_effective matches formula: V_blood + sum(Kp_i * V_i)
  - t_half ≈ 0.693 * Vd / CL_total
  - High Kp_fat → higher vd_effective → longer t_half
  - Monotone decrease after IV bolus dose
  - tissue_concentration_at_t: C_tissue = Kp * C_blood
  - c_fat_at_24h > c_blood at 24h (fat accumulation when Kp_fat > 1)
  - c_liver_at_1h > c_blood at 1h (Kp_liver > 1)
  - 2× dose → 2× AUC (linearity)
  - Result field types
  - Oral vs IV: oral has delayed Cmax
  - tissue_concentration_at_t invalid tissue raises ValueError
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from omega_pbpk.core.flow_limited_pbpk import (
    FlowLimitedPBPKResult,
    simulate_flow_limited_pbpk,
    tissue_concentration_at_t,
)

# Physiological organ volumes (must match module constants)
_V_LIVER = 1.7
_V_KIDNEY = 0.3
_V_MUSCLE = 35.0
_V_FAT = 15.0
_V_BLOOD = 5.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


BASE_KWARGS = dict(
    drug_name="testdrug",
    dose_mg=100.0,
    kp_liver=3.0,
    kp_kidney=2.0,
    kp_muscle=0.5,
    kp_fat=10.0,
    cl_hepatic_L_per_h=5.0,
    cl_renal_L_per_h=1.0,
    route="iv",
    ka_per_h=1.0,
    f_oral=1.0,
    t_end_h=24.0,
    dt_h=0.1,
)


def _sim(**kwargs):
    kw = {**BASE_KWARGS, **kwargs}
    return simulate_flow_limited_pbpk(**kw)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        _sim(dose_mg=0.0)


def test_validation_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        _sim(dose_mg=-50.0)


def test_validation_kp_liver_zero():
    with pytest.raises(ValueError, match="kp_liver"):
        _sim(kp_liver=0.0)


def test_validation_kp_fat_negative():
    with pytest.raises(ValueError, match="kp_fat"):
        _sim(kp_fat=-1.0)


def test_validation_cl_hepatic_zero():
    with pytest.raises(ValueError, match="cl_hepatic_L_per_h"):
        _sim(cl_hepatic_L_per_h=0.0)


def test_validation_cl_renal_negative():
    with pytest.raises(ValueError, match="cl_renal_L_per_h"):
        _sim(cl_renal_L_per_h=-0.5)


def test_validation_invalid_route():
    with pytest.raises(ValueError, match="route"):
        _sim(route="subcutaneous")


# ---------------------------------------------------------------------------
# vd_effective formula
# ---------------------------------------------------------------------------


def test_vd_effective_formula():
    """Vd_eff = V_blood + Kp_liver*V_liver + Kp_kidney*V_kidney + Kp_muscle*V_muscle + Kp_fat*V_fat."""
    kp_l, kp_k, kp_m, kp_f = 3.0, 2.0, 0.5, 10.0
    expected = _V_BLOOD + kp_l * _V_LIVER + kp_k * _V_KIDNEY + kp_m * _V_MUSCLE + kp_f * _V_FAT
    r = _sim(kp_liver=kp_l, kp_kidney=kp_k, kp_muscle=kp_m, kp_fat=kp_f)
    assert r.vd_effective_L == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# t_half
# ---------------------------------------------------------------------------


def test_t_half_formula():
    """t_half ≈ 0.693 * Vd_eff / CL_total."""
    r = _sim()
    expected = math.log(2.0) * r.vd_effective_L / (5.0 + 1.0)
    assert r.t_half_h == pytest.approx(expected, rel=1e-6)


def test_high_kp_fat_longer_thalf():
    """Higher Kp_fat → larger Vd → longer t_half."""
    r_low = _sim(kp_fat=1.0)
    r_high = _sim(kp_fat=50.0)
    assert r_high.t_half_h > r_low.t_half_h
    assert r_high.vd_effective_L > r_low.vd_effective_L


# ---------------------------------------------------------------------------
# Monotone decrease after IV bolus
# ---------------------------------------------------------------------------


def test_iv_monotone_decrease():
    """After IV bolus, blood concentration should monotonically decrease."""
    r = _sim(route="iv", dt_h=0.05)
    c = r.c_blood_mg_L
    # Allow tiny floating point rounding; check no significant increases
    for i in range(1, len(c)):
        assert c[i] <= c[i - 1] + 1e-9, f"Non-monotone at index {i}: {c[i - 1]:.6f} → {c[i]:.6f}"


def test_iv_cmax_at_t0():
    """IV bolus: Cmax occurs at t=0."""
    r = _sim(route="iv")
    assert r.tmax_h == pytest.approx(0.0, abs=0.1)


# ---------------------------------------------------------------------------
# tissue_concentration_at_t
# ---------------------------------------------------------------------------


def test_tissue_concentration_liver_at_1h():
    """C_liver(1h) = Kp_liver * C_blood(1h)."""
    r = _sim(kp_liver=3.0)
    c_liver = tissue_concentration_at_t(r, "liver", 1.0)
    # Blood concentration at 1h via interpolation
    t_arr = np.asarray(r.times_h)
    c_arr = np.asarray(r.c_blood_mg_L)
    c_blood_1h = float(np.interp(1.0, t_arr, c_arr))
    assert c_liver == pytest.approx(3.0 * c_blood_1h, rel=1e-4)


def test_tissue_concentration_fat_at_12h():
    """C_fat(12h) = Kp_fat * C_blood(12h)."""
    r = _sim(kp_fat=10.0)
    c_fat = tissue_concentration_at_t(r, "fat", 12.0)
    t_arr = np.asarray(r.times_h)
    c_arr = np.asarray(r.c_blood_mg_L)
    c_blood_12h = float(np.interp(12.0, t_arr, c_arr))
    assert c_fat == pytest.approx(10.0 * c_blood_12h, rel=1e-4)


def test_tissue_concentration_invalid_tissue():
    """Invalid tissue name raises ValueError."""
    r = _sim()
    with pytest.raises(ValueError, match="tissue"):
        tissue_concentration_at_t(r, "brain", 1.0)


def test_tissue_concentration_at_zero():
    """t=0 → tissue concentration is Kp * C_blood[0]."""
    r = _sim(route="iv", kp_kidney=2.0)
    c_k = tissue_concentration_at_t(r, "kidney", 0.0)
    assert c_k == pytest.approx(2.0 * r.c_blood_mg_L[0], rel=1e-4)


# ---------------------------------------------------------------------------
# Fat accumulation
# ---------------------------------------------------------------------------


def test_c_fat_at_24h_gt_c_blood():
    """Fat concentration at 24h > blood concentration (Kp_fat > 1)."""
    r = _sim(kp_fat=10.0, t_end_h=24.0)
    # C_blood at 24h
    c_blood_24h = r.c_blood_mg_L[-1]
    assert r.c_fat_at_24h > c_blood_24h


def test_c_liver_at_1h_gt_c_blood():
    """Liver concentration at 1h > blood concentration (Kp_liver=3 > 1)."""
    r = _sim(kp_liver=3.0)
    t_arr = np.asarray(r.times_h)
    c_arr = np.asarray(r.c_blood_mg_L)
    c_blood_1h = float(np.interp(1.0, t_arr, c_arr))
    assert r.c_liver_at_1h > c_blood_1h


# ---------------------------------------------------------------------------
# Dose proportionality (2× dose → 2× AUC)
# ---------------------------------------------------------------------------


def test_dose_linearity_auc():
    """Doubling dose doubles AUC (linear model)."""
    r1 = _sim(dose_mg=100.0)
    r2 = _sim(dose_mg=200.0)
    assert r2.auc_blood == pytest.approx(2.0 * r1.auc_blood, rel=1e-3)


def test_dose_linearity_cmax():
    """Doubling dose doubles Cmax."""
    r1 = _sim(dose_mg=100.0)
    r2 = _sim(dose_mg=200.0)
    assert r2.cmax_blood == pytest.approx(2.0 * r1.cmax_blood, rel=1e-3)


# ---------------------------------------------------------------------------
# Oral vs IV
# ---------------------------------------------------------------------------


def test_oral_delayed_tmax():
    """Oral route produces delayed Tmax (absorption lag)."""
    r_iv = _sim(route="iv")
    r_oral = _sim(route="oral", ka_per_h=0.5, f_oral=1.0)
    assert r_oral.tmax_h > r_iv.tmax_h


# ---------------------------------------------------------------------------
# Result field types
# ---------------------------------------------------------------------------


def test_result_field_types():
    r = _sim()
    assert isinstance(r, FlowLimitedPBPKResult)
    assert isinstance(r.drug_name, str)
    assert isinstance(r.dose_mg, float)
    assert isinstance(r.route, str)
    assert isinstance(r.kp_liver, float)
    assert isinstance(r.kp_kidney, float)
    assert isinstance(r.kp_muscle, float)
    assert isinstance(r.kp_fat, float)
    assert isinstance(r.vd_effective_L, float)
    assert isinstance(r.times_h, list)
    assert isinstance(r.c_blood_mg_L, list)
    assert isinstance(r.cmax_blood, float)
    assert isinstance(r.tmax_h, float)
    assert isinstance(r.auc_blood, float)
    assert isinstance(r.c_liver_at_1h, float)
    assert isinstance(r.c_kidney_at_1h, float)
    assert isinstance(r.c_fat_at_24h, float)
    assert isinstance(r.t_half_h, float)
    assert isinstance(r.notes, list)
