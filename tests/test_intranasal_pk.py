"""Tests for Phase 727 — Intranasal Drug Delivery PK."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.core.intranasal_pk import (
    IntranasalPKResult,
    compare_intranasal_vs_oral,
    simulate_intranasal_pk,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sim(**kwargs) -> IntranasalPKResult:
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=1.0,
        kabs_nasal_per_h=1.5,
        kclear_per_h=0.5,
        f_nasal=0.6,
        f_swallowed_oral=0.3,
        ka_oral_per_h=0.5,
        cl_L_per_h=5.0,
        vd_L=50.0,
        t_end_h=12.0,
        dt_h=0.05,
    )
    defaults.update(kwargs)
    return simulate_intranasal_pk(**defaults)


# ---------------------------------------------------------------------------
# Basic return type and structure
# ---------------------------------------------------------------------------


def test_returns_intranasal_pk_result():
    r = _sim()
    assert isinstance(r, IntranasalPKResult)


def test_cmax_positive():
    r = _sim()
    assert r.cmax_mg_L > 0.0


def test_auc_positive():
    r = _sim()
    assert r.auc_mg_h_per_L > 0.0


def test_time_series_length_consistent():
    r = _sim(t_end_h=6.0, dt_h=0.1)
    assert len(r.times_h) == len(r.c_plasma_mg_L)
    assert len(r.times_h) == len(r.a_nasal_mg)


def test_time_starts_at_zero():
    r = _sim()
    assert r.times_h[0] == pytest.approx(0.0, abs=1e-10)


def test_nasal_amount_starts_at_dose():
    r = _sim(dose_mg=2.0)
    assert r.a_nasal_mg[0] == pytest.approx(2.0, rel=1e-6)


def test_plasma_starts_at_zero():
    r = _sim()
    assert r.c_plasma_mg_L[0] == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Linear dose scaling
# ---------------------------------------------------------------------------


def test_linear_dose_scaling_cmax():
    r1 = _sim(dose_mg=1.0)
    r2 = _sim(dose_mg=2.0)
    assert r2.cmax_mg_L == pytest.approx(2.0 * r1.cmax_mg_L, rel=1e-3)


def test_linear_dose_scaling_auc():
    r1 = _sim(dose_mg=1.0)
    r2 = _sim(dose_mg=2.0)
    assert r2.auc_mg_h_per_L == pytest.approx(2.0 * r1.auc_mg_h_per_L, rel=1e-3)


# ---------------------------------------------------------------------------
# Kinetic behaviour
# ---------------------------------------------------------------------------


def test_higher_kabs_nasal_faster_tmax():
    r_slow = _sim(kabs_nasal_per_h=0.5, kclear_per_h=0.1)
    r_fast = _sim(kabs_nasal_per_h=3.0, kclear_per_h=0.1)
    assert r_fast.tmax_h <= r_slow.tmax_h


def test_higher_kclear_reduces_auc():
    r_low = _sim(kclear_per_h=0.1)
    r_high = _sim(kclear_per_h=5.0)
    assert r_high.auc_mg_h_per_L < r_low.auc_mg_h_per_L


def test_higher_kclear_reduces_cmax():
    r_low = _sim(kclear_per_h=0.1)
    r_high = _sim(kclear_per_h=5.0)
    assert r_high.cmax_mg_L < r_low.cmax_mg_L


def test_high_f_nasal_increases_auc():
    r_low = _sim(f_nasal=0.1, f_swallowed_oral=0.0)
    r_high = _sim(f_nasal=0.95, f_swallowed_oral=0.0)
    assert r_high.auc_mg_h_per_L > r_low.auc_mg_h_per_L


# ---------------------------------------------------------------------------
# Bioavailability metrics
# ---------------------------------------------------------------------------


def test_f_systemic_nasal_equals_f_nasal():
    f_n = 0.75
    r = _sim(f_nasal=f_n)
    assert r.f_systemic_nasal == pytest.approx(f_n, rel=1e-9)


def test_f_systemic_total_formula():
    f_n = 0.6
    f_sw = 0.3
    r = _sim(f_nasal=f_n, f_swallowed_oral=f_sw)
    expected = f_n + (1.0 - f_n) * f_sw
    assert r.f_systemic_total == pytest.approx(expected, rel=1e-9)


def test_f_systemic_total_greater_than_f_nasal_when_oral_positive():
    r = _sim(f_nasal=0.6, f_swallowed_oral=0.3)
    assert r.f_systemic_total > r.f_systemic_nasal


# ---------------------------------------------------------------------------
# Half-life
# ---------------------------------------------------------------------------


def test_t_half_h_from_cl_vd():
    cl = 5.0
    vd = 50.0
    r = _sim(cl_L_per_h=cl, vd_L=vd)
    expected_t_half = math.log(2.0) / (cl / vd)
    assert r.t_half_h == pytest.approx(expected_t_half, rel=1e-6)


# ---------------------------------------------------------------------------
# Notes field
# ---------------------------------------------------------------------------


def test_notes_nonempty():
    r = _sim()
    assert len(r.notes) > 0


def test_notes_is_string():
    r = _sim()
    assert isinstance(r.notes, str)


# ---------------------------------------------------------------------------
# compare_intranasal_vs_oral
# ---------------------------------------------------------------------------


def test_compare_returns_dict():
    result = compare_intranasal_vs_oral(drug_name="Drug", dose_mg=1.0)
    assert isinstance(result, dict)


def test_compare_has_required_keys():
    result = compare_intranasal_vs_oral(drug_name="Drug", dose_mg=1.0)
    for key in ("intranasal_result", "oral_result", "auc_ratio", "cmax_ratio", "notes"):
        assert key in result, f"Missing key: {key}"


def test_compare_auc_ratio_positive():
    result = compare_intranasal_vs_oral(drug_name="Drug", dose_mg=1.0)
    assert result["auc_ratio"] > 0.0


def test_compare_cmax_ratio_positive():
    result = compare_intranasal_vs_oral(drug_name="Drug", dose_mg=1.0)
    assert result["cmax_ratio"] > 0.0


def test_compare_intranasal_result_type():
    result = compare_intranasal_vs_oral(drug_name="Drug", dose_mg=1.0)
    assert isinstance(result["intranasal_result"], IntranasalPKResult)


def test_compare_oral_result_type():
    result = compare_intranasal_vs_oral(drug_name="Drug", dose_mg=1.0)
    assert isinstance(result["oral_result"], IntranasalPKResult)


def test_compare_notes_nonempty():
    result = compare_intranasal_vs_oral(drug_name="Drug", dose_mg=1.0)
    assert len(result["notes"]) > 0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validation_dose_zero_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        _sim(dose_mg=0.0)


def test_validation_dose_negative_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        _sim(dose_mg=-1.0)


def test_validation_f_nasal_zero_raises():
    with pytest.raises(ValueError, match="f_nasal"):
        _sim(f_nasal=0.0)


def test_validation_f_nasal_one_raises():
    with pytest.raises(ValueError, match="f_nasal"):
        _sim(f_nasal=1.0)


def test_validation_f_nasal_above_one_raises():
    with pytest.raises(ValueError, match="f_nasal"):
        _sim(f_nasal=1.1)


def test_validation_cl_zero_raises():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        _sim(cl_L_per_h=0.0)


def test_validation_vd_zero_raises():
    with pytest.raises(ValueError, match="vd_L"):
        _sim(vd_L=0.0)


def test_validation_kabs_zero_raises():
    with pytest.raises(ValueError, match="kabs_nasal_per_h"):
        _sim(kabs_nasal_per_h=0.0)
