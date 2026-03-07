"""Tests for Phase 728 — Intestinal Metabolism PK."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.core.intestinal_metabolism_pk import (
    IntestinalMetabPKResult,
    assess_gut_wall_extraction,
    simulate_intestinal_metabolism_pk,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sim(**kwargs) -> IntestinalMetabPKResult:
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=1.0,
        ka_per_h=1.0,
        fg=0.7,
        fh=0.7,
        f_abs=1.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
        ke_portal_per_h=5.0,
        t_end_h=12.0,
        dt_h=0.05,
    )
    defaults.update(kwargs)
    return simulate_intestinal_metabolism_pk(**defaults)


# ---------------------------------------------------------------------------
# Return type and basic structure
# ---------------------------------------------------------------------------


def test_returns_intestinal_metab_pk_result():
    r = _sim()
    assert isinstance(r, IntestinalMetabPKResult)


def test_cmax_positive():
    r = _sim()
    assert r.cmax_mg_L > 0.0


def test_auc_positive():
    r = _sim()
    assert r.auc_mg_h_per_L > 0.0


def test_time_series_length_consistent():
    r = _sim(t_end_h=6.0, dt_h=0.1)
    assert len(r.times_h) == len(r.c_plasma_mg_L)
    assert len(r.times_h) == len(r.a_gut_mg)
    assert len(r.times_h) == len(r.a_portal_mg)


def test_portal_list_length_equals_times_length():
    r = _sim(t_end_h=4.0, dt_h=0.05)
    assert len(r.a_portal_mg) == len(r.times_h)


def test_time_starts_at_zero():
    r = _sim()
    assert r.times_h[0] == pytest.approx(0.0, abs=1e-10)


def test_plasma_starts_at_zero():
    r = _sim()
    assert r.c_plasma_mg_L[0] == pytest.approx(0.0, abs=1e-10)


def test_portal_starts_at_zero():
    r = _sim()
    assert r.a_portal_mg[0] == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# f_oral = f_abs * fg * fh
# ---------------------------------------------------------------------------


def test_f_oral_formula():
    r = _sim(f_abs=0.8, fg=0.6, fh=0.5)
    expected = 0.8 * 0.6 * 0.5
    assert r.f_oral == pytest.approx(expected, rel=1e-9)


def test_f_oral_unity_when_all_one():
    r = _sim(f_abs=1.0, fg=1.0, fh=1.0)
    assert r.f_oral == pytest.approx(1.0, rel=1e-9)


def test_fg_stored_correctly():
    r = _sim(fg=0.42)
    assert r.fg == pytest.approx(0.42, rel=1e-9)


def test_fh_stored_correctly():
    r = _sim(fh=0.88)
    assert r.fh == pytest.approx(0.88, rel=1e-9)


# ---------------------------------------------------------------------------
# Bioavailability effects on exposure
# ---------------------------------------------------------------------------


def test_fg_one_fh_one_gives_highest_auc():
    r_full = _sim(fg=1.0, fh=1.0)
    r_partial = _sim(fg=0.5, fh=0.5)
    assert r_full.auc_mg_h_per_L > r_partial.auc_mg_h_per_L


def test_low_fg_gives_low_cmax():
    r_high = _sim(fg=0.9, fh=0.9)
    r_low = _sim(fg=0.1, fh=0.9)
    assert r_low.cmax_mg_L < r_high.cmax_mg_L


def test_low_fg_gives_low_auc():
    r_high = _sim(fg=0.9, fh=0.9)
    r_low = _sim(fg=0.1, fh=0.9)
    assert r_low.auc_mg_h_per_L < r_high.auc_mg_h_per_L


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
# Half-life
# ---------------------------------------------------------------------------


def test_t_half_h_from_cl_vd():
    cl = 5.0
    vd = 50.0
    r = _sim(cl_L_per_h=cl, vd_L=vd)
    expected = math.log(2.0) / (cl / vd)
    assert r.t_half_h == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


def test_notes_nonempty():
    r = _sim()
    assert len(r.notes) > 0


def test_notes_is_string():
    r = _sim()
    assert isinstance(r.notes, str)


# ---------------------------------------------------------------------------
# assess_gut_wall_extraction
# ---------------------------------------------------------------------------


def test_assess_returns_dict():
    result = assess_gut_wall_extraction(ka_per_h=1.0, clint_gut_L_per_h=10.0, fu_gut=0.5)
    assert isinstance(result, dict)


def test_assess_has_fg_key():
    result = assess_gut_wall_extraction(ka_per_h=1.0, clint_gut_L_per_h=10.0, fu_gut=0.5)
    assert "fg" in result


def test_assess_fg_in_range():
    result = assess_gut_wall_extraction(ka_per_h=1.0, clint_gut_L_per_h=10.0, fu_gut=0.5)
    assert 0.0 < result["fg"] <= 1.0


def test_assess_gut_extraction_ratio_in_range():
    result = assess_gut_wall_extraction(ka_per_h=1.0, clint_gut_L_per_h=10.0, fu_gut=0.5)
    assert "gut_extraction_ratio" in result
    assert 0.0 <= result["gut_extraction_ratio"] < 1.0


def test_assess_high_clint_gives_low_fg():
    low_clint = assess_gut_wall_extraction(ka_per_h=1.0, clint_gut_L_per_h=1.0, fu_gut=0.5)
    high_clint = assess_gut_wall_extraction(ka_per_h=1.0, clint_gut_L_per_h=100.0, fu_gut=0.5)
    assert high_clint["fg"] < low_clint["fg"]


def test_assess_zero_clint_gives_fg_one():
    result = assess_gut_wall_extraction(ka_per_h=1.0, clint_gut_L_per_h=0.0, fu_gut=0.5)
    assert result["fg"] == pytest.approx(1.0, rel=1e-9)


def test_assess_fg_plus_extraction_equals_one():
    result = assess_gut_wall_extraction(ka_per_h=1.0, clint_gut_L_per_h=10.0, fu_gut=0.5)
    assert result["fg"] + result["gut_extraction_ratio"] == pytest.approx(1.0, rel=1e-9)


def test_assess_notes_nonempty():
    result = assess_gut_wall_extraction(ka_per_h=1.0, clint_gut_L_per_h=10.0, fu_gut=0.5)
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


def test_validation_fg_zero_raises():
    with pytest.raises(ValueError, match="fg"):
        _sim(fg=0.0)


def test_validation_fg_above_one_raises():
    with pytest.raises(ValueError, match="fg"):
        _sim(fg=1.1)


def test_validation_fh_zero_raises():
    with pytest.raises(ValueError, match="fh"):
        _sim(fh=0.0)


def test_validation_cl_zero_raises():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        _sim(cl_L_per_h=0.0)


def test_validation_vd_zero_raises():
    with pytest.raises(ValueError, match="vd_L"):
        _sim(vd_L=0.0)
