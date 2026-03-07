"""Tests for Phase 644 — flip_flop_characterizer.py (flip-flop PK)."""

import math

import pytest

from omega_pbpk.core.flip_flop_characterizer import (
    FlipFlopResult,
    detect_flip_flop,
    simulate_flip_flop_pk,
)

VALID_KWARGS = dict(
    drug_name="TestDrug",
    dose_mg=100.0,
    ka_per_h=0.5,
    ke_per_h=0.3,
    vd_L=50.0,
    f_oral=1.0,
    t_end_h=48.0,
    dt_h=0.05,
)


def make_result(**overrides):
    kwargs = dict(VALID_KWARGS)
    kwargs.update(overrides)
    return simulate_flip_flop_pk(**kwargs)


# --- Basic return type ---
def test_returns_flip_flop_result():
    r = make_result()
    assert isinstance(r, FlipFlopResult)


def test_cmax_positive():
    r = make_result()
    assert r.cmax_mg_L > 0.0


def test_auc_positive():
    r = make_result()
    assert r.auc_mg_h_per_L > 0.0


def test_tmax_positive():
    r = make_result()
    assert r.tmax_h > 0.0


def test_true_t_half_positive():
    r = make_result()
    assert r.true_t_half_h > 0.0


def test_apparent_t_half_positive():
    r = make_result()
    assert r.apparent_t_half_h > 0.0


def test_flip_flop_ratio_equals_ke_over_ka():
    r = make_result(ka_per_h=0.5, ke_per_h=0.3)
    expected = 0.3 / 0.5
    assert abs(r.flip_flop_ratio - expected) < 1e-9


def test_notes_nonempty():
    r = make_result()
    assert len(r.notes) > 0


# --- Flip-flop detection ---
def test_ka_lt_ke_is_flip_flop_true():
    r = make_result(ka_per_h=0.1, ke_per_h=1.0)
    assert r.is_flip_flop is True


def test_ka_gt_ke_is_flip_flop_false():
    r = make_result(ka_per_h=1.0, ke_per_h=0.2)
    assert r.is_flip_flop is False


def test_flip_flop_apparent_t_half_reflects_ka():
    """In flip-flop, terminal slope reflects ka, so apparent t½ >> true t½."""
    r = make_result(ka_per_h=0.1, ke_per_h=1.0)
    assert r.apparent_t_half_h > r.true_t_half_h


def test_non_flip_flop_apparent_t_half_close_to_true():
    """Normal PK: apparent t½ should be within factor of 2 of true t½."""
    r = make_result(ka_per_h=1.0, ke_per_h=0.2)
    ratio = r.apparent_t_half_h / r.true_t_half_h
    assert 0.3 < ratio < 3.0


# --- Linearity ---
def test_double_dose_double_cmax():
    r1 = make_result(dose_mg=100.0)
    r2 = make_result(dose_mg=200.0)
    assert abs(r2.cmax_mg_L / r1.cmax_mg_L - 2.0) < 0.05


def test_double_dose_double_auc():
    r1 = make_result(dose_mg=100.0)
    r2 = make_result(dose_mg=200.0)
    assert abs(r2.auc_mg_h_per_L / r1.auc_mg_h_per_L - 2.0) < 0.05


def test_higher_f_oral_higher_cmax():
    r_low = make_result(f_oral=0.5)
    r_high = make_result(f_oral=1.0)
    assert r_high.cmax_mg_L > r_low.cmax_mg_L


# --- detect_flip_flop ---
def test_detect_flip_flop_returns_dict():
    times = [float(i) for i in range(20)]
    concs = [math.exp(-0.1 * t) for t in times]
    result = detect_flip_flop(times, concs)
    assert isinstance(result, dict)


def test_detect_flip_flop_keys():
    times = [float(i) for i in range(20)]
    concs = [math.exp(-0.1 * t) for t in times]
    result = detect_flip_flop(times, concs)
    assert "is_flip_flop" in result
    assert "apparent_ke" in result
    assert "confidence" in result
    assert "notes" in result


def test_detect_flip_flop_with_iv_ke_slow_oral():
    """Slow oral data (apparent_ke~0.1) vs IV ke=1.0 → flip-flop."""
    times = [float(i) * 2 for i in range(20)]
    concs = [5.0 * math.exp(-0.1 * t) for t in times]
    result = detect_flip_flop(times, concs, iv_ke_per_h=1.0)
    assert result["is_flip_flop"] is True


def test_detect_flip_flop_confidence_valid():
    times = [float(i) for i in range(20)]
    concs = [math.exp(-0.3 * t) for t in times]
    result = detect_flip_flop(times, concs)
    assert result["confidence"] in {"high", "moderate", "low"}


# --- Validation errors ---
def test_validation_dose_zero_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        make_result(dose_mg=0.0)


def test_validation_ka_zero_raises():
    with pytest.raises(ValueError, match="ka_per_h"):
        make_result(ka_per_h=0.0)


def test_validation_ke_zero_raises():
    with pytest.raises(ValueError, match="ke_per_h"):
        make_result(ke_per_h=0.0)


def test_validation_f_oral_above_one_raises():
    with pytest.raises(ValueError, match="f_oral"):
        make_result(f_oral=1.5)
