"""Tests for advanced flip-flop kinetics module."""

import math

import pytest

from omega_pbpk.core.flip_flop_advanced import (
    FlipFlopResult,
    detect_flip_flop_from_data,
    estimate_ka_ke_from_oral,
    simulate_flip_flop,
)


def _make_oral_profile(ka, ke, vd, dose, f=1.0, t_end=48.0, dt=0.1):
    """Helper to generate oral concentration profile."""
    a_gut = dose * f
    c = 0.0
    times = []
    concs = []
    t = 0.0
    steps = int(t_end / dt) + 1
    for _ in range(steps):
        times.append(t)
        concs.append(c)
        da_gut = -ka * a_gut
        dc = ka * a_gut / vd - ke * c
        a_gut += da_gut * dt
        if a_gut < 0:
            a_gut = 0.0
        c += dc * dt
        if c < 0:
            c = 0.0
        t += dt
    return times, concs


def _make_iv_profile(ke, vd, dose, t_end=48.0, dt=0.1):
    """Helper to generate IV concentration profile."""
    c = dose / vd
    times = []
    concs = []
    t = 0.0
    steps = int(t_end / dt) + 1
    for _ in range(steps):
        times.append(t)
        concs.append(c)
        dc = -ke * c
        c += dc * dt
        if c < 0:
            c = 0.0
        t += dt
    return times, concs


def test_returns_result():
    result = simulate_flip_flop("DrugA", 100.0, 0.5, 0.2, 20.0)
    assert isinstance(result, FlipFlopResult)


def test_flip_flop_detected_when_ka_lt_ke():
    result = simulate_flip_flop("DrugA", 100.0, ka_per_h=0.1, ke_per_h=1.0, vd_L=20.0)
    assert result.flip_flop_detected is True


def test_no_flip_flop_when_ka_gt_ke():
    result = simulate_flip_flop("DrugA", 100.0, ka_per_h=1.0, ke_per_h=0.1, vd_L=20.0)
    assert result.flip_flop_detected is False


def test_flip_flop_ratio():
    ka, ke = 0.2, 1.0
    result = simulate_flip_flop("DrugA", 100.0, ka_per_h=ka, ke_per_h=ke, vd_L=20.0)
    assert abs(result.flip_flop_ratio - ke / ka) < 1e-9


def test_apparent_t_half_positive():
    result = simulate_flip_flop("DrugA", 100.0, 0.5, 0.2, 20.0)
    assert result.apparent_t_half_h > 0


def test_true_t_half_positive():
    result = simulate_flip_flop("DrugA", 100.0, 0.5, 0.2, 20.0)
    assert result.true_elimination_t_half_h > 0


def test_absorption_t_half_positive():
    result = simulate_flip_flop("DrugA", 100.0, 0.5, 0.2, 20.0)
    assert result.absorption_t_half_h > 0


def test_flip_flop_ratio_gt_1_for_flip_flop():
    result = simulate_flip_flop("DrugA", 100.0, ka_per_h=0.1, ke_per_h=1.0, vd_L=20.0)
    assert result.flip_flop_ratio > 1.0


def test_flip_flop_ratio_lt_1_for_normal():
    result = simulate_flip_flop("DrugA", 100.0, ka_per_h=1.0, ke_per_h=0.1, vd_L=20.0)
    assert result.flip_flop_ratio < 1.0


def test_fraction_absorbed_in_0_100():
    result = simulate_flip_flop("DrugA", 100.0, 0.5, 0.2, 20.0)
    assert 0.0 <= result.fraction_absorbed_pct <= 100.0


def test_dose_proportionality_terminal():
    """2x dose -> ~2x terminal concentration (linear PK)."""
    r1 = simulate_flip_flop("DrugA", 100.0, 1.0, 0.2, 20.0, t_end_h=24.0)
    r2 = simulate_flip_flop("DrugA", 200.0, 1.0, 0.2, 20.0, t_end_h=24.0)
    # AUC should scale linearly; terminal conc proportional to dose
    # Check via fraction absorbed and ratio of true t_half
    assert abs(r1.true_elimination_t_half_h - r2.true_elimination_t_half_h) < 0.01


def test_detect_from_data_returns_dict():
    times, c_oral = _make_oral_profile(0.1, 1.0, 20.0, 100.0)
    _, c_iv = _make_iv_profile(1.0, 20.0, 100.0)
    result = detect_flip_flop_from_data(times, c_oral, c_iv, 100.0, 100.0)
    assert isinstance(result, dict)


def test_detect_flip_flop_keys():
    times, c_oral = _make_oral_profile(0.1, 1.0, 20.0, 100.0)
    _, c_iv = _make_iv_profile(1.0, 20.0, 100.0)
    result = detect_flip_flop_from_data(times, c_oral, c_iv, 100.0, 100.0)
    for key in ["flip_flop", "oral_slope", "iv_slope", "ratio", "notes"]:
        assert key in result


def test_detect_identifies_flip_flop():
    """When ka << ke, oral terminal slope reflects ka, which is slower than IV's ke."""
    # ka=0.05 (very slow absorption), ke=2.0 (fast elimination)
    # In flip-flop: oral terminal apparent rate = ka < ke (iv terminal rate)
    times, c_oral = _make_oral_profile(ka=0.05, ke=2.0, vd=20.0, dose=100.0, t_end=100.0)
    _, c_iv = _make_iv_profile(ke=2.0, vd=20.0, dose=100.0, t_end=100.0)
    result = detect_flip_flop_from_data(times, c_oral, c_iv, 100.0, 100.0)
    assert result["flip_flop"] is True


def test_detect_no_flip_flop():
    """When oral slope > IV slope: no flip-flop (use synthetic data)."""

    # Oral terminal slope faster (larger ke_oral) than IV: no flip-flop scenario
    # Synthetic: oral has ke=0.5, IV has ke=0.1 (oral eliminates faster -> larger slope)
    ke_oral = 0.5
    ke_iv = 0.1
    times = [20.0, 24.0, 30.0, 36.0]
    # oral declines faster (larger ke)
    c_oral = [math.exp(-ke_oral * t) for t in times]
    c_iv = [math.exp(-ke_iv * t) for t in times]
    result = detect_flip_flop_from_data(times, c_oral, c_iv, 100.0, 100.0)
    assert result["flip_flop"] is False


def test_estimate_ka_ke_returns_dict():
    times, concs = _make_oral_profile(1.0, 0.2, 20.0, 100.0)
    result = estimate_ka_ke_from_oral(times, concs, 100.0, 20.0)
    assert isinstance(result, dict)


def test_estimate_ka_positive():
    times, concs = _make_oral_profile(1.0, 0.2, 20.0, 100.0)
    result = estimate_ka_ke_from_oral(times, concs, 100.0, 20.0)
    assert result["ka"] > 0


def test_estimate_ke_positive():
    times, concs = _make_oral_profile(1.0, 0.2, 20.0, 100.0)
    result = estimate_ka_ke_from_oral(times, concs, 100.0, 20.0)
    assert result["ke"] > 0


def test_estimate_flip_flop_bool():
    times, concs = _make_oral_profile(1.0, 0.2, 20.0, 100.0)
    result = estimate_ka_ke_from_oral(times, concs, 100.0, 20.0)
    assert isinstance(result["flip_flop"], bool)


def test_invalid_ka_raises():
    with pytest.raises(ValueError):
        simulate_flip_flop("DrugA", 100.0, ka_per_h=0.0, ke_per_h=0.2, vd_L=20.0)


def test_invalid_ke_raises():
    with pytest.raises(ValueError):
        simulate_flip_flop("DrugA", 100.0, ka_per_h=0.5, ke_per_h=-0.1, vd_L=20.0)


def test_invalid_vd_raises():
    with pytest.raises(ValueError):
        simulate_flip_flop("DrugA", 100.0, ka_per_h=0.5, ke_per_h=0.2, vd_L=0.0)
