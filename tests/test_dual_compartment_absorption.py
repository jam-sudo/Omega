"""Tests for Phase 579 — dual_compartment_absorption."""

import math

import pytest

from omega_pbpk.core.dual_compartment_absorption import (
    biphasic_absorption_signature,
    compare_absorption_profiles,
    simulate_dual_absorption,
)

# ---------------------------------------------------------------------------
# simulate_dual_absorption — basic structure
# ---------------------------------------------------------------------------


def test_returns_expected_keys():
    res = simulate_dual_absorption("Drug", 100.0, 0.5, 2.0, 0.2, 5.0, 20.0)
    for key in (
        "drug_name",
        "times_h",
        "c_plasma_mg_L",
        "cmax",
        "tmax_h",
        "auc",
        "t_half_h",
        "notes",
    ):
        assert key in res


def test_times_and_concentrations_same_length():
    res = simulate_dual_absorption("Drug", 100.0, 0.5, 2.0, 0.2, 5.0, 20.0)
    assert len(res["times_h"]) == len(res["c_plasma_mg_L"])


def test_cmax_positive():
    res = simulate_dual_absorption("Drug", 100.0, 0.5, 2.0, 0.2, 5.0, 20.0)
    assert res["cmax"] > 0


def test_auc_positive():
    res = simulate_dual_absorption("Drug", 100.0, 0.5, 2.0, 0.2, 5.0, 20.0)
    assert res["auc"] > 0


def test_t_half_equals_log2_over_ke():
    cl = 5.0
    vd = 20.0
    ke = cl / vd
    res = simulate_dual_absorption("Drug", 100.0, 0.5, 2.0, 0.2, cl, vd)
    assert abs(res["t_half_h"] - math.log(2) / ke) < 1e-9


def test_concentrations_non_negative():
    res = simulate_dual_absorption("Drug", 100.0, 0.5, 2.0, 0.2, 5.0, 20.0)
    assert all(c >= 0 for c in res["c_plasma_mg_L"])


def test_starts_at_zero_concentration():
    res = simulate_dual_absorption("Drug", 100.0, 0.5, 2.0, 0.2, 5.0, 20.0)
    assert res["c_plasma_mg_L"][0] == 0.0


# ---------------------------------------------------------------------------
# f_fast effect on tmax
# ---------------------------------------------------------------------------


def test_higher_f_fast_yields_earlier_tmax():
    res_high = simulate_dual_absorption("Drug", 100.0, 0.8, 2.0, 0.2, 5.0, 20.0)
    res_low = simulate_dual_absorption("Drug", 100.0, 0.2, 2.0, 0.2, 5.0, 20.0)
    assert res_high["tmax_h"] <= res_low["tmax_h"]


def test_f_fast_close_to_one_approximates_fast_single_compartment():
    """f_fast=0.99 should give a tmax close to that of fast-only absorption."""
    res_near_fast = simulate_dual_absorption("Drug", 100.0, 0.99, 2.0, 0.2, 5.0, 20.0)
    res_mid = simulate_dual_absorption("Drug", 100.0, 0.5, 2.0, 0.2, 5.0, 20.0)
    # Near-fast should peak earlier than a balanced mix
    assert res_near_fast["tmax_h"] <= res_mid["tmax_h"]


def test_f_fast_close_to_zero_approximates_slow_absorption():
    """f_fast=0.01 should give later tmax than f_fast=0.99."""
    res_slow = simulate_dual_absorption("Drug", 100.0, 0.01, 2.0, 0.2, 5.0, 20.0)
    res_fast = simulate_dual_absorption("Drug", 100.0, 0.99, 2.0, 0.2, 5.0, 20.0)
    assert res_slow["tmax_h"] >= res_fast["tmax_h"]


# ---------------------------------------------------------------------------
# Dose proportionality
# ---------------------------------------------------------------------------


def test_double_dose_doubles_cmax():
    res1 = simulate_dual_absorption("Drug", 100.0, 0.5, 2.0, 0.2, 5.0, 20.0)
    res2 = simulate_dual_absorption("Drug", 200.0, 0.5, 2.0, 0.2, 5.0, 20.0)
    assert abs(res2["cmax"] / res1["cmax"] - 2.0) < 0.01


def test_double_dose_doubles_auc():
    res1 = simulate_dual_absorption("Drug", 100.0, 0.5, 2.0, 0.2, 5.0, 20.0)
    res2 = simulate_dual_absorption("Drug", 200.0, 0.5, 2.0, 0.2, 5.0, 20.0)
    assert abs(res2["auc"] / res1["auc"] - 2.0) < 0.01


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_invalid_f_fast_zero_raises():
    with pytest.raises(ValueError, match="f_fast"):
        simulate_dual_absorption("Drug", 100.0, 0.0, 2.0, 0.2, 5.0, 20.0)


def test_invalid_f_fast_one_raises():
    with pytest.raises(ValueError, match="f_fast"):
        simulate_dual_absorption("Drug", 100.0, 1.0, 2.0, 0.2, 5.0, 20.0)


def test_invalid_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_dual_absorption("Drug", 0.0, 0.5, 2.0, 0.2, 5.0, 20.0)


def test_ka_fast_must_exceed_ka_slow():
    with pytest.raises(ValueError):
        simulate_dual_absorption("Drug", 100.0, 0.5, 0.1, 0.2, 5.0, 20.0)


# ---------------------------------------------------------------------------
# compare_absorption_profiles
# ---------------------------------------------------------------------------


def test_compare_profiles_returns_correct_length():
    f_vals = [0.1, 0.3, 0.5, 0.7, 0.9]
    results = compare_absorption_profiles(100.0, 5.0, 20.0, f_vals)
    assert len(results) == len(f_vals)


def test_compare_profiles_contains_expected_keys():
    results = compare_absorption_profiles(100.0, 5.0, 20.0, [0.3, 0.7])
    for r in results:
        assert "f_fast" in r and "cmax" in r and "tmax_h" in r and "auc" in r


def test_compare_profiles_cmax_all_positive():
    results = compare_absorption_profiles(100.0, 5.0, 20.0, [0.2, 0.5, 0.8])
    assert all(r["cmax"] > 0 for r in results)


# ---------------------------------------------------------------------------
# biphasic_absorption_signature
# ---------------------------------------------------------------------------


def test_single_peak_not_biphasic():
    # Unimodal bell curve
    times = list(range(25))
    c = [
        0,
        1,
        3,
        5,
        7,
        8,
        7,
        5,
        3,
        2,
        1.5,
        1.2,
        1.0,
        0.9,
        0.8,
        0.7,
        0.6,
        0.5,
        0.4,
        0.35,
        0.3,
        0.25,
        0.2,
        0.15,
        0.1,
    ]
    sig = biphasic_absorption_signature(times, c)
    assert sig["is_biphasic"] is False
    assert sig["n_peaks"] == 1


def test_biphasic_pattern_detected():
    # Two distinct peaks
    times = list(range(10))
    c = [0.0, 3.0, 2.0, 4.0, 2.5, 1.5, 1.0, 0.7, 0.4, 0.2]
    sig = biphasic_absorption_signature(times, c)
    assert sig["is_biphasic"] is True
    assert sig["n_peaks"] >= 2


def test_biphasic_signature_short_input():
    sig = biphasic_absorption_signature([0, 1], [0.0, 1.0])
    assert sig["is_biphasic"] is False
    assert sig["n_peaks"] == 0
