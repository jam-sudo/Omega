"""Tests for flip_flop_pk2 module (Phase 697)."""

import math

import pytest

from omega_pbpk.core.flip_flop_pk2 import (
    FlipFlopResult,
    identify_flip_flop_from_data,
    screen_flip_flop_risk,
    simulate_flip_flop_pk,
)

# ---------------------------------------------------------------------------
# Basic return type and structure
# ---------------------------------------------------------------------------


def test_returns_flip_flop_result():
    result = simulate_flip_flop_pk("DrugA", dose_mg=100, ka_per_h=0.2, ke_per_h=1.0, vd_L=10.0)
    assert isinstance(result, FlipFlopResult)


def test_times_and_concs_are_lists():
    result = simulate_flip_flop_pk("DrugA", dose_mg=100, ka_per_h=0.2, ke_per_h=1.0, vd_L=10.0)
    assert isinstance(result.times_h, list)
    assert isinstance(result.c_plasma_mg_L, list)
    assert len(result.times_h) == len(result.c_plasma_mg_L)
    assert len(result.times_h) > 0


# ---------------------------------------------------------------------------
# Flip-flop detection
# ---------------------------------------------------------------------------


def test_is_flip_flop_true_when_ka_less_than_ke():
    # ka=0.2 < ke=1.0 → flip-flop
    result = simulate_flip_flop_pk("FF", dose_mg=100, ka_per_h=0.2, ke_per_h=1.0, vd_L=10.0)
    assert result.is_flip_flop is True


def test_is_flip_flop_false_when_ka_greater_than_ke():
    # ka=2.0 > ke=0.5 → no flip-flop
    result = simulate_flip_flop_pk("NFF", dose_mg=100, ka_per_h=2.0, ke_per_h=0.5, vd_L=10.0)
    assert result.is_flip_flop is False


def test_ka_ke_ratio_correct():
    ka, ke = 0.3, 1.5
    result = simulate_flip_flop_pk("Drug", dose_mg=100, ka_per_h=ka, ke_per_h=ke, vd_L=10.0)
    assert abs(result.ka_ke_ratio - ka / ke) < 1e-9


def test_ka_ke_ratio_less_than_one_for_flip_flop():
    result = simulate_flip_flop_pk("FF", dose_mg=100, ka_per_h=0.2, ke_per_h=1.0, vd_L=10.0)
    assert result.ka_ke_ratio < 1.0


# ---------------------------------------------------------------------------
# True half-life and absorption half-life
# ---------------------------------------------------------------------------


def test_true_t_half_equals_ln2_over_ke():
    ke = 0.693
    result = simulate_flip_flop_pk("Drug", dose_mg=100, ka_per_h=0.1, ke_per_h=ke, vd_L=10.0)
    expected = math.log(2.0) / ke
    assert abs(result.true_t_half_h - expected) < 1e-9


def test_absorption_t_half_equals_ln2_over_ka():
    ka = 0.3
    result = simulate_flip_flop_pk("Drug", dose_mg=100, ka_per_h=ka, ke_per_h=1.5, vd_L=10.0)
    expected = math.log(2.0) / ka
    assert abs(result.absorption_t_half_h - expected) < 1e-9


def test_apparent_t_half_longer_than_true_t_half_in_flip_flop():
    # When flip-flop, apparent terminal ~ absorption half-life (longer)
    result = simulate_flip_flop_pk(
        "FF", dose_mg=100, ka_per_h=0.1, ke_per_h=2.0, vd_L=10.0, t_end_h=48.0
    )
    assert result.apparent_t_half_h > result.true_t_half_h


# ---------------------------------------------------------------------------
# PK metrics
# ---------------------------------------------------------------------------


def test_cmax_positive():
    result = simulate_flip_flop_pk("Drug", dose_mg=100, ka_per_h=1.0, ke_per_h=0.5, vd_L=10.0)
    assert result.cmax_mg_L > 0.0


def test_tmax_positive():
    result = simulate_flip_flop_pk("Drug", dose_mg=100, ka_per_h=1.0, ke_per_h=0.5, vd_L=10.0)
    assert result.tmax_h > 0.0


def test_auc_positive():
    result = simulate_flip_flop_pk("Drug", dose_mg=100, ka_per_h=1.0, ke_per_h=0.5, vd_L=10.0)
    assert result.auc_mg_h_per_L > 0.0


def test_double_dose_doubles_cmax():
    r1 = simulate_flip_flop_pk("Drug", dose_mg=100, ka_per_h=1.0, ke_per_h=0.5, vd_L=10.0)
    r2 = simulate_flip_flop_pk("Drug", dose_mg=200, ka_per_h=1.0, ke_per_h=0.5, vd_L=10.0)
    assert abs(r2.cmax_mg_L / r1.cmax_mg_L - 2.0) < 0.05


# ---------------------------------------------------------------------------
# Misidentification risk
# ---------------------------------------------------------------------------


def test_misidentification_risk_high_when_ratio_below_0_5():
    # ka/ke = 0.1/1.0 = 0.1 < 0.5 → high
    result = simulate_flip_flop_pk("FF", dose_mg=100, ka_per_h=0.1, ke_per_h=1.0, vd_L=10.0)
    assert result.misidentification_risk == "high"


def test_misidentification_risk_moderate_when_flip_flop_but_ratio_above_0_5():
    # ka/ke = 0.6/1.0 = 0.6 → moderate
    result = simulate_flip_flop_pk("FF", dose_mg=100, ka_per_h=0.6, ke_per_h=1.0, vd_L=10.0)
    assert result.misidentification_risk == "moderate"


def test_misidentification_risk_low_when_no_flip_flop():
    result = simulate_flip_flop_pk("NFF", dose_mg=100, ka_per_h=2.0, ke_per_h=0.5, vd_L=10.0)
    assert result.misidentification_risk == "low"


# ---------------------------------------------------------------------------
# Screen function
# ---------------------------------------------------------------------------


def test_screen_sorted_ascending_by_ka_ke_ratio():
    compounds = [
        {"compound_name": "A", "ka_per_h": 0.5, "ke_per_h": 1.0, "dose_mg": 100, "vd_L": 10},
        {"compound_name": "B", "ka_per_h": 0.1, "ke_per_h": 1.0, "dose_mg": 100, "vd_L": 10},
        {"compound_name": "C", "ka_per_h": 0.8, "ke_per_h": 1.0, "dose_mg": 100, "vd_L": 10},
    ]
    results = screen_flip_flop_risk(compounds)
    ratios = [r.ka_ke_ratio for r in results]
    assert ratios == sorted(ratios)


def test_screen_first_is_most_flip_flop_prone():
    compounds = [
        {"compound_name": "A", "ka_per_h": 0.8, "ke_per_h": 1.0, "dose_mg": 100, "vd_L": 10},
        {"compound_name": "B", "ka_per_h": 0.05, "ke_per_h": 1.0, "dose_mg": 100, "vd_L": 10},
    ]
    results = screen_flip_flop_risk(compounds)
    assert results[0].drug_name == "B"


# ---------------------------------------------------------------------------
# identify_flip_flop_from_data
# ---------------------------------------------------------------------------


def test_identify_flip_flop_from_data_returns_dict():
    times = [0, 1, 2, 4, 8, 12, 24]
    concs = [0, 2, 3, 2.5, 1.5, 0.8, 0.1]
    result = identify_flip_flop_from_data(times, concs, ke_true=1.0)
    assert isinstance(result, dict)
    assert "apparent_ke" in result
    assert "true_ke" in result
    assert "is_flip_flop" in result
    assert "confidence" in result


def test_identify_flip_flop_true_ke_matches_input():
    times = [0, 1, 2, 4, 8]
    concs = [0, 1, 1.5, 1.2, 0.6]
    ke_true = 0.5
    result = identify_flip_flop_from_data(times, concs, ke_true=ke_true)
    assert result["true_ke"] == ke_true


def test_identify_flip_flop_detected_when_apparent_slow():
    # Simulate flip-flop profile where slow absorption drives terminal slope
    # Build a flip-flop PK profile: ka=0.1, ke=2.0
    sim = simulate_flip_flop_pk(
        "FF", dose_mg=100, ka_per_h=0.1, ke_per_h=2.0, vd_L=10.0, t_end_h=48.0
    )
    result = identify_flip_flop_from_data(sim.times_h, sim.c_plasma_mg_L, ke_true=2.0)
    assert result["is_flip_flop"] is True


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validation_dose_zero_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_flip_flop_pk("Drug", dose_mg=0, ka_per_h=1.0, ke_per_h=0.5, vd_L=10.0)


def test_validation_dose_negative_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_flip_flop_pk("Drug", dose_mg=-100, ka_per_h=1.0, ke_per_h=0.5, vd_L=10.0)


def test_validation_ka_zero_raises():
    with pytest.raises(ValueError, match="ka_per_h"):
        simulate_flip_flop_pk("Drug", dose_mg=100, ka_per_h=0.0, ke_per_h=0.5, vd_L=10.0)


def test_validation_ke_zero_raises():
    with pytest.raises(ValueError, match="ke_per_h"):
        simulate_flip_flop_pk("Drug", dose_mg=100, ka_per_h=1.0, ke_per_h=0.0, vd_L=10.0)
