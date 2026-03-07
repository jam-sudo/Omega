"""Tests for Phase 561: flip-flop pharmacokinetics."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.core.flip_flop_pk import (
    compare_formulations,
    detect_flip_flop,
    flip_flop_ratio,
    simulate_flip_flop_pk,
    true_half_life_from_iv,
)

# ---------------------------------------------------------------------------
# detect_flip_flop
# ---------------------------------------------------------------------------


def test_detect_flip_flop_true_when_ka_less_than_ke():
    assert detect_flip_flop(ka_per_h=0.2, ke_per_h=1.0) is True


def test_detect_flip_flop_false_when_ka_greater_than_ke():
    assert detect_flip_flop(ka_per_h=2.0, ke_per_h=0.5) is False


def test_detect_flip_flop_false_when_ka_equals_ke():
    # ka == ke is NOT flip-flop by definition (not strictly less)
    assert detect_flip_flop(ka_per_h=1.0, ke_per_h=1.0) is False


def test_detect_flip_flop_invalid_ka():
    with pytest.raises(ValueError, match="ka_per_h"):
        detect_flip_flop(ka_per_h=0.0, ke_per_h=1.0)


def test_detect_flip_flop_invalid_ke():
    with pytest.raises(ValueError, match="ke_per_h"):
        detect_flip_flop(ka_per_h=1.0, ke_per_h=-0.1)


# ---------------------------------------------------------------------------
# flip_flop_ratio
# ---------------------------------------------------------------------------


def test_flip_flop_ratio_less_than_one_in_flip_flop():
    ratio = flip_flop_ratio(ka_per_h=0.3, ke_per_h=1.5)
    assert ratio < 1.0
    assert ratio == pytest.approx(0.3 / 1.5)


def test_flip_flop_ratio_greater_than_one_in_normal_pk():
    ratio = flip_flop_ratio(ka_per_h=2.0, ke_per_h=0.5)
    assert ratio > 1.0
    assert ratio == pytest.approx(4.0)


def test_flip_flop_ratio_invalid_ka():
    with pytest.raises(ValueError, match="ka_per_h"):
        flip_flop_ratio(ka_per_h=0.0, ke_per_h=1.0)


def test_flip_flop_ratio_invalid_ke():
    with pytest.raises(ValueError, match="ke_per_h"):
        flip_flop_ratio(ka_per_h=1.0, ke_per_h=0.0)


# ---------------------------------------------------------------------------
# true_half_life_from_iv
# ---------------------------------------------------------------------------


def test_true_half_life_from_iv_formula():
    ke = 0.5
    expected = math.log(2.0) / ke
    assert true_half_life_from_iv(ke) == pytest.approx(expected)


def test_true_half_life_from_iv_invalid_ke():
    with pytest.raises(ValueError, match="ke_per_h"):
        true_half_life_from_iv(ke_per_h=0.0)


# ---------------------------------------------------------------------------
# simulate_flip_flop_pk — input validation
# ---------------------------------------------------------------------------


def test_simulate_invalid_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_flip_flop_pk("Drug", dose_mg=0, ka_per_h=0.5, ke_per_h=1.0, vd_L=20.0)


def test_simulate_invalid_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_flip_flop_pk("Drug", dose_mg=-10, ka_per_h=0.5, ke_per_h=1.0, vd_L=20.0)


def test_simulate_invalid_ka_zero():
    with pytest.raises(ValueError, match="ka_per_h"):
        simulate_flip_flop_pk("Drug", dose_mg=100, ka_per_h=0.0, ke_per_h=1.0, vd_L=20.0)


def test_simulate_invalid_ke_zero():
    with pytest.raises(ValueError, match="ke_per_h"):
        simulate_flip_flop_pk("Drug", dose_mg=100, ka_per_h=1.0, ke_per_h=0.0, vd_L=20.0)


def test_simulate_invalid_vd_zero():
    with pytest.raises(ValueError, match="vd_L"):
        simulate_flip_flop_pk("Drug", dose_mg=100, ka_per_h=1.0, ke_per_h=0.5, vd_L=0.0)


def test_simulate_invalid_f_oral_zero():
    with pytest.raises(ValueError, match="f_oral"):
        simulate_flip_flop_pk(
            "Drug", dose_mg=100, ka_per_h=1.0, ke_per_h=0.5, vd_L=20.0, f_oral=0.0
        )


def test_simulate_invalid_f_oral_above_one():
    with pytest.raises(ValueError, match="f_oral"):
        simulate_flip_flop_pk(
            "Drug", dose_mg=100, ka_per_h=1.0, ke_per_h=0.5, vd_L=20.0, f_oral=1.2
        )


# ---------------------------------------------------------------------------
# simulate_flip_flop_pk — is_flip_flop flag
# ---------------------------------------------------------------------------


def test_simulate_is_flip_flop_true():
    res = simulate_flip_flop_pk("FlipDrug", dose_mg=100, ka_per_h=0.2, ke_per_h=2.0, vd_L=20.0)
    assert res["is_flip_flop"] is True


def test_simulate_is_flip_flop_false():
    res = simulate_flip_flop_pk("NormalDrug", dose_mg=100, ka_per_h=2.0, ke_per_h=0.2, vd_L=20.0)
    assert res["is_flip_flop"] is False


# ---------------------------------------------------------------------------
# simulate_flip_flop_pk — output values
# ---------------------------------------------------------------------------


def test_simulate_cmax_positive():
    res = simulate_flip_flop_pk("Drug", dose_mg=200, ka_per_h=0.5, ke_per_h=1.5, vd_L=30.0)
    assert res["cmax"] > 0


def test_simulate_auc_positive():
    res = simulate_flip_flop_pk("Drug", dose_mg=100, ka_per_h=0.3, ke_per_h=1.0, vd_L=20.0)
    assert res["auc"] > 0


def test_simulate_tmax_in_range():
    res = simulate_flip_flop_pk(
        "Drug", dose_mg=100, ka_per_h=0.5, ke_per_h=2.0, vd_L=15.0, t_end_h=24.0
    )
    assert 0.0 < res["tmax_h"] <= 24.0


def test_simulate_double_dose_doubles_cmax():
    res1 = simulate_flip_flop_pk("Drug", dose_mg=100, ka_per_h=0.5, ke_per_h=1.5, vd_L=25.0)
    res2 = simulate_flip_flop_pk("Drug", dose_mg=200, ka_per_h=0.5, ke_per_h=1.5, vd_L=25.0)
    assert res2["cmax"] == pytest.approx(2.0 * res1["cmax"], rel=0.01)


def test_simulate_double_dose_doubles_auc():
    res1 = simulate_flip_flop_pk("Drug", dose_mg=100, ka_per_h=0.5, ke_per_h=1.5, vd_L=25.0)
    res2 = simulate_flip_flop_pk("Drug", dose_mg=200, ka_per_h=0.5, ke_per_h=1.5, vd_L=25.0)
    assert res2["auc"] == pytest.approx(2.0 * res1["auc"], rel=0.01)


def test_simulate_t_half_apparent_equals_ln2_over_ka_in_flip_flop():
    ka, ke = 0.2, 2.0
    res = simulate_flip_flop_pk("Drug", dose_mg=100, ka_per_h=ka, ke_per_h=ke, vd_L=20.0)
    expected = math.log(2.0) / ka
    assert res["t_half_apparent_h"] == pytest.approx(expected, rel=1e-6)


def test_simulate_t_half_apparent_equals_ln2_over_ke_in_normal_pk():
    ka, ke = 2.0, 0.3
    res = simulate_flip_flop_pk("Drug", dose_mg=100, ka_per_h=ka, ke_per_h=ke, vd_L=20.0)
    expected = math.log(2.0) / ke
    assert res["t_half_apparent_h"] == pytest.approx(expected, rel=1e-6)


def test_simulate_times_length_correct():
    res = simulate_flip_flop_pk(
        "Drug", dose_mg=100, ka_per_h=1.0, ke_per_h=0.5, vd_L=20.0, t_end_h=10.0, dt_h=0.5
    )
    assert len(res["times_h"]) == len(res["c_plasma_mg_L"])


def test_simulate_notes_contains_drug_name():
    res = simulate_flip_flop_pk("Morphine", dose_mg=10, ka_per_h=0.1, ke_per_h=1.0, vd_L=250.0)
    assert "Morphine" in res["notes"]


def test_simulate_flip_flop_notes_mention_flip_flop():
    res = simulate_flip_flop_pk("Drug", dose_mg=100, ka_per_h=0.15, ke_per_h=2.0, vd_L=20.0)
    assert "flip" in res["notes"].lower() or "FLIP" in res["notes"]


# ---------------------------------------------------------------------------
# compare_formulations
# ---------------------------------------------------------------------------


def test_compare_formulations_returns_correct_count():
    ka_list = [0.2, 0.5, 1.0, 2.0]
    results = compare_formulations("Drug", dose_mg=100, ke_per_h=1.0, vd_L=20.0, ka_values=ka_list)
    assert len(results) == 4


def test_compare_formulations_sorted_by_ka_ascending():
    ka_list = [2.0, 0.5, 1.5, 0.1]
    results = compare_formulations("Drug", dose_mg=100, ke_per_h=1.0, vd_L=20.0, ka_values=ka_list)
    kas = [r["ka"] for r in results]
    assert kas == sorted(kas)


def test_compare_formulations_flip_flop_flags_correct():
    ke = 1.0
    ka_list = [0.3, 2.0]  # 0.3 < ke => flip-flop; 2.0 > ke => normal
    results = compare_formulations("Drug", dose_mg=100, ke_per_h=ke, vd_L=20.0, ka_values=ka_list)
    # sorted ascending: [0.3, 2.0]
    assert results[0]["is_flip_flop"] is True
    assert results[1]["is_flip_flop"] is False


def test_compare_formulations_empty_ka_list():
    results = compare_formulations("Drug", dose_mg=100, ke_per_h=1.0, vd_L=20.0, ka_values=[])
    assert results == []


def test_compare_formulations_cmax_positive():
    results = compare_formulations(
        "Drug", dose_mg=50, ke_per_h=0.5, vd_L=15.0, ka_values=[0.3, 1.0]
    )
    for r in results:
        assert r["cmax"] > 0
