"""Tests for concentration-effect relationship modeling (Phase 657)."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.clinical.conc_effect_p657 import (
    ConcEffectResult,
    evaluate_effect,
    model_concentration_effect,
)

# --- Frozen dataclass tests ---


def test_result_is_frozen():
    r = model_concentration_effect("DrugA", "emax", 10.0)
    with pytest.raises(AttributeError):
        r.drug_name = "other"


def test_result_type():
    r = model_concentration_effect("DrugA", "emax", 10.0)
    assert isinstance(r, ConcEffectResult)


# --- Emax model ---


def test_emax_effect_at_ec50():
    r = model_concentration_effect("DrugA", "emax", 10.0, emax=1.0, baseline_effect=0.0)
    assert math.isclose(r.effect_at_ec50, 0.5, rel_tol=1e-6)


def test_emax_effect_at_ec50_with_baseline():
    r = model_concentration_effect("DrugA", "emax", 10.0, emax=2.0, baseline_effect=1.0)
    assert math.isclose(r.effect_at_ec50, 2.0, rel_tol=1e-6)


def test_emax_ec90_gt_ec50():
    r = model_concentration_effect("DrugA", "emax", 10.0)
    assert r.ec90_mg_L > r.ec50_mg_L


def test_emax_ec50_gt_ec10():
    r = model_concentration_effect("DrugA", "emax", 10.0)
    assert r.ec50_mg_L > r.ec10_mg_L


def test_emax_ec90_value():
    r = model_concentration_effect("DrugA", "emax", 10.0)
    assert math.isclose(r.ec90_mg_L, 90.0, rel_tol=1e-6)


def test_emax_ec10_value():
    r = model_concentration_effect("DrugA", "emax", 10.0)
    assert math.isclose(r.ec10_mg_L, 10.0 / 9.0, rel_tol=1e-6)


# --- Sigmoid model ---


def test_sigmoid_hill1_same_as_emax():
    r_emax = model_concentration_effect("DrugA", "emax", 10.0, emax=2.0)
    r_sig = model_concentration_effect("DrugA", "sigmoid", 10.0, emax=2.0, hill_n=1.0)
    assert math.isclose(r_emax.effect_at_ec50, r_sig.effect_at_ec50, rel_tol=1e-6)


def test_sigmoid_higher_hill_steeper():
    """Higher hill_n should give a steeper curve (lower EC10, higher EC90 ratio)."""
    r1 = model_concentration_effect("DrugA", "sigmoid", 10.0, hill_n=1.0)
    r4 = model_concentration_effect("DrugA", "sigmoid", 10.0, hill_n=4.0)
    # EC90/EC50 ratio should be smaller with higher hill_n (steeper)
    ratio1 = r1.ec90_mg_L / r1.ec50_mg_L
    ratio4 = r4.ec90_mg_L / r4.ec50_mg_L
    assert ratio4 < ratio1


def test_sigmoid_ec90_gt_ec50():
    r = model_concentration_effect("DrugA", "sigmoid", 10.0, hill_n=2.0)
    assert r.ec90_mg_L > r.ec50_mg_L


def test_sigmoid_ec50_gt_ec10():
    r = model_concentration_effect("DrugA", "sigmoid", 10.0, hill_n=2.0)
    assert r.ec50_mg_L > r.ec10_mg_L


def test_sigmoid_effect_at_ec50():
    r = model_concentration_effect("DrugA", "sigmoid", 5.0, emax=1.0, hill_n=3.0)
    assert math.isclose(r.effect_at_ec50, 0.5, rel_tol=1e-6)


# --- Linear model ---


def test_linear_proportional():
    effects = evaluate_effect("DrugA", "linear", 10.0, [5.0, 10.0, 20.0])
    assert math.isclose(effects[1][1], 2.0 * effects[0][1], rel_tol=1e-6)


def test_linear_model_equation():
    r = model_concentration_effect("DrugA", "linear", 10.0)
    assert "linear" in r.notes


def test_linear_effect_at_ec50():
    r = model_concentration_effect("DrugA", "linear", 10.0, emax=1.0)
    # At C = ec50: E = 0 + 1.0 * 10 / 10 = 1.0
    assert math.isclose(r.effect_at_ec50, 1.0, rel_tol=1e-6)


# --- Indirect model ---


def test_indirect_inhibitory_at_high_conc():
    """Indirect model: at very high C, effect approaches baseline (full inhibition)."""
    effects = evaluate_effect("DrugA", "indirect", 10.0, [0.0, 1000.0], emax=1.0)
    # At C=0: E = baseline + emax * (1 - 0) = baseline + emax
    # At C=very high: E = baseline + emax * (1 - ~1) ≈ baseline
    assert effects[0][1] > effects[1][1]


def test_indirect_effect_at_ec50():
    r = model_concentration_effect("DrugA", "indirect", 10.0, emax=1.0, baseline_effect=0.0)
    # E(ec50) = 0 + 1.0 * (1 - 10/(10+10)) = 0.5
    assert math.isclose(r.effect_at_ec50, 0.5, rel_tol=1e-6)


def test_indirect_ec90():
    r = model_concentration_effect("DrugA", "indirect", 10.0)
    assert math.isclose(r.ec90_mg_L, 90.0, rel_tol=1e-6)


# --- Therapeutic index ---


def test_therapeutic_index_with_threshold():
    r = model_concentration_effect("DrugA", "emax", 10.0, toxic_effect_threshold=50.0)
    assert math.isclose(r.therapeutic_index_approx, 5.0, rel_tol=1e-6)


def test_therapeutic_index_no_threshold():
    r = model_concentration_effect("DrugA", "emax", 10.0)
    assert r.therapeutic_index_approx == -1.0


# --- evaluate_effect ---


def test_evaluate_effect_length():
    concs = [0.0, 1.0, 5.0, 10.0, 50.0, 100.0]
    result = evaluate_effect("DrugA", "emax", 10.0, concs)
    assert len(result) == len(concs)


def test_evaluate_effect_returns_tuples():
    result = evaluate_effect("DrugA", "emax", 10.0, [1.0, 5.0])
    for item in result:
        assert isinstance(item, tuple)
        assert len(item) == 2


def test_evaluate_effect_non_negative():
    concs = [0.0, 1.0, 5.0, 10.0, 50.0]
    result = evaluate_effect("DrugA", "emax", 10.0, concs)
    for _, effect in result:
        assert effect >= 0.0


# --- Validation ---


def test_invalid_model_type():
    with pytest.raises(ValueError, match="model_type"):
        model_concentration_effect("DrugA", "invalid", 10.0)


def test_ec50_zero_raises():
    with pytest.raises(ValueError, match="ec50"):
        model_concentration_effect("DrugA", "emax", 0.0)


def test_ec50_negative_raises():
    with pytest.raises(ValueError, match="ec50"):
        model_concentration_effect("DrugA", "emax", -5.0)


def test_emax_zero_raises():
    with pytest.raises(ValueError, match="emax"):
        model_concentration_effect("DrugA", "emax", 10.0, emax=0.0)


def test_hill_n_zero_raises():
    with pytest.raises(ValueError, match="hill_n"):
        model_concentration_effect("DrugA", "sigmoid", 10.0, hill_n=0.0)
