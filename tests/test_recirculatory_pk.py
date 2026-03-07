"""Tests for Phase 264: recirculatory pharmacokinetic model."""

from __future__ import annotations

import pytest

from omega_pbpk.core.recirculatory_pk import (
    RecirculatoryPKResult,
    simulate_recirculatory_pk,
)

# ---------------------------------------------------------------------------
# Default parameter set for convenience
# ---------------------------------------------------------------------------

_BASE_KWARGS = dict(
    drug_name="TestDrug",
    dose_mg=100.0,
    cl_central_L_per_h=5.0,
    v_central_L=10.0,
    q_fast_L_per_h=20.0,
    v_fast_L=15.0,
    q_slow_L_per_h=5.0,
    v_slow_L=30.0,
    ke0_per_h=1.0,
    v_effect_L=0.5,
    t_end_h=12.0,
    dt_h=0.05,
)


def _run(**overrides):
    kw = dict(_BASE_KWARGS)
    kw.update(overrides)
    return simulate_recirculatory_pk(**kw)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "param",
    [
        "dose_mg",
        "cl_central_L_per_h",
        "v_central_L",
        "q_fast_L_per_h",
        "v_fast_L",
        "q_slow_L_per_h",
        "v_slow_L",
        "ke0_per_h",
        "v_effect_L",
    ],
)
def test_invalid_param_raises(param):
    with pytest.raises(ValueError, match=param):
        _run(**{param: 0.0})


def test_negative_param_raises():
    with pytest.raises(ValueError):
        _run(dose_mg=-50.0)


# ---------------------------------------------------------------------------
# Initial condition
# ---------------------------------------------------------------------------


def test_c_central_initial_equals_dose_over_volume():
    result = _run(dose_mg=100.0, v_central_L=10.0)
    assert abs(result.c_central_mg_L[0] - 10.0) < 1e-6


# ---------------------------------------------------------------------------
# Basic positivity / structure
# ---------------------------------------------------------------------------


def test_cmax_central_positive():
    result = _run()
    assert result.cmax_central > 0.0


def test_auc_central_positive():
    result = _run()
    assert result.auc_central > 0.0


def test_effect_cmax_positive():
    result = _run()
    assert result.effect_cmax > 0.0


def test_effect_tmax_positive():
    result = _run()
    assert result.effect_tmax_h > 0.0


def test_t_half_positive():
    result = _run()
    assert result.t_half_central_h > 0.0


# ---------------------------------------------------------------------------
# Array lengths
# ---------------------------------------------------------------------------


def test_array_lengths_consistent():
    result = _run()
    n = len(result.times_h)
    assert len(result.c_central_mg_L) == n
    assert len(result.c_fast_mg_L) == n
    assert len(result.c_slow_mg_L) == n
    assert len(result.c_effect_mg_L) == n


# ---------------------------------------------------------------------------
# Linearity
# ---------------------------------------------------------------------------


def test_double_dose_doubles_cmax():
    r1 = _run(dose_mg=100.0)
    r2 = _run(dose_mg=200.0)
    assert abs(r2.cmax_central / r1.cmax_central - 2.0) < 0.01


def test_double_dose_doubles_auc():
    r1 = _run(dose_mg=100.0)
    r2 = _run(dose_mg=200.0)
    assert abs(r2.auc_central / r1.auc_central - 2.0) < 0.01


# ---------------------------------------------------------------------------
# Distribution
# ---------------------------------------------------------------------------


def test_fast_compartment_has_drug():
    result = _run()
    assert max(result.c_fast_mg_L) > 0.0


def test_slow_compartment_has_drug():
    result = _run()
    assert max(result.c_slow_mg_L) > 0.0


# ---------------------------------------------------------------------------
# Drug name stored
# ---------------------------------------------------------------------------


def test_drug_name_stored():
    result = _run(drug_name="Propofol")
    assert result.drug_name == "Propofol"


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


def test_returns_correct_type():
    result = _run()
    assert isinstance(result, RecirculatoryPKResult)


# ---------------------------------------------------------------------------
# Notes field
# ---------------------------------------------------------------------------


def test_notes_contains_drug_name():
    result = _run(drug_name="Fentanyl")
    assert "Fentanyl" in result.notes
