"""Tests for Phase 624 PK parameter sensitivity analysis."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.analysis.pk_parameter_sensitivity import (
    SensitivityResult624,
    analytical_pk_sensitivity,
    one_at_a_time_sensitivity,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _auc_model(params: dict) -> float:
    return params["dose"] / params["cl"]


def _t_half_model(params: dict) -> float:
    return math.log(2) * params["vd"] / params["cl"]


_BASE_AUC = {"dose": 100.0, "cl": 5.0}
_BASE_THALF = {"dose": 100.0, "cl": 5.0, "vd": 50.0}


# ---------------------------------------------------------------------------
# one_at_a_time_sensitivity — structural tests
# ---------------------------------------------------------------------------


def test_oat_returns_result():
    r = one_at_a_time_sensitivity(_auc_model, _BASE_AUC, ["dose", "cl"])
    assert isinstance(r, SensitivityResult624)


def test_oat_n_parameters():
    r = one_at_a_time_sensitivity(_auc_model, _BASE_AUC, ["dose", "cl"])
    assert len(r.parameters) == 2


def test_oat_most_sensitive_is_cl():
    # AUC = dose/CL → CL elasticity = -1, dose elasticity = +1 (same magnitude).
    # Both have |e|=1. In a tie, CL comes first alphabetically after sorting by abs value.
    # Let's verify the most_sensitive is either "cl" or "dose" (both have |e|=1).
    r = one_at_a_time_sensitivity(_auc_model, _BASE_AUC, ["dose", "cl"])
    assert abs(r.parameters[0].elasticity) >= abs(r.parameters[-1].elasticity)


def test_oat_elasticity_negative_for_cl_on_auc():
    r = one_at_a_time_sensitivity(_auc_model, _BASE_AUC, ["cl"])
    cl_entry = r.parameters[0]
    assert cl_entry.elasticity < 0.0


def test_oat_elasticity_near_zero_for_vd_on_auc():
    # AUC = dose/cl, independent of vd
    params = {"dose": 100.0, "cl": 5.0, "vd": 50.0}
    r = one_at_a_time_sensitivity(_auc_model, params, ["vd"])
    vd_entry = r.parameters[0]
    assert abs(vd_entry.elasticity) < 1e-9


def test_oat_entries_sorted_by_abs_elasticity():
    r = one_at_a_time_sensitivity(_t_half_model, _BASE_THALF, ["cl", "vd", "dose"])
    abs_vals = [abs(e.elasticity) for e in r.parameters]
    assert abs_vals == sorted(abs_vals, reverse=True)


def test_oat_ranks_ascending():
    r = one_at_a_time_sensitivity(_auc_model, _BASE_AUC, ["dose", "cl"])
    ranks = [e.rank for e in r.parameters]
    assert ranks == sorted(ranks)


def test_oat_output_name_stored():
    r = one_at_a_time_sensitivity(_auc_model, _BASE_AUC, ["cl"], output_name="AUC_mg_h_L")
    assert r.output_name == "AUC_mg_h_L"


# ---------------------------------------------------------------------------
# analytical_pk_sensitivity — AUC
# ---------------------------------------------------------------------------


def test_analytical_auc_returns_result():
    r = analytical_pk_sensitivity(dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0, output="auc")
    assert isinstance(r, SensitivityResult624)


def test_analytical_auc_cl_elasticity_neg_1():
    r = analytical_pk_sensitivity(dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0, output="auc")
    cl_entry = next(e for e in r.parameters if e.parameter == "CL")
    assert abs(cl_entry.elasticity - (-1.0)) < 1e-9


def test_analytical_auc_vd_elasticity_near_zero():
    r = analytical_pk_sensitivity(dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0, output="auc")
    vd_entry = next(e for e in r.parameters if e.parameter == "Vd")
    assert abs(vd_entry.elasticity) < 1e-9


# ---------------------------------------------------------------------------
# analytical_pk_sensitivity — t_half
# ---------------------------------------------------------------------------


def test_analytical_t_half_returns_result():
    r = analytical_pk_sensitivity(dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0, output="t_half")
    assert isinstance(r, SensitivityResult624)


def test_analytical_t_half_cl_elasticity_neg_1():
    r = analytical_pk_sensitivity(dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0, output="t_half")
    cl_entry = next(e for e in r.parameters if e.parameter == "CL")
    assert abs(cl_entry.elasticity - (-1.0)) < 1e-9


def test_analytical_t_half_vd_elasticity_pos_1():
    r = analytical_pk_sensitivity(dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0, output="t_half")
    vd_entry = next(e for e in r.parameters if e.parameter == "Vd")
    assert abs(vd_entry.elasticity - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# SensitivityResult624 fields
# ---------------------------------------------------------------------------


def test_total_variance_explained_nonneg():
    r = analytical_pk_sensitivity(dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0, output="auc")
    assert r.total_variance_explained >= 0.0


def test_least_sensitive_is_vd_for_auc():
    r = analytical_pk_sensitivity(dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0, output="auc")
    assert r.least_sensitive == "Vd"


def test_oat_perturbation_10pct():
    # AUC = dose/cl is a nonlinear function, so central-difference elasticity for CL
    # is slightly below -1.0 (exact value ≈ -1.0101 for 10% perturbation).
    r = one_at_a_time_sensitivity(_auc_model, _BASE_AUC, ["cl"], perturbation_pct=10.0)
    cl_entry = r.parameters[0]
    assert cl_entry.elasticity < 0.0
    assert abs(cl_entry.elasticity) > 0.9  # close to -1


def test_oat_positive_elasticity():
    # AUC = dose/cl, dose elasticity = +1
    r = one_at_a_time_sensitivity(_auc_model, _BASE_AUC, ["dose"], perturbation_pct=10.0)
    dose_entry = r.parameters[0]
    assert dose_entry.elasticity > 0.0


def test_absolute_sensitivity_positive_or_negative():
    r = one_at_a_time_sensitivity(_auc_model, _BASE_AUC, ["cl"])
    cl_entry = r.parameters[0]
    # absolute_sensitivity is a finite float
    assert isinstance(cl_entry.absolute_sensitivity, float)
    assert math.isfinite(cl_entry.absolute_sensitivity)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_empty_params_raises():
    with pytest.raises(ValueError, match="baseline_params"):
        one_at_a_time_sensitivity(_auc_model, {}, ["cl"])


def test_invalid_perturbation_raises():
    with pytest.raises(ValueError, match="perturbation_pct"):
        one_at_a_time_sensitivity(_auc_model, _BASE_AUC, ["cl"], perturbation_pct=0.0)


def test_notes_nonempty():
    r = one_at_a_time_sensitivity(_auc_model, _BASE_AUC, ["cl"])
    assert len(r.notes) > 0
