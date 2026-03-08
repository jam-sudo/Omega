"""Tests for Phase 539 — Subcutaneous Drug Absorption PK."""

import math

import pytest

from omega_pbpk.core.subcutaneous_absorption_pk import (
    SubcutaneousAbsorptionResult,
    compare_sc_formulations,
    simulate_subcutaneous_absorption,
)

# --- Helpers -----------------------------------------------------------------

BASE_PARAMS = dict(
    drug_name="TestDrug",
    dose_mg=100.0,
    mw_Da=300.0,
    formulation_type="solution",
    cl_L_per_h=5.0,
    vd_sys_L=50.0,
)


def _sim(**overrides):
    p = {**BASE_PARAMS, **overrides}
    return simulate_subcutaneous_absorption(**p)


# --- Return type & fields ----------------------------------------------------


def test_return_type():
    result = _sim()
    assert isinstance(result, SubcutaneousAbsorptionResult)


def test_has_all_fields():
    r = _sim()
    assert hasattr(r, "drug_name")
    assert hasattr(r, "dose_mg")
    assert hasattr(r, "mw_Da")
    assert hasattr(r, "formulation_type")
    assert hasattr(r, "ka_sc_per_h")
    assert hasattr(r, "f_sc")
    assert hasattr(r, "f_lymph")
    assert hasattr(r, "t_half_absorption_h")
    assert hasattr(r, "times_h")
    assert hasattr(r, "c_systemic_mg_L")
    assert hasattr(r, "cmax_mg_L")
    assert hasattr(r, "tmax_h")
    assert hasattr(r, "auc_mg_h_per_L")
    assert hasattr(r, "bioavailability_pct")
    assert hasattr(r, "relative_to_iv_auc")
    assert hasattr(r, "notes")


# --- Initial conditions -------------------------------------------------------


def test_c_sys_starts_at_zero():
    r = _sim()
    assert r.c_systemic_mg_L[0] == pytest.approx(0.0)


def test_times_start_at_zero():
    r = _sim()
    assert r.times_h[0] == pytest.approx(0.0)


def test_time_series_length():
    r = _sim()
    assert len(r.times_h) == len(r.c_systemic_mg_L)
    assert len(r.times_h) > 1


# --- Dynamics ----------------------------------------------------------------


def test_c_sys_rises_then_falls():
    r = _sim()
    # Peak should not be at t=0 or at the very end
    assert r.tmax_h > 0.0
    assert r.cmax_mg_L > r.c_systemic_mg_L[0]
    assert r.cmax_mg_L > r.c_systemic_mg_L[-1]


def test_cmax_positive():
    r = _sim()
    assert r.cmax_mg_L > 0.0


def test_auc_positive():
    r = _sim()
    assert r.auc_mg_h_per_L > 0.0


# --- Formulation-specific parameters ------------------------------------------


def test_solution_ka():
    r = _sim(formulation_type="solution")
    assert r.ka_sc_per_h == pytest.approx(0.3)


def test_suspension_ka():
    r = _sim(formulation_type="suspension")
    assert r.ka_sc_per_h == pytest.approx(0.05)


def test_biologic_ka():
    r = _sim(formulation_type="biologic", mw_Da=15000.0)
    assert r.ka_sc_per_h == pytest.approx(0.02)


def test_solution_f_sc():
    r = _sim(formulation_type="solution")
    assert r.f_sc == pytest.approx(0.8)


def test_biologic_f_sc():
    r = _sim(formulation_type="biologic", mw_Da=15000.0)
    assert r.f_sc == pytest.approx(0.6)


def test_solution_f_lymph():
    r = _sim(formulation_type="solution")
    assert r.f_lymph == pytest.approx(0.05)


def test_biologic_f_lymph():
    r = _sim(formulation_type="biologic", mw_Da=15000.0)
    assert r.f_lymph == pytest.approx(0.2)


def test_bioavailability_pct_solution():
    r = _sim(formulation_type="solution")
    assert r.bioavailability_pct == pytest.approx(80.0)


def test_bioavailability_pct_biologic():
    r = _sim(formulation_type="biologic", mw_Da=15000.0)
    assert r.bioavailability_pct == pytest.approx(60.0)


# --- Absorption half-life -----------------------------------------------------


def test_t_half_absorption_solution():
    r = _sim(formulation_type="solution")
    expected = math.log(2) / 0.3
    assert r.t_half_absorption_h == pytest.approx(expected, rel=1e-4)


def test_t_half_absorption_biologic():
    r = _sim(formulation_type="biologic", mw_Da=15000.0)
    expected = math.log(2) / 0.02
    assert r.t_half_absorption_h == pytest.approx(expected, rel=1e-4)


# --- Comparison across formulations ------------------------------------------


def test_solution_tmax_less_than_suspension():
    sol = _sim(formulation_type="solution")
    susp = _sim(formulation_type="suspension")
    # Solution absorbs faster → earlier Tmax
    assert sol.tmax_h < susp.tmax_h


def test_suspension_tmax_less_than_biologic():
    susp = _sim(formulation_type="suspension")
    bio = _sim(formulation_type="biologic", mw_Da=15000.0)
    assert susp.tmax_h < bio.tmax_h


def test_biologic_lower_auc_than_solution():
    """Biologic has lower F_sc (0.6 vs 0.8), same dose → lower AUC."""
    sol = _sim(formulation_type="solution", mw_Da=300.0)
    bio = _sim(formulation_type="biologic", mw_Da=15000.0)
    assert bio.auc_mg_h_per_L < sol.auc_mg_h_per_L


# --- Dose linearity ----------------------------------------------------------


def test_dose_linearity_cmax():
    r1 = _sim(dose_mg=50.0)
    r2 = _sim(dose_mg=100.0)
    assert r2.cmax_mg_L == pytest.approx(2.0 * r1.cmax_mg_L, rel=0.01)


def test_dose_linearity_auc():
    r1 = _sim(dose_mg=50.0)
    r2 = _sim(dose_mg=100.0)
    assert r2.auc_mg_h_per_L == pytest.approx(2.0 * r1.auc_mg_h_per_L, rel=0.01)


# --- relative_to_iv_auc -------------------------------------------------------


def test_relative_to_iv_auc_approx_f_sc():
    """relative_to_iv_auc = AUC * CL / dose; should approximate F_sc."""
    r = _sim(formulation_type="solution", t_end_h=200.0)
    assert r.relative_to_iv_auc == pytest.approx(r.f_sc, rel=0.05)


# --- compare_sc_formulations -------------------------------------------------


def test_compare_returns_three():
    results = compare_sc_formulations("Drug", 100.0, 300.0, 5.0, 50.0)
    assert len(results) == 3


def test_compare_sorted_by_cmax_descending():
    results = compare_sc_formulations("Drug", 100.0, 300.0, 5.0, 50.0)
    cmaxes = [r.cmax_mg_L for r in results]
    assert cmaxes == sorted(cmaxes, reverse=True)


def test_compare_returns_subcutaneous_results():
    results = compare_sc_formulations("Drug", 100.0, 300.0, 5.0, 50.0)
    for r in results:
        assert isinstance(r, SubcutaneousAbsorptionResult)


# --- Validation errors -------------------------------------------------------


def test_validation_negative_dose():
    with pytest.raises(ValueError, match="dose_mg"):
        _sim(dose_mg=-10.0)


def test_validation_zero_cl():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        _sim(cl_L_per_h=0.0)


def test_validation_zero_vd():
    with pytest.raises(ValueError, match="vd_sys_L"):
        _sim(vd_sys_L=0.0)


def test_validation_bad_formulation():
    with pytest.raises(ValueError):
        _sim(formulation_type="inhaled")
