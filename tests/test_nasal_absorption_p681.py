"""Tests for Phase 681 — Nasal Drug Absorption."""

import pytest

from omega_pbpk.core.nasal_absorption_p681 import (
    NasalAbsorptionResult,
    simulate_nasal_absorption,
)

# Default test parameters
_DEFAULTS = dict(
    drug_name="TestDrug",
    dose_mg=10.0,
    formulation="solution",
    mw_Da=300.0,
    logP=2.0,
    cl_sys_L_per_h=5.0,
    vd_sys_L=50.0,
)


def _run(**overrides):
    params = {**_DEFAULTS, **overrides}
    return simulate_nasal_absorption(**params)


class TestReturnType:
    def test_returns_nasal_absorption_result(self):
        r = _run()
        assert isinstance(r, NasalAbsorptionResult)

    def test_drug_name(self):
        r = _run()
        assert r.drug_name == "TestDrug"

    def test_dose_mg(self):
        r = _run()
        assert r.dose_mg == 10.0

    def test_formulation(self):
        r = _run()
        assert r.formulation == "solution"


class TestFieldTypes:
    def test_times_is_list(self):
        r = _run()
        assert isinstance(r.times_h, list)

    def test_c_plasma_is_list(self):
        r = _run()
        assert isinstance(r.c_plasma_mg_L, list)

    def test_c_nasal_depot_is_list(self):
        r = _run()
        assert isinstance(r.c_nasal_depot_mg, list)

    def test_mucociliary_cleared_is_list(self):
        r = _run()
        assert isinstance(r.mucociliary_cleared_mg, list)

    def test_bioavailability_is_float(self):
        r = _run()
        assert isinstance(r.bioavailability_nasal, float)

    def test_cmax_is_float(self):
        r = _run()
        assert isinstance(r.cmax_mg_L, float)

    def test_tmax_is_float(self):
        r = _run()
        assert isinstance(r.tmax_h, float)

    def test_auc_is_float(self):
        r = _run()
        assert isinstance(r.auc_mg_h_per_L, float)

    def test_notes_is_string(self):
        r = _run()
        assert isinstance(r.notes, str)


class TestInitialConditions:
    def test_initial_plasma_zero(self):
        r = _run()
        assert r.c_plasma_mg_L[0] == 0.0

    def test_initial_depot_equals_dose(self):
        r = _run()
        assert r.c_nasal_depot_mg[0] == pytest.approx(10.0)

    def test_initial_mucociliary_zero(self):
        r = _run()
        assert r.mucociliary_cleared_mg[0] == 0.0


class TestNonNegativity:
    def test_plasma_non_negative(self):
        r = _run()
        assert all(c >= 0 for c in r.c_plasma_mg_L)

    def test_depot_non_negative(self):
        r = _run()
        assert all(c >= 0 for c in r.c_nasal_depot_mg)

    def test_mucociliary_non_negative(self):
        r = _run()
        assert all(c >= 0 for c in r.mucociliary_cleared_mg)


class TestPKMetrics:
    def test_cmax_positive(self):
        r = _run()
        assert r.cmax_mg_L > 0

    def test_auc_positive(self):
        r = _run()
        assert r.auc_mg_h_per_L > 0

    def test_tmax_positive(self):
        r = _run()
        assert r.tmax_h > 0

    def test_t_half_plasma_positive(self):
        r = _run()
        assert r.t_half_plasma_h > 0


class TestFormulationEffects:
    def test_solution_faster_than_gel(self):
        r_sol = _run(formulation="solution")
        r_gel = _run(formulation="gel")
        assert r_sol.cmax_mg_L > r_gel.cmax_mg_L

    def test_solution_faster_than_powder(self):
        r_sol = _run(formulation="solution")
        r_pow = _run(formulation="powder")
        assert r_sol.cmax_mg_L > r_pow.cmax_mg_L


class TestBioavailability:
    def test_bioavailability_in_range(self):
        r = _run()
        assert 0 < r.bioavailability_nasal <= 1.0

    def test_higher_mw_lower_bioavailability(self):
        r_low = _run(mw_Da=200.0)
        r_high = _run(mw_Da=800.0)
        assert r_low.bioavailability_nasal > r_high.bioavailability_nasal


class TestDoseLinearity:
    def test_double_dose_double_cmax(self):
        r1 = _run(dose_mg=10.0)
        r2 = _run(dose_mg=20.0)
        assert r2.cmax_mg_L == pytest.approx(2.0 * r1.cmax_mg_L, rel=0.05)

    def test_double_dose_double_auc(self):
        r1 = _run(dose_mg=10.0)
        r2 = _run(dose_mg=20.0)
        assert r2.auc_mg_h_per_L == pytest.approx(2.0 * r1.auc_mg_h_per_L, rel=0.05)


class TestValidation:
    def test_zero_dose_raises(self):
        with pytest.raises(ValueError):
            _run(dose_mg=0)

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError):
            _run(dose_mg=-1)

    def test_zero_clearance_raises(self):
        with pytest.raises(ValueError):
            _run(cl_sys_L_per_h=0)

    def test_zero_vd_raises(self):
        with pytest.raises(ValueError):
            _run(vd_sys_L=0)

    def test_invalid_formulation_raises(self):
        with pytest.raises(ValueError):
            _run(formulation="tablet")
