"""Tests for Phase 682 — Intramuscular Depot PK."""

import pytest

from omega_pbpk.core.im_depot_pk_p682 import (
    IMDepotPKResult,
    simulate_im_depot_pk,
)

_DEFAULTS = dict(
    drug_name="TestDrug",
    dose_mg=100.0,
    formulation_type="aqueous",
    cl_sys_L_per_h=5.0,
    vd_sys_L=50.0,
)


def _run(**overrides):
    params = {**_DEFAULTS, **overrides}
    return simulate_im_depot_pk(**params)


class TestReturnType:
    def test_returns_im_depot_pk_result(self):
        r = _run()
        assert isinstance(r, IMDepotPKResult)

    def test_drug_name(self):
        r = _run()
        assert r.drug_name == "TestDrug"

    def test_dose_mg(self):
        r = _run()
        assert r.dose_mg == 100.0

    def test_formulation_type(self):
        r = _run()
        assert r.formulation_type == "aqueous"


class TestFieldTypes:
    def test_times_is_list(self):
        r = _run()
        assert isinstance(r.times_h, list)

    def test_c_depot_is_list(self):
        r = _run()
        assert isinstance(r.c_depot_mg, list)

    def test_c_plasma_is_list(self):
        r = _run()
        assert isinstance(r.c_plasma_mg_L, list)

    def test_cmax_is_float(self):
        r = _run()
        assert isinstance(r.cmax_mg_L, float)

    def test_flip_flop_is_bool(self):
        r = _run()
        assert isinstance(r.flip_flop, bool)

    def test_bioavailability_is_float(self):
        r = _run()
        assert isinstance(r.bioavailability_pct, float)

    def test_notes_is_string(self):
        r = _run()
        assert isinstance(r.notes, str)


class TestInitialConditions:
    def test_initial_depot_equals_dose(self):
        r = _run()
        assert r.c_depot_mg[0] == pytest.approx(100.0)

    def test_initial_plasma_zero(self):
        r = _run()
        assert r.c_plasma_mg_L[0] == 0.0


class TestNonNegativity:
    def test_depot_non_negative(self):
        r = _run()
        assert all(c >= 0 for c in r.c_depot_mg)

    def test_plasma_non_negative(self):
        r = _run()
        assert all(c >= 0 for c in r.c_plasma_mg_L)


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

    def test_t_half_absorption_positive(self):
        r = _run()
        assert r.t_half_absorption_h > 0

    def test_t_half_plasma_positive(self):
        r = _run()
        assert r.t_half_plasma_h > 0


class TestFormulationEffects:
    def test_oil_slower_than_aqueous(self):
        r_aq = _run(formulation_type="aqueous")
        r_oil = _run(formulation_type="oil")
        assert r_oil.tmax_h > r_aq.tmax_h

    def test_microsphere_slower_than_aqueous(self):
        r_aq = _run(formulation_type="aqueous")
        r_ms = _run(formulation_type="microsphere")
        assert r_ms.tmax_h > r_aq.tmax_h


class TestFlipFlop:
    def test_aqueous_no_flip_flop(self):
        # ka=1.5, ke=5/50=0.1 → t_half_abs < t_half_plasma → no flip-flop
        r = _run(formulation_type="aqueous")
        assert r.flip_flop is False

    def test_oil_flip_flop(self):
        # ka=0.05, ke=5/50=0.1 → t_half_abs > t_half_plasma → flip-flop
        r = _run(formulation_type="oil")
        assert r.flip_flop is True

    def test_microsphere_flip_flop(self):
        r = _run(formulation_type="microsphere")
        assert r.flip_flop is True


class TestBioavailability:
    def test_bioavailability_85(self):
        r = _run()
        assert r.bioavailability_pct == pytest.approx(85.0)


class TestDoseLinearity:
    def test_double_dose_double_cmax(self):
        r1 = _run(dose_mg=100.0)
        r2 = _run(dose_mg=200.0)
        assert r2.cmax_mg_L == pytest.approx(2.0 * r1.cmax_mg_L, rel=0.05)

    def test_double_dose_double_auc(self):
        r1 = _run(dose_mg=100.0)
        r2 = _run(dose_mg=200.0)
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
            _run(formulation_type="patch")
