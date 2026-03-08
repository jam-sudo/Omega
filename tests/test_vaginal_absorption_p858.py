"""Tests for Phase 858 — Vaginal Drug Absorption."""

import math
import pytest
from omega_pbpk.core.vaginal_absorption_p858 import (
    VaginalAbsorptionResult,
    simulate_vaginal_absorption,
)

DEFAULT = dict(
    drug_name="TestDrug",
    dose_mg=10.0,
    logP=2.0,
    mw_Da=300.0,
    cl_sys_L_per_h=10.0,
    vd_sys_L=50.0,
)


def _run(**overrides):
    params = {**DEFAULT, **overrides}
    return simulate_vaginal_absorption(**params)


class TestReturnType:
    def test_returns_dataclass(self):
        r = _run()
        assert isinstance(r, VaginalAbsorptionResult)

    def test_has_drug_name(self):
        r = _run()
        assert r.drug_name == "TestDrug"

    def test_has_dose(self):
        r = _run()
        assert r.dose_mg == 10.0

    def test_has_formulation(self):
        r = _run()
        assert r.formulation == "gel"

    def test_times_is_list(self):
        r = _run()
        assert isinstance(r.times_h, list)
        assert len(r.times_h) > 0

    def test_lists_same_length(self):
        r = _run()
        assert len(r.times_h) == len(r.c_vaginal_depot_mg) == len(r.c_plasma_mg_L)


class TestFormulations:
    def test_gel(self):
        r = _run(formulation="gel")
        assert r.cmax_mg_L > 0

    def test_tablet(self):
        r = _run(formulation="tablet")
        assert r.cmax_mg_L > 0

    def test_ring(self):
        r = _run(formulation="ring")
        assert r.cmax_mg_L > 0

    def test_cream(self):
        r = _run(formulation="cream")
        assert r.cmax_mg_L > 0

    def test_invalid_formulation(self):
        with pytest.raises(ValueError):
            _run(formulation="suppository")

    def test_ring_slowest_absorption(self):
        r_ring = _run(formulation="ring")
        r_gel = _run(formulation="gel")
        # Ring has lowest ka, so t_half_absorption should be longest
        assert r_ring.t_half_absorption_h > r_gel.t_half_absorption_h


class TestConcentrations:
    def test_non_negative_depot(self):
        r = _run()
        assert all(c >= 0 for c in r.c_vaginal_depot_mg)

    def test_non_negative_plasma(self):
        r = _run()
        assert all(c >= 0 for c in r.c_plasma_mg_L)

    def test_depot_decreases(self):
        r = _run()
        assert r.c_vaginal_depot_mg[-1] < r.c_vaginal_depot_mg[0]

    def test_plasma_rises(self):
        r = _run()
        assert r.cmax_mg_L > 0


class TestPKParameters:
    def test_cmax_positive(self):
        r = _run()
        assert r.cmax_mg_L > 0

    def test_auc_positive(self):
        r = _run()
        assert r.auc_mg_h_per_L > 0

    def test_tmax_positive(self):
        r = _run()
        assert r.tmax_h > 0


class TestBioavailability:
    def test_in_range(self):
        r = _run()
        assert 0 < r.bioavailability_pct <= 100.0

    def test_different_formulations_different_bioavail(self):
        r_gel = _run(formulation="gel")
        r_ring = _run(formulation="ring")
        assert r_gel.bioavailability_pct != r_ring.bioavailability_pct


class TestLocalConc:
    def test_local_conc_equals_dose_over_2(self):
        r = _run(dose_mg=10.0)
        assert abs(r.local_conc_mg_mL - 5.0) < 1e-10

    def test_local_conc_scales_with_dose(self):
        r = _run(dose_mg=20.0)
        assert abs(r.local_conc_mg_mL - 10.0) < 1e-10


class TestDoseProportionality:
    def test_double_dose_double_cmax(self):
        r1 = _run(dose_mg=10.0)
        r2 = _run(dose_mg=20.0)
        ratio = r2.cmax_mg_L / r1.cmax_mg_L
        assert abs(ratio - 2.0) < 0.1

    def test_double_dose_double_auc(self):
        r1 = _run(dose_mg=10.0)
        r2 = _run(dose_mg=20.0)
        ratio = r2.auc_mg_h_per_L / r1.auc_mg_h_per_L
        assert abs(ratio - 2.0) < 0.1


class TestValidation:
    def test_zero_dose(self):
        with pytest.raises(ValueError):
            _run(dose_mg=0)

    def test_negative_dose(self):
        with pytest.raises(ValueError):
            _run(dose_mg=-1)

    def test_zero_cl(self):
        with pytest.raises(ValueError):
            _run(cl_sys_L_per_h=0)

    def test_zero_vd(self):
        with pytest.raises(ValueError):
            _run(vd_sys_L=0)


class TestNotes:
    def test_notes_is_string(self):
        r = _run()
        assert isinstance(r.notes, str)
        assert len(r.notes) > 0
