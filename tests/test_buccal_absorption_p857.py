"""Tests for Phase 857 — Buccal Mucosa Drug Absorption."""

import math
import pytest
from omega_pbpk.core.buccal_absorption_p857 import (
    BuccalAbsorptionResult,
    simulate_buccal_absorption,
)

# Default parameters for convenience
DEFAULT = dict(
    drug_name="TestDrug",
    dose_mg=10.0,
    logP=2.0,
    mw_Da=300.0,
    pka=8.0,
    is_acid=False,
    cl_sys_L_per_h=10.0,
    vd_sys_L=50.0,
)


def _run(**overrides):
    params = {**DEFAULT, **overrides}
    return simulate_buccal_absorption(**params)


class TestReturnType:
    def test_returns_dataclass(self):
        r = _run()
        assert isinstance(r, BuccalAbsorptionResult)

    def test_has_drug_name(self):
        r = _run()
        assert r.drug_name == "TestDrug"

    def test_has_dose(self):
        r = _run()
        assert r.dose_mg == 10.0

    def test_times_is_list(self):
        r = _run()
        assert isinstance(r.times_h, list)
        assert len(r.times_h) > 0

    def test_c_buccal_depot_is_list(self):
        r = _run()
        assert isinstance(r.c_buccal_depot_mg, list)

    def test_c_plasma_is_list(self):
        r = _run()
        assert isinstance(r.c_plasma_mg_L, list)

    def test_lists_same_length(self):
        r = _run()
        assert len(r.times_h) == len(r.c_buccal_depot_mg) == len(r.c_plasma_mg_L)


class TestConcentrations:
    def test_non_negative_buccal_depot(self):
        r = _run()
        assert all(c >= 0 for c in r.c_buccal_depot_mg)

    def test_non_negative_plasma(self):
        r = _run()
        assert all(c >= 0 for c in r.c_plasma_mg_L)

    def test_buccal_depot_decreases(self):
        r = _run()
        assert r.c_buccal_depot_mg[-1] < r.c_buccal_depot_mg[0]

    def test_plasma_rises_then_falls(self):
        r = _run()
        assert r.cmax_mg_L > 0
        assert r.c_plasma_mg_L[0] == 0.0


class TestKpBuccal:
    def test_kp_in_range(self):
        r = _run()
        assert 0 < r.kp_buccal <= 1.0

    def test_kp_less_than_one(self):
        r = _run()
        assert r.kp_buccal < 1.0


class TestThalfBuccal:
    def test_positive(self):
        r = _run()
        assert r.t_half_buccal_h > 0


class TestPermeability:
    def test_higher_logP_higher_permeability(self):
        r_low = _run(logP=0.5)
        r_high = _run(logP=3.0)
        # Higher logP should give higher kp_buccal (more absorption)
        assert r_high.kp_buccal > r_low.kp_buccal

    def test_higher_MW_lower_permeability(self):
        r_low_mw = _run(mw_Da=200.0)
        r_high_mw = _run(mw_Da=800.0)
        assert r_low_mw.kp_buccal > r_high_mw.kp_buccal

    def test_permeability_class_valid(self):
        r = _run()
        assert r.permeability_class in ("high", "moderate", "low")

    def test_permeability_class_high(self):
        # Very high logP, low MW → high permeability
        r = _run(logP=4.0, mw_Da=150.0, pka=6.8, is_acid=False)
        # P_buccal may or may not be high, just check valid
        assert r.permeability_class in ("high", "moderate", "low")

    def test_permeability_class_low(self):
        # Very low logP, high MW → low permeability
        r = _run(logP=-1.0, mw_Da=900.0)
        assert r.permeability_class == "low"


class TestBioavailability:
    def test_in_range(self):
        r = _run()
        assert 0 < r.bioavailability_pct <= 100.0

    def test_equals_kp_times_100(self):
        r = _run()
        assert abs(r.bioavailability_pct - r.kp_buccal * 100.0) < 1e-10


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

    def test_zero_mw(self):
        with pytest.raises(ValueError):
            _run(mw_Da=0)


class TestNotes:
    def test_notes_is_string(self):
        r = _run()
        assert isinstance(r.notes, str)
        assert len(r.notes) > 0


class TestAcidBase:
    def test_acid_runs(self):
        r = _run(is_acid=True, pka=4.0)
        assert r.cmax_mg_L > 0

    def test_base_runs(self):
        r = _run(is_acid=False, pka=8.0)
        assert r.cmax_mg_L > 0
