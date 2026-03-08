"""Tests for Phase 864 — Drug Penetration into Biofilm."""

import pytest

from omega_pbpk.core.biofilm_penetration_p864 import (
    BiofilmPenetrationResult,
    simulate_biofilm_penetration,
)

BASE = dict(
    drug_name="TestAbx", dose_mg=500.0, logP=1.0, mw_Da=400.0, cl_sys_L_per_h=10.0, vd_sys_L=40.0
)


def _run(**overrides):
    params = {**BASE, **overrides}
    return simulate_biofilm_penetration(**params)


class TestBiofilmBasic:
    def test_returns_result(self):
        r = _run()
        assert isinstance(r, BiofilmPenetrationResult)

    def test_drug_name(self):
        r = _run()
        assert r.drug_name == "TestAbx"

    def test_dose_mg(self):
        r = _run()
        assert r.dose_mg == 500.0

    def test_times_non_empty(self):
        r = _run()
        assert len(r.times_h) > 0

    def test_times_start_zero(self):
        r = _run()
        assert r.times_h[0] == 0.0

    def test_plasma_non_negative(self):
        r = _run()
        assert all(c >= 0 for c in r.c_plasma_mg_L)

    def test_surface_non_negative(self):
        r = _run()
        assert all(c >= 0 for c in r.c_biofilm_surface_mg_L)

    def test_deep_non_negative(self):
        r = _run()
        assert all(c >= 0 for c in r.c_biofilm_deep_mg_L)

    def test_cmax_plasma_positive(self):
        r = _run()
        assert r.cmax_plasma_mg_L > 0

    def test_cmax_surface_positive(self):
        r = _run()
        assert r.cmax_surface_mg_L > 0

    def test_cmax_deep_positive(self):
        r = _run()
        assert r.cmax_deep_mg_L > 0

    def test_surface_greater_than_deep(self):
        r = _run()
        assert r.cmax_surface_mg_L >= r.cmax_deep_mg_L

    def test_auc_plasma_positive(self):
        r = _run()
        assert r.auc_plasma_mg_h_per_L > 0

    def test_auc_surface_positive(self):
        r = _run()
        assert r.auc_surface_mg_h_per_L > 0

    def test_notes_is_string(self):
        r = _run()
        assert isinstance(r.notes, str) and len(r.notes) > 0


class TestBiofilmPenetration:
    def test_penetration_ratio_range(self):
        r = _run()
        assert 0 < r.penetration_ratio <= 1.0

    def test_thicker_biofilm_lower_deep(self):
        r_thin = _run(biofilm_thickness_um=50.0)
        r_thick = _run(biofilm_thickness_um=500.0)
        assert r_thin.cmax_deep_mg_L >= r_thick.cmax_deep_mg_L

    def test_mbec_ratio_positive(self):
        r = _run()
        assert r.mbec_ratio > 0


class TestBiofilmEfficacy:
    def test_therapeutic_efficacy_valid(self):
        r = _run()
        assert r.therapeutic_efficacy in ("effective", "partially_effective", "ineffective")

    def test_high_dose_effective(self):
        r = _run(dose_mg=5000.0, mbec_mg_L=1.0)
        assert r.therapeutic_efficacy == "effective"

    def test_low_dose_ineffective(self):
        r = _run(dose_mg=1.0, mbec_mg_L=1000.0)
        assert r.therapeutic_efficacy == "ineffective"


class TestBiofilmDoseLinearity:
    def test_double_dose_doubles_cmax(self):
        r1 = _run(dose_mg=500.0)
        r2 = _run(dose_mg=1000.0)
        ratio = r2.cmax_plasma_mg_L / r1.cmax_plasma_mg_L
        assert 1.8 < ratio < 2.2


class TestBiofilmValidation:
    def test_zero_dose(self):
        with pytest.raises(ValueError):
            _run(dose_mg=0)

    def test_negative_dose(self):
        with pytest.raises(ValueError):
            _run(dose_mg=-10)

    def test_zero_clearance(self):
        with pytest.raises(ValueError):
            _run(cl_sys_L_per_h=0)

    def test_zero_vd(self):
        with pytest.raises(ValueError):
            _run(vd_sys_L=0)

    def test_zero_thickness(self):
        with pytest.raises(ValueError):
            _run(biofilm_thickness_um=0)

    def test_zero_mbec(self):
        with pytest.raises(ValueError):
            _run(mbec_mg_L=0)


class TestBiofilmListLengths:
    def test_list_lengths_match(self):
        r = _run()
        n = len(r.times_h)
        assert len(r.c_plasma_mg_L) == n
        assert len(r.c_biofilm_surface_mg_L) == n
        assert len(r.c_biofilm_deep_mg_L) == n
