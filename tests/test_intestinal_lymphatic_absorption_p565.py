"""Tests for Phase 565 - Intestinal Lymphatic Drug Absorption."""

import pytest

from omega_pbpk.core.intestinal_lymphatic_absorption_p565 import (
    LymphaticAbsorptionResult,
    simulate_lymphatic_absorption,
)


def _default_result(**kwargs):
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        logP=6.0,
        ka_per_h=1.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
    )
    defaults.update(kwargs)
    return simulate_lymphatic_absorption(**defaults)


class TestReturnType:
    def test_returns_lymphatic_absorption_result(self):
        r = _default_result()
        assert isinstance(r, LymphaticAbsorptionResult)

    def test_list_lengths_equal(self):
        r = _default_result()
        n = len(r.times_h)
        assert len(r.c_portal_mg_L) == n
        assert len(r.c_lymph_mg_L) == n
        assert len(r.c_systemic_mg_L) == n

    def test_drug_name_preserved(self):
        r = _default_result(drug_name="Halofantrine")
        assert r.drug_name == "Halofantrine"

    def test_dose_preserved(self):
        r = _default_result(dose_mg=200.0)
        assert r.dose_mg == 200.0


class TestLymphaticFraction:
    def test_logP_below_4_gives_zero_lymphatic(self):
        r = _default_result(logP=3.0)
        assert r.f_lymphatic == 0.0

    def test_logP_equal_4_gives_zero_lymphatic(self):
        r = _default_result(logP=4.0)
        assert r.f_lymphatic == 0.0

    def test_logP_6_gives_0_2(self):
        r = _default_result(logP=6.0)
        assert abs(r.f_lymphatic - 0.2) < 1e-10

    def test_logP_8_gives_0_4(self):
        r = _default_result(logP=8.0)
        assert abs(r.f_lymphatic - 0.4) < 1e-10

    def test_logP_very_high_capped_at_0_8(self):
        r = _default_result(logP=20.0)
        assert abs(r.f_lymphatic - 0.8) < 1e-10

    def test_f_portal_plus_f_lymphatic_equals_one(self):
        for lp in [2.0, 4.0, 5.0, 6.0, 8.0, 12.0]:
            r = _default_result(logP=lp)
            assert abs(r.f_portal + r.f_lymphatic - 1.0) < 1e-10


class TestFirstPassBypass:
    def test_logP_below_4_no_bypass(self):
        r = _default_result(logP=3.0)
        assert r.first_pass_bypass_fraction == 0.0

    def test_logP_above_4_higher_bypass(self):
        r = _default_result(logP=6.0)
        assert r.first_pass_bypass_fraction > 0.0

    def test_bypass_equals_f_lymphatic(self):
        r = _default_result(logP=7.0)
        assert r.first_pass_bypass_fraction == r.f_lymphatic


class TestAUC:
    def test_auc_portal_positive(self):
        r = _default_result()
        assert r.auc_portal_mg_h_per_L > 0

    def test_auc_systemic_positive(self):
        r = _default_result()
        assert r.auc_systemic_mg_h_per_L > 0

    def test_dose_linearity_systemic(self):
        r1 = _default_result(dose_mg=100.0)
        r2 = _default_result(dose_mg=200.0)
        ratio = r2.auc_systemic_mg_h_per_L / r1.auc_systemic_mg_h_per_L
        assert abs(ratio - 2.0) < 0.1

    def test_dose_linearity_portal(self):
        r1 = _default_result(dose_mg=100.0)
        r2 = _default_result(dose_mg=200.0)
        ratio = r2.auc_portal_mg_h_per_L / r1.auc_portal_mg_h_per_L
        assert abs(ratio - 2.0) < 0.1


class TestValidation:
    def test_dose_zero_raises(self):
        with pytest.raises(ValueError):
            _default_result(dose_mg=0)

    def test_dose_negative_raises(self):
        with pytest.raises(ValueError):
            _default_result(dose_mg=-10)

    def test_ka_zero_raises(self):
        with pytest.raises(ValueError):
            _default_result(ka_per_h=0)

    def test_cl_zero_raises(self):
        with pytest.raises(ValueError):
            _default_result(cl_L_per_h=0)

    def test_vd_zero_raises(self):
        with pytest.raises(ValueError):
            _default_result(vd_L=0)


class TestNotes:
    def test_notes_contains_drug_name(self):
        r = _default_result(drug_name="DrugX")
        assert "DrugX" in r.notes
