"""Tests for Phase 675 — Intraperitoneal Drug Absorption PK."""

import pytest

from omega_pbpk.core.intraperitoneal_pk_p675 import (
    IntraperitonealPKResult,
    simulate_intraperitoneal_pk,
)


def _default_sim(**kwargs):
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=10.0,
        cl_sys_L_per_h=0.5,
        vd_sys_L=1.0,
        logP=2.0,
        mw_Da=300.0,
        species="mouse",
    )
    defaults.update(kwargs)
    return simulate_intraperitoneal_pk(**defaults)


class TestReturnType:
    def test_returns_result_type(self):
        r = _default_sim()
        assert isinstance(r, IntraperitonealPKResult)

    def test_drug_name(self):
        r = _default_sim(drug_name="Aspirin")
        assert r.drug_name == "Aspirin"

    def test_species_stored(self):
        r = _default_sim(species="rat")
        assert r.species == "rat"


class TestListLengths:
    def test_lists_same_length(self):
        r = _default_sim()
        n = len(r.times_h)
        assert len(r.c_peritoneal_mg_L) == n
        assert len(r.c_portal_mg_L) == n
        assert len(r.c_systemic_mg_L) == n

    def test_lists_nonempty(self):
        r = _default_sim()
        assert len(r.times_h) > 0


class TestNonNegative:
    def test_concentrations_non_negative(self):
        r = _default_sim()
        assert all(c >= 0 for c in r.c_peritoneal_mg_L)
        assert all(c >= 0 for c in r.c_portal_mg_L)
        assert all(c >= 0 for c in r.c_systemic_mg_L)

    def test_auc_non_negative(self):
        r = _default_sim()
        assert r.auc_systemic_mg_h_per_L >= 0
        assert r.auc_peritoneal_mg_h_per_L >= 0


class TestCmaxTmax:
    def test_cmax_positive(self):
        r = _default_sim()
        assert r.cmax_systemic_mg_L > 0

    def test_tmax_positive(self):
        r = _default_sim()
        assert r.tmax_systemic_h > 0


class TestPeritonealDecreases:
    def test_peritoneal_decreases(self):
        r = _default_sim()
        # First value should be highest
        assert r.c_peritoneal_mg_L[0] >= r.c_peritoneal_mg_L[-1]
        # Check monotonic decrease
        for i in range(1, len(r.c_peritoneal_mg_L)):
            assert r.c_peritoneal_mg_L[i] <= r.c_peritoneal_mg_L[i - 1] + 1e-12


class TestSystemicRiseThenFall:
    def test_systemic_rises_then_falls(self):
        r = _default_sim()
        idx_max = r.c_systemic_mg_L.index(max(r.c_systemic_mg_L))
        # Should rise to max
        assert idx_max > 0
        # Should fall after max
        assert r.c_systemic_mg_L[-1] < r.cmax_systemic_mg_L


class TestDoseProportionality:
    def test_double_dose_doubles_cmax(self):
        r1 = _default_sim(dose_mg=10.0)
        r2 = _default_sim(dose_mg=20.0)
        ratio = r2.cmax_systemic_mg_L / r1.cmax_systemic_mg_L
        assert abs(ratio - 2.0) < 0.1

    def test_double_dose_doubles_auc(self):
        r1 = _default_sim(dose_mg=10.0)
        r2 = _default_sim(dose_mg=20.0)
        ratio = r2.auc_systemic_mg_h_per_L / r1.auc_systemic_mg_h_per_L
        assert abs(ratio - 2.0) < 0.1


class TestFirstPassEffect:
    def test_first_pass_in_range(self):
        r = _default_sim()
        assert 0 <= r.first_pass_effect_pct <= 100

    def test_first_pass_not_zero(self):
        r = _default_sim()
        assert r.first_pass_effect_pct > 0


class TestAbsorptionHalfLife:
    def test_absorption_half_life_positive(self):
        r = _default_sim()
        assert r.absorption_half_life_h > 0


class TestLogPEffect:
    def test_higher_logP_faster_absorption(self):
        r1 = _default_sim(logP=1.0)
        r2 = _default_sim(logP=3.0)
        assert r2.absorption_half_life_h < r1.absorption_half_life_h


class TestSpecies:
    def test_mouse(self):
        r = _default_sim(species="mouse")
        assert r.species == "mouse"

    def test_rat(self):
        r = _default_sim(species="rat")
        assert r.cmax_systemic_mg_L > 0

    def test_human(self):
        r = _default_sim(species="human", vd_sys_L=50.0, cl_sys_L_per_h=5.0)
        assert r.cmax_systemic_mg_L > 0

    def test_invalid_species(self):
        with pytest.raises(ValueError, match="species"):
            _default_sim(species="dog")


class TestValidation:
    def test_zero_dose(self):
        with pytest.raises(ValueError):
            _default_sim(dose_mg=0)

    def test_negative_dose(self):
        with pytest.raises(ValueError):
            _default_sim(dose_mg=-1)

    def test_zero_clearance(self):
        with pytest.raises(ValueError):
            _default_sim(cl_sys_L_per_h=0)

    def test_zero_vd(self):
        with pytest.raises(ValueError):
            _default_sim(vd_sys_L=0)


class TestNotes:
    def test_notes_present(self):
        r = _default_sim()
        assert len(r.notes) > 0


class TestAUCRatio:
    def test_peritoneal_to_systemic_ratio_positive(self):
        r = _default_sim()
        assert r.peritoneal_to_systemic_auc_ratio > 0
