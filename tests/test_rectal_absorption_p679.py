"""Tests for Phase 679 — Rectal Drug Absorption."""

import pytest

from omega_pbpk.core.rectal_absorption_p679 import (
    RectalAbsorptionResult,
    simulate_rectal_absorption,
)

_BASE = dict(
    drug_name="TestDrug",
    dose_mg=100.0,
    logP=2.0,
    mw_Da=300.0,
    cl_sys_L_per_h=10.0,
    vd_sys_L=50.0,
)


def _run(**overrides):
    kw = {**_BASE, **overrides}
    return simulate_rectal_absorption(**kw)


class TestReturnType:
    def test_returns_result_dataclass(self):
        r = _run()
        assert isinstance(r, RectalAbsorptionResult)

    def test_drug_name_preserved(self):
        r = _run()
        assert r.drug_name == "TestDrug"

    def test_formulation_type_preserved(self):
        r = _run(formulation_type="enema")
        assert r.formulation_type == "enema"


class TestListLengths:
    def test_times_length(self):
        r = _run(t_end_h=6.0, dt_h=0.1)
        expected = int(6.0 / 0.1) + 1
        assert len(r.times_h) == expected

    def test_all_lists_same_length(self):
        r = _run()
        n = len(r.times_h)
        assert len(r.c_rectal_depot_mg) == n
        assert len(r.c_portal_mg_L) == n
        assert len(r.c_systemic_mg_L) == n


class TestNonNegative:
    def test_concentrations_non_negative(self):
        r = _run()
        assert all(c >= 0 for c in r.c_rectal_depot_mg)
        assert all(c >= 0 for c in r.c_portal_mg_L)
        assert all(c >= 0 for c in r.c_systemic_mg_L)

    def test_cmax_positive(self):
        r = _run()
        assert r.cmax_systemic_mg_L > 0

    def test_tmax_positive(self):
        r = _run()
        assert r.tmax_systemic_h > 0

    def test_auc_positive(self):
        r = _run()
        assert r.auc_systemic_mg_h_per_L > 0


class TestFormulationComparison:
    def test_enema_faster_onset_than_suppository(self):
        r_supp = _run(formulation_type="suppository")
        r_enema = _run(formulation_type="enema")
        assert r_enema.tmax_systemic_h < r_supp.tmax_systemic_h

    def test_gel_intermediate_tmax(self):
        r_supp = _run(formulation_type="suppository")
        r_gel = _run(formulation_type="gel")
        # Gel should be faster than suppository
        assert r_gel.tmax_systemic_h <= r_supp.tmax_systemic_h


class TestFirstPassBypass:
    def test_first_pass_avoided_positive(self):
        r = _run()
        assert r.first_pass_avoided_pct > 0

    def test_first_pass_avoided_suppository(self):
        r = _run(formulation_type="suppository")
        assert r.first_pass_avoided_pct == 50.0

    def test_first_pass_avoided_enema(self):
        r = _run(formulation_type="enema")
        assert r.first_pass_avoided_pct == 60.0


class TestDepotDecreases:
    def test_depot_decreases_over_time(self):
        r = _run()
        # Depot should decrease monotonically
        for i in range(1, len(r.c_rectal_depot_mg)):
            assert r.c_rectal_depot_mg[i] <= r.c_rectal_depot_mg[i - 1] + 1e-12


class TestSystemicProfile:
    def test_systemic_rises_then_falls(self):
        r = _run(t_end_h=12.0)
        # Find peak index
        peak_idx = r.c_systemic_mg_L.index(max(r.c_systemic_mg_L))
        assert peak_idx > 0
        # After peak, at least some values should be lower
        assert r.c_systemic_mg_L[-1] < r.c_systemic_mg_L[peak_idx]

    def test_starts_at_zero(self):
        r = _run()
        assert r.c_systemic_mg_L[0] == 0.0


class TestDoseProportionality:
    def test_double_dose_doubles_cmax(self):
        r1 = _run(dose_mg=100.0)
        r2 = _run(dose_mg=200.0)
        ratio = r2.cmax_systemic_mg_L / r1.cmax_systemic_mg_L
        assert 1.9 < ratio < 2.1

    def test_double_dose_doubles_auc(self):
        r1 = _run(dose_mg=100.0)
        r2 = _run(dose_mg=200.0)
        ratio = r2.auc_systemic_mg_h_per_L / r1.auc_systemic_mg_h_per_L
        assert 1.9 < ratio < 2.1


class TestBioavailability:
    def test_bioavailability_in_range(self):
        r = _run()
        assert 0 <= r.bioavailability_pct <= 100

    def test_bioavailability_positive(self):
        r = _run()
        assert r.bioavailability_pct > 0

    def test_high_clearance_lowers_bioavailability(self):
        r_low = _run(cl_sys_L_per_h=5.0)
        r_high = _run(cl_sys_L_per_h=80.0)
        assert r_high.bioavailability_pct < r_low.bioavailability_pct


class TestValidation:
    def test_negative_dose_raises(self):
        with pytest.raises(ValueError):
            _run(dose_mg=-10.0)

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError):
            _run(dose_mg=0.0)

    def test_negative_clearance_raises(self):
        with pytest.raises(ValueError):
            _run(cl_sys_L_per_h=-1.0)

    def test_negative_vd_raises(self):
        with pytest.raises(ValueError):
            _run(vd_sys_L=-1.0)

    def test_invalid_formulation_raises(self):
        with pytest.raises(ValueError):
            _run(formulation_type="tablet")
