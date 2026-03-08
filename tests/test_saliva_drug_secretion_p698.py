"""Tests for Phase 698 — Drug Saliva Secretion."""

import pytest

from omega_pbpk.core.saliva_drug_secretion_p698 import (
    SalivaDrugSecretionResult,
    simulate_saliva_drug_secretion,
)

BASE = dict(
    drug_name="TestDrug",
    dose_mg=10.0,
    logP=2.0,
    pka=8.0,
    is_acid=False,
    fu_plasma=0.5,
    cl_sys_L_per_h=5.0,
    vd_sys_L=50.0,
)


def _run(**kw):
    params = {**BASE, **kw}
    return simulate_saliva_drug_secretion(**params)


class TestReturnType:
    def test_returns_result(self):
        r = _run()
        assert isinstance(r, SalivaDrugSecretionResult)

    def test_drug_name(self):
        r = _run()
        assert r.drug_name == "TestDrug"

    def test_dose_mg(self):
        r = _run()
        assert r.dose_mg == 10.0

    def test_times_list(self):
        r = _run()
        assert isinstance(r.times_h, list) and len(r.times_h) > 1

    def test_plasma_list(self):
        r = _run()
        assert isinstance(r.c_plasma_mg_L, list) and len(r.c_plasma_mg_L) == len(r.times_h)

    def test_saliva_list(self):
        r = _run()
        assert isinstance(r.c_saliva_mg_L, list) and len(r.c_saliva_mg_L) == len(r.times_h)


class TestConcentrations:
    def test_plasma_nonneg(self):
        r = _run()
        assert all(c >= 0 for c in r.c_plasma_mg_L)

    def test_saliva_nonneg(self):
        r = _run()
        assert all(c >= 0 for c in r.c_saliva_mg_L)

    def test_cmax_plasma_positive(self):
        r = _run()
        assert r.cmax_plasma_mg_L > 0

    def test_cmax_saliva_positive(self):
        r = _run()
        assert r.cmax_saliva_mg_L > 0

    def test_auc_plasma_positive(self):
        r = _run()
        assert r.auc_plasma_mg_h_per_L > 0

    def test_auc_saliva_positive(self):
        r = _run()
        assert r.auc_saliva_mg_h_per_L > 0

    def test_saliva_follows_plasma(self):
        """C_saliva should be proportional to C_plasma."""
        r = _run()
        # At each time point, c_saliva = fu * c_plasma * ratio
        # So saliva should peak roughly when plasma peaks
        idx_peak_plasma = r.c_plasma_mg_L.index(max(r.c_plasma_mg_L))
        idx_peak_saliva = r.c_saliva_mg_L.index(max(r.c_saliva_mg_L))
        assert abs(idx_peak_plasma - idx_peak_saliva) <= 1


class TestFuEffect:
    def test_higher_fu_higher_saliva(self):
        r1 = _run(fu_plasma=0.1)
        r2 = _run(fu_plasma=0.9)
        assert r2.cmax_saliva_mg_L > r1.cmax_saliva_mg_L


class TestMonitoringFeasibility:
    def test_feasibility_is_string(self):
        r = _run()
        assert r.monitoring_feasibility in ("excellent", "good", "poor")

    def test_neutral_drug_ph74_excellent(self):
        """pH 7.4 saliva → ratio ≈ 1 for neutral drugs (pKa >> plasma pH)."""
        r = _run(pka=14.0, is_acid=False, saliva_ph=7.4)
        assert r.monitoring_feasibility == "excellent"

    def test_acid_low_pka(self):
        """Acid with pKa < saliva_ph → ratio < 1."""
        r = _run(pka=4.0, is_acid=True, saliva_ph=6.8)
        # ratio = (1+10^(6.8-4))/(1+10^(7.4-4)) = (1+631)/(1+2512) ≈ 0.25
        assert r.monitoring_feasibility == "poor"


class TestRatioUnbound:
    def test_ratio_positive(self):
        r = _run()
        assert r.saliva_to_plasma_ratio_unbound > 0


class TestDoseLinearity:
    def test_double_dose_double_cmax(self):
        r1 = _run(dose_mg=10.0)
        r2 = _run(dose_mg=20.0)
        ratio = r2.cmax_plasma_mg_L / r1.cmax_plasma_mg_L
        assert 1.9 < ratio < 2.1


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

    def test_bad_fu(self):
        with pytest.raises(ValueError):
            _run(fu_plasma=0.0)

    def test_bad_flow(self):
        with pytest.raises(ValueError):
            _run(salivary_flow_mL_per_h=0)


class TestNotes:
    def test_notes_is_string(self):
        r = _run()
        assert isinstance(r.notes, str) and len(r.notes) > 0
