"""Tests for Phase 861 — Otic (Ear) Drug Delivery."""

import pytest

from omega_pbpk.core.otic_drug_delivery_p861 import (
    OticDrugDeliveryResult,
    simulate_otic_drug_delivery,
)

DEFAULTS = dict(
    drug_name="TestDrug",
    dose_mg=1.0,
    logP=2.0,
    mw_Da=300.0,
    cl_sys_L_per_h=5.0,
    vd_sys_L=50.0,
)


def _run(**overrides):
    params = {**DEFAULTS, **overrides}
    return simulate_otic_drug_delivery(**params)


class TestReturnType:
    def test_returns_dataclass(self):
        r = _run()
        assert isinstance(r, OticDrugDeliveryResult)

    def test_drug_name(self):
        r = _run()
        assert r.drug_name == "TestDrug"

    def test_dose_mg(self):
        r = _run()
        assert r.dose_mg == 1.0

    def test_times_is_list(self):
        r = _run()
        assert isinstance(r.times_h, list)
        assert len(r.times_h) > 10

    def test_all_concentration_lists_same_length(self):
        r = _run()
        n = len(r.times_h)
        assert len(r.c_ear_canal_mg_mL) == n
        assert len(r.c_middle_ear_mg_L) == n
        assert len(r.c_inner_ear_mg_L) == n
        assert len(r.c_plasma_mg_L) == n


class TestConcentrations:
    def test_non_negative_ear_canal(self):
        r = _run()
        assert all(c >= 0 for c in r.c_ear_canal_mg_mL)

    def test_non_negative_middle_ear(self):
        r = _run()
        assert all(c >= 0 for c in r.c_middle_ear_mg_L)

    def test_non_negative_inner_ear(self):
        r = _run()
        assert all(c >= 0 for c in r.c_inner_ear_mg_L)

    def test_non_negative_plasma(self):
        r = _run()
        assert all(c >= 0 for c in r.c_plasma_mg_L)

    def test_ear_canal_decays(self):
        r = _run()
        assert r.c_ear_canal_mg_mL[0] > r.c_ear_canal_mg_mL[-1]

    def test_ear_canal_initial_high(self):
        r = _run()
        assert r.c_ear_canal_mg_mL[0] > 0

    def test_middle_ear_rises_then_falls(self):
        r = _run()
        max_idx = r.c_middle_ear_mg_L.index(max(r.c_middle_ear_mg_L))
        assert max_idx > 0
        assert max_idx < len(r.c_middle_ear_mg_L) - 1


class TestMetrics:
    def test_cmax_middle_ear_positive(self):
        r = _run()
        assert r.cmax_middle_ear_mg_L > 0

    def test_cmax_inner_ear_positive(self):
        r = _run()
        assert r.cmax_inner_ear_mg_L > 0

    def test_auc_middle_ear_positive(self):
        r = _run()
        assert r.auc_middle_ear_mg_h_per_L > 0

    def test_auc_inner_ear_positive(self):
        r = _run()
        assert r.auc_inner_ear_mg_h_per_L > 0

    def test_systemic_absorption_in_range(self):
        r = _run()
        assert 0 <= r.systemic_absorption_pct <= 100

    def test_tympanic_permeability_positive(self):
        r = _run()
        assert r.tympanic_membrane_permeability > 0

    def test_round_window_positive(self):
        r = _run()
        assert r.round_window_permeability > 0

    def test_round_window_less_than_tympanic(self):
        r = _run()
        assert r.round_window_permeability < r.tympanic_membrane_permeability


class TestPermeability:
    def test_higher_logP_higher_tympanic(self):
        r_low = _run(logP=0.5)
        r_high = _run(logP=3.0)
        assert r_high.tympanic_membrane_permeability >= r_low.tympanic_membrane_permeability

    def test_higher_logP_higher_round_window(self):
        r_low = _run(logP=0.5)
        r_high = _run(logP=3.0)
        assert r_high.round_window_permeability >= r_low.round_window_permeability


class TestDoseLinearity:
    def test_double_dose_double_cmax(self):
        r1 = _run(dose_mg=1.0)
        r2 = _run(dose_mg=2.0)
        ratio = r2.cmax_middle_ear_mg_L / r1.cmax_middle_ear_mg_L
        assert abs(ratio - 2.0) < 0.1


class TestValidation:
    def test_zero_dose_raises(self):
        with pytest.raises(ValueError):
            _run(dose_mg=0)

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError):
            _run(dose_mg=-1)

    def test_zero_cl_raises(self):
        with pytest.raises(ValueError):
            _run(cl_sys_L_per_h=0)

    def test_zero_vd_raises(self):
        with pytest.raises(ValueError):
            _run(vd_sys_L=0)

    def test_zero_ear_canal_volume_raises(self):
        with pytest.raises(ValueError):
            _run(ear_canal_volume_mL=0)


class TestNotes:
    def test_notes_is_string(self):
        r = _run()
        assert isinstance(r.notes, str)
        assert len(r.notes) > 0
