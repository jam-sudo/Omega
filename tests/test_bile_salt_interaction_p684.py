"""Tests for Phase 684 — Bile Salt Interaction."""

import pytest

from omega_pbpk.core.bile_salt_interaction_p684 import (
    BileSaltInteractionResult,
    simulate_bile_salt_interaction,
)

DEFAULTS = dict(
    drug_name="TestDrug",
    dose_mg=100.0,
    logP=2.5,
    mw_Da=350.0,
    cl_sys_L_per_h=5.0,
    vd_sys_L=50.0,
    bile_salt_conc_mM=10.0,
)


def _run(**overrides):
    params = {**DEFAULTS, **overrides}
    return simulate_bile_salt_interaction(**params)


# --- Return type and structure ---


class TestReturnType:
    def test_return_type(self):
        r = _run()
        assert isinstance(r, BileSaltInteractionResult)

    def test_drug_name(self):
        r = _run()
        assert r.drug_name == "TestDrug"

    def test_dose_mg(self):
        r = _run()
        assert r.dose_mg == 100.0

    def test_times_is_list(self):
        r = _run()
        assert isinstance(r.times_h, list)
        assert len(r.times_h) > 0

    def test_c_free_gut_is_list(self):
        r = _run()
        assert isinstance(r.c_free_gut_mg_L, list)
        assert len(r.c_free_gut_mg_L) == len(r.times_h)

    def test_c_micellar_is_list(self):
        r = _run()
        assert isinstance(r.c_micellar_mg_L, list)
        assert len(r.c_micellar_mg_L) == len(r.times_h)

    def test_c_plasma_is_list(self):
        r = _run()
        assert isinstance(r.c_plasma_mg_L, list)
        assert len(r.c_plasma_mg_L) == len(r.times_h)


# --- f_micellar ---


class TestFMicellar:
    def test_f_micellar_in_range(self):
        r = _run()
        assert 0 <= r.f_micellar < 1.0

    def test_high_logP_higher_f_micellar(self):
        r_low = _run(logP=0.5)
        r_high = _run(logP=4.0)
        assert r_high.f_micellar > r_low.f_micellar

    def test_zero_bile_salt_zero_f_micellar(self):
        r = _run(bile_salt_conc_mM=0.0)
        assert r.f_micellar == 0.0


# --- Effective solubility ---


class TestEffectiveSolubility:
    def test_effective_solubility_positive(self):
        r = _run()
        assert r.effective_solubility_mg_L > 0


# --- Non-negativity ---


class TestNonNegativity:
    def test_c_free_gut_non_negative(self):
        r = _run()
        assert all(c >= 0 for c in r.c_free_gut_mg_L)

    def test_c_micellar_non_negative(self):
        r = _run()
        assert all(c >= 0 for c in r.c_micellar_mg_L)

    def test_c_plasma_non_negative(self):
        r = _run()
        assert all(c >= 0 for c in r.c_plasma_mg_L)


# --- Cmax and AUC ---


class TestCmaxAUC:
    def test_cmax_positive(self):
        r = _run()
        assert r.cmax_mg_L > 0

    def test_auc_positive(self):
        r = _run()
        assert r.auc_mg_h_per_L > 0


# --- Absorption enhancement ---


class TestAbsorptionEnhancement:
    def test_enhancement_ge_1_lipophilic(self):
        r = _run(logP=3.0, bile_salt_conc_mM=15.0)
        assert r.absorption_enhancement_fold >= 1.0

    def test_no_bile_enhancement_is_1(self):
        r = _run(bile_salt_conc_mM=0.0)
        assert abs(r.absorption_enhancement_fold - 1.0) < 0.01

    def test_more_bile_more_enhancement(self):
        r_low = _run(bile_salt_conc_mM=5.0, logP=3.0)
        r_high = _run(bile_salt_conc_mM=20.0, logP=3.0)
        assert r_high.absorption_enhancement_fold >= r_low.absorption_enhancement_fold


# --- Dose linearity ---


class TestDoseLinearity:
    def test_double_dose_double_cmax(self):
        # Use small doses to stay well below solubility limit
        r1 = _run(dose_mg=5.0)
        r2 = _run(dose_mg=10.0)
        ratio = r2.cmax_mg_L / r1.cmax_mg_L
        assert abs(ratio - 2.0) < 0.15


# --- Validation ---


class TestValidation:
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

    def test_negative_bile_salt(self):
        with pytest.raises(ValueError):
            _run(bile_salt_conc_mM=-1)


# --- Notes ---


class TestNotes:
    def test_notes_is_string(self):
        r = _run()
        assert isinstance(r.notes, str)
        assert len(r.notes) > 0
