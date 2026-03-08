"""Tests for Phase 683 — CSF Drug Penetration."""

import pytest

from omega_pbpk.core.csf_penetration_p683 import (
    CSFPenetrationResult,
    simulate_csf_penetration,
)

# Default parameters for a typical drug
DEFAULTS = dict(
    drug_name="TestDrug",
    dose_mg=100.0,
    logP=2.0,
    mw_Da=300.0,
    psa_A2=60.0,
    cl_sys_L_per_h=5.0,
    vd_sys_L=50.0,
)


def _run(**overrides):
    params = {**DEFAULTS, **overrides}
    return simulate_csf_penetration(**params)


# --- Return type and structure ---


class TestReturnType:
    def test_return_type(self):
        r = _run()
        assert isinstance(r, CSFPenetrationResult)

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

    def test_c_plasma_is_list(self):
        r = _run()
        assert isinstance(r.c_plasma_mg_L, list)
        assert len(r.c_plasma_mg_L) == len(r.times_h)

    def test_c_csf_is_list(self):
        r = _run()
        assert isinstance(r.c_csf_mg_L, list)
        assert len(r.c_csf_mg_L) == len(r.times_h)


# --- Initial conditions ---


class TestInitialConditions:
    def test_plasma_starts_zero(self):
        r = _run()
        assert r.c_plasma_mg_L[0] == 0.0

    def test_csf_starts_zero(self):
        r = _run()
        assert r.c_csf_mg_L[0] == 0.0


# --- Non-negativity ---


class TestNonNegativity:
    def test_plasma_non_negative(self):
        r = _run()
        assert all(c >= 0 for c in r.c_plasma_mg_L)

    def test_csf_non_negative(self):
        r = _run()
        assert all(c >= 0 for c in r.c_csf_mg_L)


# --- Cmax and AUC ---


class TestCmaxAUC:
    def test_cmax_plasma_positive(self):
        r = _run()
        assert r.cmax_plasma_mg_L > 0

    def test_cmax_csf_positive(self):
        r = _run()
        assert r.cmax_csf_mg_L > 0

    def test_auc_plasma_positive(self):
        r = _run()
        assert r.auc_plasma_mg_h_per_L > 0

    def test_auc_csf_positive(self):
        r = _run()
        assert r.auc_csf_mg_h_per_L > 0


# --- Physicochemistry effects on kp_csf ---


class TestPhysicochemistry:
    def test_higher_logP_higher_kp(self):
        r_low = _run(logP=0.5)
        r_high = _run(logP=3.0)
        assert r_high.kp_csf > r_low.kp_csf

    def test_higher_psa_lower_kp(self):
        r_low_psa = _run(psa_A2=40.0)
        r_high_psa = _run(psa_A2=150.0)
        assert r_low_psa.kp_csf > r_high_psa.kp_csf

    def test_higher_mw_lower_kp(self):
        r_low_mw = _run(mw_Da=200.0)
        r_high_mw = _run(mw_Da=800.0)
        assert r_low_mw.kp_csf > r_high_mw.kp_csf


# --- Penetration class ---


class TestPenetrationClass:
    def test_valid_class(self):
        r = _run()
        assert r.penetration_class in ("high", "moderate", "low")

    def test_high_penetration(self):
        r = _run(logP=3.0, mw_Da=200.0, psa_A2=30.0)
        assert r.penetration_class == "high"

    def test_low_penetration(self):
        r = _run(logP=-1.0, mw_Da=900.0, psa_A2=200.0)
        assert r.penetration_class == "low"


# --- CSF to plasma ratio ---


class TestCSFPlasmaRatio:
    def test_ratio_positive(self):
        r = _run()
        assert r.csf_to_plasma_ratio > 0

    def test_ratio_bounded(self):
        r = _run()
        assert 0 < r.csf_to_plasma_ratio < 2.0


# --- t_half_csf ---


class TestHalfLife:
    def test_t_half_csf_positive(self):
        r = _run()
        assert r.t_half_csf_h > 0


# --- Dose linearity ---


class TestDoseLinearity:
    def test_double_dose_double_cmax(self):
        r1 = _run(dose_mg=100.0)
        r2 = _run(dose_mg=200.0)
        ratio = r2.cmax_plasma_mg_L / r1.cmax_plasma_mg_L
        assert abs(ratio - 2.0) < 0.1


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

    def test_zero_mw(self):
        with pytest.raises(ValueError):
            _run(mw_Da=0)

    def test_negative_psa(self):
        with pytest.raises(ValueError):
            _run(psa_A2=-1)


# --- Notes ---


class TestNotes:
    def test_notes_is_string(self):
        r = _run()
        assert isinstance(r.notes, str)
        assert len(r.notes) > 0
