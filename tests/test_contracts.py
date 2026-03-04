"""Unit tests for omega_pbpk.contracts module."""

import pytest

from omega_pbpk.contracts import DrugSpec, PatientSpec, Regimen

# ---------------------------------------------------------------------------
# DrugSpec.validate()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDrugSpecValidate:
    def test_valid_default(self):
        DrugSpec(name="test").validate()

    def test_fup_zero_raises(self):
        with pytest.raises(ValueError, match="fup"):
            DrugSpec(name="x", fup=0).validate()

    def test_fup_negative_raises(self):
        with pytest.raises(ValueError, match="fup"):
            DrugSpec(name="x", fup=-0.1).validate()

    def test_fup_above_one_raises(self):
        with pytest.raises(ValueError, match="fup"):
            DrugSpec(name="x", fup=1.1).validate()

    def test_fup_exactly_one_ok(self):
        DrugSpec(name="x", fup=1.0).validate()

    def test_mw_negative_raises(self):
        with pytest.raises(ValueError, match="mw"):
            DrugSpec(name="x", mw=-1).validate()

    def test_mw_zero_raises(self):
        with pytest.raises(ValueError, match="mw"):
            DrugSpec(name="x", mw=0).validate()

    def test_clint_negative_raises(self):
        with pytest.raises(ValueError, match="clint_hepatic"):
            DrugSpec(name="x", clint_hepatic_L_per_h=-1).validate()

    def test_clr_negative_raises(self):
        with pytest.raises(ValueError, match="clr"):
            DrugSpec(name="x", clr_L_per_h=-0.5).validate()

    def test_peff_negative_raises(self):
        with pytest.raises(ValueError, match="peff"):
            DrugSpec(name="x", peff=-0.1).validate()

    def test_solubility_zero_raises(self):
        with pytest.raises(ValueError, match="solubility"):
            DrugSpec(name="x", solubility_mg_mL=0).validate()

    def test_kp_negative_raises(self):
        with pytest.raises(ValueError, match="kp\\[liver\\]"):
            DrugSpec(name="x", kp={"liver": -0.5}).validate()

    def test_kp_zero_ok(self):
        DrugSpec(name="x", kp={"liver": 0.0}).validate()

    def test_rbp_zero_raises(self):
        with pytest.raises(ValueError, match="rbp"):
            DrugSpec(name="x", rbp=0).validate()

    def test_rbp_negative_raises(self):
        with pytest.raises(ValueError, match="rbp"):
            DrugSpec(name="x", rbp=-1).validate()


# ---------------------------------------------------------------------------
# PatientSpec.validate()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPatientSpecValidate:
    def test_valid_default(self):
        PatientSpec().validate()

    def test_body_weight_zero_raises(self):
        with pytest.raises(ValueError, match="body_weight_kg"):
            PatientSpec(body_weight_kg=0).validate()

    def test_body_weight_negative_raises(self):
        with pytest.raises(ValueError, match="body_weight_kg"):
            PatientSpec(body_weight_kg=-10).validate()

    def test_age_negative_raises(self):
        with pytest.raises(ValueError, match="age_years"):
            PatientSpec(age_years=-1).validate()

    def test_age_zero_ok(self):
        PatientSpec(age_years=0).validate()

    def test_gfr_negative_raises(self):
        with pytest.raises(ValueError, match="gfr_mL_min"):
            PatientSpec(gfr_mL_min=-1).validate()

    def test_gfr_zero_ok(self):
        PatientSpec(gfr_mL_min=0).validate()

    def test_cardiac_output_zero_raises(self):
        with pytest.raises(ValueError, match="cardiac_output"):
            PatientSpec(cardiac_output_L_h=0).validate()


# ---------------------------------------------------------------------------
# Regimen.validate()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRegimenValidate:
    def test_valid(self):
        Regimen(dose_mg=100).validate()

    def test_dose_zero_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            Regimen(dose_mg=0).validate()

    def test_dose_negative_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            Regimen(dose_mg=-50).validate()

    def test_n_doses_zero_raises(self):
        with pytest.raises(ValueError, match="n_doses"):
            Regimen(dose_mg=100, n_doses=0).validate()

    def test_interval_zero_raises(self):
        with pytest.raises(ValueError, match="interval_h"):
            Regimen(dose_mg=100, interval_h=0).validate()

    def test_n_doses_one_ok(self):
        Regimen(dose_mg=100, n_doses=1).validate()
