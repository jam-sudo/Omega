"""Tests for Phase 872: Extended BCS Classification."""

import pytest

from omega_pbpk.biopharmaceutics.bcs_extended import (
    BCSExtendedResult,
    classify_bcs_extended,
    screen_bcs_portfolio,
)


class TestReturnType:
    def test_returns_bcs_extended_result(self):
        result = classify_bcs_extended("Drug", dose_mg=10.0, solubility_mg_mL=1.0, peff_cm_s=2e-4)
        assert isinstance(result, BCSExtendedResult)

    def test_fields_preserved(self):
        result = classify_bcs_extended(
            "Ibuprofen", dose_mg=200.0, solubility_mg_mL=0.021, peff_cm_s=4e-4, f_metabolized=0.9
        )
        assert result.drug_name == "Ibuprofen"
        assert result.dose_mg == pytest.approx(200.0)
        assert result.solubility_mg_mL == pytest.approx(0.021)
        assert result.f_metabolized == pytest.approx(0.9)

    def test_frozen_dataclass(self):
        result = classify_bcs_extended("Drug", dose_mg=10.0, solubility_mg_mL=1.0, peff_cm_s=2e-4)
        with pytest.raises((AttributeError, TypeError)):
            result.drug_name = "Other"


class TestDoseAndAbsorptionNumbers:
    def test_dose_number_computed(self):
        # Do = 10 / (1.0 * 250) = 0.04
        result = classify_bcs_extended("X", dose_mg=10.0, solubility_mg_mL=1.0, peff_cm_s=2e-4)
        assert result.dose_number == pytest.approx(0.04, rel=1e-4)

    def test_absorption_number_computed(self):
        # An = Peff * 3600 * 3.5 / 0.175
        peff = 2e-4
        expected_an = peff * 3600 * 3.5 / 0.175
        result = classify_bcs_extended("X", dose_mg=10.0, solubility_mg_mL=1.0, peff_cm_s=peff)
        assert result.absorption_number == pytest.approx(expected_an, rel=1e-4)

    def test_absorption_number_zero_peff(self):
        result = classify_bcs_extended("X", dose_mg=10.0, solubility_mg_mL=1.0, peff_cm_s=0.0)
        assert result.absorption_number == pytest.approx(0.0)


class TestBCSClassification:
    def test_class_I_high_sol_high_perm(self):
        # Do < 1: dose_mg / (sol * 250) < 1 → sol > dose/250
        # An > 1: Peff > 0.175 / (3600 * 3.5) ≈ 1.39e-5
        result = classify_bcs_extended("ClassI", dose_mg=10.0, solubility_mg_mL=5.0, peff_cm_s=1e-4)
        assert result.bcs_class == "I"
        assert result.high_solubility is True
        assert result.high_permeability is True

    def test_class_II_low_sol_high_perm(self):
        # Do >= 1: dose_mg / (sol * 250) >= 1 → sol small
        # An > 1
        result = classify_bcs_extended(
            "ClassII", dose_mg=500.0, solubility_mg_mL=0.001, peff_cm_s=1e-4
        )
        assert result.bcs_class == "II"
        assert result.high_solubility is False
        assert result.high_permeability is True

    def test_class_III_high_sol_low_perm(self):
        # Do < 1, An <= 1 (very low Peff)
        result = classify_bcs_extended(
            "ClassIII", dose_mg=10.0, solubility_mg_mL=5.0, peff_cm_s=1e-7
        )
        assert result.bcs_class == "III"
        assert result.high_solubility is True
        assert result.high_permeability is False

    def test_class_IV_low_sol_low_perm(self):
        # Do >= 1, An <= 1
        result = classify_bcs_extended(
            "ClassIV", dose_mg=500.0, solubility_mg_mL=0.001, peff_cm_s=1e-7
        )
        assert result.bcs_class == "IV"
        assert result.high_solubility is False
        assert result.high_permeability is False


class TestBDDCS:
    def test_bddcs_class_1_high_sol_extensive(self):
        result = classify_bcs_extended(
            "BDDCS1", dose_mg=10.0, solubility_mg_mL=5.0, peff_cm_s=1e-4, f_metabolized=0.9
        )
        assert result.bddcs_class == "1"

    def test_bddcs_class_2_low_sol_extensive(self):
        result = classify_bcs_extended(
            "BDDCS2", dose_mg=500.0, solubility_mg_mL=0.001, peff_cm_s=1e-4, f_metabolized=0.85
        )
        assert result.bddcs_class == "2"

    def test_bddcs_class_3_high_sol_poor(self):
        result = classify_bcs_extended(
            "BDDCS3", dose_mg=10.0, solubility_mg_mL=5.0, peff_cm_s=1e-7, f_metabolized=0.2
        )
        assert result.bddcs_class == "3"

    def test_bddcs_class_4_low_sol_poor(self):
        result = classify_bcs_extended(
            "BDDCS4", dose_mg=500.0, solubility_mg_mL=0.001, peff_cm_s=1e-7, f_metabolized=0.1
        )
        assert result.bddcs_class == "4"

    def test_bddcs_threshold_extensive(self):
        # f_metabolized = 0.7 exactly → extensive
        result = classify_bcs_extended(
            "Thresh", dose_mg=10.0, solubility_mg_mL=5.0, peff_cm_s=1e-4, f_metabolized=0.7
        )
        assert result.bddcs_class == "1"


class TestBiowaiver:
    def test_class_I_rapidly_dissolving_eligible(self):
        result = classify_bcs_extended(
            "ClassI", dose_mg=10.0, solubility_mg_mL=5.0, peff_cm_s=1e-4, rapidly_dissolving=True
        )
        assert result.bcs_class == "I"
        assert result.biowaiver_eligible is True

    def test_class_I_not_rapidly_dissolving_not_eligible(self):
        result = classify_bcs_extended(
            "ClassI_slow",
            dose_mg=10.0,
            solubility_mg_mL=5.0,
            peff_cm_s=1e-4,
            rapidly_dissolving=False,
        )
        assert result.biowaiver_eligible is False

    def test_class_II_not_eligible(self):
        result = classify_bcs_extended(
            "ClassII",
            dose_mg=500.0,
            solubility_mg_mL=0.001,
            peff_cm_s=1e-4,
            rapidly_dissolving=True,
        )
        assert result.biowaiver_eligible is False

    def test_class_IV_not_eligible(self):
        result = classify_bcs_extended(
            "ClassIV",
            dose_mg=500.0,
            solubility_mg_mL=0.001,
            peff_cm_s=1e-7,
            rapidly_dissolving=True,
        )
        assert result.biowaiver_eligible is False

    def test_class_III_rapidly_dissolving_eligible(self):
        result = classify_bcs_extended(
            "ClassIII", dose_mg=10.0, solubility_mg_mL=5.0, peff_cm_s=1e-7, rapidly_dissolving=True
        )
        assert result.bcs_class == "III"
        assert result.biowaiver_eligible is True


class TestPredictedFa:
    def test_predicted_fa_in_range(self):
        result = classify_bcs_extended("X", dose_mg=10.0, solubility_mg_mL=1.0, peff_cm_s=2e-4)
        assert 0.0 <= result.predicted_fa <= 1.0

    def test_predicted_fa_zero_peff(self):
        result = classify_bcs_extended("X", dose_mg=10.0, solubility_mg_mL=1.0, peff_cm_s=0.0)
        assert result.predicted_fa == pytest.approx(0.0)

    def test_predicted_fa_high_perm(self):
        # High Peff → fa close to 1
        result = classify_bcs_extended("X", dose_mg=10.0, solubility_mg_mL=1.0, peff_cm_s=1e-2)
        assert result.predicted_fa > 0.9


class TestScreenPortfolio:
    def test_returns_dict_with_class_keys(self):
        compounds = [
            {"drug_name": "A", "dose_mg": 10.0, "solubility_mg_mL": 5.0, "peff_cm_s": 1e-4},
            {"drug_name": "B", "dose_mg": 500.0, "solubility_mg_mL": 0.001, "peff_cm_s": 1e-7},
        ]
        result = screen_bcs_portfolio(compounds)
        assert "class_I" in result
        assert "class_II" in result
        assert "class_III" in result
        assert "class_IV" in result
        assert "biowaiver_candidates" in result

    def test_compounds_routed_correctly(self):
        compounds = [
            {"drug_name": "ClassI", "dose_mg": 10.0, "solubility_mg_mL": 5.0, "peff_cm_s": 1e-4},
            {
                "drug_name": "ClassIV",
                "dose_mg": 500.0,
                "solubility_mg_mL": 0.001,
                "peff_cm_s": 1e-7,
            },
        ]
        result = screen_bcs_portfolio(compounds)
        assert len(result["class_I"]) == 1
        assert len(result["class_IV"]) == 1
        assert result["class_I"][0].drug_name == "ClassI"

    def test_biowaiver_candidates_populated(self):
        compounds = [
            {
                "drug_name": "BioOK",
                "dose_mg": 10.0,
                "solubility_mg_mL": 5.0,
                "peff_cm_s": 1e-4,
                "rapidly_dissolving": True,
            },
            {
                "drug_name": "BioNo",
                "dose_mg": 500.0,
                "solubility_mg_mL": 0.001,
                "peff_cm_s": 1e-4,
                "rapidly_dissolving": True,
            },
        ]
        result = screen_bcs_portfolio(compounds)
        names = [r.drug_name for r in result["biowaiver_candidates"]]
        assert "BioOK" in names
        assert "BioNo" not in names

    def test_empty_list_returns_empty_dict(self):
        result = screen_bcs_portfolio([])
        assert result["class_I"] == []
        assert result["class_II"] == []
        assert result["class_III"] == []
        assert result["class_IV"] == []
        assert result["biowaiver_candidates"] == []


class TestValidation:
    def test_dose_mg_zero(self):
        with pytest.raises(ValueError, match="dose_mg must be > 0"):
            classify_bcs_extended("X", dose_mg=0.0, solubility_mg_mL=1.0, peff_cm_s=1e-4)

    def test_dose_mg_negative(self):
        with pytest.raises(ValueError, match="dose_mg must be > 0"):
            classify_bcs_extended("X", dose_mg=-5.0, solubility_mg_mL=1.0, peff_cm_s=1e-4)

    def test_solubility_zero(self):
        with pytest.raises(ValueError, match="solubility_mg_mL must be > 0"):
            classify_bcs_extended("X", dose_mg=10.0, solubility_mg_mL=0.0, peff_cm_s=1e-4)

    def test_peff_negative(self):
        with pytest.raises(ValueError, match="peff_cm_s must be >= 0"):
            classify_bcs_extended("X", dose_mg=10.0, solubility_mg_mL=1.0, peff_cm_s=-1e-4)
