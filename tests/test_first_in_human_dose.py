"""Tests for Phase 362 — First-in-Human Dose Prediction."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.first_in_human_dose import (
    FIHDosePredictionResult,
    predict_fih_dose,
)

_VALID_LIMITING_FACTORS = {"HED", "MABEL", "MTED"}
_VALID_RISK_CATEGORIES = {"low", "moderate", "high"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rat_basic() -> FIHDosePredictionResult:
    return predict_fih_dose(
        drug_name="TestDrug",
        noael_mg_per_kg_animal=10.0,
        animal_species="rat",
    )


def _monkey_tox() -> FIHDosePredictionResult:
    return predict_fih_dose(
        drug_name="ToxDrug",
        noael_mg_per_kg_animal=5.0,
        animal_species="monkey",
        target_organ_toxicity=True,
    )


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_fih_result(self):
        result = _rat_basic()
        assert isinstance(result, FIHDosePredictionResult)

    def test_drug_name_preserved(self):
        result = _rat_basic()
        assert result.drug_name == "TestDrug"

    def test_animal_species_preserved(self):
        result = _rat_basic()
        assert result.animal_species == "rat"

    def test_noael_preserved(self):
        result = _rat_basic()
        assert result.noael_mg_per_kg == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# HED calculation — rat: Km=6, human Km=37
# ---------------------------------------------------------------------------


class TestHEDCalculation:
    def test_hed_mg_per_kg_rat(self):
        # NOAEL=10, Km_rat=6, Km_human=37 → HED = 10 * 6/37
        result = _rat_basic()
        expected = 10.0 * 6.0 / 37.0
        assert result.hed_mg_per_kg == pytest.approx(expected, rel=1e-5)

    def test_hed_mg_rat(self):
        result = _rat_basic()
        expected = (10.0 * 6.0 / 37.0) * 70.0
        assert result.hed_mg == pytest.approx(expected, rel=1e-5)

    def test_hed_mg_per_kg_mouse(self):
        # Km_mouse=3
        result = predict_fih_dose("X", 20.0, "mouse")
        expected = 20.0 * 3.0 / 37.0
        assert result.hed_mg_per_kg == pytest.approx(expected, rel=1e-5)

    def test_hed_mg_per_kg_dog(self):
        # Km_dog=20
        result = predict_fih_dose("X", 5.0, "dog")
        expected = 5.0 * 20.0 / 37.0
        assert result.hed_mg_per_kg == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# MRSD
# ---------------------------------------------------------------------------


class TestMRSD:
    def test_mrsd_positive(self):
        result = _rat_basic()
        assert result.mrsd_mg > 0.0

    def test_mrsd_leq_hed_div_sf(self):
        result = _rat_basic()
        # Use unrounded HED for the comparison: NOAEL=10, Km_rat=6, Km_human=37, BW=70
        hed_unrounded = 10.0 * 6.0 / 37.0 * 70.0
        assert result.mrsd_mg <= hed_unrounded / result.safety_factor + 1e-9

    def test_mrsd_without_constraints_equals_hed_div_sf(self):
        result = _rat_basic()
        expected = result.hed_mg / result.safety_factor
        assert result.mrsd_mg == pytest.approx(expected, rel=1e-5)

    def test_mabel_constraint_applied(self):
        # MABEL / 10 should cap MRSD
        result = predict_fih_dose(
            drug_name="MabelDrug",
            noael_mg_per_kg_animal=10.0,
            animal_species="rat",
            mabel_mg=1.0,  # MABEL/10 = 0.1 mg — very small
        )
        assert result.mrsd_mg <= 1.0 / 10.0 + 1e-9

    def test_mabel_is_limiting_factor(self):
        result = predict_fih_dose(
            drug_name="MabelDrug",
            noael_mg_per_kg_animal=10.0,
            animal_species="rat",
            mabel_mg=0.5,
        )
        assert result.mrsd_limiting_factor == "MABEL"

    def test_mted_constraint_applied(self):
        result = predict_fih_dose(
            drug_name="MtedDrug",
            noael_mg_per_kg_animal=10.0,
            animal_species="rat",
            mted_mg=0.6,  # MTED/6 = 0.1 mg
        )
        assert result.mrsd_mg <= 0.6 / 6.0 + 1e-9

    def test_mted_is_limiting_factor(self):
        result = predict_fih_dose(
            drug_name="MtedDrug",
            noael_mg_per_kg_animal=10.0,
            animal_species="rat",
            mted_mg=0.6,
        )
        assert result.mrsd_limiting_factor == "MTED"


# ---------------------------------------------------------------------------
# Safety factor / risk category
# ---------------------------------------------------------------------------


class TestSafetyFactor:
    def test_default_sf_is_10(self):
        result = _rat_basic()
        assert result.safety_factor == pytest.approx(10.0)

    def test_target_organ_toxicity_raises_sf_to_50(self):
        result = _monkey_tox()
        assert result.safety_factor == pytest.approx(50.0)

    def test_tox_sf_gives_lower_mrsd_than_default(self):
        base = predict_fih_dose("X", 10.0, "rat")
        tox = predict_fih_dose("X", 10.0, "rat", target_organ_toxicity=True)
        assert tox.mrsd_mg < base.mrsd_mg

    def test_risk_category_low_for_default(self):
        result = _rat_basic()
        assert result.risk_category == "low"

    def test_risk_category_high_for_tox(self):
        result = _monkey_tox()
        assert result.risk_category == "high"

    def test_risk_category_in_valid_set(self):
        result = _rat_basic()
        assert result.risk_category in _VALID_RISK_CATEGORIES


# ---------------------------------------------------------------------------
# Dose escalation scheme
# ---------------------------------------------------------------------------


class TestDoseEscalation:
    def test_dose_escalation_has_four_steps(self):
        result = _rat_basic()
        assert len(result.dose_escalation_mg) == 4

    def test_dose_escalation_step1_is_mrsd(self):
        result = _rat_basic()
        assert result.dose_escalation_mg[0] == pytest.approx(result.mrsd_mg, rel=1e-5)

    def test_dose_escalation_step2_is_3x_mrsd(self):
        result = _rat_basic()
        assert result.dose_escalation_mg[1] == pytest.approx(result.mrsd_mg * 3, rel=1e-5)

    def test_dose_escalation_step4_is_20x_mrsd(self):
        result = _rat_basic()
        assert result.dose_escalation_mg[3] == pytest.approx(result.mrsd_mg * 20, rel=1e-5)

    def test_dose_escalation_increasing(self):
        result = _rat_basic()
        doses = result.dose_escalation_mg
        for i in range(1, len(doses)):
            assert doses[i] > doses[i - 1]


# ---------------------------------------------------------------------------
# mrsd_limiting_factor
# ---------------------------------------------------------------------------


class TestLimitingFactor:
    def test_hed_limiting_by_default(self):
        result = _rat_basic()
        assert result.mrsd_limiting_factor == "HED"

    def test_limiting_factor_in_valid_set(self):
        result = _rat_basic()
        assert result.mrsd_limiting_factor in _VALID_LIMITING_FACTORS


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidationErrors:
    def test_negative_noael_raises(self):
        with pytest.raises(ValueError, match="noael"):
            predict_fih_dose("X", -1.0, "rat")

    def test_zero_noael_raises(self):
        with pytest.raises(ValueError, match="noael"):
            predict_fih_dose("X", 0.0, "rat")

    def test_invalid_species_raises(self):
        with pytest.raises(ValueError, match="animal_species"):
            predict_fih_dose("X", 10.0, "hamster")

    def test_invalid_species_human_raises(self):
        # "human" is not a valid input animal species
        with pytest.raises(ValueError, match="animal_species"):
            predict_fih_dose("X", 10.0, "human")
