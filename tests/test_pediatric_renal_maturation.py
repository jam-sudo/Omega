"""Tests for Phase 825 — pediatric_renal_maturation module."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.clinical.pediatric_renal_maturation import (
    PediatricRenalResult,
    dose_adjustment_renal_immature,
    predict_gfr_maturation,
)

# ---------------------------------------------------------------------------
# Helper defaults
# ---------------------------------------------------------------------------

TERM_NEONATE = {"age_weeks_postnatal": 1.0, "gestational_age_weeks": 40.0, "weight_kg": 3.5}
PRETERM = {"age_weeks_postnatal": 0.0, "gestational_age_weeks": 28.0, "weight_kg": 1.2}
INFANT_3MO = {"age_weeks_postnatal": 12.0, "gestational_age_weeks": 40.0, "weight_kg": 6.0}
TODDLER_1Y = {"age_weeks_postnatal": 52.0, "gestational_age_weeks": 40.0, "weight_kg": 10.0}
OLDER_CHILD = {"age_weeks_postnatal": 300.0, "gestational_age_weeks": 40.0, "weight_kg": 30.0}
ADULT_EQUIV = {"age_weeks_postnatal": 800.0, "gestational_age_weeks": 40.0, "weight_kg": 70.0}


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_pediatric_renal_result(self):
        result = predict_gfr_maturation(**TERM_NEONATE)
        assert isinstance(result, PediatricRenalResult)

    def test_result_is_frozen(self):
        result = predict_gfr_maturation(**TERM_NEONATE)
        with pytest.raises((AttributeError, TypeError)):
            result.gfr_mL_per_min = 999.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# GFR fraction bounds
# ---------------------------------------------------------------------------


class TestGFRFractionBounds:
    def test_gfr_fraction_positive(self):
        result = predict_gfr_maturation(**TERM_NEONATE)
        assert result.gfr_fraction_of_adult > 0.0

    def test_gfr_fraction_le_1(self):
        result = predict_gfr_maturation(**ADULT_EQUIV)
        assert result.gfr_fraction_of_adult <= 1.0

    def test_gfr_absolute_positive(self):
        result = predict_gfr_maturation(**TERM_NEONATE)
        assert result.gfr_mL_per_min > 0.0

    def test_gfr_normalized_positive(self):
        result = predict_gfr_maturation(**TERM_NEONATE)
        assert result.gfr_normalized_mL_per_min_per_1_73m2 > 0.0


# ---------------------------------------------------------------------------
# Maturation gradient: preterm < term < adult
# ---------------------------------------------------------------------------


class TestGFRMaturationGradient:
    def test_preterm_lower_gfr_fraction_than_term(self):
        preterm_result = predict_gfr_maturation(**PRETERM)
        term_result = predict_gfr_maturation(**TERM_NEONATE)
        assert preterm_result.gfr_fraction_of_adult < term_result.gfr_fraction_of_adult

    def test_infant_higher_gfr_fraction_than_neonate(self):
        neonate = predict_gfr_maturation(**TERM_NEONATE)
        infant = predict_gfr_maturation(**INFANT_3MO)
        assert infant.gfr_fraction_of_adult > neonate.gfr_fraction_of_adult

    def test_adult_gfr_fraction_close_to_1(self):
        result = predict_gfr_maturation(**ADULT_EQUIV)
        assert result.gfr_fraction_of_adult > 0.95

    def test_preterm_gfr_fraction_well_below_1(self):
        result = predict_gfr_maturation(**PRETERM)
        assert result.gfr_fraction_of_adult < 0.5


# ---------------------------------------------------------------------------
# Postmenstrual age field
# ---------------------------------------------------------------------------


class TestPMAField:
    def test_pma_equals_ga_plus_postnatal(self):
        result = predict_gfr_maturation(
            age_weeks_postnatal=4.0, gestational_age_weeks=36.0, weight_kg=2.5
        )
        assert math.isclose(result.age_weeks_pma, 40.0, rel_tol=1e-9)

    def test_postnatal_age_stored(self):
        result = predict_gfr_maturation(**TERM_NEONATE)
        assert math.isclose(result.age_weeks_postnatal, TERM_NEONATE["age_weeks_postnatal"])

    def test_gestational_age_stored(self):
        result = predict_gfr_maturation(**TERM_NEONATE)
        assert math.isclose(result.gestational_age_weeks, TERM_NEONATE["gestational_age_weeks"])


# ---------------------------------------------------------------------------
# Renal maturation stage
# ---------------------------------------------------------------------------


VALID_STAGES = {"neonatal", "infant", "toddler", "child", "mature"}


class TestRenalMaturationStage:
    def test_stage_is_valid_string(self):
        result = predict_gfr_maturation(**TERM_NEONATE)
        assert result.renal_maturation_stage in VALID_STAGES

    def test_preterm_is_neonatal(self):
        result = predict_gfr_maturation(**PRETERM)
        assert result.renal_maturation_stage == "neonatal"

    def test_adult_is_mature(self):
        result = predict_gfr_maturation(**ADULT_EQUIV)
        assert result.renal_maturation_stage == "mature"

    def test_toddler_stage(self):
        result = predict_gfr_maturation(**TODDLER_1Y)
        assert result.renal_maturation_stage in {"toddler", "infant"}


# ---------------------------------------------------------------------------
# Serum creatinine
# ---------------------------------------------------------------------------


class TestSerumCreatinine:
    def test_creatinine_positive(self):
        result = predict_gfr_maturation(**TERM_NEONATE)
        assert result.creatinine_mg_dL > 0.0

    def test_creatinine_physiological_range(self):
        # Neonatal creatinine typically 0.3-1.5 mg/dL
        result = predict_gfr_maturation(**TERM_NEONATE)
        assert 0.1 <= result.creatinine_mg_dL <= 10.0

    def test_creatinine_preterm_positive(self):
        result = predict_gfr_maturation(**PRETERM)
        assert result.creatinine_mg_dL > 0.0


# ---------------------------------------------------------------------------
# Renal CL adjustment factor
# ---------------------------------------------------------------------------


class TestRenalCLAdjustment:
    def test_adjustment_factor_equals_gfr_fraction(self):
        result = predict_gfr_maturation(**TERM_NEONATE)
        assert math.isclose(
            result.renal_cl_adjustment_factor,
            result.gfr_fraction_of_adult,
            rel_tol=1e-9,
        )

    def test_adjustment_factor_in_0_1(self):
        result = predict_gfr_maturation(**TERM_NEONATE)
        assert 0.0 < result.renal_cl_adjustment_factor <= 1.0


# ---------------------------------------------------------------------------
# dose_adjustment_renal_immature
# ---------------------------------------------------------------------------


class TestDoseAdjustmentRenalImmature:
    def test_returns_dict(self):
        result = dose_adjustment_renal_immature(
            drug_name="amikacin",
            adult_dose_mg=500.0,
            adult_gfr_mL_per_min=100.0,
            **TERM_NEONATE,
            fe_renal=0.9,
        )
        assert isinstance(result, dict)

    def test_dict_has_required_keys(self):
        result = dose_adjustment_renal_immature(
            drug_name="amikacin",
            adult_dose_mg=500.0,
            adult_gfr_mL_per_min=100.0,
            **TERM_NEONATE,
            fe_renal=0.9,
        )
        assert "pediatric_dose_mg" in result
        assert "dose_fraction" in result
        assert "gfr_fraction" in result

    def test_dose_fraction_in_range_for_renal_drug(self):
        result = dose_adjustment_renal_immature(
            drug_name="gentamicin",
            adult_dose_mg=240.0,
            adult_gfr_mL_per_min=100.0,
            **PRETERM,
            fe_renal=1.0,
        )
        assert 0.0 < result["dose_fraction"] <= 1.0

    def test_pediatric_dose_less_than_adult_for_immature_renal(self):
        result = dose_adjustment_renal_immature(
            drug_name="vancomycin",
            adult_dose_mg=1000.0,
            adult_gfr_mL_per_min=100.0,
            **PRETERM,
            fe_renal=0.95,
        )
        # Preterm neonate with low GFR + highly renal drug → dose should be < adult
        assert result["pediatric_dose_mg"] < 1000.0

    def test_fe_renal_zero_dose_fraction_is_1(self):
        result = dose_adjustment_renal_immature(
            drug_name="hepatic_drug",
            adult_dose_mg=100.0,
            adult_gfr_mL_per_min=100.0,
            **PRETERM,
            fe_renal=0.0,
        )
        assert math.isclose(result["dose_fraction"], 1.0, rel_tol=1e-9)

    def test_notes_nonempty(self):
        result = dose_adjustment_renal_immature(
            drug_name="test",
            adult_dose_mg=100.0,
            adult_gfr_mL_per_min=100.0,
            **TERM_NEONATE,
            fe_renal=0.5,
        )
        assert len(result["notes"]) > 0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidationErrors:
    def test_negative_postnatal_age_raises(self):
        with pytest.raises(ValueError):
            predict_gfr_maturation(
                age_weeks_postnatal=-1.0,
                gestational_age_weeks=40.0,
                weight_kg=3.5,
            )

    def test_ga_too_low_raises(self):
        with pytest.raises(ValueError):
            predict_gfr_maturation(
                age_weeks_postnatal=0.0,
                gestational_age_weeks=15.0,
                weight_kg=1.0,
            )

    def test_ga_too_high_raises(self):
        with pytest.raises(ValueError):
            predict_gfr_maturation(
                age_weeks_postnatal=0.0,
                gestational_age_weeks=50.0,
                weight_kg=3.5,
            )

    def test_zero_weight_raises(self):
        with pytest.raises(ValueError):
            predict_gfr_maturation(
                age_weeks_postnatal=0.0,
                gestational_age_weeks=40.0,
                weight_kg=0.0,
            )

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError):
            predict_gfr_maturation(
                age_weeks_postnatal=0.0,
                gestational_age_weeks=40.0,
                weight_kg=-1.0,
            )
