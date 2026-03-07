"""Tests for omega_pbpk.clinical.renal_dosing module."""

import math

import pytest

from omega_pbpk.clinical.renal_dosing import (
    PhaseRenalDosingResult,
    RenalDosingResult,
    _ckd_stage,
    _cockcroft_gault,
    adjust_dose_renal,
    adjust_renal_dose,
    cockroft_gault,
    renal_dose_from_patient,
)

# ---------------------------------------------------------------------------
# Helper constants
# ---------------------------------------------------------------------------
DRUG = "vancomycin"
DOSE = 1000.0  # mg
INTERVAL = 12.0  # h


# ---------------------------------------------------------------------------
# _ckd_stage tests
# ---------------------------------------------------------------------------


class TestCkdStage:
    def test_normal_at_90(self):
        assert _ckd_stage(90.0) == "normal"

    def test_normal_above_90(self):
        assert _ckd_stage(120.0) == "normal"

    def test_mild_at_60(self):
        assert _ckd_stage(60.0) == "mild"

    def test_mild_between_60_and_89(self):
        assert _ckd_stage(75.0) == "mild"

    def test_moderate_at_30(self):
        assert _ckd_stage(30.0) == "moderate"

    def test_moderate_between_30_and_59(self):
        assert _ckd_stage(45.0) == "moderate"

    def test_severe_at_15(self):
        assert _ckd_stage(15.0) == "severe"

    def test_severe_between_15_and_29(self):
        assert _ckd_stage(20.0) == "severe"

    def test_esrd_below_15(self):
        assert _ckd_stage(14.9) == "ESRD"

    def test_esrd_at_zero(self):
        assert _ckd_stage(0.0) == "ESRD"


# ---------------------------------------------------------------------------
# _cockcroft_gault tests
# ---------------------------------------------------------------------------


class TestCockcroftGault:
    def test_positive_male(self):
        crcl = _cockcroft_gault(age_years=50, weight_kg=70, serum_creatinine_mg_dL=1.0, sex="male")
        assert crcl > 0

    def test_positive_female(self):
        crcl = _cockcroft_gault(
            age_years=50, weight_kg=70, serum_creatinine_mg_dL=1.0, sex="female"
        )
        assert crcl > 0

    def test_female_lower_than_male(self):
        male = _cockcroft_gault(age_years=50, weight_kg=70, serum_creatinine_mg_dL=1.0, sex="male")
        female = _cockcroft_gault(
            age_years=50, weight_kg=70, serum_creatinine_mg_dL=1.0, sex="female"
        )
        assert female < male

    def test_higher_scr_lower_crcl(self):
        low_scr = _cockcroft_gault(
            age_years=50, weight_kg=70, serum_creatinine_mg_dL=0.8, sex="male"
        )
        high_scr = _cockcroft_gault(
            age_years=50, weight_kg=70, serum_creatinine_mg_dL=2.0, sex="male"
        )
        assert high_scr < low_scr

    def test_higher_age_lower_crcl(self):
        young = _cockcroft_gault(age_years=30, weight_kg=70, serum_creatinine_mg_dL=1.0, sex="male")
        old = _cockcroft_gault(age_years=80, weight_kg=70, serum_creatinine_mg_dL=1.0, sex="male")
        assert old < young

    def test_returns_float(self):
        result = _cockcroft_gault(
            age_years=45, weight_kg=80, serum_creatinine_mg_dL=1.1, sex="male"
        )
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# adjust_renal_dose — return type and fields
# ---------------------------------------------------------------------------


class TestAdjustRenalDoseReturnType:
    def test_returns_renal_dosing_result(self):
        result = adjust_renal_dose(DRUG, DOSE, INTERVAL, crcl_mL_per_min=80.0)
        assert isinstance(result, RenalDosingResult)

    def test_drug_name_preserved(self):
        result = adjust_renal_dose(DRUG, DOSE, INTERVAL, crcl_mL_per_min=80.0)
        assert result.drug_name == DRUG

    def test_crcl_stored(self):
        result = adjust_renal_dose(DRUG, DOSE, INTERVAL, crcl_mL_per_min=45.0)
        assert result.crcl_mL_per_min == pytest.approx(45.0)

    def test_ckd_stage_field_populated(self):
        result = adjust_renal_dose(DRUG, DOSE, INTERVAL, crcl_mL_per_min=80.0)
        assert result.ckd_stage in {"normal", "mild", "moderate", "severe", "ESRD"}

    def test_recommendation_is_string(self):
        result = adjust_renal_dose(DRUG, DOSE, INTERVAL, crcl_mL_per_min=80.0)
        assert isinstance(result.recommendation, str)

    def test_primary_method_is_string(self):
        result = adjust_renal_dose(DRUG, DOSE, INTERVAL, crcl_mL_per_min=80.0)
        assert isinstance(result.primary_method, str)


# ---------------------------------------------------------------------------
# CKD stage assignment via adjust_renal_dose
# ---------------------------------------------------------------------------


class TestCkdStageAssignment:
    def test_normal_crcl_90(self):
        result = adjust_renal_dose(DRUG, DOSE, INTERVAL, crcl_mL_per_min=90.0)
        assert result.ckd_stage == "normal"

    def test_normal_crcl_above_90(self):
        result = adjust_renal_dose(DRUG, DOSE, INTERVAL, crcl_mL_per_min=110.0)
        assert result.ckd_stage == "normal"

    def test_esrd_crcl_below_15(self):
        result = adjust_renal_dose(DRUG, DOSE, INTERVAL, crcl_mL_per_min=10.0)
        assert result.ckd_stage == "ESRD"

    def test_esrd_crcl_zero(self):
        result = adjust_renal_dose(DRUG, DOSE, INTERVAL, crcl_mL_per_min=0.0)
        assert result.ckd_stage == "ESRD"


# ---------------------------------------------------------------------------
# Dose adjustment factor
# ---------------------------------------------------------------------------


class TestDoseAdjustmentFactor:
    def test_factor_never_exceeds_1(self):
        for crcl in [0, 15, 45, 80, 100, 130]:
            result = adjust_renal_dose(DRUG, DOSE, INTERVAL, crcl_mL_per_min=float(crcl))
            assert result.dose_adjustment_factor <= 1.0

    def test_factor_at_least_0_1(self):
        result = adjust_renal_dose(DRUG, DOSE, INTERVAL, crcl_mL_per_min=0.0, fraction_renal=1.0)
        assert result.dose_adjustment_factor >= 0.1

    def test_factor_equals_1_when_crcl_100(self):
        result = adjust_renal_dose(DRUG, DOSE, INTERVAL, crcl_mL_per_min=100.0)
        assert result.dose_adjustment_factor == pytest.approx(1.0)

    def test_factor_equals_1_when_fraction_renal_0(self):
        result = adjust_renal_dose(DRUG, DOSE, INTERVAL, crcl_mL_per_min=30.0, fraction_renal=0.0)
        assert result.dose_adjustment_factor == pytest.approx(1.0)

    def test_high_fraction_renal_low_crcl_gives_lower_factor(self):
        low_fr = adjust_renal_dose(DRUG, DOSE, INTERVAL, crcl_mL_per_min=20.0, fraction_renal=0.1)
        high_fr = adjust_renal_dose(DRUG, DOSE, INTERVAL, crcl_mL_per_min=20.0, fraction_renal=0.9)
        assert high_fr.dose_adjustment_factor < low_fr.dose_adjustment_factor

    def test_factor_formula(self):
        crcl = 50.0
        fr = 0.6
        expected = 1 - fr * (1 - crcl / 100)
        result = adjust_renal_dose(DRUG, DOSE, INTERVAL, crcl_mL_per_min=crcl, fraction_renal=fr)
        assert result.dose_adjustment_factor == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Adjusted dose
# ---------------------------------------------------------------------------


class TestAdjustedDose:
    def test_adjusted_dose_equals_dose_times_factor(self):
        result = adjust_renal_dose(DRUG, DOSE, INTERVAL, crcl_mL_per_min=50.0, fraction_renal=0.5)
        assert result.adjusted_dose_mg == pytest.approx(DOSE * result.dose_adjustment_factor)

    def test_adjusted_dose_full_function_at_normal_crcl(self):
        result = adjust_renal_dose(DRUG, DOSE, INTERVAL, crcl_mL_per_min=100.0)
        assert result.adjusted_dose_mg == pytest.approx(DOSE)


# ---------------------------------------------------------------------------
# Adjusted interval
# ---------------------------------------------------------------------------


class TestAdjustedInterval:
    def test_adjusted_interval_capped_at_48h(self):
        # Very low factor -> interval would blow up; must cap at 48h
        result = adjust_renal_dose(DRUG, DOSE, INTERVAL, crcl_mL_per_min=0.0, fraction_renal=1.0)
        assert result.adjusted_interval_h <= 48.0

    def test_adjusted_interval_not_capped_for_normal_renal(self):
        result = adjust_renal_dose(DRUG, DOSE, INTERVAL, crcl_mL_per_min=100.0)
        assert result.adjusted_interval_h == pytest.approx(INTERVAL)


# ---------------------------------------------------------------------------
# Dialysis supplementation flag
# ---------------------------------------------------------------------------


class TestDialysisSupplementation:
    def test_requires_dialysis_esrd_high_renal_fraction(self):
        result = adjust_renal_dose(DRUG, DOSE, INTERVAL, crcl_mL_per_min=5.0, fraction_renal=0.7)
        assert result.requires_dialysis_supplementation is True

    def test_no_dialysis_esrd_low_renal_fraction(self):
        result = adjust_renal_dose(DRUG, DOSE, INTERVAL, crcl_mL_per_min=5.0, fraction_renal=0.3)
        assert result.requires_dialysis_supplementation is False

    def test_no_dialysis_normal_crcl_high_renal_fraction(self):
        result = adjust_renal_dose(DRUG, DOSE, INTERVAL, crcl_mL_per_min=95.0, fraction_renal=0.9)
        assert result.requires_dialysis_supplementation is False


# ---------------------------------------------------------------------------
# ValueError conditions
# ---------------------------------------------------------------------------


class TestValueErrors:
    def test_raises_on_zero_dose(self):
        with pytest.raises(ValueError):
            adjust_renal_dose(DRUG, 0.0, INTERVAL, crcl_mL_per_min=80.0)

    def test_raises_on_negative_dose(self):
        with pytest.raises(ValueError):
            adjust_renal_dose(DRUG, -500.0, INTERVAL, crcl_mL_per_min=80.0)

    def test_raises_on_negative_crcl(self):
        with pytest.raises(ValueError):
            adjust_renal_dose(DRUG, DOSE, INTERVAL, crcl_mL_per_min=-1.0)


# ---------------------------------------------------------------------------
# renal_dose_from_patient
# ---------------------------------------------------------------------------


class TestRenalDoseFromPatient:
    def test_returns_renal_dosing_result(self):
        result = renal_dose_from_patient(
            drug_name=DRUG,
            dose_mg=DOSE,
            dosing_interval_h=INTERVAL,
            age_years=55,
            weight_kg=75.0,
            serum_creatinine_mg_dL=1.2,
            sex="male",
        )
        assert isinstance(result, RenalDosingResult)

    def test_female_patient_returns_result(self):
        result = renal_dose_from_patient(
            drug_name=DRUG,
            dose_mg=DOSE,
            dosing_interval_h=INTERVAL,
            age_years=60,
            weight_kg=65.0,
            serum_creatinine_mg_dL=0.9,
            sex="female",
        )
        assert isinstance(result, RenalDosingResult)

    def test_patient_crcl_stored_in_result(self):
        result = renal_dose_from_patient(
            drug_name=DRUG,
            dose_mg=DOSE,
            dosing_interval_h=INTERVAL,
            age_years=70,
            weight_kg=60.0,
            serum_creatinine_mg_dL=2.5,
        )
        assert result.crcl_mL_per_min > 0

    def test_elderly_high_creatinine_likely_moderate_or_worse(self):
        result = renal_dose_from_patient(
            drug_name=DRUG,
            dose_mg=DOSE,
            dosing_interval_h=INTERVAL,
            age_years=85,
            weight_kg=55.0,
            serum_creatinine_mg_dL=3.0,
        )
        assert result.ckd_stage in {"moderate", "severe", "ESRD"}

    def test_young_healthy_normal_stage(self):
        result = renal_dose_from_patient(
            drug_name=DRUG,
            dose_mg=DOSE,
            dosing_interval_h=INTERVAL,
            age_years=25,
            weight_kg=80.0,
            serum_creatinine_mg_dL=0.8,
            sex="male",
        )
        assert result.ckd_stage == "normal"


# ===========================================================================
# Phase 407 Tests — adjust_dose_renal & cockroft_gault
# ===========================================================================


class TestCockroftGault407:
    """Tests for the public cockroft_gault() function (Phase 407)."""

    def test_male_returns_positive(self):
        crcl = cockroft_gault(scr_mg_dL=1.0, age=50, weight_kg=70, sex="male")
        assert crcl > 0

    def test_female_returns_positive(self):
        crcl = cockroft_gault(scr_mg_dL=1.0, age=50, weight_kg=70, sex="female")
        assert crcl > 0

    def test_female_lower_than_male(self):
        m = cockroft_gault(scr_mg_dL=1.0, age=50, weight_kg=70, sex="male")
        f = cockroft_gault(scr_mg_dL=1.0, age=50, weight_kg=70, sex="female")
        assert f == pytest.approx(m * 0.85)

    def test_high_scr_low_crcl(self):
        low = cockroft_gault(scr_mg_dL=0.8, age=40, weight_kg=70, sex="male")
        high = cockroft_gault(scr_mg_dL=3.0, age=40, weight_kg=70, sex="male")
        assert high < low

    def test_formula_correctness_male(self):
        expected = (140 - 50) * 70 / (72 * 1.0)
        result = cockroft_gault(scr_mg_dL=1.0, age=50, weight_kg=70, sex="male")
        assert result == pytest.approx(expected)

    def test_formula_correctness_female(self):
        expected = (140 - 50) * 70 / (72 * 1.0) * 0.85
        result = cockroft_gault(scr_mg_dL=1.0, age=50, weight_kg=70, sex="female")
        assert result == pytest.approx(expected)

    def test_raises_zero_scr(self):
        with pytest.raises(ValueError, match="scr_mg_dL"):
            cockroft_gault(scr_mg_dL=0.0, age=50, weight_kg=70, sex="male")

    def test_raises_negative_scr(self):
        with pytest.raises(ValueError):
            cockroft_gault(scr_mg_dL=-1.0, age=50, weight_kg=70, sex="male")

    def test_raises_zero_age(self):
        with pytest.raises(ValueError):
            cockroft_gault(scr_mg_dL=1.0, age=0, weight_kg=70, sex="male")

    def test_raises_zero_weight(self):
        with pytest.raises(ValueError):
            cockroft_gault(scr_mg_dL=1.0, age=50, weight_kg=0, sex="male")

    def test_raises_invalid_sex(self):
        with pytest.raises(ValueError):
            cockroft_gault(scr_mg_dL=1.0, age=50, weight_kg=70, sex="other")


class TestAdjustDoseRenal407:
    """Tests for adjust_dose_renal() Phase 407."""

    def test_returns_phase_renal_dosing_result(self):
        r = adjust_dose_renal("vancomycin", 1000.0, 12.0, 60.0, 0.7)
        assert isinstance(r, PhaseRenalDosingResult)

    def test_drug_name_preserved(self):
        r = adjust_dose_renal("gentamicin", 80.0, 8.0, 45.0, 0.9)
        assert r.drug_name == "gentamicin"

    def test_gfr_category_normal(self):
        r = adjust_dose_renal("drug", 100.0, 24.0, 95.0, 0.5)
        assert r.gfr_category == "normal"

    def test_gfr_category_mild(self):
        r = adjust_dose_renal("drug", 100.0, 24.0, 75.0, 0.5)
        assert r.gfr_category == "mild"

    def test_gfr_category_moderate(self):
        r = adjust_dose_renal("drug", 100.0, 24.0, 45.0, 0.5)
        assert r.gfr_category == "moderate"

    def test_gfr_category_severe(self):
        r = adjust_dose_renal("drug", 100.0, 24.0, 20.0, 0.5)
        assert r.gfr_category == "severe"

    def test_gfr_category_esrd(self):
        r = adjust_dose_renal("drug", 100.0, 24.0, 5.0, 0.5)
        assert r.gfr_category == "ESRD"

    def test_dose_factor_at_normal_gfr(self):
        r = adjust_dose_renal("drug", 100.0, 24.0, 90.0, 0.8)
        assert r.dose_factor == pytest.approx(1.0)

    def test_dose_factor_clamped_min(self):
        r = adjust_dose_renal("drug", 100.0, 24.0, 0.0, 1.0)
        assert r.dose_factor >= 0.1

    def test_dose_factor_formula(self):
        gfr = 45.0
        fre = 0.6
        expected = max(0.1, min(1.0, 1 - fre * (1 - gfr / 90.0)))
        r = adjust_dose_renal("drug", 100.0, 24.0, gfr, fre)
        assert r.dose_factor == pytest.approx(expected)

    def test_method_dose_reduction_keeps_interval(self):
        r = adjust_dose_renal("drug", 500.0, 12.0, 40.0, 0.7, method="dose_reduction")
        assert r.adjusted_interval_h == pytest.approx(12.0)
        assert r.adjusted_dose_mg < 500.0

    def test_method_interval_extension_keeps_dose(self):
        r = adjust_dose_renal("drug", 500.0, 12.0, 40.0, 0.7, method="interval_extension")
        assert r.adjusted_dose_mg == pytest.approx(500.0)
        assert r.adjusted_interval_h > 12.0

    def test_method_both_splits_between_dose_and_interval(self):
        r = adjust_dose_renal("drug", 500.0, 12.0, 40.0, 0.7, method="both")
        assert r.adjusted_dose_mg < 500.0
        assert r.adjusted_interval_h > 12.0

    def test_method_both_uses_sqrt(self):
        gfr = 30.0
        fre = 0.8
        dose_factor = max(0.1, min(1.0, 1 - fre * (1 - gfr / 90.0)))
        sqrt_f = math.sqrt(dose_factor)
        r = adjust_dose_renal("drug", 200.0, 8.0, gfr, fre, method="both")
        assert r.adjusted_dose_mg == pytest.approx(200.0 * sqrt_f, rel=1e-5)
        assert r.adjusted_interval_h == pytest.approx(8.0 / sqrt_f, rel=1e-5)

    def test_t_half_formula(self):
        """Verify t_half = 0.693 / (ke_normal / dose_factor) as per spec."""
        import math

        r = adjust_dose_renal("drug", 100.0, 24.0, 45.0, 0.7)
        ke_normal = 0.1  # module default
        expected = math.log(2) / (ke_normal / r.dose_factor)
        assert r.t_half_adjusted_h == pytest.approx(expected, rel=1e-5)

    def test_notes_is_list(self):
        r = adjust_dose_renal("drug", 100.0, 24.0, 30.0, 0.5)
        assert isinstance(r.notes, list)

    def test_esrd_note_present(self):
        r = adjust_dose_renal("drug", 100.0, 24.0, 5.0, 0.5)
        notes_text = " ".join(r.notes).lower()
        assert "esrd" in notes_text or "dialysis" in notes_text

    def test_high_renal_fraction_note(self):
        r = adjust_dose_renal("drug", 100.0, 24.0, 30.0, 0.9)
        notes_text = " ".join(r.notes)
        assert "80%" in notes_text or "renal excretion" in notes_text.lower()

    def test_raises_empty_drug_name(self):
        with pytest.raises(ValueError):
            adjust_dose_renal("", 100.0, 24.0, 60.0, 0.5)

    def test_raises_nonpositive_dose(self):
        with pytest.raises(ValueError):
            adjust_dose_renal("drug", 0.0, 24.0, 60.0, 0.5)

    def test_raises_nonpositive_interval(self):
        with pytest.raises(ValueError):
            adjust_dose_renal("drug", 100.0, 0.0, 60.0, 0.5)

    def test_raises_negative_gfr(self):
        with pytest.raises(ValueError):
            adjust_dose_renal("drug", 100.0, 24.0, -1.0, 0.5)

    def test_raises_f_renal_out_of_range_high(self):
        with pytest.raises(ValueError):
            adjust_dose_renal("drug", 100.0, 24.0, 60.0, 1.5)

    def test_raises_f_renal_out_of_range_low(self):
        with pytest.raises(ValueError):
            adjust_dose_renal("drug", 100.0, 24.0, 60.0, -0.1)

    def test_raises_invalid_method(self):
        with pytest.raises(ValueError, match="method"):
            adjust_dose_renal("drug", 100.0, 24.0, 60.0, 0.5, method="unknown")

    def test_zero_f_renal_no_adjustment(self):
        r = adjust_dose_renal("drug", 100.0, 24.0, 5.0, 0.0, method="dose_reduction")
        assert r.adjusted_dose_mg == pytest.approx(100.0)

    def test_gfr_at_90_no_adjustment(self):
        r = adjust_dose_renal("drug", 200.0, 12.0, 90.0, 0.9, method="dose_reduction")
        assert r.adjusted_dose_mg == pytest.approx(200.0)
