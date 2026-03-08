"""
Tests for Phase 509 — Pediatric Weight-Based Dosing
"""

import math

import pytest

from omega_pbpk.clinical.pediatric_weight_based_dosing import (
    PediatricDosingResult,
    calculate_pediatric_dose,
    dose_range_by_age,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

DRUG = "TestDrug"
ADULT_DOSE = 500.0  # mg
DOSE_MG_KG = 10.0  # mg/kg
DOSE_MG_M2 = 200.0  # mg/m²


def make_result(
    age_years=5.0,
    weight_kg=20.0,
    height_cm=110.0,
    serum_cr=0.5,
    sex="male",
):
    return calculate_pediatric_dose(
        drug_name=DRUG,
        adult_dose_mg=ADULT_DOSE,
        dose_mg_per_kg=DOSE_MG_KG,
        dose_mg_per_m2=DOSE_MG_M2,
        patient_weight_kg=weight_kg,
        patient_age_years=age_years,
        patient_height_cm=height_cm,
        serum_creatinine_mg_dL=serum_cr,
        sex=sex,
    )


# ---------------------------------------------------------------------------
# Return type and field presence
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_dataclass(self):
        result = make_result()
        assert isinstance(result, PediatricDosingResult)

    def test_field_drug_name(self):
        result = make_result()
        assert result.drug_name == DRUG

    def test_field_weight(self):
        result = make_result(weight_kg=20.0)
        assert result.patient_weight_kg == 20.0

    def test_field_age(self):
        result = make_result(age_years=5.0)
        assert result.patient_age_years == 5.0

    def test_field_height(self):
        result = make_result(height_cm=110.0)
        assert result.patient_height_cm == 110.0

    def test_field_notes_is_str(self):
        result = make_result()
        assert isinstance(result.notes, str)


# ---------------------------------------------------------------------------
# Weight-based dose calculation
# ---------------------------------------------------------------------------


class TestWeightBasedDose:
    def test_dose_weight_equals_mgkg_times_weight(self):
        result = make_result(weight_kg=20.0)
        assert result.dose_weight_based_mg == pytest.approx(DOSE_MG_KG * 20.0)

    def test_dose_weight_different_weight(self):
        result = make_result(weight_kg=30.0)
        assert result.dose_weight_based_mg == pytest.approx(DOSE_MG_KG * 30.0)


# ---------------------------------------------------------------------------
# BSA calculation
# ---------------------------------------------------------------------------


class TestBSA:
    def test_bsa_formula(self):
        result = make_result(weight_kg=20.0, height_cm=110.0)
        expected_bsa = math.sqrt(110.0 * 20.0 / 3600.0)
        assert result.bsa_m2 == pytest.approx(expected_bsa, rel=1e-6)

    def test_dose_bsa_based(self):
        result = make_result(weight_kg=20.0, height_cm=110.0)
        expected_bsa = math.sqrt(110.0 * 20.0 / 3600.0)
        assert result.dose_bsa_based_mg == pytest.approx(DOSE_MG_M2 * expected_bsa, rel=1e-6)


# ---------------------------------------------------------------------------
# Clark's rule
# ---------------------------------------------------------------------------


class TestClarksRule:
    def test_clark_dose_formula(self):
        result = make_result(weight_kg=20.0)
        expected = ADULT_DOSE * 20.0 / 70.0
        assert result.dose_clark_mg == pytest.approx(expected, rel=1e-6)

    def test_clark_dose_70kg_equals_adult(self):
        result = make_result(weight_kg=70.0)
        assert result.dose_clark_mg == pytest.approx(ADULT_DOSE, rel=1e-6)


# ---------------------------------------------------------------------------
# Young's rule
# ---------------------------------------------------------------------------


class TestYoungsRule:
    def test_young_dose_formula(self):
        result = make_result(age_years=5.0)
        expected = ADULT_DOSE * 5.0 / (5.0 + 12.0)
        assert result.dose_young_mg == pytest.approx(expected, rel=1e-6)

    def test_young_dose_increases_with_age(self):
        r5 = make_result(age_years=5.0)
        r10 = make_result(age_years=10.0)
        assert r10.dose_young_mg > r5.dose_young_mg


# ---------------------------------------------------------------------------
# Age group classification
# ---------------------------------------------------------------------------


class TestAgeGroupClassification:
    def test_neonate_less_than_1_month(self):
        result = make_result(age_years=0.05)  # ~0.6 months
        assert result.age_group == "neonate"

    def test_infant_1_to_12_months(self):
        result = make_result(age_years=0.5)  # 6 months
        assert result.age_group == "infant"

    def test_child_1_to_12_years(self):
        result = make_result(age_years=5.0)
        assert result.age_group == "child"

    def test_adolescent_12_plus(self):
        result = make_result(age_years=15.0)
        assert result.age_group == "adolescent"


# ---------------------------------------------------------------------------
# Dose capping
# ---------------------------------------------------------------------------


class TestDoseCapping:
    def test_neonate_capped_at_25_percent(self):
        # neonate: age = 0.05 years, weight = 3.5 kg → weight dose = 35 mg
        # adult dose = 500, cap = 0.25 * 500 = 125 mg → no cap (35 < 125)
        # Use high dose_mg_per_kg to force cap
        result = calculate_pediatric_dose(
            drug_name=DRUG,
            adult_dose_mg=100.0,
            dose_mg_per_kg=50.0,  # 50 * 3.5 = 175 mg
            dose_mg_per_m2=DOSE_MG_M2,
            patient_weight_kg=3.5,
            patient_age_years=0.05,
            patient_height_cm=50.0,
        )
        assert result.age_group == "neonate"
        assert result.recommended_dose_mg == pytest.approx(0.25 * 100.0, rel=1e-6)

    def test_adolescent_not_capped(self):
        # adolescent cap = 1.0 → recommended = weight-based if < adult dose
        result = calculate_pediatric_dose(
            drug_name=DRUG,
            adult_dose_mg=500.0,
            dose_mg_per_kg=5.0,  # 5 * 60 = 300 mg < 500 → no cap
            dose_mg_per_m2=DOSE_MG_M2,
            patient_weight_kg=60.0,
            patient_age_years=15.0,
            patient_height_cm=165.0,
        )
        assert result.age_group == "adolescent"
        assert result.recommended_dose_mg == pytest.approx(300.0, rel=1e-6)

    def test_child_cap_at_75_percent(self):
        result = calculate_pediatric_dose(
            drug_name=DRUG,
            adult_dose_mg=100.0,
            dose_mg_per_kg=10.0,  # 10 * 20 = 200 mg  → cap to 75 mg
            dose_mg_per_m2=DOSE_MG_M2,
            patient_weight_kg=20.0,
            patient_age_years=5.0,
            patient_height_cm=110.0,
        )
        assert result.age_group == "child"
        assert result.recommended_dose_mg == pytest.approx(0.75 * 100.0, rel=1e-6)

    def test_adult_dose_fraction_range(self):
        result = make_result()
        assert 0.0 <= result.adult_dose_fraction <= 1.0


# ---------------------------------------------------------------------------
# Renal dose adjustment — Schwartz CrCl
# ---------------------------------------------------------------------------


class TestRenalAdjustment:
    def test_high_creatinine_triggers_flag(self):
        result = make_result(age_years=5.0, height_cm=110.0, serum_cr=5.0)
        # CrCl = 0.55 * 110 / 5.0 = 12.1 < 50 → flag
        assert result.renal_dose_adjustment is True

    def test_normal_creatinine_no_flag(self):
        result = make_result(age_years=5.0, height_cm=110.0, serum_cr=0.4)
        # CrCl = 0.55 * 110 / 0.4 = 151.25 > 50 → no flag
        assert result.renal_dose_adjustment is False

    def test_schwartz_formula_child(self):
        result = make_result(age_years=5.0, height_cm=110.0, serum_cr=0.5)
        expected_crcl = 0.55 * 110.0 / 0.5
        assert result.crcl_ml_per_min == pytest.approx(expected_crcl, rel=1e-6)

    def test_schwartz_formula_infant(self):
        result = make_result(age_years=0.5, height_cm=65.0, serum_cr=0.4)
        expected_crcl = 0.45 * 65.0 / 0.4
        assert result.crcl_ml_per_min == pytest.approx(expected_crcl, rel=1e-6)

    def test_schwartz_formula_adolescent_male(self):
        result = make_result(age_years=15.0, height_cm=165.0, serum_cr=0.8, sex="male")
        expected_crcl = 0.70 * 165.0 / 0.8
        assert result.crcl_ml_per_min == pytest.approx(expected_crcl, rel=1e-6)

    def test_schwartz_formula_adolescent_female(self):
        result = make_result(age_years=15.0, height_cm=165.0, serum_cr=0.8, sex="female")
        expected_crcl = 0.55 * 165.0 / 0.8
        assert result.crcl_ml_per_min == pytest.approx(expected_crcl, rel=1e-6)


# ---------------------------------------------------------------------------
# dose_range_by_age
# ---------------------------------------------------------------------------


class TestDoseRangeByAge:
    def test_returns_list_of_4(self):
        results = dose_range_by_age(DRUG, ADULT_DOSE, DOSE_MG_KG, DOSE_MG_M2)
        assert len(results) == 4

    def test_all_are_results(self):
        results = dose_range_by_age(DRUG, ADULT_DOSE, DOSE_MG_KG, DOSE_MG_M2)
        for r in results:
            assert isinstance(r, PediatricDosingResult)

    def test_age_groups_in_order(self):
        results = dose_range_by_age(DRUG, ADULT_DOSE, DOSE_MG_KG, DOSE_MG_M2)
        assert results[0].age_group == "neonate"
        assert results[1].age_group == "infant"
        assert results[2].age_group == "child"
        assert results[3].age_group == "adolescent"

    def test_weights_match_representative_patients(self):
        results = dose_range_by_age(DRUG, ADULT_DOSE, DOSE_MG_KG, DOSE_MG_M2)
        expected_weights = [3.5, 7.0, 20.0, 60.0]
        for r, ew in zip(results, expected_weights):
            assert r.patient_weight_kg == pytest.approx(ew)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidationErrors:
    def test_zero_weight_raises(self):
        with pytest.raises(ValueError, match="weight"):
            make_result(weight_kg=0.0)

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError, match="weight"):
            make_result(weight_kg=-5.0)

    def test_negative_age_raises(self):
        with pytest.raises(ValueError, match="age"):
            make_result(age_years=-1.0)

    def test_zero_height_raises(self):
        with pytest.raises(ValueError, match="height"):
            make_result(height_cm=0.0)

    def test_invalid_sex_raises(self):
        with pytest.raises(ValueError, match="sex"):
            make_result(sex="other")
