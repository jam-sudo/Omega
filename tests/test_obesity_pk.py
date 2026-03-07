"""Tests for clinical/obesity_pk.py — Phase 419."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.clinical.obesity_pk import (
    ObesityPKResult,
    compute_lean_body_weight,
    obesity_pk_scaling,
)

# ---------------------------------------------------------------------------
# compute_lean_body_weight tests
# ---------------------------------------------------------------------------


class TestComputeLeanBodyWeight:
    def test_male_standard_height(self):
        # 180 cm male: height_in = 70.87 in, inches_over_60 = 10.87
        # LBW = 50 + 2.3 * 10.87 = 50 + 25.0 = 75.0 kg
        lbw = compute_lean_body_weight(180.0, 80.0, "male")
        assert lbw == pytest.approx(50 + 2.3 * (180.0 / 2.54 - 60), rel=1e-4)

    def test_female_standard_height(self):
        lbw = compute_lean_body_weight(165.0, 65.0, "female")
        expected = 45.5 + 2.3 * (165.0 / 2.54 - 60.0)
        assert lbw == pytest.approx(expected, rel=1e-4)

    def test_male_70kg_reference(self):
        # 178 cm ~ 70 in male
        height_for_70_lbw = (70.0 - 50.0) / 2.3 + 60.0  # inches
        height_cm = height_for_70_lbw * 2.54
        lbw = compute_lean_body_weight(height_cm, 80.0, "male")
        assert lbw == pytest.approx(70.0, rel=0.01)

    def test_minimum_clamp(self):
        # Very short person → clamp to 30 kg
        lbw = compute_lean_body_weight(100.0, 40.0, "male")
        assert lbw == 30.0

    def test_female_minimum_clamp(self):
        lbw = compute_lean_body_weight(90.0, 30.0, "female")
        assert lbw == 30.0

    def test_male_female_differ(self):
        lbw_m = compute_lean_body_weight(170.0, 70.0, "male")
        lbw_f = compute_lean_body_weight(170.0, 70.0, "female")
        assert lbw_m > lbw_f

    def test_case_insensitive_sex(self):
        lbw1 = compute_lean_body_weight(170.0, 70.0, "Male")
        lbw2 = compute_lean_body_weight(170.0, 70.0, "male")
        assert lbw1 == pytest.approx(lbw2)

    def test_invalid_sex(self):
        with pytest.raises(ValueError, match="sex"):
            compute_lean_body_weight(170.0, 70.0, "unknown")

    def test_invalid_height(self):
        with pytest.raises(ValueError, match="height_cm"):
            compute_lean_body_weight(0.0, 70.0, "male")

    def test_invalid_weight(self):
        with pytest.raises(ValueError, match="weight_kg"):
            compute_lean_body_weight(170.0, -5.0, "male")

    def test_negative_height(self):
        with pytest.raises(ValueError):
            compute_lean_body_weight(-10.0, 70.0, "female")

    def test_lbw_increases_with_height(self):
        lbw1 = compute_lean_body_weight(160.0, 70.0, "male")
        lbw2 = compute_lean_body_weight(180.0, 70.0, "male")
        assert lbw2 > lbw1


# ---------------------------------------------------------------------------
# obesity_pk_scaling tests
# ---------------------------------------------------------------------------


def _default_result(
    bmi: float = 32.0,
    weight_kg: float = 100.0,
    sex: str = "male",
    distribution_type: str = "hydrophilic",
) -> ObesityPKResult:
    """Helper to produce a result with sensible defaults."""
    return obesity_pk_scaling(
        drug_name="TestDrug",
        normal_cl_L_per_h=5.0,
        normal_vd_L=50.0,
        bmi=bmi,
        height_cm=175.0,
        weight_kg=weight_kg,
        sex=sex,
        distribution_type=distribution_type,
    )


class TestObesityPKScaling:
    def test_returns_dataclass(self):
        result = _default_result()
        assert isinstance(result, ObesityPKResult)

    def test_drug_name_preserved(self):
        result = _default_result()
        assert result.drug_name == "TestDrug"

    def test_bmi_stored(self):
        result = _default_result(bmi=35.0)
        assert result.bmi == 35.0

    def test_tbw_equals_weight_kg(self):
        result = _default_result(weight_kg=110.0)
        assert result.tbw_kg == 110.0

    # BMI categories
    def test_bmi_normal(self):
        result = _default_result(bmi=22.0)
        assert result.bmi_category == "normal"

    def test_bmi_overweight(self):
        result = _default_result(bmi=27.0)
        assert result.bmi_category == "overweight"

    def test_bmi_obese(self):
        result = _default_result(bmi=35.0)
        assert result.bmi_category == "obese"

    def test_bmi_morbidly_obese(self):
        result = _default_result(bmi=45.0)
        assert result.bmi_category == "morbidly_obese"

    # CL scaling
    def test_cl_scales_by_lbw(self):
        result = _default_result()
        expected_cl = 5.0 * (result.lbw_kg / 70.0)
        assert result.cl_adjusted_L_per_h == pytest.approx(expected_cl, rel=1e-6)

    def test_cl_independent_of_distribution_type(self):
        r_h = _default_result(distribution_type="hydrophilic")
        r_l = _default_result(distribution_type="lipophilic")
        r_m = _default_result(distribution_type="mixed")
        assert r_h.cl_adjusted_L_per_h == pytest.approx(r_l.cl_adjusted_L_per_h)
        assert r_h.cl_adjusted_L_per_h == pytest.approx(r_m.cl_adjusted_L_per_h)

    # Vd scaling per distribution type
    def test_hydrophilic_vd_uses_lbw(self):
        result = obesity_pk_scaling(
            "Drug", 5.0, 50.0, 35.0, 175.0, 120.0, "male", "hydrophilic"
        )
        expected_vd = 50.0 * (result.lbw_kg / 70.0)
        assert result.vd_adjusted_L == pytest.approx(expected_vd, rel=1e-6)

    def test_lipophilic_vd_uses_tbw(self):
        result = obesity_pk_scaling(
            "Drug", 5.0, 50.0, 35.0, 175.0, 120.0, "male", "lipophilic"
        )
        expected_vd = 50.0 * (120.0 / 70.0)
        assert result.vd_adjusted_L == pytest.approx(expected_vd, rel=1e-6)

    def test_mixed_vd_blends_lbw_tbw(self):
        result = obesity_pk_scaling(
            "Drug", 5.0, 50.0, 35.0, 175.0, 120.0, "male", "mixed"
        )
        lbw = result.lbw_kg
        tbw = 120.0
        excess_fat = tbw - lbw
        eff_weight = lbw + 0.3 * excess_fat
        expected_vd = 50.0 * (eff_weight / 70.0)
        assert result.vd_adjusted_L == pytest.approx(expected_vd, rel=1e-6)

    def test_lipophilic_vd_gt_hydrophilic_for_obese(self):
        r_h = _default_result(bmi=40.0, weight_kg=130.0, distribution_type="hydrophilic")
        r_l = _default_result(bmi=40.0, weight_kg=130.0, distribution_type="lipophilic")
        assert r_l.vd_adjusted_L > r_h.vd_adjusted_L

    def test_mixed_vd_between_hydrophilic_and_lipophilic(self):
        r_h = _default_result(bmi=40.0, weight_kg=130.0, distribution_type="hydrophilic")
        r_l = _default_result(bmi=40.0, weight_kg=130.0, distribution_type="lipophilic")
        r_m = _default_result(bmi=40.0, weight_kg=130.0, distribution_type="mixed")
        assert r_h.vd_adjusted_L <= r_m.vd_adjusted_L <= r_l.vd_adjusted_L

    # Half-life
    def test_t_half_positive(self):
        result = _default_result()
        assert result.t_half_adjusted_h > 0.0

    def test_t_half_formula(self):
        result = _default_result()
        expected = math.log(2.0) * result.vd_adjusted_L / result.cl_adjusted_L_per_h
        assert result.t_half_adjusted_h == pytest.approx(expected, rel=1e-6)

    # Dose recommendation
    def test_dose_recommendation_present(self):
        result = _default_result()
        assert isinstance(result.dose_recommendation, str)
        assert len(result.dose_recommendation) > 0

    def test_normal_bmi_recommendation(self):
        result = _default_result(bmi=22.0)
        assert "standard dose" in result.dose_recommendation.lower()

    def test_morbidly_obese_recommendation(self):
        result = _default_result(bmi=45.0, weight_kg=150.0)
        assert "morbid obesity" in result.dose_recommendation.lower()

    # Notes
    def test_notes_contains_bmi(self):
        result = _default_result(bmi=32.0)
        assert "32" in result.notes

    # Validation
    def test_invalid_drug_name(self):
        with pytest.raises(ValueError, match="drug_name"):
            obesity_pk_scaling("", 5.0, 50.0, 32.0, 175.0, 100.0, "male", "hydrophilic")

    def test_invalid_cl(self):
        with pytest.raises(ValueError, match="normal_cl_L_per_h"):
            obesity_pk_scaling(
                "Drug", -1.0, 50.0, 32.0, 175.0, 100.0, "male", "hydrophilic"
            )

    def test_invalid_vd(self):
        with pytest.raises(ValueError, match="normal_vd_L"):
            obesity_pk_scaling(
                "Drug", 5.0, 0.0, 32.0, 175.0, 100.0, "male", "hydrophilic"
            )

    def test_invalid_bmi(self):
        with pytest.raises(ValueError, match="bmi"):
            obesity_pk_scaling(
                "Drug", 5.0, 50.0, -5.0, 175.0, 100.0, "male", "hydrophilic"
            )

    def test_invalid_distribution_type(self):
        with pytest.raises(ValueError, match="distribution_type"):
            obesity_pk_scaling(
                "Drug", 5.0, 50.0, 32.0, 175.0, 100.0, "male", "amphiphilic"
            )

    def test_distribution_type_case_insensitive(self):
        result = obesity_pk_scaling(
            "Drug", 5.0, 50.0, 32.0, 175.0, 100.0, "male", "Hydrophilic"
        )
        assert result.distribution_type == "hydrophilic"

    def test_female_patient(self):
        result = obesity_pk_scaling(
            "Drug", 3.0, 30.0, 30.0, 160.0, 90.0, "female", "lipophilic"
        )
        assert result.cl_adjusted_L_per_h > 0
        assert result.vd_adjusted_L > 0
