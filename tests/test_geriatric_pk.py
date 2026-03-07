"""Tests for src/omega_pbpk/clinical/geriatric_pk.py (Phase 445)."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.geriatric_pk import (
    GeriatricPKResult,
    _calc_co_factor,
    _calc_cyp_factor,
    _calc_fat_fraction,
    _calc_gfr,
    _calc_muscle_fraction,
    geriatric_pk_scaling,
)

# ---------------------------------------------------------------------------
# Helper defaults
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    drug_name="TestDrug",
    age_years=70,
    weight_kg=70,
    sex="male",
    scr_mg_dL=1.0,
    cl_normal_L_per_h=5.0,
    vd_normal_L=50.0,
    distribution_type="neutral",
)


def run(**overrides):
    kwargs = {**DEFAULTS, **overrides}
    return geriatric_pk_scaling(**kwargs)


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestCalcGFR:
    def test_age_40_is_120(self):
        assert _calc_gfr(40) == pytest.approx(120.0)

    def test_gfr_declines_with_age(self):
        gfr_70 = _calc_gfr(70)
        gfr_40 = _calc_gfr(40)
        assert gfr_70 < gfr_40

    def test_gfr_clamped_at_15(self):
        # At very old age GFR should be clamped at 15
        assert _calc_gfr(200) == 15.0

    def test_gfr_below_40_stays_at_120(self):
        # Age 30 should give 120 (no decline below 40)
        assert _calc_gfr(30) == pytest.approx(120.0)

    def test_gfr_age_70(self):
        expected = 120.0 * (1.0 - 30 * 0.008)
        assert _calc_gfr(70) == pytest.approx(expected)


class TestCalcCYPFactor:
    def test_no_decline_before_60(self):
        assert _calc_cyp_factor(60) == pytest.approx(1.0)

    def test_declines_after_60(self):
        assert _calc_cyp_factor(70) < _calc_cyp_factor(60)

    def test_clamped_at_04(self):
        assert _calc_cyp_factor(200) == pytest.approx(0.4)

    def test_age_80(self):
        expected = max(0.4, 1.0 - 20 * 0.01)
        assert _calc_cyp_factor(80) == pytest.approx(expected)


class TestCalcCOFactor:
    def test_no_decline_before_60(self):
        assert _calc_co_factor(60) == pytest.approx(1.0)

    def test_declines_after_60(self):
        assert _calc_co_factor(75) < _calc_co_factor(60)

    def test_clamped_at_05(self):
        assert _calc_co_factor(200) == pytest.approx(0.5)


class TestBodyComposition:
    def test_fat_fraction_increases_with_age(self):
        assert _calc_fat_fraction(70) > _calc_fat_fraction(40)

    def test_fat_fraction_clamped_at_045(self):
        assert _calc_fat_fraction(200) == pytest.approx(0.45)

    def test_fat_fraction_base_at_40(self):
        assert _calc_fat_fraction(40) == pytest.approx(0.15)

    def test_muscle_fraction_decreases_with_age(self):
        assert _calc_muscle_fraction(70) < _calc_muscle_fraction(40)

    def test_muscle_fraction_clamped_at_020(self):
        assert _calc_muscle_fraction(200) == pytest.approx(0.20)

    def test_muscle_fraction_base_at_40(self):
        assert _calc_muscle_fraction(40) == pytest.approx(0.40)


# ---------------------------------------------------------------------------
# Main function tests
# ---------------------------------------------------------------------------

class TestGeriatricPKScaling:
    def test_returns_correct_type(self):
        result = run()
        assert isinstance(result, GeriatricPKResult)

    def test_basic_fields_populated(self):
        result = run()
        assert result.drug_name == "TestDrug"
        assert result.age_years == 70
        assert result.sex == "male"
        assert result.weight_kg == 70

    def test_cl_geriatric_less_than_normal_at_70(self):
        result = run(age_years=70)
        assert result.cl_geriatric_L_per_h < DEFAULTS["cl_normal_L_per_h"]

    def test_t_half_positive(self):
        result = run()
        assert result.t_half_h > 0

    def test_t_half_formula(self):
        result = run()
        expected = 0.693 * result.vd_geriatric_L / result.cl_geriatric_L_per_h
        assert result.t_half_h == pytest.approx(expected, rel=1e-5)

    def test_dose_adjustment_factor_less_than_1_at_70(self):
        result = run(age_years=70)
        assert result.dose_adjustment_factor < 1.0

    def test_dose_adjustment_factor_closer_to_1_at_young(self):
        result_young = run(age_years=40)
        result_old = run(age_years=80)
        assert result_young.dose_adjustment_factor > result_old.dose_adjustment_factor

    def test_hydrophilic_vd_reduced_at_old_age(self):
        result = run(distribution_type="hydrophilic")
        # muscle fraction < 0.40 → Vd reduced
        assert result.vd_geriatric_L < DEFAULTS["vd_normal_L"]

    def test_lipophilic_vd_increased_at_old_age(self):
        result = run(distribution_type="lipophilic", age_years=70)
        # fat fraction > 0.15 → Vd increased
        assert result.vd_geriatric_L > DEFAULTS["vd_normal_L"]

    def test_neutral_vd_reduced_via_albumin(self):
        result = run(distribution_type="neutral", age_years=80)
        assert result.vd_geriatric_L < DEFAULTS["vd_normal_L"]

    def test_gfr_in_result_matches_formula(self):
        result = run(age_years=70)
        expected_gfr = 120.0 * (1 - 30 * 0.008)
        assert result.gfr_estimated_mL_per_min == pytest.approx(expected_gfr)

    def test_cyp_factor_in_result(self):
        result = run(age_years=75)
        expected = max(0.4, 1.0 - 15 * 0.01)
        assert result.cyp_factor == pytest.approx(expected)

    def test_co_factor_in_result(self):
        result = run(age_years=70)
        expected = max(0.5, 1.0 - 10 * 0.01)
        assert result.co_factor == pytest.approx(expected)

    def test_notes_is_list(self):
        result = run()
        assert isinstance(result.notes, list)

    def test_dose_recommendation_is_str(self):
        result = run()
        assert isinstance(result.dose_recommendation, str)
        assert len(result.dose_recommendation) > 0

    def test_very_old_patient_notes(self):
        result = run(age_years=85)
        joined = " ".join(result.notes)
        assert "80" in joined or "very" in joined.lower() or "elderly" in joined.lower()

    def test_female_sex_accepted(self):
        result = run(sex="female")
        assert result.sex == "female"

    def test_sex_case_insensitive(self):
        result = run(sex="Male")
        assert result.sex == "male"

    def test_cl_renal_and_hep_sum_correctly(self):
        age = 70
        cl_n = 5.0
        gfr = _calc_gfr(age)
        cyp = _calc_cyp_factor(age)
        co = _calc_co_factor(age)
        cl_renal = cl_n * 0.5 * (gfr / 120.0)
        cl_hep = cl_n * 0.5 * cyp * co
        expected_cl = cl_renal + cl_hep
        result = run(age_years=age, cl_normal_L_per_h=cl_n)
        assert result.cl_geriatric_L_per_h == pytest.approx(expected_cl)


# ---------------------------------------------------------------------------
# Validation error tests
# ---------------------------------------------------------------------------

class TestGeriatricPKValidation:
    def test_empty_drug_name(self):
        with pytest.raises(ValueError, match="drug_name"):
            run(drug_name="")

    def test_blank_drug_name(self):
        with pytest.raises(ValueError, match="drug_name"):
            run(drug_name="   ")

    def test_age_below_18(self):
        with pytest.raises(ValueError, match="age_years"):
            run(age_years=10)

    def test_negative_weight(self):
        with pytest.raises(ValueError, match="weight_kg"):
            run(weight_kg=-1)

    def test_zero_weight(self):
        with pytest.raises(ValueError, match="weight_kg"):
            run(weight_kg=0)

    def test_invalid_sex(self):
        with pytest.raises(ValueError, match="sex"):
            run(sex="unknown")

    def test_zero_scr(self):
        with pytest.raises(ValueError, match="scr_mg_dL"):
            run(scr_mg_dL=0)

    def test_negative_scr(self):
        with pytest.raises(ValueError, match="scr_mg_dL"):
            run(scr_mg_dL=-0.5)

    def test_zero_cl_normal(self):
        with pytest.raises(ValueError, match="cl_normal_L_per_h"):
            run(cl_normal_L_per_h=0)

    def test_zero_vd_normal(self):
        with pytest.raises(ValueError, match="vd_normal_L"):
            run(vd_normal_L=0)

    def test_invalid_distribution_type(self):
        with pytest.raises(ValueError, match="distribution_type"):
            run(distribution_type="quantum")
