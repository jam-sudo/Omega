"""Tests for Phase 519 — Creatinine Clearance & Renal Function Estimation."""

import pytest

from omega_pbpk.clinical.renal_function_calc import (
    ckd_epi_egfr,
    classify_ckd_stage,
    cockcroft_gault,
    mdrd_egfr,
    renal_dose_adjustment,
    schwartz_egfr,
)

# ---------------------------------------------------------------------------
# cockcroft_gault
# ---------------------------------------------------------------------------


class TestCockcroftGault:
    def test_reference_male(self):
        # 70 yo, 70 kg, male, scr=1.0 → (140-70)*70 / (72*1.0) = 4900/72 ≈ 68.06
        result = cockcroft_gault(70, 70, 1.0, "male")
        assert abs(result - (140 - 70) * 70 / (72 * 1.0)) < 0.01

    def test_female_multiplier(self):
        male_val = cockcroft_gault(50, 60, 1.0, "male")
        female_val = cockcroft_gault(50, 60, 1.0, "female")
        assert abs(female_val - male_val * 0.85) < 0.001

    def test_higher_scr_lower_crcl(self):
        low_scr = cockcroft_gault(50, 70, 0.8, "male")
        high_scr = cockcroft_gault(50, 70, 2.0, "male")
        assert low_scr > high_scr

    def test_older_age_lower_crcl(self):
        young = cockcroft_gault(30, 70, 1.0, "male")
        old = cockcroft_gault(70, 70, 1.0, "male")
        assert young > old

    def test_case_insensitive_sex(self):
        r1 = cockcroft_gault(50, 70, 1.0, "Male")
        r2 = cockcroft_gault(50, 70, 1.0, "male")
        assert abs(r1 - r2) < 0.001

    def test_invalid_age(self):
        with pytest.raises(ValueError, match="age_years"):
            cockcroft_gault(-1, 70, 1.0, "male")

    def test_invalid_weight(self):
        with pytest.raises(ValueError, match="weight_kg"):
            cockcroft_gault(50, 0, 1.0, "male")

    def test_invalid_scr(self):
        with pytest.raises(ValueError, match="scr_mg_dL"):
            cockcroft_gault(50, 70, -0.5, "male")

    def test_invalid_sex(self):
        with pytest.raises(ValueError, match="sex"):
            cockcroft_gault(50, 70, 1.0, "other")


# ---------------------------------------------------------------------------
# ckd_epi_egfr
# ---------------------------------------------------------------------------


class TestCkdEpiEgfr:
    def test_basic_male(self):
        result = ckd_epi_egfr(50, 1.0, "male")
        assert result > 0

    def test_female_multiplier_applied(self):
        male = ckd_epi_egfr(50, 1.0, "male")
        female = ckd_epi_egfr(50, 1.0, "female")
        # female eGFR should be higher due to 1.018 multiplier
        # (kappa difference also affects the result)
        assert female != male

    def test_black_race_multiplier(self):
        non_black = ckd_epi_egfr(50, 1.0, "male", race="non_black")
        black = ckd_epi_egfr(50, 1.0, "male", race="black")
        assert abs(black - non_black * 1.159) < 0.01

    def test_higher_scr_lower_egfr(self):
        low = ckd_epi_egfr(50, 0.8, "male")
        high = ckd_epi_egfr(50, 2.5, "male")
        assert low > high

    def test_invalid_scr(self):
        with pytest.raises(ValueError, match="scr_mg_dL"):
            ckd_epi_egfr(50, 0, "male")

    def test_invalid_race(self):
        with pytest.raises(ValueError, match="race"):
            ckd_epi_egfr(50, 1.0, "male", race="asian")


# ---------------------------------------------------------------------------
# mdrd_egfr
# ---------------------------------------------------------------------------


class TestMdrdEgfr:
    def test_basic_male(self):
        result = mdrd_egfr(50, 1.0, "male")
        assert result > 0

    def test_female_factor(self):
        male = mdrd_egfr(50, 1.0, "male")
        female = mdrd_egfr(50, 1.0, "female")
        assert abs(female - male * 0.742) < 0.01

    def test_black_race_factor(self):
        non_black = mdrd_egfr(50, 1.0, "male")
        black = mdrd_egfr(50, 1.0, "male", race="black")
        assert abs(black - non_black * 1.212) < 0.01

    def test_invalid_age(self):
        with pytest.raises(ValueError, match="age_years"):
            mdrd_egfr(0, 1.0, "male")

    def test_invalid_scr(self):
        with pytest.raises(ValueError, match="scr_mg_dL"):
            mdrd_egfr(50, -1.0, "male")


# ---------------------------------------------------------------------------
# schwartz_egfr
# ---------------------------------------------------------------------------


class TestSchwartzEgfr:
    def test_reference_child(self):
        # 100 cm height, scr=0.5 → 0.413 * 100 / 0.5 = 82.6
        result = schwartz_egfr(100, 0.5)
        assert abs(result - 82.6) < 0.001

    def test_higher_scr_lower_egfr(self):
        low = schwartz_egfr(120, 0.5)
        high = schwartz_egfr(120, 1.5)
        assert low > high

    def test_invalid_height(self):
        with pytest.raises(ValueError, match="height_cm"):
            schwartz_egfr(0, 0.5)

    def test_invalid_scr(self):
        with pytest.raises(ValueError, match="scr_mg_dL"):
            schwartz_egfr(100, -0.1)


# ---------------------------------------------------------------------------
# classify_ckd_stage
# ---------------------------------------------------------------------------


class TestClassifyCkdStage:
    def test_g1_normal(self):
        result = classify_ckd_stage(95)
        assert result["stage"] == "G1"
        assert result["dose_adjustment_needed"] is False

    def test_g1_boundary(self):
        result = classify_ckd_stage(90)
        assert result["stage"] == "G1"

    def test_g2(self):
        result = classify_ckd_stage(75)
        assert result["stage"] == "G2"
        assert result["dose_adjustment_needed"] is False

    def test_g2_boundary(self):
        result = classify_ckd_stage(60)
        assert result["stage"] == "G2"

    def test_g3a(self):
        result = classify_ckd_stage(52)
        assert result["stage"] == "G3a"
        assert result["dose_adjustment_needed"] is True

    def test_g3b(self):
        result = classify_ckd_stage(35)
        assert result["stage"] == "G3b"
        assert result["dose_adjustment_needed"] is True

    def test_g3b_boundary(self):
        result = classify_ckd_stage(30)
        assert result["stage"] == "G3b"

    def test_g4(self):
        result = classify_ckd_stage(20)
        assert result["stage"] == "G4"
        assert result["dose_adjustment_needed"] is True

    def test_g5_kidney_failure(self):
        result = classify_ckd_stage(10)
        assert result["stage"] == "G5"
        assert result["dose_adjustment_needed"] is True

    def test_g5_zero(self):
        result = classify_ckd_stage(0)
        assert result["stage"] == "G5"

    def test_negative_egfr_raises(self):
        with pytest.raises(ValueError):
            classify_ckd_stage(-5)

    def test_returns_description(self):
        result = classify_ckd_stage(75)
        assert "description" in result
        assert len(result["description"]) > 0


# ---------------------------------------------------------------------------
# renal_dose_adjustment
# ---------------------------------------------------------------------------


class TestRenalDoseAdjustment:
    def test_dose_factor_50_percent(self):
        result = renal_dose_adjustment(10.0, 50.0, egfr_normal=100.0)
        assert abs(result["dose_factor"] - 0.5) < 0.001

    def test_normal_function_no_adjustment(self):
        result = renal_dose_adjustment(10.0, 100.0)
        assert abs(result["dose_factor"] - 1.0) < 0.001

    def test_adjusted_cl(self):
        result = renal_dose_adjustment(20.0, 50.0, egfr_normal=100.0)
        assert abs(result["adjusted_cl_L_per_h"] - 10.0) < 0.001

    def test_low_egfr_major_adjustment(self):
        result = renal_dose_adjustment(10.0, 10.0)
        assert result["dose_factor"] < 0.25

    def test_ckd_stage_in_result(self):
        result = renal_dose_adjustment(10.0, 50.0)
        assert "ckd_stage" in result

    def test_invalid_cl(self):
        with pytest.raises(ValueError, match="cl_normal"):
            renal_dose_adjustment(0, 50.0)

    def test_invalid_egfr_patient(self):
        with pytest.raises(ValueError, match="egfr_patient"):
            renal_dose_adjustment(10.0, -5.0)
