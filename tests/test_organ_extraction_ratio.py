"""Tests for clinical/organ_extraction_ratio.py — Phase 556."""

import pytest

from omega_pbpk.clinical.organ_extraction_ratio import (
    first_pass_oral_bioavailability,
    hepatic_extraction_ratio,
    organ_clearance_budget,
    renal_extraction_ratio,
    sensitivity_to_blood_flow,
)

# ---------------------------------------------------------------------------
# hepatic_extraction_ratio
# ---------------------------------------------------------------------------


class TestHepaticExtractionRatio:
    def test_high_cl_int_gives_high_er(self):
        result = hepatic_extraction_ratio(10000.0, q_liver_L_per_h=90.0, fu_plasma=1.0)
        assert result["er"] > 0.7
        assert result["hepatic_class"] == "high"

    def test_low_cl_int_gives_low_er(self):
        result = hepatic_extraction_ratio(1.0, q_liver_L_per_h=90.0, fu_plasma=1.0)
        assert result["er"] < 0.3
        assert result["hepatic_class"] == "low"

    def test_cl_int_equals_q_over_fu_gives_er_half(self):
        # ER = fu*CLint / (Q + fu*CLint); if fu*CLint = Q → ER = Q/(Q+Q) = 0.5
        q = 90.0
        fu = 1.0
        cl_int = q / fu  # 90.0
        result = hepatic_extraction_ratio(cl_int, q_liver_L_per_h=q, fu_plasma=fu)
        assert abs(result["er"] - 0.5) < 1e-9

    def test_intermediate_class(self):
        # er~0.5 should be intermediate
        q = 90.0
        fu = 1.0
        cl_int = q  # er=0.5
        result = hepatic_extraction_ratio(cl_int, q_liver_L_per_h=q, fu_plasma=fu)
        assert result["hepatic_class"] == "intermediate"

    def test_f_oral_estimate_is_1_minus_er(self):
        result = hepatic_extraction_ratio(90.0, q_liver_L_per_h=90.0, fu_plasma=1.0)
        assert abs(result["f_oral_estimate"] - (1.0 - result["er"])) < 1e-9

    def test_cl_hepatic_equals_q_times_er(self):
        q = 90.0
        result = hepatic_extraction_ratio(100.0, q_liver_L_per_h=q, fu_plasma=1.0)
        assert abs(result["cl_hepatic"] - q * result["er"]) < 1e-9

    def test_lower_fu_reduces_er(self):
        result_fu1 = hepatic_extraction_ratio(50.0, fu_plasma=1.0)
        result_fu05 = hepatic_extraction_ratio(50.0, fu_plasma=0.5)
        # lower fu → less extraction
        assert result_fu05["er"] < result_fu1["er"]

    def test_return_keys(self):
        result = hepatic_extraction_ratio(100.0)
        expected = {"er", "cl_hepatic", "cl_intrinsic_scaled", "hepatic_class", "f_oral_estimate"}
        assert set(result.keys()) == expected

    def test_invalid_cl_int_raises(self):
        with pytest.raises(ValueError):
            hepatic_extraction_ratio(-10.0)

    def test_invalid_q_raises(self):
        with pytest.raises(ValueError):
            hepatic_extraction_ratio(100.0, q_liver_L_per_h=0.0)

    def test_invalid_fu_zero_raises(self):
        with pytest.raises(ValueError):
            hepatic_extraction_ratio(100.0, fu_plasma=0.0)

    def test_invalid_fu_above_1_raises(self):
        with pytest.raises(ValueError):
            hepatic_extraction_ratio(100.0, fu_plasma=1.5)


# ---------------------------------------------------------------------------
# renal_extraction_ratio
# ---------------------------------------------------------------------------


class TestRenalExtractionRatio:
    def test_high_cl_renal_gives_high_er(self):
        result = renal_extraction_ratio(200.0, q_renal_L_per_h=72.0)
        # Capped at 1.0
        assert result["er"] == 1.0
        assert result["renal_class"] == "high"

    def test_low_cl_renal_gives_low_er(self):
        result = renal_extraction_ratio(1.0, q_renal_L_per_h=72.0)
        assert result["er"] < 0.3
        assert result["renal_class"] == "low"

    def test_er_capped_at_1(self):
        result = renal_extraction_ratio(1000.0, q_renal_L_per_h=72.0)
        assert result["er"] <= 1.0

    def test_return_keys(self):
        result = renal_extraction_ratio(10.0)
        assert set(result.keys()) == {"er", "cl_renal", "renal_class"}

    def test_er_equals_cl_over_q_when_under_1(self):
        cl = 20.0
        q = 72.0
        result = renal_extraction_ratio(cl, q)
        assert abs(result["er"] - cl / q) < 1e-9

    def test_invalid_negative_cl_raises(self):
        with pytest.raises(ValueError):
            renal_extraction_ratio(-5.0)


# ---------------------------------------------------------------------------
# organ_clearance_budget
# ---------------------------------------------------------------------------


class TestOrganClearanceBudget:
    def test_fractions_sum_to_1(self):
        result = organ_clearance_budget(60.0, 30.0, 10.0)
        total = result["frac_hepatic"] + result["frac_renal"] + result["frac_other"]
        assert abs(total - 1.0) < 1e-9

    def test_dominant_organ_hepatic(self):
        result = organ_clearance_budget(80.0, 10.0, 5.0)
        assert result["dominant_organ"] == "hepatic"

    def test_dominant_organ_renal(self):
        result = organ_clearance_budget(10.0, 80.0, 5.0)
        assert result["dominant_organ"] == "renal"

    def test_cl_total_correct(self):
        result = organ_clearance_budget(60.0, 30.0, 10.0)
        assert abs(result["cl_total"] - 100.0) < 1e-9

    def test_no_other_cl(self):
        result = organ_clearance_budget(60.0, 40.0)
        assert abs(result["frac_other"]) < 1e-9
        assert abs(result["frac_hepatic"] + result["frac_renal"] - 1.0) < 1e-9

    def test_invalid_negative_cl_raises(self):
        with pytest.raises(ValueError):
            organ_clearance_budget(-10.0, 30.0)


# ---------------------------------------------------------------------------
# first_pass_oral_bioavailability
# ---------------------------------------------------------------------------


class TestFirstPassOralBioavailability:
    def test_no_extraction_gives_f_1(self):
        f = first_pass_oral_bioavailability(0.0, 0.0)
        assert f == 1.0

    def test_complete_hepatic_extraction_gives_f_0(self):
        f = first_pass_oral_bioavailability(0.0, 1.0)
        assert f == 0.0

    def test_complete_gut_extraction_gives_f_0(self):
        f = first_pass_oral_bioavailability(1.0, 0.0)
        assert f == 0.0

    def test_half_half_gives_f_quarter(self):
        f = first_pass_oral_bioavailability(0.5, 0.5)
        assert abs(f - 0.25) < 1e-9

    def test_formula_consistency(self):
        er_gut = 0.3
        er_hep = 0.6
        f = first_pass_oral_bioavailability(er_gut, er_hep)
        expected = (1 - er_gut) * (1 - er_hep)
        assert abs(f - expected) < 1e-9

    def test_invalid_er_gut_raises(self):
        with pytest.raises(ValueError):
            first_pass_oral_bioavailability(-0.1, 0.5)

    def test_invalid_er_hepatic_raises(self):
        with pytest.raises(ValueError):
            first_pass_oral_bioavailability(0.5, 1.5)


# ---------------------------------------------------------------------------
# sensitivity_to_blood_flow
# ---------------------------------------------------------------------------


class TestSensitivityToBloodFlow:
    def test_returns_correct_length(self):
        q_range = [60.0, 90.0, 120.0, 150.0]
        result = sensitivity_to_blood_flow(90.0, 1.0, q_range)
        assert len(result) == len(q_range)

    def test_each_entry_has_required_keys(self):
        result = sensitivity_to_blood_flow(90.0, 1.0, [60.0, 90.0])
        for entry in result:
            assert "q" in entry
            assert "er" in entry
            assert "cl_hepatic" in entry

    def test_higher_q_gives_lower_er(self):
        # For fixed CLint, higher blood flow → lower ER (dilution effect)
        result = sensitivity_to_blood_flow(90.0, 1.0, [60.0, 90.0, 120.0])
        ers = [r["er"] for r in result]
        # ER should decrease as Q increases
        assert ers[0] > ers[1] > ers[2]

    def test_at_q_equals_cl_int_er_is_half(self):
        cl_int = 90.0
        fu = 1.0
        result = sensitivity_to_blood_flow(cl_int, fu, [cl_int])
        # fu*CLint = Q → ER = 0.5
        assert abs(result[0]["er"] - 0.5) < 1e-9

    def test_empty_q_range_returns_empty_list(self):
        result = sensitivity_to_blood_flow(90.0, 1.0, [])
        assert result == []
