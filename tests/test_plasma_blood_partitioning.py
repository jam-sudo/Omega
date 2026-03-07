"""Tests for clinical/plasma_blood_partitioning.py."""

import pytest

from omega_pbpk.clinical.plasma_blood_partitioning import (
    BloodPartitioningResult,
    blood_partitioning_impact,
    compare_oral_iv_plasma_blood,
    correct_cl_for_blood,
    correct_vd_for_blood,
    predict_rb,
)


class TestPredictRb:
    def test_low_logp_rb_near_plasma_fraction(self):
        """Low logP drug barely partitions into RBCs; Rb ≈ (1 - Hct) + small RBC term."""
        rb = predict_rb(logP=-2.0, fu_plasma=1.0, hematocrit=0.45)
        # fu_rbc ≈ 1/(1+10^-1) = 0.909; Rb ≈ 0.55 + 0.45*1/0.909 ≈ 0.55 + 0.495 ≈ 1.04
        # For very low logP: fu_rbc large → small RBC partitioning
        assert 0.3 <= rb <= 1.5

    def test_high_logp_rb_greater_than_one(self):
        """High logP drug partitions into RBCs, pushing Rb above 1."""
        rb = predict_rb(logP=4.0, fu_plasma=0.1, hematocrit=0.45)
        assert rb > 1.0

    def test_rb_clamped_to_min(self):
        """Rb cannot fall below 0.3."""
        # Very hydrophilic, high protein binding: fu_plasma tiny, low logP
        rb = predict_rb(logP=-5.0, fu_plasma=0.001, hematocrit=0.45)
        assert rb >= 0.3

    def test_rb_clamped_to_max(self):
        """Rb cannot exceed 5.0."""
        # Extreme lipophilicity
        rb = predict_rb(logP=10.0, fu_plasma=1.0, hematocrit=0.45)
        assert rb <= 5.0

    def test_rb_returns_float(self):
        rb = predict_rb(logP=1.0, fu_plasma=0.5)
        assert isinstance(rb, float)

    def test_rb_invalid_fu_plasma_zero(self):
        with pytest.raises(ValueError):
            predict_rb(logP=1.0, fu_plasma=0.0)

    def test_rb_invalid_fu_plasma_negative(self):
        with pytest.raises(ValueError):
            predict_rb(logP=1.0, fu_plasma=-0.1)

    def test_rb_invalid_fu_plasma_above_one(self):
        with pytest.raises(ValueError):
            predict_rb(logP=1.0, fu_plasma=1.1)

    def test_rb_invalid_hematocrit_zero(self):
        with pytest.raises(ValueError):
            predict_rb(logP=1.0, fu_plasma=0.5, hematocrit=0.0)

    def test_rb_invalid_hematocrit_one(self):
        with pytest.raises(ValueError):
            predict_rb(logP=1.0, fu_plasma=0.5, hematocrit=1.0)

    def test_rb_default_hematocrit(self):
        """Default hematocrit of 0.45 is used."""
        rb1 = predict_rb(logP=1.0, fu_plasma=0.5)
        rb2 = predict_rb(logP=1.0, fu_plasma=0.5, hematocrit=0.45)
        assert rb1 == rb2


class TestCorrectClForBlood:
    def test_rb_two_halves_cl(self):
        """Rb=2 → CL_blood = CL_plasma/2."""
        cl_blood = correct_cl_for_blood(cl_plasma_L_per_h=20.0, rb=2.0)
        assert cl_blood == pytest.approx(10.0)

    def test_rb_one_unchanged(self):
        cl_blood = correct_cl_for_blood(cl_plasma_L_per_h=15.0, rb=1.0)
        assert cl_blood == pytest.approx(15.0)

    def test_invalid_rb_zero(self):
        with pytest.raises(ValueError):
            correct_cl_for_blood(cl_plasma_L_per_h=10.0, rb=0.0)


class TestCorrectVdForBlood:
    def test_rb_two_doubles_vd(self):
        """Rb=2 → Vd_blood = 2*Vd_plasma (simplified correction)."""
        vd_blood = correct_vd_for_blood(vd_plasma_L=50.0, rb=2.0, fu_plasma=0.5)
        assert vd_blood == pytest.approx(100.0)

    def test_rb_one_unchanged(self):
        vd_blood = correct_vd_for_blood(vd_plasma_L=50.0, rb=1.0, fu_plasma=0.5)
        assert vd_blood == pytest.approx(50.0)

    def test_explicit_fu_blood(self):
        """When fu_blood is provided directly."""
        # fu_blood = 0.25, fu_plasma = 0.5 → ratio = 2 → Vd_blood = 2 * Vd_plasma
        vd_blood = correct_vd_for_blood(vd_plasma_L=30.0, rb=1.0, fu_plasma=0.5, fu_blood=0.25)
        assert vd_blood == pytest.approx(60.0)

    def test_invalid_rb(self):
        with pytest.raises(ValueError):
            correct_vd_for_blood(vd_plasma_L=50.0, rb=0.0, fu_plasma=0.5)


class TestBloodPartitioningImpact:
    def test_returns_dataclass(self):
        result = blood_partitioning_impact(
            drug_name="TestDrug",
            logP=2.0,
            fu_plasma=0.5,
            cl_plasma_L_per_h=10.0,
            vd_plasma_L=50.0,
        )
        assert isinstance(result, BloodPartitioningResult)

    def test_drug_name_preserved(self):
        result = blood_partitioning_impact("Warfarin", 2.7, 0.01, 0.3, 8.0)
        assert result.drug_name == "Warfarin"

    def test_rb_in_valid_range(self):
        result = blood_partitioning_impact("Drug", 1.0, 0.5, 10.0, 100.0)
        assert 0.3 <= result.rb <= 5.0

    def test_cl_blood_equals_cl_plasma_divided_by_rb(self):
        result = blood_partitioning_impact("Drug", 1.0, 0.5, 10.0, 100.0)
        assert result.cl_blood_L_per_h == pytest.approx(result.cl_plasma_L_per_h / result.rb)

    def test_vd_blood_equals_vd_plasma_times_rb(self):
        result = blood_partitioning_impact("Drug", 1.0, 0.5, 10.0, 100.0)
        assert result.vd_blood_L == pytest.approx(result.vd_plasma_L * result.rb)

    def test_t_half_positive(self):
        result = blood_partitioning_impact("Drug", 1.0, 0.5, 5.0, 70.0)
        assert result.t_half_plasma_h > 0
        assert result.t_half_blood_h > 0

    def test_notes_populated(self):
        result = blood_partitioning_impact("Drug", 1.0, 0.5, 10.0, 50.0)
        assert len(result.notes) > 0

    def test_high_logp_note_mentions_rbc(self):
        result = blood_partitioning_impact("LipophilicDrug", 5.0, 0.9, 20.0, 200.0)
        # High logP → rb > 1.5 → note about RBC partitioning
        if result.rb > 1.5:
            assert "RBC" in result.notes


class TestCompareOralIvPlasmaBlood:
    def test_no_correction_when_rb_one(self):
        """When Rb=1, AUC_blood_oral = AUC_plasma_oral; F same as uncorrected."""
        result = compare_oral_iv_plasma_blood(
            iv_auc_blood=100.0,
            oral_auc_plasma=80.0,
            dose_iv=100.0,
            dose_oral=100.0,
            rb=1.0,
        )
        assert result["F_pct"] == pytest.approx(80.0)

    def test_rb_doubles_oral_auc_blood(self):
        """Rb=2 doubles oral plasma AUC to get blood AUC."""
        result = compare_oral_iv_plasma_blood(
            iv_auc_blood=100.0,
            oral_auc_plasma=50.0,
            dose_iv=100.0,
            dose_oral=100.0,
            rb=2.0,
        )
        assert result["auc_blood_oral"] == pytest.approx(100.0)
        assert result["F_pct"] == pytest.approx(100.0)

    def test_returns_dict_with_expected_keys(self):
        result = compare_oral_iv_plasma_blood(100.0, 80.0, 100.0, 100.0, 1.0)
        assert "F_pct" in result
        assert "auc_blood_oral" in result
        assert "correction_note" in result

    def test_invalid_rb_zero(self):
        with pytest.raises(ValueError):
            compare_oral_iv_plasma_blood(100.0, 80.0, 100.0, 100.0, 0.0)

    def test_invalid_dose_zero(self):
        with pytest.raises(ValueError):
            compare_oral_iv_plasma_blood(100.0, 80.0, 0.0, 100.0, 1.0)
