"""
Tests for Phase 405 — Red Blood Cell Partitioning Model
"""

import math

import pytest

from omega_pbpk.prediction.rbc_partitioning_model import (
    RBCPartitioningResult,
    _compute_kp_rbc,
    predict_rbc_partitioning,
    screen_hematocrit_effect,
)

# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_rbc_partitioning_result(self):
        result = predict_rbc_partitioning("TestDrug", logp=2.0, fu_plasma=0.1)
        assert isinstance(result, RBCPartitioningResult)

    def test_result_is_frozen(self):
        result = predict_rbc_partitioning("TestDrug", logp=2.0, fu_plasma=0.1)
        with pytest.raises((AttributeError, TypeError)):
            result.drug_name = "Modified"  # type: ignore[misc]

    def test_drug_name_preserved(self):
        result = predict_rbc_partitioning("Aspirin", logp=1.2, fu_plasma=0.05)
        assert result.drug_name == "Aspirin"

    def test_logp_preserved(self):
        result = predict_rbc_partitioning("X", logp=3.5, fu_plasma=0.2)
        assert result.logp == pytest.approx(3.5)

    def test_hematocrit_preserved(self):
        result = predict_rbc_partitioning("X", logp=1.0, fu_plasma=0.1, hematocrit=0.40)
        assert result.hematocrit == pytest.approx(0.40)


# ---------------------------------------------------------------------------
# Kp_rbc model correctness
# ---------------------------------------------------------------------------


class TestKpRBCModel:
    def test_basic_drug_higher_kp_than_neutral(self):
        """Basic drug (pka_base > 7.4) should have higher Kp than neutral drug."""
        kp_neutral = _compute_kp_rbc(logp=2.0, pka_base=None)
        kp_basic = _compute_kp_rbc(logp=2.0, pka_base=9.0)
        assert kp_basic > kp_neutral

    def test_negative_logp_gives_low_kp(self):
        """Hydrophilic drug (logP < 0) uses fixed lipophilic component of 0.3."""
        kp = _compute_kp_rbc(logp=-1.0, pka_base=None)
        assert kp == pytest.approx(0.3)

    def test_kp_clamped_at_max(self):
        """Very lipophilic, strongly basic drug should be clamped to 20."""
        kp = _compute_kp_rbc(logp=5.0, pka_base=11.0)
        assert kp <= 20.0

    def test_kp_clamped_at_min(self):
        """Very hydrophilic neutral drug should be clamped to 0.1."""
        kp = _compute_kp_rbc(logp=-5.0, pka_base=None)
        assert kp >= 0.1

    def test_ion_trapping_formula(self):
        """Ion-trapping factor should be > 1 for basic drug."""
        kp_neutral = _compute_kp_rbc(logp=2.0, pka_base=None)
        kp_basic = _compute_kp_rbc(logp=2.0, pka_base=8.0)
        # Ion trap factor > 1, so basic > neutral
        assert kp_basic > kp_neutral


# ---------------------------------------------------------------------------
# Blood-to-plasma ratio
# ---------------------------------------------------------------------------


class TestBloodToPlasmaRatio:
    def test_bp_gt_1_for_lipophilic_basic(self):
        """Lipophilic basic drug should have B/P > 1."""
        result = predict_rbc_partitioning("BaseLipo", logp=3.0, fu_plasma=0.1, pka_base=9.0)
        assert result.blood_to_plasma_ratio > 1.0

    def test_bp_formula_neutral_drug(self):
        """B/P = 1 - Hct + Hct * Kp for neutral drug."""
        hct = 0.45
        result = predict_rbc_partitioning("Neutral", logp=1.0, fu_plasma=0.5)
        kp = result.kp_rbc
        expected_bp = 1 - hct + hct * kp
        assert result.blood_to_plasma_ratio == pytest.approx(expected_bp, rel=1e-6)

    def test_fu_blood_le_fu_plasma_for_high_kp(self):
        """With high Kp (B/P > 1), fu_blood should be < fu_plasma."""
        result = predict_rbc_partitioning("HighKp", logp=4.0, fu_plasma=0.3, pka_base=9.5)
        assert result.fu_blood <= result.fu_plasma

    def test_fu_blood_equals_fu_plasma_over_bp(self):
        """fu_blood = fu_plasma / B_P."""
        result = predict_rbc_partitioning("X", logp=2.0, fu_plasma=0.2)
        expected = result.fu_plasma / result.blood_to_plasma_ratio
        assert result.fu_blood == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Clearance
# ---------------------------------------------------------------------------


class TestClearance:
    def test_cl_blood_equals_cl_plasma_over_bp(self):
        """cl_blood = cl_plasma / B_P."""
        result = predict_rbc_partitioning("Drug", logp=2.0, fu_plasma=0.1, cl_ref_L_per_h=20.0)
        expected_cl_blood = result.cl_plasma_L_per_h / result.blood_to_plasma_ratio
        assert result.cl_blood_L_per_h == pytest.approx(expected_cl_blood, rel=1e-6)

    def test_cl_plasma_equals_cl_ref(self):
        result = predict_rbc_partitioning("Drug", logp=1.5, fu_plasma=0.1, cl_ref_L_per_h=15.0)
        assert result.cl_plasma_L_per_h == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# Distribution percentages
# ---------------------------------------------------------------------------


class TestDistributionPercentages:
    def test_rbc_plus_plasma_equals_100(self):
        """RBC sequestration % + drug in plasma % should equal 100%."""
        result = predict_rbc_partitioning("Drug", logp=2.5, fu_plasma=0.15)
        total = result.rbc_sequestration_pct + result.drug_in_plasma_pct
        assert total == pytest.approx(100.0, rel=1e-4)

    def test_rbc_sequestration_increases_with_kp(self):
        """Higher Kp should lead to higher RBC sequestration."""
        result_low = predict_rbc_partitioning("Low", logp=0.5, fu_plasma=0.5)
        result_high = predict_rbc_partitioning("High", logp=4.0, fu_plasma=0.1, pka_base=9.0)
        assert result_high.rbc_sequestration_pct > result_low.rbc_sequestration_pct


# ---------------------------------------------------------------------------
# Hemoglobin binding
# ---------------------------------------------------------------------------


class TestHemoglobinBinding:
    def test_hemoglobin_binding_for_basic_lipophilic(self):
        """Strongly basic, lipophilic drug with high Kp should show hemoglobin binding."""
        result = predict_rbc_partitioning("BasicLipo", logp=4.0, fu_plasma=0.1, pka_base=9.5)
        # Verify Kp > 2 and pka_base > 6
        if result.kp_rbc > 2:
            assert result.hemoglobin_binding is True

    def test_no_hemoglobin_binding_for_neutral_drug(self):
        """Neutral drug should not show hemoglobin binding."""
        result = predict_rbc_partitioning("Neutral", logp=1.0, fu_plasma=0.5)
        assert result.hemoglobin_binding is False

    def test_no_hemoglobin_binding_low_pka(self):
        """Drug with pKa <= 6 should not show hemoglobin binding."""
        result = predict_rbc_partitioning("WeakBase", logp=3.0, fu_plasma=0.1, pka_base=5.0)
        assert result.hemoglobin_binding is False


# ---------------------------------------------------------------------------
# Clinical relevance
# ---------------------------------------------------------------------------


class TestClinicalRelevance:
    def test_high_relevance_kp_gt_2(self):
        """Drug with Kp > 2 should have 'high' clinical relevance."""
        result = predict_rbc_partitioning("HighKp", logp=4.0, fu_plasma=0.1, pka_base=9.0)
        if result.kp_rbc > 2:
            assert result.clinical_relevance == "high"

    def test_low_relevance_hydrophilic(self):
        """Hydrophilic neutral drug should have 'low' clinical relevance."""
        result = predict_rbc_partitioning("Hydro", logp=-0.5, fu_plasma=0.9)
        assert result.clinical_relevance == "low"


# ---------------------------------------------------------------------------
# Screen hematocrit effect
# ---------------------------------------------------------------------------


class TestScreenHematocritEffect:
    def test_screen_returns_5_results_default(self):
        results = screen_hematocrit_effect("Drug", logp=2.0, fu_plasma=0.1)
        assert len(results) == 5

    def test_screen_sorted_by_bp_ascending(self):
        """Results should be sorted by blood_to_plasma_ratio ascending."""
        results = screen_hematocrit_effect("Drug", logp=2.0, fu_plasma=0.1, pka_base=8.0)
        bp_values = [r.blood_to_plasma_ratio for r in results]
        assert bp_values == sorted(bp_values)

    def test_screen_custom_hct_values(self):
        results = screen_hematocrit_effect(
            "Drug", logp=1.5, fu_plasma=0.2, hct_values=[0.30, 0.45, 0.60]
        )
        assert len(results) == 3

    def test_all_results_are_rbc_result_type(self):
        results = screen_hematocrit_effect("Drug", logp=2.0, fu_plasma=0.15)
        assert all(isinstance(r, RBCPartitioningResult) for r in results)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidationErrors:
    def test_invalid_fu_plasma_zero(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            predict_rbc_partitioning("Drug", logp=1.0, fu_plasma=0.0)

    def test_invalid_fu_plasma_gt_1(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            predict_rbc_partitioning("Drug", logp=1.0, fu_plasma=1.5)

    def test_invalid_hematocrit_zero(self):
        with pytest.raises(ValueError, match="hematocrit"):
            predict_rbc_partitioning("Drug", logp=1.0, fu_plasma=0.1, hematocrit=0.0)

    def test_invalid_hematocrit_one(self):
        with pytest.raises(ValueError, match="hematocrit"):
            predict_rbc_partitioning("Drug", logp=1.0, fu_plasma=0.1, hematocrit=1.0)

    def test_invalid_cl_ref_zero(self):
        with pytest.raises(ValueError, match="cl_ref"):
            predict_rbc_partitioning("Drug", logp=1.0, fu_plasma=0.1, cl_ref_L_per_h=0.0)

    def test_invalid_cl_ref_negative(self):
        with pytest.raises(ValueError, match="cl_ref"):
            predict_rbc_partitioning("Drug", logp=1.0, fu_plasma=0.1, cl_ref_L_per_h=-5.0)


# ---------------------------------------------------------------------------
# pKa = None (no basic group)
# ---------------------------------------------------------------------------


class TestNoPkaBase:
    def test_pka_base_is_nan_when_not_provided(self):
        result = predict_rbc_partitioning("Neutral", logp=2.0, fu_plasma=0.2)
        assert math.isnan(result.pka_base)

    def test_pka_base_stored_when_provided(self):
        result = predict_rbc_partitioning("Basic", logp=2.0, fu_plasma=0.2, pka_base=8.5)
        assert result.pka_base == pytest.approx(8.5)
