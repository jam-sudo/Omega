"""Tests for Phase 561 — Cardiac Output Effect on Drug Distribution."""

import pytest

from omega_pbpk.core.cardiac_output_distribution import (
    CardiacOutputDistributionResult,
    compare_cardiac_outputs,
    predict_co_distribution_effect,
)


@pytest.fixture
def base_params():
    return dict(
        drug_name="DrugA",
        cardiac_output_L_per_h=90.0,
        baseline_cl_hepatic_normal_L_per_h=10.0,
        fup=0.5,
        baseline_vd_L=50.0,
    )


class TestReturnType:
    def test_returns_correct_type(self, base_params):
        result = predict_co_distribution_effect(**base_params)
        assert isinstance(result, CardiacOutputDistributionResult)

    def test_drug_name_preserved(self, base_params):
        result = predict_co_distribution_effect(**base_params)
        assert result.drug_name == "DrugA"


class TestExtractionRatio:
    def test_e_hepatic_in_range(self, base_params):
        result = predict_co_distribution_effect(**base_params)
        assert 0 <= result.e_hepatic <= 1

    def test_high_extraction_flow_limited(self):
        result = predict_co_distribution_effect(
            drug_name="HighE",
            cardiac_output_L_per_h=45.0,
            baseline_cl_hepatic_normal_L_per_h=20.0,
            fup=0.5,
            baseline_vd_L=50.0,
            e_hepatic_normal=0.8,
        )
        # High E: cl_hepatic = Q_hepatic * E
        expected_q = 72.0 * (45.0 / 90.0)
        assert abs(result.cl_hepatic_L_per_h - expected_q * 0.8) < 0.01

    def test_low_extraction_independent(self):
        result_low_co = predict_co_distribution_effect(
            drug_name="LowE",
            cardiac_output_L_per_h=45.0,
            baseline_cl_hepatic_normal_L_per_h=5.0,
            fup=0.5,
            baseline_vd_L=50.0,
            e_hepatic_normal=0.2,
        )
        result_high_co = predict_co_distribution_effect(
            drug_name="LowE",
            cardiac_output_L_per_h=135.0,
            baseline_cl_hepatic_normal_L_per_h=5.0,
            fup=0.5,
            baseline_vd_L=50.0,
            e_hepatic_normal=0.2,
        )
        # Low E: hepatic CL should be same regardless of CO
        assert abs(result_low_co.cl_hepatic_L_per_h - result_high_co.cl_hepatic_L_per_h) < 0.01


class TestHalfLife:
    def test_t_half_positive(self, base_params):
        result = predict_co_distribution_effect(**base_params)
        assert result.t_half_h > 0

    def test_vd_preserved(self, base_params):
        result = predict_co_distribution_effect(**base_params)
        assert result.vd_L == 50.0


class TestLowCO:
    def test_low_co_higher_auc(self, base_params):
        """Low CO should lead to higher AUC (reduced clearance)."""
        base_params["cardiac_output_L_per_h"] = 40.0
        result = predict_co_distribution_effect(**base_params)
        assert result.auc_ratio_vs_normal > 1.0

    def test_low_co_class(self, base_params):
        base_params["cardiac_output_L_per_h"] = 40.0
        result = predict_co_distribution_effect(**base_params)
        assert result.co_class == "low"


class TestHighCO:
    def test_high_co_lower_auc(self, base_params):
        """High CO should lead to lower AUC (increased clearance)."""
        base_params["cardiac_output_L_per_h"] = 150.0
        result = predict_co_distribution_effect(**base_params)
        assert result.auc_ratio_vs_normal < 1.0

    def test_high_co_class(self, base_params):
        base_params["cardiac_output_L_per_h"] = 150.0
        result = predict_co_distribution_effect(**base_params)
        assert result.co_class == "high"


class TestNormalCO:
    def test_normal_co_class(self, base_params):
        result = predict_co_distribution_effect(**base_params)
        assert result.co_class == "normal"

    def test_normal_co_auc_ratio_approx_1(self, base_params):
        result = predict_co_distribution_effect(**base_params)
        assert abs(result.auc_ratio_vs_normal - 1.0) < 0.01


class TestAUCRatio:
    def test_auc_ratio_positive(self, base_params):
        result = predict_co_distribution_effect(**base_params)
        assert result.auc_ratio_vs_normal > 0

    def test_significant_impact(self):
        result = predict_co_distribution_effect(
            drug_name="X",
            cardiac_output_L_per_h=30.0,
            baseline_cl_hepatic_normal_L_per_h=10.0,
            fup=0.5,
            baseline_vd_L=50.0,
            e_hepatic_normal=0.5,
        )
        # Very low CO should create significant impact
        assert result.clinical_impact in ("moderate", "significant")

    def test_minimal_impact_at_normal(self, base_params):
        result = predict_co_distribution_effect(**base_params)
        assert result.clinical_impact == "minimal"


class TestCompare:
    def test_compare_sorted_descending(self):
        results = compare_cardiac_outputs(
            drug_name="DrugA",
            co_values=[45.0, 90.0, 135.0],
            baseline_cl_hepatic_normal_L_per_h=10.0,
            fup=0.5,
            baseline_vd_L=50.0,
        )
        ratios = [r.auc_ratio_vs_normal for r in results]
        assert ratios == sorted(ratios, reverse=True)

    def test_compare_returns_list(self):
        results = compare_cardiac_outputs(
            drug_name="DrugA",
            co_values=[60.0, 120.0],
            baseline_cl_hepatic_normal_L_per_h=10.0,
            fup=0.5,
            baseline_vd_L=50.0,
        )
        assert isinstance(results, list)
        assert len(results) == 2


class TestValidation:
    def test_co_must_be_positive(self, base_params):
        base_params["cardiac_output_L_per_h"] = 0
        with pytest.raises(ValueError):
            predict_co_distribution_effect(**base_params)

    def test_co_negative(self, base_params):
        base_params["cardiac_output_L_per_h"] = -10
        with pytest.raises(ValueError):
            predict_co_distribution_effect(**base_params)

    def test_fup_zero(self, base_params):
        base_params["fup"] = 0
        with pytest.raises(ValueError):
            predict_co_distribution_effect(**base_params)

    def test_fup_above_one(self, base_params):
        base_params["fup"] = 1.5
        with pytest.raises(ValueError):
            predict_co_distribution_effect(**base_params)

    def test_vd_must_be_positive(self, base_params):
        base_params["baseline_vd_L"] = 0
        with pytest.raises(ValueError):
            predict_co_distribution_effect(**base_params)
