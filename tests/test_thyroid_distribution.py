"""Tests for Phase 917 — Drug Thyroid Gland Distribution."""

import pytest

from omega_pbpk.prediction.thyroid_distribution import (
    ThyroidDistributionResult,
    predict_thyroid_distribution,
    screen_thyroid_safety,
)


class TestThyroidDistributionResultDataclass:
    def test_is_frozen(self):
        result = predict_thyroid_distribution("TestDrug")
        with pytest.raises((AttributeError, TypeError)):
            result.drug_name = "Other"  # type: ignore[misc]

    def test_fields_present(self):
        result = predict_thyroid_distribution("TestDrug")
        assert hasattr(result, "drug_name")
        assert hasattr(result, "logp")
        assert hasattr(result, "mw_Da")
        assert hasattr(result, "iodine_content_pct")
        assert hasattr(result, "thyroid_plasma_ratio")
        assert hasattr(result, "predicted_thyroid_conc_ug_g")
        assert hasattr(result, "plasma_conc_input_mg_L")
        assert hasattr(result, "nis_substrate_probability")
        assert hasattr(result, "thyroid_disruption_score")
        assert hasattr(result, "thyroid_toxicity_concern")
        assert hasattr(result, "notes")


class TestValidation:
    def test_mw_zero_raises(self):
        with pytest.raises(ValueError, match="mw_Da"):
            predict_thyroid_distribution("Drug", mw_Da=0.0)

    def test_mw_negative_raises(self):
        with pytest.raises(ValueError, match="mw_Da"):
            predict_thyroid_distribution("Drug", mw_Da=-100.0)

    def test_plasma_conc_zero_raises(self):
        with pytest.raises(ValueError, match="plasma_conc_mg_L"):
            predict_thyroid_distribution("Drug", plasma_conc_mg_L=0.0)

    def test_plasma_conc_negative_raises(self):
        with pytest.raises(ValueError, match="plasma_conc_mg_L"):
            predict_thyroid_distribution("Drug", plasma_conc_mg_L=-1.0)

    def test_iodine_negative_raises(self):
        with pytest.raises(ValueError, match="iodine_content_pct"):
            predict_thyroid_distribution("Drug", iodine_content_pct=-5.0)


class TestDefaultDrug:
    def test_returns_result_type(self):
        result = predict_thyroid_distribution("Generic")
        assert isinstance(result, ThyroidDistributionResult)

    def test_drug_name_stored(self):
        result = predict_thyroid_distribution("Aspirin")
        assert result.drug_name == "Aspirin"

    def test_logp_stored(self):
        result = predict_thyroid_distribution("Drug", logp=1.5)
        assert result.logp == 1.5

    def test_mw_stored(self):
        result = predict_thyroid_distribution("Drug", mw_Da=350.0)
        assert result.mw_Da == 350.0

    def test_plasma_conc_stored(self):
        result = predict_thyroid_distribution("Drug", plasma_conc_mg_L=2.5)
        assert result.plasma_conc_input_mg_L == 2.5

    def test_iodine_stored(self):
        result = predict_thyroid_distribution("Drug", iodine_content_pct=5.0)
        assert result.iodine_content_pct == 5.0

    def test_nis_base_probability(self):
        # No iodine, mw > 300: should get base 0.05
        result = predict_thyroid_distribution("Drug", logp=2.0, mw_Da=500.0, iodine_content_pct=0.0)
        assert result.nis_substrate_probability == pytest.approx(0.05)

    def test_nis_low_thyroid_ratio_no_iodine(self):
        result = predict_thyroid_distribution("Drug", iodine_content_pct=0.0)
        # baseline 0.5 (no multiplier since nis_prob = 0.05 <= 0.3)
        assert result.thyroid_plasma_ratio == pytest.approx(0.5)


class TestAmiodaroneLike:
    """Amiodarone has ~37% iodine content."""

    def test_high_iodine_nis_probability(self):
        result = predict_thyroid_distribution("Amiodarone", iodine_content_pct=37.0)
        # base 0.05 + 0.5 = 0.55
        assert result.nis_substrate_probability == pytest.approx(0.55)

    def test_high_iodine_thyroid_ratio(self):
        result = predict_thyroid_distribution("Amiodarone", iodine_content_pct=37.0)
        # 0.5 * 20 = 10.0
        assert result.thyroid_plasma_ratio == pytest.approx(10.0)

    def test_high_iodine_disruption_score(self):
        result = predict_thyroid_distribution("Amiodarone", iodine_content_pct=37.0)
        # 37*2 + 10.0*0.5 + 0.55*20 = 74 + 5 + 11 = 90
        assert result.thyroid_disruption_score == pytest.approx(90.0)

    def test_high_iodine_toxicity_concern(self):
        result = predict_thyroid_distribution("Amiodarone", iodine_content_pct=37.0)
        assert result.thyroid_toxicity_concern is True

    def test_high_iodine_notes(self):
        result = predict_thyroid_distribution("Amiodarone", iodine_content_pct=37.0)
        assert "High iodine content" in result.notes
        assert "Monitor thyroid function tests" in result.notes

    def test_predicted_concentration_calculation(self):
        result = predict_thyroid_distribution(
            "Amiodarone",
            plasma_conc_mg_L=1.0,
            iodine_content_pct=37.0,
            fu_plasma=0.5,
        )
        # ratio=10.0, plasma=1.0, fu=0.5, density=1.05
        expected = 10.0 * 1.0 * 0.5 * 1000.0 / 1.05
        assert result.predicted_thyroid_conc_ug_g == pytest.approx(expected, rel=1e-6)


class TestSmallPolarMolecule:
    """Small, polar molecule (mw < 300, logp < 0) gets extra NIS probability."""

    def test_nis_boost_small_polar(self):
        result = predict_thyroid_distribution(
            "Iodide", logp=-1.0, mw_Da=127.0, iodine_content_pct=0.0
        )
        # base 0.05 + 0.2 = 0.25 (no iodine bonus)
        assert result.nis_substrate_probability == pytest.approx(0.25)

    def test_nis_boost_plus_iodine(self):
        result = predict_thyroid_distribution(
            "IodideSalt", logp=-1.0, mw_Da=200.0, iodine_content_pct=15.0
        )
        # base 0.05 + 0.5 + 0.2 = 0.75
        assert result.nis_substrate_probability == pytest.approx(0.75)

    def test_nis_clamped_at_1(self):
        # Make sure clamping works
        result = predict_thyroid_distribution(
            "MaxNIS", logp=-2.0, mw_Da=100.0, iodine_content_pct=50.0
        )
        assert result.nis_substrate_probability <= 1.0

    def test_moderate_nis_multiplier(self):
        # nis_prob = 0.25 > 0.3? No. So no multiplier.
        result = predict_thyroid_distribution(
            "Iodide", logp=-1.0, mw_Da=127.0, iodine_content_pct=0.0
        )
        assert result.thyroid_plasma_ratio == pytest.approx(0.5)


class TestLowRiskDrug:
    def test_low_disruption_score(self):
        result = predict_thyroid_distribution(
            "LowRisk", logp=1.0, mw_Da=300.0, iodine_content_pct=0.0
        )
        assert result.thyroid_disruption_score < 40.0

    def test_no_toxicity_concern(self):
        result = predict_thyroid_distribution(
            "LowRisk", logp=1.0, mw_Da=300.0, iodine_content_pct=0.0
        )
        assert result.thyroid_toxicity_concern is False

    def test_low_risk_notes(self):
        result = predict_thyroid_distribution(
            "LowRisk", logp=1.0, mw_Da=300.0, iodine_content_pct=0.0
        )
        assert result.notes == "Low thyroid distribution risk"


class TestBoundaryConditions:
    def test_disruption_score_clamped_at_100(self):
        result = predict_thyroid_distribution("ExtremeIodine", iodine_content_pct=100.0)
        assert result.thyroid_disruption_score <= 100.0

    def test_thyroid_ratio_min_clamped(self):
        # Very low logp, no iodine — ratio should be >= 0.1
        result = predict_thyroid_distribution(
            "Drug", logp=-10.0, mw_Da=1000.0, iodine_content_pct=0.0
        )
        assert result.thyroid_plasma_ratio >= 0.1

    def test_thyroid_conc_non_negative(self):
        result = predict_thyroid_distribution("Drug", plasma_conc_mg_L=0.001, fu_plasma=0.01)
        assert result.predicted_thyroid_conc_ug_g >= 0.0


class TestScreenThyroidSafety:
    def test_returns_sorted_list(self):
        candidates = [
            {"name": "LowRisk", "logp": 1.0, "iodine_content_pct": 0.0},
            {"name": "Amiodarone", "logp": 3.0, "iodine_content_pct": 37.0},
            {"name": "MedRisk", "logp": 1.5, "iodine_content_pct": 5.0},
        ]
        results = screen_thyroid_safety(candidates)
        assert results[0].drug_name == "Amiodarone"
        scores = [r.thyroid_disruption_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_default_mw_used(self):
        candidates = [{"name": "Drug", "logp": 1.0}]
        results = screen_thyroid_safety(candidates)
        assert results[0].mw_Da == 400.0

    def test_default_iodine_zero(self):
        candidates = [{"name": "Drug", "logp": 1.0}]
        results = screen_thyroid_safety(candidates)
        assert results[0].iodine_content_pct == 0.0

    def test_empty_list(self):
        results = screen_thyroid_safety([])
        assert results == []
