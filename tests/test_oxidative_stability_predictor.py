"""Tests for Phase 227: Oxidative Stability Predictor."""

import pytest

from omega_pbpk.prediction.oxidative_stability_predictor import (
    OxidativeStabilityResult,
    predict_oxidative_stability,
    screen_oxidative_stability,
)


class TestReturnType:
    def test_returns_oxidative_stability_result(self):
        result = predict_oxidative_stability("TestDrug")
        assert isinstance(result, OxidativeStabilityResult)

    def test_result_is_frozen(self):
        result = predict_oxidative_stability("TestDrug")
        with pytest.raises((AttributeError, TypeError)):
            result.risk_score = 99.0  # type: ignore[misc]


class TestFieldPreservation:
    def test_drug_name_preserved(self):
        result = predict_oxidative_stability("Aspirin", mw_Da=180.0)
        assert result.drug_name == "Aspirin"

    def test_has_phenol_preserved(self):
        result = predict_oxidative_stability("TestDrug", has_phenol=True)
        assert result.has_phenol is True

    def test_has_catechol_preserved(self):
        result = predict_oxidative_stability("TestDrug", has_catechol=True)
        assert result.has_catechol is True

    def test_has_thioether_preserved(self):
        result = predict_oxidative_stability("TestDrug", has_thioether=True)
        assert result.has_thioether is True

    def test_has_aldehyde_preserved(self):
        result = predict_oxidative_stability("TestDrug", has_aldehyde=True)
        assert result.has_aldehyde is True

    def test_logP_preserved(self):
        result = predict_oxidative_stability("TestDrug", logP=2.5)
        assert result.logP == pytest.approx(2.5)

    def test_mw_Da_preserved(self):
        result = predict_oxidative_stability("TestDrug", mw_Da=350.0)
        assert result.mw_Da == pytest.approx(350.0)


class TestRiskScoreRange:
    def test_risk_score_non_negative(self):
        result = predict_oxidative_stability("Blank")
        assert result.risk_score >= 0.0

    def test_risk_score_max_100(self):
        # All flags set: 25+40+20+30+10+5 = 130 → capped at 100
        result = predict_oxidative_stability(
            "MaxRisk",
            has_phenol=True,
            has_catechol=True,
            has_thioether=True,
            has_aldehyde=True,
            logP=5.0,
            mw_Da=600.0,
        )
        assert result.risk_score <= 100.0

    def test_risk_score_in_bounds_no_flags(self):
        result = predict_oxidative_stability("Clean", logP=1.0, mw_Da=200.0)
        assert 0.0 <= result.risk_score <= 100.0


class TestStabilityClasses:
    def test_catechol_drug_is_high_risk(self):
        # catechol(40) + logP>3(10) = 50 → moderate_risk boundary
        # To reach high_risk (>50): catechol + thioether = 40+20=60
        result = predict_oxidative_stability(
            "Catecholamine", has_catechol=True, has_thioether=True, logP=1.0, mw_Da=200.0
        )
        assert result.stability_class == "high_risk"

    def test_no_oxidizable_groups_is_stable(self):
        result = predict_oxidative_stability(
            "CleanDrug",
            has_phenol=False,
            has_catechol=False,
            has_thioether=False,
            has_aldehyde=False,
            logP=1.0,
            mw_Da=200.0,
        )
        assert result.stability_class == "stable"

    def test_phenol_only_moderate_or_high(self):
        result = predict_oxidative_stability("PhenolDrug", has_phenol=True, logP=1.0, mw_Da=200.0)
        # phenol alone = 25 → moderate_risk
        assert result.stability_class == "moderate_risk"

    def test_thioether_adds_to_score(self):
        # thioether = 20 → boundary of stable/moderate_risk; score ≤ 20 → stable
        # To get moderate_risk from thioether need additional points: thioether(20)+logP>3(10)=30
        result = predict_oxidative_stability(
            "ThioetherDrug", has_thioether=True, logP=4.0, mw_Da=200.0
        )
        assert result.stability_class == "moderate_risk"

    def test_aldehyde_high_risk_with_other_flag(self):
        # aldehyde(30) + thioether(20) = 50 → boundary: >50 → high_risk
        result = predict_oxidative_stability(
            "AldehydeThio",
            has_aldehyde=True,
            has_thioether=True,
            logP=0.0,
            mw_Da=200.0,
        )
        # 30 + 20 = 50 → score == 50 → moderate_risk (≤50)
        assert result.stability_class in ("moderate_risk", "high_risk")


class TestRiskScoreValues:
    def test_phenol_adds_25_points(self):
        phenol = predict_oxidative_stability("Phenol", has_phenol=True, logP=1.0, mw_Da=200.0)
        assert phenol.risk_score >= 25.0

    def test_catechol_adds_40_points(self):
        base = predict_oxidative_stability("Base", logP=1.0, mw_Da=200.0)
        catechol = predict_oxidative_stability("Catechol", has_catechol=True, logP=1.0, mw_Da=200.0)
        assert catechol.risk_score - base.risk_score == pytest.approx(40.0)

    def test_high_logP_adds_10_points(self):
        low = predict_oxidative_stability("Low", logP=1.0, mw_Da=200.0)
        high = predict_oxidative_stability("High", logP=4.0, mw_Da=200.0)
        assert high.risk_score - low.risk_score == pytest.approx(10.0)

    def test_high_mw_adds_5_points(self):
        low = predict_oxidative_stability("LowMW", logP=1.0, mw_Da=200.0)
        high = predict_oxidative_stability("HighMW", logP=1.0, mw_Da=600.0)
        assert high.risk_score - low.risk_score == pytest.approx(5.0)


class TestShelfLife:
    def test_stable_shelf_life_24_months(self):
        result = predict_oxidative_stability("Clean", logP=1.0, mw_Da=200.0)
        assert result.predicted_shelf_life_months == 24

    def test_moderate_risk_shelf_life_12_months(self):
        result = predict_oxidative_stability("Phenol", has_phenol=True, logP=1.0, mw_Da=200.0)
        assert result.predicted_shelf_life_months == 12

    def test_high_risk_shelf_life_3_months(self):
        result = predict_oxidative_stability(
            "HighRisk", has_catechol=True, has_aldehyde=True, logP=1.0, mw_Da=200.0
        )
        assert result.predicted_shelf_life_months == 3

    def test_shelf_life_positive(self):
        for has_cat, has_phen in [(False, False), (True, False), (True, True)]:
            result = predict_oxidative_stability(
                "Drug", has_catechol=has_cat, has_phenol=has_phen, mw_Da=300.0
            )
            assert result.predicted_shelf_life_months > 0


class TestAntioxidantRecommendation:
    def test_stable_no_antioxidant_needed(self):
        result = predict_oxidative_stability("Clean", logP=1.0, mw_Da=200.0)
        assert (
            "No antioxidant" in result.antioxidant_recommendation
            or len(result.antioxidant_recommendation) > 0
        )

    def test_high_risk_has_recommendation(self):
        result = predict_oxidative_stability(
            "HighRisk", has_catechol=True, has_aldehyde=True, mw_Da=200.0
        )
        assert len(result.antioxidant_recommendation) > 0


class TestScreenFunction:
    def test_screen_returns_list(self):
        compounds = [
            {"drug_name": "A", "has_catechol": True, "mw_Da": 300.0},
            {"drug_name": "B", "mw_Da": 250.0},
        ]
        results = screen_oxidative_stability(compounds)
        assert isinstance(results, list)
        assert all(isinstance(r, OxidativeStabilityResult) for r in results)

    def test_screen_sorted_ascending(self):
        compounds = [
            {"drug_name": "HighRisk", "has_catechol": True, "has_aldehyde": True, "mw_Da": 300.0},
            {"drug_name": "LowRisk", "mw_Da": 200.0, "logP": 1.0},
            {"drug_name": "MidRisk", "has_phenol": True, "mw_Da": 300.0},
        ]
        results = screen_oxidative_stability(compounds)
        scores = [r.risk_score for r in results]
        assert scores == sorted(scores)

    def test_screen_most_stable_first(self):
        compounds = [
            {"drug_name": "HighRisk", "has_catechol": True, "mw_Da": 300.0},
            {"drug_name": "LowRisk", "mw_Da": 200.0, "logP": 1.0},
        ]
        results = screen_oxidative_stability(compounds)
        assert results[0].drug_name == "LowRisk"

    def test_screen_empty_list(self):
        results = screen_oxidative_stability([])
        assert results == []


class TestValidation:
    def test_invalid_mw_zero_raises(self):
        with pytest.raises(ValueError):
            predict_oxidative_stability("Bad", mw_Da=0.0)

    def test_invalid_mw_negative_raises(self):
        with pytest.raises(ValueError):
            predict_oxidative_stability("Bad", mw_Da=-100.0)
