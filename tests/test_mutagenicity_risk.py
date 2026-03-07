"""Tests for mutagenicity risk assessment module (Phase 690)."""

import math

import pytest

from omega_pbpk.risk.mutagenicity_risk import (
    MutagenicityResult,
    assess_mutagenicity_risk,
    calculate_td50_estimate,
    screen_mutagenicity,
)


class TestAssessMutagenicityRisk:
    def test_returns_mutagenicity_result(self):
        result = assess_mutagenicity_risk("TestCompound", {})
        assert isinstance(result, MutagenicityResult)

    def test_no_alerts_score_zero(self):
        result = assess_mutagenicity_risk("SafeDrug", {})
        assert result.mutagenicity_score == 0.0

    def test_no_alerts_risk_class_none(self):
        result = assess_mutagenicity_risk("SafeDrug", {})
        assert result.risk_class == "none"

    def test_epoxide_score_30(self):
        result = assess_mutagenicity_risk("Epoxide", {"n_epoxides": 1})
        assert math.isclose(result.mutagenicity_score, 30.0, rel_tol=1e-9)

    def test_epoxide_risk_class_moderate(self):
        result = assess_mutagenicity_risk("Epoxide", {"n_epoxides": 1})
        assert result.risk_class == "moderate"

    def test_multiple_alerts_accumulate(self):
        # epoxide=30 + aziridine=35 = 65 → high
        result = assess_mutagenicity_risk("Multi", {"n_epoxides": 1, "n_aziridines": 1})
        assert math.isclose(result.mutagenicity_score, 65.0, rel_tol=1e-9)
        assert result.risk_class == "high"

    def test_score_capped_at_100(self):
        # Very many alerts to exceed 100
        result = assess_mutagenicity_risk(
            "Toxic",
            {
                "n_alkylating_agents": 2,  # 80
                "n_epoxides": 2,  # 60
                "n_aziridines": 1,  # 35
            },
        )
        assert result.mutagenicity_score <= 100.0

    def test_requires_ames_test_true_when_score_gt_20(self):
        result = assess_mutagenicity_risk("Epoxide", {"n_epoxides": 1})
        # score=30 > 20
        assert result.requires_ames_test is True

    def test_requires_ames_test_false_when_score_le_20(self):
        result = assess_mutagenicity_risk("LowRisk", {"n_aldehydes": 1})
        # score=15 <= 20
        assert result.requires_ames_test is False

    def test_ames_prediction_likely_positive_when_gt_30(self):
        result = assess_mutagenicity_risk("Epoxide", {"n_epoxides": 1})
        # score=30: boundary — "likely_positive" if score > 30
        # score=30 is NOT > 30, so "likely_negative"
        assert result.ames_prediction == "likely_negative"

    def test_ames_prediction_likely_positive_when_score_above_30(self):
        result = assess_mutagenicity_risk("Alkylator", {"n_alkylating_agents": 1})
        # score=40 > 30
        assert result.ames_prediction == "likely_positive"

    def test_ames_prediction_likely_negative_when_low(self):
        result = assess_mutagenicity_risk("Low", {"n_aldehydes": 1})
        assert result.ames_prediction == "likely_negative"

    def test_primary_alert_most_impactful(self):
        # aziridine=35 wins over epoxide=30
        result = assess_mutagenicity_risk("Multi", {"n_epoxides": 1, "n_aziridines": 1})
        assert result.primary_alert == "aziridines"

    def test_primary_alert_none_when_no_alerts(self):
        result = assess_mutagenicity_risk("Safe", {})
        assert result.primary_alert == "none"

    def test_mw_gt_1000_reduces_score(self):
        # n_epoxides=1 → score=30 without MW; with MW>1000 → 30*0.5=15
        r_low_mw = assess_mutagenicity_risk("Small", {"n_epoxides": 1, "mw": 300.0})
        r_high_mw = assess_mutagenicity_risk("Large", {"n_epoxides": 1, "mw": 1500.0})
        assert r_high_mw.mutagenicity_score < r_low_mw.mutagenicity_score
        assert math.isclose(r_high_mw.mutagenicity_score, 15.0, rel_tol=1e-9)

    def test_aflatoxin_scaffold_high_risk(self):
        # Aflatoxin alone = +50 → boundary of "moderate" (21-50)
        result = assess_mutagenicity_risk("Aflatoxin", {"has_aflatoxin_scaffold": True})
        assert result.mutagenicity_score == 50.0
        assert result.risk_class == "moderate"

    def test_aflatoxin_plus_alert_very_high_risk(self):
        # Aflatoxin=50 + alkylating=40 = 90 → very_high
        result = assess_mutagenicity_risk(
            "AflatoxinToxic", {"has_aflatoxin_scaffold": True, "n_alkylating_agents": 1}
        )
        assert result.risk_class == "very_high"

    def test_n_structural_alerts_count(self):
        result = assess_mutagenicity_risk(
            "Multi",
            {"n_epoxides": 2, "n_nitro_groups": 1},
        )
        # 2 epoxides + 1 nitro = 3 total alerts
        assert result.n_structural_alerts == 3

    def test_notes_nonempty(self):
        result = assess_mutagenicity_risk("AnyDrug", {})
        assert len(result.notes) > 0

    def test_compound_name_stored(self):
        result = assess_mutagenicity_risk("TestDrugXYZ", {})
        assert result.compound_name == "TestDrugXYZ"

    def test_validation_negative_n_nitro_raises(self):
        with pytest.raises(ValueError):
            assess_mutagenicity_risk("Bad", {"n_nitro_groups": -1})

    def test_validation_negative_epoxides_raises(self):
        with pytest.raises(ValueError):
            assess_mutagenicity_risk("Bad", {"n_epoxides": -2})

    def test_validation_empty_name_raises(self):
        with pytest.raises(ValueError):
            assess_mutagenicity_risk("", {})

    def test_logp_lt_minus2_reduces_score(self):
        # n_epoxides=1 → 30; logP=-3 → 30*0.7=21
        r_normal = assess_mutagenicity_risk("Normal", {"n_epoxides": 1, "logP": 0.0})
        r_hydrophilic = assess_mutagenicity_risk("Hydro", {"n_epoxides": 1, "logP": -3.0})
        assert r_hydrophilic.mutagenicity_score < r_normal.mutagenicity_score
        assert math.isclose(r_hydrophilic.mutagenicity_score, 21.0, rel_tol=1e-9)


class TestScreenMutagenicity:
    def test_screen_empty_returns_empty(self):
        result = screen_mutagenicity([])
        assert result == []

    def test_screen_sorted_descending(self):
        compounds = [
            {"compound_name": "Low", "n_aldehydes": 1},
            {"compound_name": "High", "n_alkylating_agents": 1},
            {"compound_name": "None"},
        ]
        results = screen_mutagenicity(compounds)
        scores = [r.mutagenicity_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_screen_returns_list_of_results(self):
        compounds = [
            {"compound_name": "A"},
            {"compound_name": "B", "n_epoxides": 1},
        ]
        results = screen_mutagenicity(compounds)
        assert all(isinstance(r, MutagenicityResult) for r in results)

    def test_screen_correct_count(self):
        compounds = [
            {"compound_name": "A"},
            {"compound_name": "B"},
            {"compound_name": "C"},
        ]
        results = screen_mutagenicity(compounds)
        assert len(results) == 3


class TestCalculateTd50Estimate:
    def test_high_score_low_td50(self):
        td50_high = calculate_td50_estimate(80.0)
        td50_low = calculate_td50_estimate(10.0)
        assert td50_high < td50_low

    def test_score_zero_td50_approx_3162(self):
        """score=0, mw=300 → log(TD50)=3.5 → TD50 ≈ 3162."""
        td50 = calculate_td50_estimate(0.0, mw=300.0)
        assert math.isclose(td50, 10**3.5, rel_tol=1e-6)

    def test_returns_positive_float(self):
        td50 = calculate_td50_estimate(50.0)
        assert td50 > 0

    def test_minimum_value_enforced(self):
        """Very high score should not go below 0.001."""
        td50 = calculate_td50_estimate(100.0, mw=5000.0)
        assert td50 >= 0.001
