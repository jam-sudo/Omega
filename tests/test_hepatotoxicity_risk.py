"""Tests for Phase 582 - Drug Hepatotoxicity Risk Score."""

import pytest

from omega_pbpk.risk.hepatotoxicity_risk import (
    predict_hepatotoxicity_risk,
    screen_hepatotoxicity,
)


class TestValidation:
    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="daily_dose_mg"):
            predict_hepatotoxicity_risk("Drug", 2.0, 300, 0)

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="daily_dose_mg"):
            predict_hepatotoxicity_risk("Drug", 2.0, 300, -10)

    def test_zero_mw_raises(self):
        with pytest.raises(ValueError, match="mw_Da"):
            predict_hepatotoxicity_risk("Drug", 2.0, 0, 100)

    def test_negative_mw_raises(self):
        with pytest.raises(ValueError, match="mw_Da"):
            predict_hepatotoxicity_risk("Drug", 2.0, -100, 100)


class TestBSEPAlert:
    def test_explicit_bsep_alert(self):
        result = predict_hepatotoxicity_risk("Drug", 1.0, 200, 10, bsep_alert=True)
        assert result.bile_salt_export_pump_alert is True

    def test_auto_bsep_mw_and_logP(self):
        result = predict_hepatotoxicity_risk("Drug", 3.0, 500, 10)
        assert result.bile_salt_export_pump_alert is True

    def test_no_auto_bsep_low_mw(self):
        result = predict_hepatotoxicity_risk("Drug", 3.0, 300, 10)
        assert result.bile_salt_export_pump_alert is False

    def test_no_auto_bsep_low_logP(self):
        result = predict_hepatotoxicity_risk("Drug", 1.5, 500, 10)
        assert result.bile_salt_export_pump_alert is False


class TestScoreComponents:
    def test_dose_points_capped_at_30(self):
        result = predict_hepatotoxicity_risk("Drug", 0.0, 200, 500)
        # dose_pts=30, lipo_pts=0, burden_pts=min(20,500*1/100)=5
        assert result.risk_score == pytest.approx(35.0)

    def test_lipo_points_capped_at_20(self):
        result = predict_hepatotoxicity_risk("Drug", 10.0, 200, 1)
        # dose_pts=0.3, lipo_pts=20, burden_pts=min(20,1*10/100)=0.1
        assert result.risk_score == pytest.approx(20.4)

    def test_reactive_metabolite_adds_15(self):
        r1 = predict_hepatotoxicity_risk("Drug", 1.0, 200, 10)
        r2 = predict_hepatotoxicity_risk("Drug", 1.0, 200, 10, reactive_metabolite_alert=True)
        assert abs(r2.risk_score - r1.risk_score - 15.0) < 0.01

    def test_mitochondrial_adds_10(self):
        r1 = predict_hepatotoxicity_risk("Drug", 1.0, 200, 10)
        r2 = predict_hepatotoxicity_risk("Drug", 1.0, 200, 10, mitochondrial_alert=True)
        assert abs(r2.risk_score - r1.risk_score - 10.0) < 0.01

    def test_bsep_adds_10(self):
        r1 = predict_hepatotoxicity_risk("Drug", 1.0, 200, 10)
        r2 = predict_hepatotoxicity_risk("Drug", 1.0, 200, 10, bsep_alert=True)
        assert abs(r2.risk_score - r1.risk_score - 10.0) < 0.01

    def test_score_capped_at_100(self):
        result = predict_hepatotoxicity_risk(
            "Drug",
            10.0,
            500,
            1000,
            reactive_metabolite_alert=True,
            mitochondrial_alert=True,
            bsep_alert=True,
        )
        assert result.risk_score == 100.0

    def test_score_floor_at_0(self):
        result = predict_hepatotoxicity_risk("Drug", -5.0, 200, 0.01)
        assert result.risk_score >= 0.0


class TestDILIClass:
    def test_most_concern(self):
        result = predict_hepatotoxicity_risk("Drug", 5.0, 500, 200, reactive_metabolite_alert=True)
        assert result.dili_class == "most_concern"

    def test_less_concern(self):
        # dose_pts=30, lipo_pts=5, burden_pts=2 => 37 => less_concern
        result = predict_hepatotoxicity_risk("Drug", 2.0, 200, 100)
        assert result.dili_class == "less_concern"

    def test_no_concern(self):
        result = predict_hepatotoxicity_risk("Drug", 0.5, 200, 5)
        assert result.dili_class == "no_concern"


class TestSeverity:
    def test_fatal(self):
        result = predict_hepatotoxicity_risk(
            "Drug",
            5.0,
            500,
            300,
            reactive_metabolite_alert=True,
            mitochondrial_alert=True,
        )
        assert result.severity_prediction == "fatal"

    def test_mild(self):
        result = predict_hepatotoxicity_risk("Drug", 0.5, 200, 5)
        assert result.severity_prediction == "mild"


class TestMonitoring:
    def test_most_concern_monitoring(self):
        result = predict_hepatotoxicity_risk("Drug", 5.0, 500, 200, reactive_metabolite_alert=True)
        assert result.monitoring_recommendation == "monthly LFT monitoring for first year"

    def test_less_concern_monitoring(self):
        result = predict_hepatotoxicity_risk("Drug", 2.0, 200, 100)
        assert result.monitoring_recommendation == "quarterly LFT monitoring"

    def test_no_concern_monitoring(self):
        result = predict_hepatotoxicity_risk("Drug", 0.5, 200, 5)
        assert result.monitoring_recommendation == "routine annual monitoring"


class TestTotalBodyBurden:
    def test_burden_with_high_logP(self):
        result = predict_hepatotoxicity_risk("Drug", 4.0, 300, 100)
        assert result.total_body_burden_mg == pytest.approx(400.0)

    def test_burden_with_low_logP(self):
        result = predict_hepatotoxicity_risk("Drug", 0.5, 300, 100)
        assert result.total_body_burden_mg == pytest.approx(100.0)


class TestScreenHepatotoxicity:
    def test_sorted_by_risk_descending(self):
        compounds = [
            {"drug_name": "A", "logP": 0.5, "mw_Da": 200, "daily_dose_mg": 5},
            {
                "drug_name": "B",
                "logP": 5.0,
                "mw_Da": 500,
                "daily_dose_mg": 500,
                "reactive_metabolite_alert": True,
            },
            {"drug_name": "C", "logP": 2.0, "mw_Da": 300, "daily_dose_mg": 50},
        ]
        results = screen_hepatotoxicity(compounds)
        scores = [r.risk_score for r in results]
        assert scores == sorted(scores, reverse=True)
        assert results[0].drug_name == "B"

    def test_empty_list(self):
        assert screen_hepatotoxicity([]) == []


class TestResultDataclass:
    def test_frozen(self):
        result = predict_hepatotoxicity_risk("Drug", 2.0, 300, 100)
        with pytest.raises(AttributeError):
            result.drug_name = "Other"

    def test_drug_name_preserved(self):
        result = predict_hepatotoxicity_risk("TestDrug", 2.0, 300, 100)
        assert result.drug_name == "TestDrug"
