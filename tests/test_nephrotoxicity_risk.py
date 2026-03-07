"""Tests for nephrotoxicity risk assessment module (Phase 725)."""

import pytest

from omega_pbpk.risk.nephrotoxicity_risk import (
    NephrotoxicityResult,
    assess_nephrotoxicity_risk,
    estimate_renal_excretion_fraction,
    screen_nephrotoxicity,
)


class TestAssessNephrotoxicityRisk:
    def test_returns_nephrotoxicity_result(self):
        result = assess_nephrotoxicity_risk("SafeDrug", logP=1.0, mw=300.0)
        assert isinstance(result, NephrotoxicityResult)

    def test_no_risk_factors_score_low(self):
        # No risk factors — base score should be 0 (no additive factors)
        result = assess_nephrotoxicity_risk(
            "SafeDrug",
            logP=1.0,
            mw=250.0,
            fup=0.5,
            plasma_conc_uM=1.0,
        )
        assert result.nephrotoxicity_score == 0.0

    def test_nephrotoxin_class_score_gte_40(self):
        result = assess_nephrotoxicity_risk(
            "Cisplatin",
            logP=0.5,
            mw=300.0,
            is_nephrotoxin_class=True,
        )
        assert result.nephrotoxicity_score >= 40.0

    def test_score_in_range_0_100(self):
        result = assess_nephrotoxicity_risk(
            "ToxicDrug",
            logP=5.0,
            mw=400.0,
            is_nephrotoxin_class=True,
            n_reactive_metabolite_alerts=3,
            cl_renal_L_per_h=20.0,
        )
        assert 0.0 <= result.nephrotoxicity_score <= 100.0

    def test_risk_class_minimal_for_low_score(self):
        result = assess_nephrotoxicity_risk("SafeDrug", logP=1.0, mw=200.0, fup=0.5)
        assert result.risk_class == "minimal"

    def test_risk_class_in_valid_set(self):
        valid = {"minimal", "low", "moderate", "high", "very_high"}
        result = assess_nephrotoxicity_risk("AnyDrug", logP=2.0, mw=300.0)
        assert result.risk_class in valid

    def test_c_kidney_greater_than_plasma(self):
        result = assess_nephrotoxicity_risk(
            "Drug",
            logP=1.0,
            mw=300.0,
            fup=0.5,
            plasma_conc_uM=1.0,
        )
        # kidney concentration should be >> plasma
        assert result.c_kidney_uM > 1.0

    def test_basic_drug_higher_c_kidney_than_neutral(self):
        neutral = assess_nephrotoxicity_risk(
            "NeutralDrug",
            logP=2.0,
            mw=300.0,
            pka_base=None,
            fup=0.1,
            plasma_conc_uM=1.0,
        )
        basic = assess_nephrotoxicity_risk(
            "BasicDrug",
            logP=2.0,
            mw=300.0,
            pka_base=10.0,
            fup=0.1,
            plasma_conc_uM=1.0,
        )
        assert basic.c_kidney_uM > neutral.c_kidney_uM

    def test_requires_renal_monitoring_true_when_score_gt_30(self):
        result = assess_nephrotoxicity_risk(
            "MonitorDrug",
            logP=2.0,
            mw=300.0,
            is_nephrotoxin_class=True,
        )
        # score >= 40 → requires monitoring
        assert result.requires_renal_monitoring is True

    def test_requires_renal_monitoring_false_when_score_le_30(self):
        result = assess_nephrotoxicity_risk("SafeDrug", logP=1.0, mw=200.0, fup=0.5)
        assert result.requires_renal_monitoring is False

    def test_renal_safety_margin_positive(self):
        result = assess_nephrotoxicity_risk("Drug", logP=2.0, mw=300.0)
        assert result.renal_safety_margin > 0

    def test_renal_safety_margin_formula(self):
        result = assess_nephrotoxicity_risk(
            "Drug",
            logP=2.0,
            mw=300.0,
            is_nephrotoxin_class=True,
        )
        expected_margin = 100.0 / max(1.0, result.nephrotoxicity_score)
        assert abs(result.renal_safety_margin - expected_margin) < 1e-4

    def test_reactive_metabolite_increases_score(self):
        base = assess_nephrotoxicity_risk("Drug", logP=1.0, mw=300.0)
        with_rm = assess_nephrotoxicity_risk(
            "DrugRM", logP=1.0, mw=300.0, n_reactive_metabolite_alerts=1
        )
        assert with_rm.nephrotoxicity_score > base.nephrotoxicity_score

    def test_reactive_metabolite_capped_at_30_contribution(self):
        result1 = assess_nephrotoxicity_risk(
            "Drug2RM", logP=1.0, mw=300.0, n_reactive_metabolite_alerts=2
        )
        result3 = assess_nephrotoxicity_risk(
            "Drug3RM", logP=1.0, mw=300.0, n_reactive_metabolite_alerts=3
        )
        # 2 alerts → 30; 3 alerts → also capped at 30 (same score)
        assert result1.nephrotoxicity_score == result3.nephrotoxicity_score

    def test_high_logp_increases_score(self):
        low_logp = assess_nephrotoxicity_risk("Drug", logP=1.0, mw=300.0)
        high_logp = assess_nephrotoxicity_risk("DrugHighLogP", logP=5.0, mw=300.0)
        assert high_logp.nephrotoxicity_score > low_logp.nephrotoxicity_score

    def test_notes_nonempty(self):
        result = assess_nephrotoxicity_risk("Drug", logP=2.0, mw=300.0)
        assert len(result.notes) > 0

    def test_primary_mechanism_known_nephrotoxin(self):
        result = assess_nephrotoxicity_risk(
            "Aminoglycoside", logP=1.0, mw=400.0, is_nephrotoxin_class=True
        )
        assert result.primary_mechanism == "direct_nephrotoxicity"

    def test_primary_mechanism_unknown_for_safe_drug(self):
        result = assess_nephrotoxicity_risk("SafeDrug", logP=1.0, mw=200.0)
        assert result.primary_mechanism == "unknown"

    def test_validation_fup_gt_1_raises(self):
        with pytest.raises(ValueError):
            assess_nephrotoxicity_risk("Drug", logP=1.0, mw=300.0, fup=1.5)

    def test_validation_fup_zero_raises(self):
        with pytest.raises(ValueError):
            assess_nephrotoxicity_risk("Drug", logP=1.0, mw=300.0, fup=0.0)

    def test_validation_negative_plasma_conc_raises(self):
        with pytest.raises(ValueError):
            assess_nephrotoxicity_risk("Drug", logP=1.0, mw=300.0, plasma_conc_uM=-1.0)


class TestScreenNephrotoxicity:
    def test_sorted_descending(self):
        compounds = [
            {"compound_name": "SafeDrug", "logP": 1.0, "mw": 200.0, "fup": 0.5},
            {
                "compound_name": "ToxicDrug",
                "logP": 5.0,
                "mw": 400.0,
                "is_nephrotoxin_class": True,
            },
            {
                "compound_name": "ModDrug",
                "logP": 3.0,
                "mw": 300.0,
                "n_reactive_metabolite_alerts": 1,
            },
        ]
        results = screen_nephrotoxicity(compounds)
        scores = [r.nephrotoxicity_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_returns_list(self):
        results = screen_nephrotoxicity([{"compound_name": "Drug", "logP": 2.0, "mw": 300.0}])
        assert isinstance(results, list)
        assert len(results) == 1


class TestEstimateRenalExcretionFraction:
    def test_basic_calculation(self):
        fe = estimate_renal_excretion_fraction(5.0, 10.0)
        assert abs(fe - 0.5) < 1e-9

    def test_fully_renal(self):
        fe = estimate_renal_excretion_fraction(10.0, 10.0)
        assert abs(fe - 1.0) < 1e-9

    def test_zero_renal(self):
        fe = estimate_renal_excretion_fraction(0.0, 10.0)
        assert abs(fe - 0.0) < 1e-9

    def test_invalid_total_cl_raises(self):
        with pytest.raises((ValueError, ZeroDivisionError)):
            estimate_renal_excretion_fraction(5.0, 0.0)
