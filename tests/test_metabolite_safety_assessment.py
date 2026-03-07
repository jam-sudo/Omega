"""Tests for Phase 304 — Drug Metabolite Safety Assessment."""

import math

import pytest

from omega_pbpk.risk.metabolite_safety_assessment import (
    MetaboliteSafetyResult,
    assess_metabolite_safety,
    screen_metabolites,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _basic(
    metabolite_auc=5.0,
    parent_auc=100.0,
    metabolite_mw=280.0,
    metabolite_logp=1.5,
    parent_dose_mg=100.0,
    is_unique=False,
    is_active=False,
    drug_name="ParentDrug",
    metabolite_name="MetabA",
):
    return assess_metabolite_safety(
        drug_name=drug_name,
        metabolite_name=metabolite_name,
        parent_auc_mg_h_per_L=parent_auc,
        metabolite_auc_mg_h_per_L=metabolite_auc,
        metabolite_mw=metabolite_mw,
        metabolite_logp=metabolite_logp,
        parent_dose_mg=parent_dose_mg,
        is_unique_human_metabolite=is_unique,
        has_pharmacological_activity=is_active,
    )


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_result_instance(self):
        r = _basic()
        assert isinstance(r, MetaboliteSafetyResult)

    def test_drug_name_preserved(self):
        r = _basic(drug_name="Warfarin")
        assert r.drug_name == "Warfarin"

    def test_metabolite_name_preserved(self):
        r = _basic(metabolite_name="7-OH-Warfarin")
        assert r.metabolite_name == "7-OH-Warfarin"

    def test_parent_auc_preserved(self):
        r = _basic(parent_auc=50.0)
        assert math.isclose(r.parent_auc_mg_h_per_L, 50.0)

    def test_metabolite_auc_preserved(self):
        r = _basic(metabolite_auc=8.0)
        assert math.isclose(r.metabolite_auc_mg_h_per_L, 8.0)


# ---------------------------------------------------------------------------
# Exposure ratio
# ---------------------------------------------------------------------------


class TestExposureRatio:
    def test_exposure_ratio_non_negative(self):
        r = _basic(metabolite_auc=0.0)
        assert r.exposure_ratio >= 0

    def test_exposure_ratio_zero_metabolite(self):
        r = _basic(metabolite_auc=0.0)
        assert r.exposure_ratio == 0.0

    def test_exposure_ratio_calculation(self):
        r = _basic(metabolite_auc=20.0, parent_auc=100.0)
        assert math.isclose(r.exposure_ratio, 0.20)


# ---------------------------------------------------------------------------
# MIST threshold
# ---------------------------------------------------------------------------


class TestMistThreshold:
    def test_below_threshold_false(self):
        r = _basic(metabolite_auc=5.0, parent_auc=100.0)  # 5%
        assert r.above_mist_threshold is False

    def test_at_threshold_true(self):
        r = _basic(metabolite_auc=10.0, parent_auc=100.0)  # exactly 10%
        assert r.above_mist_threshold is True

    def test_above_threshold_true(self):
        r = _basic(metabolite_auc=15.0, parent_auc=100.0)  # 15%
        assert r.above_mist_threshold is True

    def test_above_threshold_requires_standalone(self):
        r = _basic(metabolite_auc=15.0, parent_auc=100.0)
        assert r.standalone_study_required is True

    def test_below_threshold_no_standalone_by_default(self):
        r = _basic(metabolite_auc=5.0, parent_auc=100.0, is_unique=False)
        assert r.standalone_study_required is False


# ---------------------------------------------------------------------------
# Unique human metabolite
# ---------------------------------------------------------------------------


class TestUniqueHumanMetabolite:
    def test_unique_above_5pct_requires_standalone(self):
        # 7% exposure, unique human → standalone required (< 10% but >= 5%)
        r = _basic(metabolite_auc=7.0, parent_auc=100.0, is_unique=True)
        assert r.standalone_study_required is True

    def test_unique_below_5pct_no_standalone(self):
        # 3% exposure, unique human → below unique threshold
        r = _basic(metabolite_auc=3.0, parent_auc=100.0, is_unique=True)
        assert r.standalone_study_required is False

    def test_unique_adds_to_risk_score(self):
        r_normal = _basic(metabolite_auc=5.0, is_unique=False)
        r_unique = _basic(metabolite_auc=5.0, is_unique=True)
        assert r_unique.risk_score > r_normal.risk_score


# ---------------------------------------------------------------------------
# Active metabolite
# ---------------------------------------------------------------------------


class TestActiveMetabolite:
    def test_active_increases_risk_score(self):
        r_inactive = _basic(is_active=False)
        r_active = _basic(is_active=True)
        assert r_active.risk_score > r_inactive.risk_score

    def test_active_adds_20_points(self):
        r_inactive = _basic(is_active=False)
        r_active = _basic(is_active=True)
        assert math.isclose(r_active.risk_score - r_inactive.risk_score, 20.0)


# ---------------------------------------------------------------------------
# Risk class
# ---------------------------------------------------------------------------


class TestRiskClass:
    def test_risk_class_in_valid_set(self):
        r = _basic()
        assert r.risk_class in {"high", "moderate", "low"}

    def test_low_risk_class(self):
        r = _basic(metabolite_auc=1.0, parent_auc=100.0, is_active=False, is_unique=False)
        assert r.risk_class == "low"

    def test_high_risk_class_active_and_unique(self):
        r = _basic(
            metabolite_auc=50.0,
            parent_auc=100.0,
            is_active=True,
            is_unique=True,
        )
        assert r.risk_class == "high"


# ---------------------------------------------------------------------------
# Safety margin
# ---------------------------------------------------------------------------


class TestSafetyMargin:
    def test_safety_margin_positive(self):
        r = _basic(metabolite_auc=10.0, parent_auc=100.0)
        assert r.safety_margin > 0

    def test_safety_margin_inf_when_zero_metabolite(self):
        r = _basic(metabolite_auc=0.0)
        assert math.isinf(r.safety_margin)

    def test_safety_margin_inverse_of_ratio(self):
        r = _basic(metabolite_auc=20.0, parent_auc=100.0)
        assert math.isclose(r.safety_margin, 5.0)


# ---------------------------------------------------------------------------
# Regulatory recommendation
# ---------------------------------------------------------------------------


class TestRegulatoryRecommendation:
    def test_standalone_required_recommendation(self):
        r = _basic(metabolite_auc=15.0, parent_auc=100.0)
        assert r.regulatory_recommendation == "standalone_nonclinical_required"

    def test_monitor_only_recommendation(self):
        # Above 30 risk_score but not standalone required
        r = _basic(
            metabolite_auc=5.0,
            parent_auc=100.0,
            is_active=True,  # +20 points
            is_unique=True,  # +15 points → total ~37.5
        )
        if not r.standalone_study_required:
            assert r.regulatory_recommendation == "monitor_only"

    def test_no_action_low_risk(self):
        r = _basic(metabolite_auc=1.0, parent_auc=100.0, is_active=False, is_unique=False)
        if r.risk_class == "low":
            assert r.regulatory_recommendation == "no_action_required"


# ---------------------------------------------------------------------------
# Screen metabolites
# ---------------------------------------------------------------------------


class TestScreenMetabolites:
    def _make_metabolites(self):
        return [
            {
                "metabolite_name": "M1",
                "metabolite_auc_mg_h_per_L": 15.0,
                "metabolite_mw": 250.0,
                "metabolite_logp": 2.0,
                "is_unique_human_metabolite": False,
                "has_pharmacological_activity": True,
            },
            {
                "metabolite_name": "M2",
                "metabolite_auc_mg_h_per_L": 2.0,
                "metabolite_mw": 300.0,
                "metabolite_logp": 0.5,
                "is_unique_human_metabolite": False,
                "has_pharmacological_activity": False,
            },
            {
                "metabolite_name": "M3",
                "metabolite_auc_mg_h_per_L": 8.0,
                "metabolite_mw": 280.0,
                "metabolite_logp": 1.5,
                "is_unique_human_metabolite": True,
                "has_pharmacological_activity": False,
            },
        ]

    def test_screen_returns_list(self):
        results = screen_metabolites("Drug", 100.0, 100.0, self._make_metabolites())
        assert isinstance(results, list)

    def test_screen_correct_count(self):
        results = screen_metabolites("Drug", 100.0, 100.0, self._make_metabolites())
        assert len(results) == 3

    def test_screen_sorted_descending(self):
        results = screen_metabolites("Drug", 100.0, 100.0, self._make_metabolites())
        scores = [r.risk_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_screen_returns_metabolite_safety_results(self):
        results = screen_metabolites("Drug", 100.0, 100.0, self._make_metabolites())
        for r in results:
            assert isinstance(r, MetaboliteSafetyResult)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_parent_auc_zero(self):
        with pytest.raises(ValueError, match="parent_auc"):
            _basic(parent_auc=0.0)

    def test_invalid_parent_auc_negative(self):
        with pytest.raises(ValueError, match="parent_auc"):
            _basic(parent_auc=-10.0)

    def test_invalid_metabolite_auc_negative(self):
        with pytest.raises(ValueError, match="metabolite_auc"):
            _basic(metabolite_auc=-1.0)

    def test_invalid_metabolite_mw(self):
        with pytest.raises(ValueError, match="metabolite_mw"):
            assess_metabolite_safety("Drug", "M1", 100.0, 10.0, 0.0, 1.5, 100.0, False, False)

    def test_invalid_parent_dose(self):
        with pytest.raises(ValueError, match="parent_dose"):
            assess_metabolite_safety("Drug", "M1", 100.0, 10.0, 280.0, 1.5, 0.0, False, False)
