"""Tests for risk/metabolite_safety.py — Expanded MIST assessment (Phase 243)."""

from __future__ import annotations

import pytest

from omega_pbpk.risk.metabolite_safety import (
    FDA_MIST_THRESHOLD_PCT,
    MISTResult,
    assess_metabolite_mist,
    calculate_metabolite_burden,
    screen_metabolites,
)


# ---------------------------------------------------------------------------
# Basic ratio and qualification tests
# ---------------------------------------------------------------------------


class TestAssessMetaboliteMist:
    def test_ratio_below_threshold_not_qualified(self):
        """Metabolite at 5% of parent AUC should not require qualification."""
        result = assess_metabolite_mist(
            parent_name="Drug_A",
            metabolite_name="Met_A1",
            auc_parent_human=1000.0,
            auc_metabolite_human=50.0,  # 5%
        )
        assert result.requires_qualification is False
        assert result.metabolite_to_parent_ratio == pytest.approx(5.0, abs=0.01)

    def test_ratio_above_threshold_requires_qualification(self):
        """Metabolite at 15% of parent AUC should require qualification."""
        result = assess_metabolite_mist(
            parent_name="Drug_B",
            metabolite_name="Met_B1",
            auc_parent_human=1000.0,
            auc_metabolite_human=150.0,  # 15%
        )
        assert result.requires_qualification is True
        assert result.metabolite_to_parent_ratio == pytest.approx(15.0, abs=0.01)

    def test_ratio_exactly_at_threshold_not_qualified(self):
        """Metabolite at exactly 10% of parent AUC is NOT above threshold."""
        result = assess_metabolite_mist(
            parent_name="Drug_C",
            metabolite_name="Met_C1",
            auc_parent_human=1000.0,
            auc_metabolite_human=100.0,  # exactly 10%
        )
        assert result.requires_qualification is False

    def test_ratio_just_above_threshold_qualified(self):
        """Metabolite at 10.1% should require qualification."""
        result = assess_metabolite_mist(
            parent_name="Drug_D",
            metabolite_name="Met_D1",
            auc_parent_human=1000.0,
            auc_metabolite_human=101.0,  # 10.1%
        )
        assert result.requires_qualification is True

    def test_fda_threshold_constant(self):
        """FDA threshold should be 10.0%."""
        assert FDA_MIST_THRESHOLD_PCT == pytest.approx(10.0)

    def test_result_threshold_field(self):
        """Result should carry the threshold value."""
        result = assess_metabolite_mist("P", "M", 1000.0, 50.0)
        assert result.fda_mist_threshold_pct == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Disproportionality tests
# ---------------------------------------------------------------------------


class TestDisproportionality:
    def test_disproportionate_human_rat_below_threshold(self):
        """Human >10%, rat <10% → disproportionate_in_humans=True."""
        result = assess_metabolite_mist(
            parent_name="Drug_E",
            metabolite_name="Met_E1",
            auc_parent_human=1000.0,
            auc_metabolite_human=200.0,  # 20% human
            auc_metabolite_rat=50.0,  # 5% rat (below threshold)
        )
        assert result.requires_qualification is True
        assert result.disproportionate_in_humans is True
        assert result.rat_metabolite_ratio == pytest.approx(5.0, abs=0.01)

    def test_not_disproportionate_both_above_threshold(self):
        """Human >10% and rat >10% → disproportionate_in_humans=False."""
        result = assess_metabolite_mist(
            parent_name="Drug_F",
            metabolite_name="Met_F1",
            auc_parent_human=1000.0,
            auc_metabolite_human=200.0,  # 20% human
            auc_metabolite_rat=150.0,  # 15% rat (above threshold)
        )
        assert result.requires_qualification is True
        assert result.disproportionate_in_humans is False

    def test_disproportionate_dog_below_threshold(self):
        """Human >10%, dog <10% → disproportionate=True."""
        result = assess_metabolite_mist(
            parent_name="Drug_G",
            metabolite_name="Met_G1",
            auc_parent_human=1000.0,
            auc_metabolite_human=300.0,  # 30% human
            auc_metabolite_dog=80.0,  # 8% dog (below threshold)
        )
        assert result.disproportionate_in_humans is True
        assert result.dog_metabolite_ratio == pytest.approx(8.0, abs=0.01)

    def test_no_animal_data_not_disproportionate(self):
        """Without animal data, disproportionate_in_humans=False even if >10%."""
        result = assess_metabolite_mist(
            parent_name="Drug_H",
            metabolite_name="Met_H1",
            auc_parent_human=1000.0,
            auc_metabolite_human=200.0,
        )
        assert result.requires_qualification is True
        assert result.disproportionate_in_humans is False
        assert result.rat_metabolite_ratio is None
        assert result.dog_metabolite_ratio is None

    def test_rat_and_dog_both_below_threshold(self):
        """Human >10%, both rat and dog <10% → disproportionate=True."""
        result = assess_metabolite_mist(
            parent_name="Drug_I",
            metabolite_name="Met_I1",
            auc_parent_human=1000.0,
            auc_metabolite_human=250.0,  # 25% human
            auc_metabolite_rat=30.0,  # 3% rat
            auc_metabolite_dog=50.0,  # 5% dog
        )
        assert result.disproportionate_in_humans is True

    def test_below_threshold_not_disproportionate_regardless_of_animals(self):
        """Human <10% → disproportionate=False regardless of animal data."""
        result = assess_metabolite_mist(
            parent_name="Drug_J",
            metabolite_name="Met_J1",
            auc_parent_human=1000.0,
            auc_metabolite_human=50.0,  # 5% human
            auc_metabolite_rat=10.0,  # 1% rat
        )
        assert result.requires_qualification is False
        assert result.disproportionate_in_humans is False


# ---------------------------------------------------------------------------
# Recommendation and notes tests
# ---------------------------------------------------------------------------


class TestRecommendation:
    def test_recommendation_populated(self):
        """Recommendation string should always be non-empty."""
        result = assess_metabolite_mist("P", "M", 1000.0, 150.0)
        assert isinstance(result.recommendation, str)
        assert len(result.recommendation) > 0

    def test_recommendation_below_threshold(self):
        """Below-threshold recommendation should mention no qualification needed."""
        result = assess_metabolite_mist("P", "M", 1000.0, 50.0)
        assert "no additional safety qualification" in result.recommendation.lower()

    def test_recommendation_disproportionate_mentions_studies(self):
        """Disproportionate metabolite recommendation should mention studies."""
        result = assess_metabolite_mist("P", "M", 1000.0, 200.0, auc_metabolite_rat=50.0)
        assert result.disproportionate_in_humans is True
        assert (
            "dedicated" in result.recommendation.lower()
            or "studies" in result.recommendation.lower()
        )

    def test_daily_dose_adds_burden_note(self):
        """Providing daily_dose_mg should add a burden note."""
        result = assess_metabolite_mist("P", "M", 1000.0, 200.0, daily_dose_mg=100.0)
        assert any("burden" in n.lower() for n in result.notes)

    def test_no_daily_dose_no_burden_note(self):
        """Without daily_dose_mg, no burden note added."""
        result = assess_metabolite_mist("P", "M", 1000.0, 200.0)
        assert not any("burden" in n.lower() for n in result.notes)


# ---------------------------------------------------------------------------
# screen_metabolites tests
# ---------------------------------------------------------------------------


class TestScreenMetabolites:
    def test_sorted_by_ratio_descending(self):
        """Results should be sorted by metabolite_to_parent_ratio descending."""
        mets = [
            {"name": "MetA", "auc_human": 50.0},  # 5%
            {"name": "MetB", "auc_human": 300.0},  # 30%
            {"name": "MetC", "auc_human": 150.0},  # 15%
        ]
        results = screen_metabolites("Drug", mets, 1000.0)
        ratios = [r.metabolite_to_parent_ratio for r in results]
        assert ratios == sorted(ratios, reverse=True)

    def test_returns_correct_count(self):
        """Should return same number of results as metabolites."""
        mets = [{"name": f"Met{i}", "auc_human": float(i * 50)} for i in range(1, 5)]
        results = screen_metabolites("Drug", mets, 1000.0)
        assert len(results) == 4

    def test_with_optional_animal_data(self):
        """Metabolites with rat/dog AUC should have those ratios set."""
        mets = [
            {"name": "MetA", "auc_human": 200.0, "auc_rat": 50.0, "auc_dog": 80.0},
        ]
        results = screen_metabolites("Drug", mets, 1000.0)
        assert results[0].rat_metabolite_ratio is not None
        assert results[0].dog_metabolite_ratio is not None


# ---------------------------------------------------------------------------
# calculate_metabolite_burden tests
# ---------------------------------------------------------------------------


class TestCalculateMetaboliteBurden:
    def test_burden_formula_correct(self):
        """burden = (auc_met / auc_parent) * dose_mg."""
        burden = calculate_metabolite_burden(
            auc_parent_human=1000.0,
            auc_metabolite_human=200.0,
            dose_mg=100.0,
        )
        expected = (200.0 / 1000.0) * 100.0  # = 20.0
        assert burden == pytest.approx(expected, rel=1e-6)

    def test_burden_at_10pct_ratio(self):
        """10% metabolite with 100 mg dose → 10 mg-equiv/day."""
        burden = calculate_metabolite_burden(1000.0, 100.0, 100.0)
        assert burden == pytest.approx(10.0, rel=1e-6)

    def test_burden_proportional_to_dose(self):
        """Doubling the dose doubles the burden."""
        b1 = calculate_metabolite_burden(1000.0, 200.0, 50.0)
        b2 = calculate_metabolite_burden(1000.0, 200.0, 100.0)
        assert b2 == pytest.approx(b1 * 2.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Validation error tests
# ---------------------------------------------------------------------------


class TestValidation:
    def test_non_positive_parent_auc_raises(self):
        with pytest.raises(ValueError, match="auc_parent_human"):
            assess_metabolite_mist("P", "M", 0.0, 50.0)

    def test_negative_parent_auc_raises(self):
        with pytest.raises(ValueError, match="auc_parent_human"):
            assess_metabolite_mist("P", "M", -100.0, 50.0)

    def test_non_positive_metabolite_auc_raises(self):
        with pytest.raises(ValueError, match="auc_metabolite_human"):
            assess_metabolite_mist("P", "M", 1000.0, 0.0)

    def test_non_positive_rat_auc_raises(self):
        with pytest.raises(ValueError, match="auc_metabolite_rat"):
            assess_metabolite_mist("P", "M", 1000.0, 50.0, auc_metabolite_rat=0.0)

    def test_non_positive_dog_auc_raises(self):
        with pytest.raises(ValueError, match="auc_metabolite_dog"):
            assess_metabolite_mist("P", "M", 1000.0, 50.0, auc_metabolite_dog=-1.0)

    def test_non_positive_dose_raises(self):
        with pytest.raises(ValueError, match="daily_dose_mg"):
            assess_metabolite_mist("P", "M", 1000.0, 50.0, daily_dose_mg=0.0)

    def test_screen_non_positive_parent_auc_raises(self):
        with pytest.raises(ValueError, match="auc_parent_human"):
            screen_metabolites("P", [{"name": "M", "auc_human": 50.0}], 0.0)

    def test_burden_non_positive_parent_raises(self):
        with pytest.raises(ValueError, match="auc_parent_human"):
            calculate_metabolite_burden(0.0, 50.0, 100.0)

    def test_burden_non_positive_met_raises(self):
        with pytest.raises(ValueError, match="auc_metabolite_human"):
            calculate_metabolite_burden(1000.0, 0.0, 100.0)

    def test_burden_non_positive_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            calculate_metabolite_burden(1000.0, 50.0, 0.0)


# ---------------------------------------------------------------------------
# Result field type tests
# ---------------------------------------------------------------------------


class TestResultFieldTypes:
    def test_result_is_mist_result_instance(self):
        result = assess_metabolite_mist("P", "M", 1000.0, 150.0)
        assert isinstance(result, MISTResult)

    def test_ratio_is_float(self):
        result = assess_metabolite_mist("P", "M", 1000.0, 150.0)
        assert isinstance(result.metabolite_to_parent_ratio, float)

    def test_flags_are_bool(self):
        result = assess_metabolite_mist("P", "M", 1000.0, 150.0)
        assert isinstance(result.requires_qualification, bool)
        assert isinstance(result.disproportionate_in_humans, bool)

    def test_notes_is_list(self):
        result = assess_metabolite_mist("P", "M", 1000.0, 150.0)
        assert isinstance(result.notes, list)

    def test_optional_animal_ratios_none_when_not_provided(self):
        result = assess_metabolite_mist("P", "M", 1000.0, 150.0)
        assert result.auc_rat is None
        assert result.auc_dog is None
        assert result.rat_metabolite_ratio is None
        assert result.dog_metabolite_ratio is None
