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


# ---------------------------------------------------------------------------
# Phase 323 — metabolite_human_exposure_fraction
# ---------------------------------------------------------------------------


from omega_pbpk.risk.metabolite_safety import (  # noqa: E402
    MetaboliteSafetyResult,
    classify_metabolite_safety,
    metabolite_battery_screen,
    metabolite_human_exposure_fraction,
    predict_metabolite_toxicity_risk,
)


class TestMetaboliteHumanExposureFraction:
    def test_basic_calculation(self):
        """Fraction = met / (met + parent)."""
        frac = metabolite_human_exposure_fraction(100.0, 900.0)
        assert frac == pytest.approx(100.0 / 1000.0, rel=1e-6)

    def test_equal_auc(self):
        frac = metabolite_human_exposure_fraction(500.0, 500.0)
        assert frac == pytest.approx(0.5, rel=1e-6)

    def test_below_10pct(self):
        frac = metabolite_human_exposure_fraction(50.0, 950.0)
        assert frac < 0.10

    def test_above_10pct(self):
        frac = metabolite_human_exposure_fraction(200.0, 800.0)
        assert frac == pytest.approx(0.20, rel=1e-6)

    def test_zero_metabolite_raises(self):
        with pytest.raises(ValueError, match="auc_metabolite"):
            metabolite_human_exposure_fraction(0.0, 1000.0)

    def test_zero_parent_raises(self):
        with pytest.raises(ValueError, match="auc_parent"):
            metabolite_human_exposure_fraction(100.0, 0.0)


class TestClassifyMetaboliteSafety:
    def test_low_risk_no_alert(self):
        result = classify_metabolite_safety("DrugX", "MetX", 50.0, 1000.0, fm_formation=0.1)
        assert result.risk_category == "low"
        assert result.requires_qualification is False
        assert result.mist_flag is False

    def test_mist_flag_above_10pct(self):
        result = classify_metabolite_safety("DrugX", "MetY", 150.0, 1000.0, fm_formation=0.2)
        assert result.mist_flag is True
        assert result.risk_category == "moderate"
        assert result.requires_qualification is True

    def test_high_risk_above_25pct(self):
        result = classify_metabolite_safety("DrugX", "MetZ", 300.0, 1000.0, fm_formation=0.4)
        assert result.risk_category == "high"
        assert result.requires_qualification is True

    def test_structural_alert_raises_to_moderate(self):
        result = classify_metabolite_safety(
            "DrugX", "MetA", 50.0, 1000.0, fm_formation=0.05, structural_alert=True
        )
        assert result.risk_category in ("moderate", "high")
        assert result.requires_qualification is True

    def test_structural_alert_high_dose_raises_to_high(self):
        result = classify_metabolite_safety(
            "DrugX",
            "MetB",
            50.0,
            1000.0,
            fm_formation=0.05,
            structural_alert=True,
            daily_dose_mg=200.0,
        )
        assert result.risk_category == "high"

    def test_exposure_fraction_stored(self):
        result = classify_metabolite_safety("DrugX", "MetC", 200.0, 1000.0, fm_formation=0.2)
        assert result.exposure_fraction == pytest.approx(0.2, rel=1e-6)

    def test_result_is_frozen(self):
        result = classify_metabolite_safety("DrugX", "MetC", 100.0, 1000.0, fm_formation=0.1)
        with pytest.raises((AttributeError, TypeError)):
            result.risk_category = "low"  # type: ignore[misc]

    def test_invalid_auc_metabolite(self):
        with pytest.raises(ValueError, match="auc_metabolite"):
            classify_metabolite_safety("D", "M", 0.0, 1000.0, fm_formation=0.1)

    def test_invalid_auc_parent(self):
        with pytest.raises(ValueError, match="auc_parent"):
            classify_metabolite_safety("D", "M", 100.0, 0.0, fm_formation=0.1)

    def test_invalid_fm_formation(self):
        with pytest.raises(ValueError, match="fm_formation"):
            classify_metabolite_safety("D", "M", 100.0, 1000.0, fm_formation=1.5)

    def test_invalid_daily_dose(self):
        with pytest.raises(ValueError, match="daily_dose_mg"):
            classify_metabolite_safety(
                "D", "M", 100.0, 1000.0, fm_formation=0.1, daily_dose_mg=-1.0
            )


class TestPredictMetaboliteToxicityRisk:
    def test_low_burden(self):
        result = predict_metabolite_toxicity_risk("Met1", "CC", 200.0, 50.0, auc_fraction=0.05)
        assert result.risk_category == "low"
        assert result.toxic_burden < 0.5

    def test_moderate_burden(self):
        result = predict_metabolite_toxicity_risk("Met2", "CC", 200.0, 100.0, auc_fraction=0.5)
        # burden = 0.5 * 100 / 100 * 1 = 0.5 → moderate (borderline)
        assert result.risk_category in ("low", "moderate")

    def test_high_burden_with_alerts(self):
        result = predict_metabolite_toxicity_risk(
            "Met3",
            "CC",
            200.0,
            500.0,
            auc_fraction=0.8,
            structural_alerts=["furan", "aniline", "epoxide_risk"],
        )
        assert result.risk_category == "high"
        assert result.n_alerts == 3

    def test_burden_formula(self):
        result = predict_metabolite_toxicity_risk(
            "Met4", "C", 150.0, 200.0, auc_fraction=0.3, structural_alerts=["furan"]
        )
        expected = 0.3 * 200.0 / 100.0 * (1.0 + 0.5 * 1)
        assert result.toxic_burden == pytest.approx(expected, rel=1e-6)

    def test_invalid_dose(self):
        with pytest.raises(ValueError, match="daily_dose_mg"):
            predict_metabolite_toxicity_risk("M", "C", 200.0, 0.0, 0.1)

    def test_invalid_auc_fraction(self):
        with pytest.raises(ValueError, match="auc_fraction"):
            predict_metabolite_toxicity_risk("M", "C", 200.0, 100.0, 1.5)

    def test_result_is_frozen(self):
        result = predict_metabolite_toxicity_risk("M", "C", 200.0, 100.0, 0.1)
        with pytest.raises((AttributeError, TypeError)):
            result.risk_category = "low"  # type: ignore[misc]


class TestMetaboliteBatteryScreen:
    def _make_met(self, name, auc_met, auc_parent=1000.0):
        return {
            "compound_name": "Parent",
            "metabolite_name": name,
            "auc_metabolite": auc_met,
            "auc_parent": auc_parent,
            "fm_formation": 0.2,
        }

    def test_sorted_descending_by_exposure(self):
        mets = [
            self._make_met("A", 50.0),
            self._make_met("B", 300.0),
            self._make_met("C", 150.0),
        ]
        results = metabolite_battery_screen(mets)
        fracs = [r.exposure_fraction for r in results]
        assert fracs == sorted(fracs, reverse=True)

    def test_returns_correct_count(self):
        mets = [self._make_met(f"Met{i}", float(i * 50)) for i in range(1, 6)]
        results = metabolite_battery_screen(mets)
        assert len(results) == 5

    def test_all_results_are_metabolite_safety_result(self):
        mets = [self._make_met("X", 200.0)]
        results = metabolite_battery_screen(mets)
        assert all(isinstance(r, MetaboliteSafetyResult) for r in results)
