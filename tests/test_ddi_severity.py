"""Tests for clinical/ddi_severity.py — Phase 310."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.ddi_severity import (
    DDIClassificationResult,
    DDIRiskProfile,
    DDISeverity,
    batch_classify,
    classify_ddi_severity,
    recommend_clinical_action,
    summarize_risk_profile,
)


# ---------------------------------------------------------------------------
# classify_ddi_severity
# ---------------------------------------------------------------------------

class TestClassifyDdiSeverity:
    def test_returns_dataclass(self):
        result = classify_ddi_severity(1.0)
        assert isinstance(result, DDISeverity)

    def test_negligible_below_1_25(self):
        result = classify_ddi_severity(1.1)
        assert result.severity_category == "negligible"

    def test_weak_at_1_25(self):
        result = classify_ddi_severity(1.25)
        assert result.severity_category == "weak"

    def test_weak_at_1_9(self):
        result = classify_ddi_severity(1.9)
        assert result.severity_category == "weak"

    def test_moderate_at_2_0(self):
        result = classify_ddi_severity(2.0)
        assert result.severity_category == "moderate"

    def test_moderate_at_4_99(self):
        result = classify_ddi_severity(4.99)
        assert result.severity_category == "moderate"

    def test_strong_at_5_0(self):
        result = classify_ddi_severity(5.0)
        assert result.severity_category == "strong"

    def test_strong_at_large_aucr(self):
        result = classify_ddi_severity(20.0)
        assert result.severity_category == "strong"

    def test_aucr_stored(self):
        result = classify_ddi_severity(3.5)
        assert result.aucr == 3.5

    def test_clinical_significance_stored(self):
        result = classify_ddi_severity(2.0, "sensitive")
        assert result.clinical_significance == "sensitive"

    def test_recommendation_present(self):
        result = classify_ddi_severity(2.0)
        assert len(result.recommendation) > 0

    def test_notes_present(self):
        result = classify_ddi_severity(2.0)
        assert len(result.notes) > 0

    def test_narrow_ti_weak_upgraded_to_moderate(self):
        result = classify_ddi_severity(1.5, "narrow_ti")
        assert result.severity_category == "moderate"

    def test_narrow_ti_negligible_stays_negligible(self):
        result = classify_ddi_severity(1.1, "narrow_ti")
        assert result.severity_category == "negligible"

    def test_narrow_ti_strong_stays_strong(self):
        result = classify_ddi_severity(6.0, "narrow_ti")
        assert result.severity_category == "strong"

    def test_validation_aucr_zero(self):
        with pytest.raises(ValueError, match="aucr must be > 0"):
            classify_ddi_severity(0.0)

    def test_validation_aucr_negative(self):
        with pytest.raises(ValueError, match="aucr must be > 0"):
            classify_ddi_severity(-1.0)

    def test_validation_bad_significance(self):
        with pytest.raises(ValueError, match="clinical_significance must be one of"):
            classify_ddi_severity(2.0, "unknown")

    def test_frozen_dataclass(self):
        result = classify_ddi_severity(2.0)
        with pytest.raises((AttributeError, TypeError)):
            result.severity_category = "negligible"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# recommend_clinical_action
# ---------------------------------------------------------------------------

class TestRecommendClinicalAction:
    def test_negligible_no_action(self):
        rec = recommend_clinical_action("negligible", "DrugA", "DrugB")
        assert "No action" in rec

    def test_weak_monitor(self):
        rec = recommend_clinical_action("weak", "DrugA", "DrugB")
        assert "Monitor" in rec or "monitor" in rec

    def test_moderate_dose_reduction(self):
        rec = recommend_clinical_action("moderate", "DrugA", "DrugB")
        assert "50%" in rec or "alternative" in rec.lower()

    def test_strong_contraindicated_or_75(self):
        rec = recommend_clinical_action("strong", "DrugA", "DrugB")
        assert "contraindicated" in rec.lower() or "75%" in rec

    def test_drug_names_included(self):
        rec = recommend_clinical_action("weak", "Warfarin", "Fluconazole")
        assert "Warfarin" in rec
        assert "Fluconazole" in rec


# ---------------------------------------------------------------------------
# batch_classify
# ---------------------------------------------------------------------------

class TestBatchClassify:
    def _sample_list(self):
        return [
            {"substrate": "Midazolam", "perpetrator": "Itraconazole", "aucr": 6.0},
            {"substrate": "Simvastatin", "perpetrator": "Clarithromycin", "aucr": 3.0},
            {"substrate": "Warfarin", "perpetrator": "Aspirin", "aucr": 1.4},
            {"substrate": "Metformin", "perpetrator": "Cimetidine", "aucr": 1.1},
        ]

    def test_returns_list(self):
        result = batch_classify(self._sample_list())
        assert isinstance(result, list)

    def test_returns_correct_type(self):
        result = batch_classify(self._sample_list())
        for r in result:
            assert isinstance(r, DDIClassificationResult)

    def test_sorted_strong_first(self):
        result = batch_classify(self._sample_list())
        assert result[0].severity_category == "strong"
        assert result[-1].severity_category == "negligible"

    def test_length_preserved(self):
        result = batch_classify(self._sample_list())
        assert len(result) == 4

    def test_substrate_and_perpetrator_preserved(self):
        result = batch_classify(self._sample_list())
        substrates = {r.substrate for r in result}
        assert "Midazolam" in substrates

    def test_aucr_preserved(self):
        result = batch_classify(self._sample_list())
        aucr_map = {r.substrate: r.aucr for r in result}
        assert aucr_map["Midazolam"] == 6.0

    def test_empty_list(self):
        result = batch_classify([])
        assert result == []

    def test_with_clinical_significance(self):
        data = [
            {
                "substrate": "CsA",
                "perpetrator": "Verapamil",
                "aucr": 1.5,
                "clinical_significance": "narrow_ti",
            }
        ]
        result = batch_classify(data)
        assert result[0].severity_category == "moderate"


# ---------------------------------------------------------------------------
# summarize_risk_profile
# ---------------------------------------------------------------------------

class TestSummarizeRiskProfile:
    def _mixed_list(self):
        return [
            {"substrate": "A", "perpetrator": "X", "aucr": 8.0},   # strong
            {"substrate": "B", "perpetrator": "Y", "aucr": 3.0},   # moderate
            {"substrate": "C", "perpetrator": "Z", "aucr": 1.5},   # weak
            {"substrate": "D", "perpetrator": "W", "aucr": 1.1},   # negligible
        ]

    def test_returns_dataclass(self):
        result = summarize_risk_profile(self._mixed_list())
        assert isinstance(result, DDIRiskProfile)

    def test_counts_correct(self):
        result = summarize_risk_profile(self._mixed_list())
        assert result.n_strong == 1
        assert result.n_moderate == 1
        assert result.n_weak == 1
        assert result.n_negligible == 1
        assert result.n_interactions == 4

    def test_highest_risk_pair_is_strong(self):
        result = summarize_risk_profile(self._mixed_list())
        assert "A" in result.highest_risk_pair
        assert "X" in result.highest_risk_pair

    def test_overall_risk_score_range(self):
        result = summarize_risk_profile(self._mixed_list())
        assert 0.0 <= result.overall_risk_score <= 1.0

    def test_all_negligible_low_score(self):
        data = [
            {"substrate": f"Drug{i}", "perpetrator": "X", "aucr": 1.0}
            for i in range(4)
        ]
        result = summarize_risk_profile(data)
        assert result.overall_risk_score == 0.0
        assert result.n_negligible == 4

    def test_all_strong_high_score(self):
        data = [
            {"substrate": f"Drug{i}", "perpetrator": "X", "aucr": 10.0}
            for i in range(4)
        ]
        result = summarize_risk_profile(data)
        assert result.overall_risk_score == 1.0
        assert result.n_strong == 4

    def test_notes_non_empty(self):
        result = summarize_risk_profile(self._mixed_list())
        assert len(result.notes) > 0

    def test_empty_list(self):
        result = summarize_risk_profile([])
        assert result.n_interactions == 0
        assert result.overall_risk_score == 0.0
        assert result.highest_risk_pair == "none"

    def test_frozen_dataclass(self):
        result = summarize_risk_profile(self._mixed_list())
        with pytest.raises((AttributeError, TypeError)):
            result.n_strong = 99  # type: ignore[misc]
