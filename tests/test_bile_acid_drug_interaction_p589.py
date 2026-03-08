"""
Tests for Phase 589 — Bile Acid Drug Interaction.
"""

import pytest

from omega_pbpk.clinical.bile_acid_drug_interaction_p589 import (
    BileAcidDrugInteractionResult,
    predict_bile_acid_interaction,
    screen_bile_acid_interactions,
)

# ---------------------------------------------------------------------------
# Helper: a baseline "safe" drug
# ---------------------------------------------------------------------------


def _safe_drug(**kwargs):
    defaults = dict(drug_name="SafeDrug", mw_Da=300.0, logP=1.0, dose_uM=0.01)
    defaults.update(kwargs)
    return predict_bile_acid_interaction(**defaults)


# ---------------------------------------------------------------------------
# 1. Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_correct_type(self):
        result = _safe_drug()
        assert isinstance(result, BileAcidDrugInteractionResult)

    def test_has_all_fields(self):
        result = _safe_drug()
        assert hasattr(result, "drug_name")
        assert hasattr(result, "ic50_ntcp_uM")
        assert hasattr(result, "ic50_bsep_uM")
        assert hasattr(result, "ic50_asbt_uM")
        assert hasattr(result, "inhibition_risk")
        assert hasattr(result, "fold_increase_bile_acids")
        assert hasattr(result, "cholestasis_risk_score")
        assert hasattr(result, "dili_risk_class")
        assert hasattr(result, "notes")

    def test_drug_name_preserved(self):
        result = predict_bile_acid_interaction("TestDrug", 400.0, 2.0, 1.0)
        assert result.drug_name == "TestDrug"

    def test_notes_is_string(self):
        result = _safe_drug()
        assert isinstance(result.notes, str)
        assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# 2. IC50 estimation
# ---------------------------------------------------------------------------


class TestIC50Estimation:
    def test_ic50_ntcp_formula(self):
        mw, logP = 400.0, 2.0
        result = predict_bile_acid_interaction("X", mw, logP, 1.0)
        expected = max(0.1, 50.0 / (1.0 + logP * mw / 10000.0))
        assert abs(result.ic50_ntcp_uM - expected) < 1e-9

    def test_ic50_bsep_formula(self):
        mw, logP = 400.0, 2.0
        result = predict_bile_acid_interaction("X", mw, logP, 1.0)
        expected = max(0.1, 30.0 / (1.0 + logP * mw / 8000.0))
        assert abs(result.ic50_bsep_uM - expected) < 1e-9

    def test_ic50_asbt_formula(self):
        mw, logP = 400.0, 2.0
        result = predict_bile_acid_interaction("X", mw, logP, 1.0)
        expected = max(0.1, 80.0 / (1.0 + logP * mw / 12000.0))
        assert abs(result.ic50_asbt_uM - expected) < 1e-9

    def test_ic50_minimum_floor(self):
        # Very lipophilic, heavy molecule → IC50 should floor at 0.1
        result = predict_bile_acid_interaction("Heavy", 5000.0, 10.0, 1.0)
        assert result.ic50_ntcp_uM >= 0.1
        assert result.ic50_bsep_uM >= 0.1
        assert result.ic50_asbt_uM >= 0.1

    def test_ki_override_ntcp(self):
        result = predict_bile_acid_interaction("X", 300.0, 2.0, 1.0, ki_ntcp_uM=5.0)
        assert result.ic50_ntcp_uM == 5.0

    def test_ki_override_bsep(self):
        result = predict_bile_acid_interaction("X", 300.0, 2.0, 1.0, ki_bsep_uM=7.0)
        assert result.ic50_bsep_uM == 7.0

    def test_ki_override_asbt(self):
        result = predict_bile_acid_interaction("X", 300.0, 2.0, 1.0, ki_asbt_uM=12.0)
        assert result.ic50_asbt_uM == 12.0


# ---------------------------------------------------------------------------
# 3. Risk classifications
# ---------------------------------------------------------------------------


class TestRiskClassification:
    def _result_with_bsep_fi(self, fi_target):
        """Find dose that gives roughly fi_target for BSEP.
        ic50 estimated from mw=300, logP=1:
            ic50_bsep = 30 / (1 + 1*300/8000) = 30 / 1.0375 ≈ 28.92
        dose = ic50 * fi / (1 - fi)
        """
        mw, logP = 300.0, 1.0
        ic50_bsep = max(0.1, 30.0 / (1.0 + logP * mw / 8000.0))
        if fi_target >= 1.0:
            fi_target = 0.999
        dose = ic50_bsep * fi_target / (1.0 - fi_target)
        return predict_bile_acid_interaction("X", mw, logP, dose)

    def test_low_risk(self):
        result = self._result_with_bsep_fi(0.05)
        assert result.inhibition_risk == "low"

    def test_moderate_risk(self):
        result = self._result_with_bsep_fi(0.2)
        assert result.inhibition_risk == "moderate"

    def test_high_risk(self):
        result = self._result_with_bsep_fi(0.45)
        assert result.inhibition_risk == "high"

    def test_critical_risk(self):
        result = self._result_with_bsep_fi(0.75)
        assert result.inhibition_risk == "critical"

    def test_dili_negligible(self):
        result = _safe_drug(dose_uM=0.001)
        assert result.dili_risk_class == "negligible"

    def test_dili_high_at_large_dose(self):
        result = predict_bile_acid_interaction(
            "HighDILI", 300.0, 1.0, 1000.0, ki_bsep_uM=0.5, ki_ntcp_uM=0.5, ki_asbt_uM=0.5
        )
        assert result.dili_risk_class == "high"

    def test_risk_strings_are_valid(self):
        valid_risks = {"low", "moderate", "high", "critical"}
        valid_dili = {"negligible", "low", "moderate", "high"}
        for dose in [0.01, 1.0, 100.0]:
            r = predict_bile_acid_interaction("D", 400.0, 2.0, dose)
            assert r.inhibition_risk in valid_risks
            assert r.dili_risk_class in valid_dili


# ---------------------------------------------------------------------------
# 4. Fold increase and cholestasis score
# ---------------------------------------------------------------------------


class TestFoldIncreaseAndScore:
    def test_fold_increase_minimum_is_one(self):
        # At near-zero inhibition, fold_increase ≈ 1
        result = predict_bile_acid_interaction("X", 300.0, 1.0, 0.0001)
        assert result.fold_increase_bile_acids >= 1.0

    def test_fold_increase_capped_at_ten(self):
        # Very high dose forces cap
        result = predict_bile_acid_interaction(
            "X", 300.0, 1.0, 100000.0, ki_bsep_uM=0.1, ki_ntcp_uM=0.1
        )
        assert result.fold_increase_bile_acids <= 10.0

    def test_cholestasis_score_bounded(self):
        for dose in [0.001, 1.0, 1000.0]:
            r = predict_bile_acid_interaction("D", 400.0, 3.0, dose)
            assert 0.0 <= r.cholestasis_risk_score <= 100.0

    def test_cholestasis_score_increases_with_dose(self):
        r_low = predict_bile_acid_interaction("D", 300.0, 2.0, 0.1)
        r_high = predict_bile_acid_interaction("D", 300.0, 2.0, 100.0)
        assert r_high.cholestasis_risk_score >= r_low.cholestasis_risk_score

    def test_fold_increase_increases_with_dose(self):
        r_low = predict_bile_acid_interaction("D", 300.0, 2.0, 0.1)
        r_high = predict_bile_acid_interaction("D", 300.0, 2.0, 100.0)
        assert r_high.fold_increase_bile_acids >= r_low.fold_increase_bile_acids


# ---------------------------------------------------------------------------
# 5. Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_uM"):
            predict_bile_acid_interaction("D", 300.0, 2.0, 0.0)

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_uM"):
            predict_bile_acid_interaction("D", 300.0, 2.0, -1.0)

    def test_zero_mw_raises(self):
        with pytest.raises(ValueError, match="mw_Da"):
            predict_bile_acid_interaction("D", 0.0, 2.0, 1.0)

    def test_negative_mw_raises(self):
        with pytest.raises(ValueError, match="mw_Da"):
            predict_bile_acid_interaction("D", -100.0, 2.0, 1.0)


# ---------------------------------------------------------------------------
# 6. Screening
# ---------------------------------------------------------------------------


class TestScreening:
    def _make_compounds(self):
        return [
            {"drug_name": "CompA", "mw_Da": 300.0, "logP": 1.0, "dose_uM": 0.1},
            {"drug_name": "CompB", "mw_Da": 600.0, "logP": 4.0, "dose_uM": 10.0},
            {
                "drug_name": "CompC",
                "mw_Da": 800.0,
                "logP": 5.0,
                "dose_uM": 50.0,
                "ki_bsep_uM": 0.5,
                "ki_ntcp_uM": 0.5,
            },
        ]

    def test_screen_returns_list(self):
        results = screen_bile_acid_interactions(self._make_compounds())
        assert isinstance(results, list)

    def test_screen_length(self):
        results = screen_bile_acid_interactions(self._make_compounds())
        assert len(results) == 3

    def test_screen_sorted_descending(self):
        results = screen_bile_acid_interactions(self._make_compounds())
        scores = [r.cholestasis_risk_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_screen_returns_correct_type(self):
        results = screen_bile_acid_interactions(self._make_compounds())
        for r in results:
            assert isinstance(r, BileAcidDrugInteractionResult)

    def test_screen_empty_list(self):
        results = screen_bile_acid_interactions([])
        assert results == []

    def test_screen_single_compound(self):
        compounds = [{"drug_name": "Solo", "mw_Da": 300.0, "logP": 2.0, "dose_uM": 5.0}]
        results = screen_bile_acid_interactions(compounds)
        assert len(results) == 1
        assert results[0].drug_name == "Solo"
