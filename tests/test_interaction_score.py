"""Tests for omega_pbpk.clinical.interaction_score module."""

import pytest
from omega_pbpk.clinical.interaction_score import (
    InteractionScoreResult,
    score_cyp_perpetrator,
    score_cyp_victim,
    calculate_interaction_score,
)


# ---------------------------------------------------------------------------
# Helper constants
# ---------------------------------------------------------------------------
VALID_SEVERITIES = {"minor", "moderate", "major", "contraindicated"}


# ---------------------------------------------------------------------------
# InteractionScoreResult dataclass
# ---------------------------------------------------------------------------
class TestInteractionScoreResult:
    def test_dataclass_fields_present(self):
        result = InteractionScoreResult(
            drug_a="DrugA",
            drug_b="DrugB",
            cyp_perpetrator_score=5.0,
            cyp_victim_score=5.0,
            transporter_score=0.0,
            combined_risk_score=3.33,
            severity="minor",
            aucr_estimate=2.0,
            cmax_ratio_estimate=1.5,
            clinical_significance="Low risk",
            management_recommendation="No action required",
        )
        assert result.drug_a == "DrugA"
        assert result.drug_b == "DrugB"
        assert result.cyp_perpetrator_score == 5.0
        assert result.cyp_victim_score == 5.0
        assert result.transporter_score == 0.0
        assert result.combined_risk_score == pytest.approx(3.33)
        assert result.severity == "minor"
        assert result.aucr_estimate == pytest.approx(2.0)
        assert result.cmax_ratio_estimate == pytest.approx(1.5)
        assert result.clinical_significance == "Low risk"
        assert result.management_recommendation == "No action required"


# ---------------------------------------------------------------------------
# score_cyp_perpetrator
# ---------------------------------------------------------------------------
class TestScoreCypPerpetrator:
    def test_returns_float(self):
        score = score_cyp_perpetrator("ketoconazole")
        assert isinstance(score, float)

    def test_score_in_valid_range(self):
        score = score_cyp_perpetrator("ketoconazole")
        assert 0.0 <= score <= 10.0

    def test_low_ki_yields_high_score(self):
        """Ki <= 1 uM (potent inhibitor) -> high perpetrator score."""
        high_score = score_cyp_perpetrator("strong_inhibitor", ki_uM=0.1)
        low_score = score_cyp_perpetrator("weak_inhibitor", ki_uM=100.0)
        assert high_score > low_score

    def test_ki_le_1_yields_high_perpetrator_score(self):
        """Spec: ki_uM <= 1 -> high perpetrator score (>= 7)."""
        score = score_cyp_perpetrator("potent_drug", ki_uM=0.5)
        assert score >= 7.0

    def test_high_induction_fold_increases_score(self):
        baseline = score_cyp_perpetrator("drug", induction_fold=1.0)
        induced = score_cyp_perpetrator("drug", induction_fold=10.0)
        assert induced >= baseline

    def test_no_inhibition_no_induction_returns_low_score(self):
        score = score_cyp_perpetrator("inert_drug", ki_uM=None, induction_fold=1.0)
        assert score < 5.0

    def test_score_clamped_to_10(self):
        score = score_cyp_perpetrator("extreme_inhibitor", ki_uM=0.001, induction_fold=50.0)
        assert score <= 10.0

    def test_score_clamped_to_0(self):
        score = score_cyp_perpetrator("no_effect", ki_uM=None, induction_fold=1.0)
        assert score >= 0.0

    def test_default_parameters_produce_valid_score(self):
        score = score_cyp_perpetrator("generic_drug")
        assert 0.0 <= score <= 10.0


# ---------------------------------------------------------------------------
# score_cyp_victim
# ---------------------------------------------------------------------------
class TestScoreCypVictim:
    def test_returns_float(self):
        score = score_cyp_victim("midazolam")
        assert isinstance(score, float)

    def test_score_in_valid_range(self):
        score = score_cyp_victim("midazolam", fm_cyp3a4=0.9)
        assert 0.0 <= score <= 10.0

    def test_nti_yields_higher_score_than_non_nti(self):
        """NTI drug should have higher victim score than same drug without NTI flag."""
        nti_score = score_cyp_victim("warfarin", fm_cyp2c9=0.8, narrow_therapeutic_index=True)
        non_nti_score = score_cyp_victim("warfarin", fm_cyp2c9=0.8, narrow_therapeutic_index=False)
        assert nti_score > non_nti_score

    def test_high_fm_yields_higher_victim_score(self):
        high_fm = score_cyp_victim("drug", fm_cyp3a4=0.95)
        low_fm = score_cyp_victim("drug", fm_cyp3a4=0.05)
        assert high_fm > low_fm

    def test_zero_fm_returns_low_score(self):
        score = score_cyp_victim("non_cyp_drug", fm_cyp3a4=0.0, fm_cyp2d6=0.0, fm_cyp2c9=0.0)
        assert score < 5.0

    def test_score_clamped_to_10(self):
        score = score_cyp_victim("extreme_victim", fm_cyp3a4=1.0, fm_cyp2d6=1.0,
                                  fm_cyp2c9=1.0, narrow_therapeutic_index=True)
        assert score <= 10.0

    def test_score_clamped_to_0(self):
        score = score_cyp_victim("non_metabolized")
        assert score >= 0.0

    def test_multiple_cyp_fractions_cumulative(self):
        """Multiple CYP pathways should collectively increase victim score."""
        multi_cyp = score_cyp_victim("drug", fm_cyp3a4=0.4, fm_cyp2d6=0.3, fm_cyp2c9=0.3)
        single_cyp = score_cyp_victim("drug", fm_cyp3a4=0.4, fm_cyp2d6=0.0, fm_cyp2c9=0.0)
        assert multi_cyp >= single_cyp


# ---------------------------------------------------------------------------
# calculate_interaction_score
# ---------------------------------------------------------------------------
class TestCalculateInteractionScore:
    def test_returns_interaction_score_result(self):
        result = calculate_interaction_score("DrugA", "DrugB", 5.0, 5.0)
        assert isinstance(result, InteractionScoreResult)

    def test_combined_risk_in_valid_range(self):
        result = calculate_interaction_score("A", "B", 4.0, 6.0, transporter_score=2.0)
        assert 0.0 <= result.combined_risk_score <= 10.0

    def test_combined_risk_formula(self):
        """combined_risk = (perp + victim + transporter) / 3."""
        result = calculate_interaction_score("A", "B", 6.0, 6.0, transporter_score=6.0)
        assert result.combined_risk_score == pytest.approx(6.0, rel=1e-6)

    def test_combined_risk_formula_no_transporter(self):
        result = calculate_interaction_score("A", "B", 3.0, 9.0, transporter_score=0.0)
        expected = (3.0 + 9.0 + 0.0) / 3
        assert result.combined_risk_score == pytest.approx(expected, rel=1e-6)

    def test_aucr_estimate_formula(self):
        """aucr_estimate = 1 + combined_risk * 0.3."""
        result = calculate_interaction_score("A", "B", 6.0, 6.0, transporter_score=6.0)
        expected_aucr = 1 + result.combined_risk_score * 0.3
        assert result.aucr_estimate == pytest.approx(expected_aucr, rel=1e-6)

    def test_aucr_estimate_always_ge_1(self):
        result = calculate_interaction_score("A", "B", 0.0, 0.0, transporter_score=0.0)
        assert result.aucr_estimate >= 1.0

    def test_severity_valid_string(self):
        result = calculate_interaction_score("A", "B", 5.0, 5.0)
        assert result.severity in VALID_SEVERITIES

    def test_severity_contraindicated_when_combined_risk_ge_8(self):
        """combined_risk >= 8 -> contraindicated."""
        result = calculate_interaction_score("A", "B", 10.0, 10.0, transporter_score=10.0)
        assert result.severity == "contraindicated"

    def test_severity_major_when_combined_risk_ge_6(self):
        """combined_risk >= 6 and < 8 -> major."""
        # (7 + 7 + 4) / 3 = 6.0
        result = calculate_interaction_score("A", "B", 7.0, 7.0, transporter_score=4.0)
        assert result.combined_risk_score == pytest.approx(6.0)
        assert result.severity == "major"

    def test_severity_moderate_when_combined_risk_ge_4(self):
        """combined_risk >= 4 and < 6 -> moderate."""
        # (4 + 4 + 4) / 3 = 4.0
        result = calculate_interaction_score("A", "B", 4.0, 4.0, transporter_score=4.0)
        assert result.combined_risk_score == pytest.approx(4.0)
        assert result.severity == "moderate"

    def test_severity_minor_when_combined_risk_lt_4(self):
        """combined_risk < 4 -> minor."""
        result = calculate_interaction_score("A", "B", 1.0, 1.0, transporter_score=0.0)
        assert result.severity == "minor"

    def test_high_perp_and_victim_scores_major_or_contraindicated(self):
        result = calculate_interaction_score("A", "B", 9.0, 9.0)
        assert result.severity in {"major", "contraindicated"}

    def test_low_scores_yield_minor(self):
        result = calculate_interaction_score("A", "B", 0.5, 1.0, transporter_score=0.0)
        assert result.severity == "minor"

    def test_management_recommendation_nonempty(self):
        result = calculate_interaction_score("A", "B", 5.0, 5.0)
        assert isinstance(result.management_recommendation, str)
        assert len(result.management_recommendation.strip()) > 0

    def test_drug_names_preserved(self):
        result = calculate_interaction_score("Fluoxetine", "Tramadol", 7.0, 8.0)
        assert result.drug_a == "Fluoxetine"
        assert result.drug_b == "Tramadol"

    def test_invalid_score_negative_perpetrator_raises(self):
        with pytest.raises(ValueError):
            calculate_interaction_score("A", "B", -1.0, 5.0)

    def test_invalid_score_negative_victim_raises(self):
        with pytest.raises(ValueError):
            calculate_interaction_score("A", "B", 5.0, -0.1)

    def test_invalid_score_above_10_perpetrator_raises(self):
        with pytest.raises(ValueError):
            calculate_interaction_score("A", "B", 10.1, 5.0)

    def test_invalid_score_above_10_victim_raises(self):
        with pytest.raises(ValueError):
            calculate_interaction_score("A", "B", 5.0, 11.0)

    def test_invalid_score_negative_transporter_raises(self):
        with pytest.raises(ValueError):
            calculate_interaction_score("A", "B", 5.0, 5.0, transporter_score=-1.0)

    def test_invalid_score_above_10_transporter_raises(self):
        with pytest.raises(ValueError):
            calculate_interaction_score("A", "B", 5.0, 5.0, transporter_score=10.5)

    def test_boundary_score_8_is_contraindicated(self):
        """Exactly combined_risk=8 -> contraindicated."""
        # (8+8+8)/3 = 8.0
        result = calculate_interaction_score("A", "B", 8.0, 8.0, transporter_score=8.0)
        assert result.combined_risk_score == pytest.approx(8.0)
        assert result.severity == "contraindicated"

    def test_boundary_score_6_is_major(self):
        """Exactly combined_risk=6 -> major."""
        result = calculate_interaction_score("A", "B", 6.0, 6.0, transporter_score=6.0)
        assert result.combined_risk_score == pytest.approx(6.0)
        assert result.severity == "major"

    def test_boundary_score_4_is_moderate(self):
        """Exactly combined_risk=4 -> moderate."""
        result = calculate_interaction_score("A", "B", 4.0, 4.0, transporter_score=4.0)
        assert result.combined_risk_score == pytest.approx(4.0)
        assert result.severity == "moderate"

    def test_clinical_significance_is_string(self):
        result = calculate_interaction_score("A", "B", 3.0, 3.0)
        assert isinstance(result.clinical_significance, str)

    def test_default_transporter_score_zero(self):
        """Default transporter_score=0 -> combined_risk = (perp+victim)/3."""
        result = calculate_interaction_score("A", "B", 6.0, 3.0)
        expected = (6.0 + 3.0 + 0.0) / 3
        assert result.combined_risk_score == pytest.approx(expected, rel=1e-6)
