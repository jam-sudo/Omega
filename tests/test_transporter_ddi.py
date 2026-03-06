"""Tests for omega_pbpk.clinical.transporter_ddi module."""

import pytest

from omega_pbpk.clinical.transporter_ddi import (
    TransporterDDIResult,
    _r_value,
    assess_transporter_ddi,
    batch_transporter_screen,
)

# ---------------------------------------------------------------------------
# _r_value unit tests
# ---------------------------------------------------------------------------


class TestRValueFormula:
    def test_competitive_basic(self):
        """r = 1 + I/Ki; I=0.5, Ki=0.5 -> r=2.0"""
        assert _r_value(0.5, 0.5, "competitive") == pytest.approx(2.0)

    def test_competitive_zero_inhibitor(self):
        """I=0 -> r=1.0 (no inhibition)"""
        assert _r_value(0.0, 1.0, "competitive") == pytest.approx(1.0)

    def test_competitive_large_ratio(self):
        """I=9.0, Ki=1.0 -> r=10.0"""
        assert _r_value(9.0, 1.0, "competitive") == pytest.approx(10.0)

    def test_competitive_fractional_ki(self):
        """I=0.1, Ki=0.25 -> r=1.4"""
        assert _r_value(0.1, 0.25, "competitive") == pytest.approx(1.4)

    def test_default_inhibition_type_is_competitive(self):
        """Default inhibition_type should be competitive."""
        assert _r_value(1.0, 1.0) == pytest.approx(_r_value(1.0, 1.0, "competitive"))

    def test_r_value_always_gte_one(self):
        """r_value must be >= 1.0 for non-negative inhibitor concentrations."""
        assert _r_value(0.0, 2.0) >= 1.0
        assert _r_value(0.01, 100.0) >= 1.0


# ---------------------------------------------------------------------------
# assess_transporter_ddi — return type and field preservation
# ---------------------------------------------------------------------------


class TestAssessTransporterDDIReturnType:
    def test_returns_transporter_ddi_result(self):
        result = assess_transporter_ddi("digoxin", "clarithromycin", "P-gp", ki_uM=1.0)
        assert isinstance(result, TransporterDDIResult)

    def test_victim_drug_preserved(self):
        result = assess_transporter_ddi("metformin", "rifampicin", "OCT2", ki_uM=2.0)
        assert result.victim_drug == "metformin"

    def test_perpetrator_drug_preserved(self):
        result = assess_transporter_ddi("metformin", "rifampicin", "OCT2", ki_uM=2.0)
        assert result.perpetrator_drug == "rifampicin"

    def test_transporter_string_preserved(self):
        result = assess_transporter_ddi("rosuvastatin", "cyclosporine", "OATP1B1", ki_uM=0.05)
        assert result.transporter == "OATP1B1"

    def test_inhibition_type_preserved(self):
        result = assess_transporter_ddi(
            "digoxin", "verapamil", "P-gp", ki_uM=1.0, inhibition_type="competitive"
        )
        assert result.inhibition_type == "competitive"

    def test_ki_uM_preserved(self):
        result = assess_transporter_ddi("drug_a", "drug_b", "BCRP", ki_uM=3.5)
        assert result.ki_uM == pytest.approx(3.5)

    def test_inhibitor_conc_uM_preserved(self):
        result = assess_transporter_ddi(
            "drug_a", "drug_b", "BCRP", ki_uM=1.0, inhibitor_conc_uM=0.25
        )
        assert result.inhibitor_conc_uM == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# assess_transporter_ddi — r_value and aucr_predicted calculations
# ---------------------------------------------------------------------------


class TestAssessTransporterDDICalculations:
    def test_r_value_formula(self):
        """r_value = 1 + I/Ki; I=0.5, Ki=0.5 -> 2.0"""
        result = assess_transporter_ddi(
            "drug_a", "drug_b", "P-gp", ki_uM=0.5, inhibitor_conc_uM=0.5
        )
        assert result.r_value == pytest.approx(2.0)

    def test_r_value_equals_aucr(self):
        """aucr_predicted should equal r_value."""
        result = assess_transporter_ddi(
            "drug_a", "drug_b", "P-gp", ki_uM=1.0, inhibitor_conc_uM=1.0
        )
        assert result.aucr_predicted == pytest.approx(result.r_value)

    def test_r_value_gte_one(self):
        result = assess_transporter_ddi(
            "drug_a", "drug_b", "OATP1B1", ki_uM=100.0, inhibitor_conc_uM=0.001
        )
        assert result.r_value >= 1.0

    def test_aucr_gte_one(self):
        result = assess_transporter_ddi(
            "drug_a", "drug_b", "OCT2", ki_uM=50.0, inhibitor_conc_uM=0.0
        )
        assert result.aucr_predicted >= 1.0

    def test_default_inhibitor_conc(self):
        """Default inhibitor_conc_uM=0.1; Ki=1.0 -> r=1.1"""
        result = assess_transporter_ddi("drug_a", "drug_b", "P-gp", ki_uM=1.0)
        assert result.r_value == pytest.approx(1.1)


# ---------------------------------------------------------------------------
# assess_transporter_ddi — clinical_relevance thresholds
# ---------------------------------------------------------------------------


class TestClinicalRelevance:
    def test_contraindicated_threshold(self):
        """aucr >= 5 -> contraindicated"""
        # I=4.0, Ki=1.0 -> r=5.0
        result = assess_transporter_ddi(
            "drug_v", "drug_p", "P-gp", ki_uM=1.0, inhibitor_conc_uM=4.0
        )
        assert result.clinical_relevance == "contraindicated"

    def test_strong_threshold(self):
        """2 <= aucr < 5 -> strong"""
        # I=1.0, Ki=1.0 -> r=2.0
        result = assess_transporter_ddi(
            "drug_v", "drug_p", "P-gp", ki_uM=1.0, inhibitor_conc_uM=1.0
        )
        assert result.clinical_relevance == "strong"

    def test_moderate_threshold(self):
        """1.25 <= aucr < 2 -> moderate"""
        # I=0.25, Ki=1.0 -> r=1.25
        result = assess_transporter_ddi(
            "drug_v", "drug_p", "P-gp", ki_uM=1.0, inhibitor_conc_uM=0.25
        )
        assert result.clinical_relevance == "moderate"

    def test_weak_threshold(self):
        """aucr < 1.25 -> weak"""
        # I=0.1, Ki=1.0 -> r=1.1
        result = assess_transporter_ddi(
            "drug_v", "drug_p", "P-gp", ki_uM=1.0, inhibitor_conc_uM=0.1
        )
        assert result.clinical_relevance == "weak"

    def test_high_ratio_is_contraindicated(self):
        """Very high I/Ki ratio must yield contraindicated."""
        result = assess_transporter_ddi(
            "digoxin", "quinidine", "P-gp", ki_uM=0.1, inhibitor_conc_uM=10.0
        )
        assert result.clinical_relevance == "contraindicated"

    def test_low_ratio_is_weak(self):
        """Very low I/Ki ratio must yield weak."""
        result = assess_transporter_ddi(
            "metformin", "trimethoprim", "OCT2", ki_uM=100.0, inhibitor_conc_uM=0.001
        )
        assert result.clinical_relevance == "weak"

    def test_clinical_relevance_is_valid_string(self):
        valid = {"contraindicated", "strong", "moderate", "weak"}
        for ki, conc in [(0.1, 5.0), (1.0, 1.5), (1.0, 0.3), (10.0, 0.01)]:
            result = assess_transporter_ddi("v", "p", "P-gp", ki_uM=ki, inhibitor_conc_uM=conc)
            assert result.clinical_relevance in valid


# ---------------------------------------------------------------------------
# assess_transporter_ddi — fda_recommendation
# ---------------------------------------------------------------------------


class TestFDARecommendation:
    def test_fda_recommendation_non_empty(self):
        result = assess_transporter_ddi("drug_v", "drug_p", "P-gp", ki_uM=1.0)
        assert isinstance(result.fda_recommendation, str)
        assert len(result.fda_recommendation) > 0

    def test_fda_recommendation_non_empty_for_contraindicated(self):
        result = assess_transporter_ddi(
            "drug_v", "drug_p", "P-gp", ki_uM=0.5, inhibitor_conc_uM=5.0
        )
        assert len(result.fda_recommendation) > 0


# ---------------------------------------------------------------------------
# assess_transporter_ddi — ValueError conditions
# ---------------------------------------------------------------------------


class TestValueErrors:
    def test_ki_zero_raises_value_error(self):
        with pytest.raises(ValueError):
            assess_transporter_ddi("drug_v", "drug_p", "P-gp", ki_uM=0.0)

    def test_ki_negative_raises_value_error(self):
        with pytest.raises(ValueError):
            assess_transporter_ddi("drug_v", "drug_p", "P-gp", ki_uM=-1.0)

    def test_inhibitor_conc_negative_raises_value_error(self):
        with pytest.raises(ValueError):
            assess_transporter_ddi("drug_v", "drug_p", "P-gp", ki_uM=1.0, inhibitor_conc_uM=-0.1)

    def test_ki_very_small_positive_does_not_raise(self):
        """ki_uM=1e-9 should not raise; it is positive."""
        result = assess_transporter_ddi(
            "drug_v", "drug_p", "P-gp", ki_uM=1e-9, inhibitor_conc_uM=0.0
        )
        assert result.r_value >= 1.0

    def test_inhibitor_conc_zero_does_not_raise(self):
        """inhibitor_conc_uM=0.0 is valid (no inhibitor present)."""
        result = assess_transporter_ddi(
            "drug_v", "drug_p", "P-gp", ki_uM=1.0, inhibitor_conc_uM=0.0
        )
        assert result.r_value == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# batch_transporter_screen
# ---------------------------------------------------------------------------


class TestBatchTransporterScreen:
    def setup_method(self):
        self.ki_dict = {
            "P-gp": 1.0,
            "BCRP": 0.5,
            "OATP1B1": 2.0,
            "OCT2": 5.0,
        }

    def test_batch_returns_list(self):
        results = batch_transporter_screen("perpetrator", 0.1, self.ki_dict)
        assert isinstance(results, list)

    def test_batch_len_matches_input_dict(self):
        results = batch_transporter_screen("perpetrator", 0.1, self.ki_dict)
        assert len(results) == len(self.ki_dict)

    def test_batch_sorted_by_aucr_descending(self):
        results = batch_transporter_screen("perpetrator", 0.1, self.ki_dict)
        aucrs = [r.aucr_predicted for r in results]
        assert aucrs == sorted(aucrs, reverse=True)

    def test_batch_all_results_are_transporter_ddi_result(self):
        results = batch_transporter_screen("perpetrator", 0.1, self.ki_dict)
        for r in results:
            assert isinstance(r, TransporterDDIResult)

    def test_batch_single_entry(self):
        results = batch_transporter_screen("drug_p", 1.0, {"P-gp": 1.0})
        assert len(results) == 1
        assert results[0].aucr_predicted == pytest.approx(2.0)

    def test_batch_perpetrator_preserved_in_all_results(self):
        results = batch_transporter_screen("itraconazole", 0.5, self.ki_dict)
        for r in results:
            assert r.perpetrator_drug == "itraconazole"

    def test_batch_highest_aucr_first(self):
        """Transporter with lowest Ki should appear first when inhibitor_conc is fixed."""
        ki_dict = {"low_ki": 0.1, "high_ki": 10.0}
        results = batch_transporter_screen("drug_p", 1.0, ki_dict)
        assert results[0].transporter == "low_ki"

    def test_batch_empty_dict_returns_empty_list(self):
        results = batch_transporter_screen("drug_p", 0.1, {})
        assert results == []
