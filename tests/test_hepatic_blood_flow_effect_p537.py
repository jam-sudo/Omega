"""Tests for Phase 537 — Hepatic Blood Flow Effect on Drug Clearance."""

import pytest

from omega_pbpk.core.hepatic_blood_flow_effect_p537 import (
    HepaticBloodFlowResult,
    analyze_hepatic_blood_flow_effect,
    compare_extraction_drugs,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _high_extraction_drug():
    """High CLint → flow-limited (E > 0.7)."""
    return analyze_hepatic_blood_flow_effect("HighEx", fup=0.5, cl_int_L_per_h=500.0)


def _low_extraction_drug():
    """Low CLint → enzyme-limited (E < 0.3)."""
    return analyze_hepatic_blood_flow_effect("LowEx", fup=0.05, cl_int_L_per_h=5.0)


def _intermediate_drug():
    """Intermediate extraction. fup*CLint = Q = 90 → EH = 0.5."""
    return analyze_hepatic_blood_flow_effect("MidEx", fup=0.5, cl_int_L_per_h=180.0)


# ---------------------------------------------------------------------------
# Return type & field presence
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_dataclass_instance(self):
        r = _high_extraction_drug()
        assert isinstance(r, HepaticBloodFlowResult)

    def test_has_drug_name(self):
        r = _high_extraction_drug()
        assert r.drug_name == "HighEx"

    def test_has_fup(self):
        r = _high_extraction_drug()
        assert r.fup == 0.5

    def test_has_cl_int(self):
        r = _high_extraction_drug()
        assert r.cl_int_L_per_h == 500.0

    def test_has_extraction_class(self):
        r = _high_extraction_drug()
        assert isinstance(r.extraction_class, str)

    def test_has_flow_sensitivity_index(self):
        r = _high_extraction_drug()
        assert isinstance(r.flow_sensitivity_index, float)

    def test_has_clinical_implication(self):
        r = _high_extraction_drug()
        assert isinstance(r.clinical_implication, str)
        assert len(r.clinical_implication) > 0

    def test_has_notes(self):
        r = _high_extraction_drug()
        assert isinstance(r.notes, str)

    def test_frozen(self):
        r = _high_extraction_drug()
        with pytest.raises((AttributeError, TypeError)):
            r.drug_name = "X"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Extraction classification
# ---------------------------------------------------------------------------


class TestExtractionClassification:
    def test_high_clint_is_flow_limited(self):
        r = _high_extraction_drug()
        assert r.extraction_class == "flow_limited"
        assert r.eh_normal > 0.7

    def test_low_clint_is_enzyme_limited(self):
        r = _low_extraction_drug()
        assert r.extraction_class == "enzyme_limited"
        assert r.eh_normal < 0.3

    def test_intermediate_clint(self):
        r = _intermediate_drug()
        assert r.extraction_class == "intermediate"
        assert 0.3 <= r.eh_normal <= 0.7

    def test_fh_equals_one_minus_eh(self):
        for drug in [_high_extraction_drug(), _low_extraction_drug(), _intermediate_drug()]:
            assert abs(drug.fh_normal - (1.0 - drug.eh_normal)) < 1e-9


# ---------------------------------------------------------------------------
# Well-stirred model: numerical correctness
# ---------------------------------------------------------------------------


class TestWellStirredModel:
    def test_clh_normal_formula(self):
        fup, cl_int, q = 0.5, 500.0, 90.0
        expected_clh = q * fup * cl_int / (q + fup * cl_int)
        r = analyze_hepatic_blood_flow_effect("X", fup=fup, cl_int_L_per_h=cl_int)
        assert abs(r.clh_normal_L_per_h - expected_clh) < 1e-9

    def test_clh_never_exceeds_q(self):
        r = _high_extraction_drug()
        assert r.clh_normal_L_per_h <= 90.0
        assert r.clh_heart_failure_L_per_h <= 45.0
        assert r.clh_exercise_L_per_h <= 180.0

    def test_clh_positive(self):
        r = _low_extraction_drug()
        assert r.clh_normal_L_per_h > 0


# ---------------------------------------------------------------------------
# Blood flow effect on clearance
# ---------------------------------------------------------------------------


class TestBloodFlowEffect:
    def test_heart_failure_reduces_clh(self):
        """Lower hepatic blood flow → lower clearance for flow-limited drug."""
        r = _high_extraction_drug()
        assert r.clh_heart_failure_L_per_h < r.clh_normal_L_per_h

    def test_exercise_increases_clh(self):
        """Higher hepatic blood flow → higher clearance for flow-limited drug."""
        r = _high_extraction_drug()
        assert r.clh_exercise_L_per_h > r.clh_normal_L_per_h

    def test_low_extraction_less_sensitive(self):
        """For enzyme-limited drug, flow changes have smaller absolute impact."""
        r_high = _high_extraction_drug()
        r_low = _low_extraction_drug()
        # The relative sensitivity (fraction of CLh_normal) is smaller for low extraction
        assert r_low.flow_sensitivity_index < r_high.flow_sensitivity_index

    def test_flow_sensitivity_positive_for_flow_limited(self):
        r = _high_extraction_drug()
        assert r.flow_sensitivity_index > 0

    def test_flow_sensitivity_small_for_enzyme_limited(self):
        r = _low_extraction_drug()
        # Still positive but small
        assert 0 < r.flow_sensitivity_index < 1.0


# ---------------------------------------------------------------------------
# Clinical implication
# ---------------------------------------------------------------------------


class TestClinicalImplication:
    def test_flow_limited_mentions_blood_flow(self):
        r = _high_extraction_drug()
        assert "flow" in r.clinical_implication.lower()

    def test_enzyme_limited_mentions_enzyme(self):
        r = _low_extraction_drug()
        assert "enzyme" in r.clinical_implication.lower()

    def test_intermediate_mentions_both(self):
        r = _intermediate_drug()
        impl = r.clinical_implication.lower()
        assert "blood flow" in impl or "flow" in impl


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_fup_zero_raises(self):
        with pytest.raises(ValueError, match="fup"):
            analyze_hepatic_blood_flow_effect("X", fup=0.0, cl_int_L_per_h=10.0)

    def test_fup_negative_raises(self):
        with pytest.raises(ValueError):
            analyze_hepatic_blood_flow_effect("X", fup=-0.1, cl_int_L_per_h=10.0)

    def test_fup_above_one_raises(self):
        with pytest.raises(ValueError):
            analyze_hepatic_blood_flow_effect("X", fup=1.5, cl_int_L_per_h=10.0)

    def test_cl_int_zero_raises(self):
        with pytest.raises(ValueError):
            analyze_hepatic_blood_flow_effect("X", fup=0.5, cl_int_L_per_h=0.0)

    def test_cl_int_negative_raises(self):
        with pytest.raises(ValueError):
            analyze_hepatic_blood_flow_effect("X", fup=0.5, cl_int_L_per_h=-5.0)

    def test_q_normal_zero_raises(self):
        with pytest.raises(ValueError):
            analyze_hepatic_blood_flow_effect(
                "X", fup=0.5, cl_int_L_per_h=10.0, q_normal_L_per_h=0.0
            )


# ---------------------------------------------------------------------------
# compare_extraction_drugs
# ---------------------------------------------------------------------------


class TestCompareExtractionDrugs:
    def _drugs(self):
        return [
            {"drug_name": "HighEx", "fup": 0.5, "cl_int_L_per_h": 500.0},
            {"drug_name": "LowEx", "fup": 0.05, "cl_int_L_per_h": 5.0},
            {"drug_name": "MidEx", "fup": 0.5, "cl_int_L_per_h": 180.0},
        ]

    def test_returns_list(self):
        results = compare_extraction_drugs(self._drugs())
        assert isinstance(results, list)
        assert len(results) == 3

    def test_sorted_descending_by_sensitivity(self):
        results = compare_extraction_drugs(self._drugs())
        indices = [r.flow_sensitivity_index for r in results]
        assert indices == sorted(indices, reverse=True)

    def test_high_extraction_first(self):
        results = compare_extraction_drugs(self._drugs())
        assert results[0].drug_name == "HighEx"

    def test_each_result_is_correct_type(self):
        results = compare_extraction_drugs(self._drugs())
        for r in results:
            assert isinstance(r, HepaticBloodFlowResult)

    def test_empty_list(self):
        assert compare_extraction_drugs([]) == []
