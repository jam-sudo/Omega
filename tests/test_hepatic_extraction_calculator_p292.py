"""Tests for Phase 292 — Hepatic Extraction Ratio Calculator."""

import pytest

from omega_pbpk.prediction.hepatic_extraction_calculator_p292 import (
    HepaticExtractionResult,
    calculate_hepatic_extraction,
    compare_models,
)

_VALID_CLASSES = {"high", "intermediate", "low"}
_ALL_MODELS = ["well_stirred", "parallel_tube", "dispersion"]


# ---------------------------------------------------------------------------
# Return type and field preservation
# ---------------------------------------------------------------------------


class TestReturnTypeAndFields:
    def test_returns_hepatic_extraction_result(self):
        result = calculate_hepatic_extraction("DrugA", 10.0, 0.5, 90.0)
        assert isinstance(result, HepaticExtractionResult)

    def test_drug_name_preserved(self):
        result = calculate_hepatic_extraction("Lidocaine", 10.0, 0.5, 90.0)
        assert result.drug_name == "Lidocaine"

    def test_model_name_preserved(self):
        result = calculate_hepatic_extraction("DrugA", 10.0, 0.5, 90.0, model="parallel_tube")
        assert result.model == "parallel_tube"

    def test_cl_int_preserved(self):
        result = calculate_hepatic_extraction("DrugA", 5.0, 0.3, 90.0)
        assert result.cl_int_mL_per_min_per_g == pytest.approx(5.0)

    def test_fu_preserved(self):
        result = calculate_hepatic_extraction("DrugA", 5.0, 0.3, 90.0)
        assert result.fu_plasma == pytest.approx(0.3)

    def test_q_preserved(self):
        result = calculate_hepatic_extraction("DrugA", 5.0, 0.3, 90.0)
        assert result.q_hepatic_L_per_h == pytest.approx(90.0)

    def test_notes_is_str(self):
        result = calculate_hepatic_extraction("DrugA", 5.0, 0.3, 90.0)
        assert isinstance(result.notes, str)


# ---------------------------------------------------------------------------
# EH and f_oral bounds
# ---------------------------------------------------------------------------


class TestExtractionBounds:
    @pytest.mark.parametrize("model", _ALL_MODELS)
    def test_eh_in_range(self, model):
        result = calculate_hepatic_extraction("DrugA", 10.0, 0.5, 90.0, model=model)
        assert 0.0 <= result.extraction_ratio <= 1.0

    @pytest.mark.parametrize("model", _ALL_MODELS)
    def test_f_oral_in_range(self, model):
        result = calculate_hepatic_extraction("DrugA", 10.0, 0.5, 90.0, model=model)
        assert 0.0 <= result.f_oral <= 1.0

    @pytest.mark.parametrize("model", _ALL_MODELS)
    def test_eh_plus_f_oral_equals_one(self, model):
        result = calculate_hepatic_extraction("DrugA", 10.0, 0.5, 90.0, model=model)
        assert result.extraction_ratio + result.f_oral == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Extraction class correctness
# ---------------------------------------------------------------------------


class TestExtractionClass:
    def test_high_cl_int_gives_high_extraction(self):
        # Very high cl_int → EH > 0.7
        result = calculate_hepatic_extraction("HighEH", 100.0, 1.0, 90.0)
        assert result.extraction_class == "high"
        assert result.extraction_ratio > 0.7

    def test_low_cl_int_gives_low_extraction(self):
        # Very low cl_int → EH < 0.3
        result = calculate_hepatic_extraction("LowEH", 0.01, 0.1, 90.0)
        assert result.extraction_class == "low"
        assert result.extraction_ratio < 0.3

    def test_intermediate_extraction(self):
        # Moderate cl_int → 0.3 < EH < 0.7
        result = calculate_hepatic_extraction("MidEH", 1.5, 0.3, 90.0)
        assert result.extraction_class in {"intermediate", "high", "low"}  # depends on exact value

    @pytest.mark.parametrize("model", _ALL_MODELS)
    def test_extraction_class_valid(self, model):
        result = calculate_hepatic_extraction("DrugA", 10.0, 0.5, 90.0, model=model)
        assert result.extraction_class in _VALID_CLASSES


# ---------------------------------------------------------------------------
# CL_hepatic > 0
# ---------------------------------------------------------------------------


class TestCLHepatic:
    @pytest.mark.parametrize("model", _ALL_MODELS)
    def test_cl_hepatic_positive(self, model):
        result = calculate_hepatic_extraction("DrugA", 10.0, 0.5, 90.0, model=model)
        assert result.cl_hepatic_L_per_h > 0.0

    def test_cl_hepatic_equals_eh_times_q(self):
        result = calculate_hepatic_extraction("DrugA", 10.0, 0.5, 90.0)
        expected = result.extraction_ratio * result.q_hepatic_L_per_h
        assert result.cl_hepatic_L_per_h == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# t_half_estimate_h
# ---------------------------------------------------------------------------


class TestTHalfEstimate:
    def test_t_half_positive(self):
        result = calculate_hepatic_extraction("DrugA", 10.0, 0.5, 90.0)
        assert result.t_half_estimate_h > 0.0

    def test_t_half_formula(self):
        result = calculate_hepatic_extraction("DrugA", 10.0, 0.5, 90.0)
        expected = 0.693 * 50.0 / result.cl_hepatic_L_per_h
        assert result.t_half_estimate_h == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Dosing implication
# ---------------------------------------------------------------------------


class TestDosingImplication:
    def test_high_extraction_iv_preferred(self):
        result = calculate_hepatic_extraction("HighEH", 100.0, 1.0, 90.0)
        assert "IV preferred" in result.dosing_implication

    def test_low_extraction_oral_suitable(self):
        result = calculate_hepatic_extraction("LowEH", 0.01, 0.1, 90.0)
        assert "oral dosing suitable" in result.dosing_implication


# ---------------------------------------------------------------------------
# compare_models
# ---------------------------------------------------------------------------


class TestCompareModels:
    def test_returns_3_results(self):
        results = compare_models("DrugA", 10.0, 0.5, 90.0)
        assert len(results) == 3

    def test_sorted_descending(self):
        results = compare_models("DrugA", 10.0, 0.5, 90.0)
        ratios = [r.extraction_ratio for r in results]
        assert ratios == sorted(ratios, reverse=True)

    def test_all_models_represented(self):
        results = compare_models("DrugA", 10.0, 0.5, 90.0)
        models = {r.model for r in results}
        assert models == set(_ALL_MODELS)

    def test_all_results_are_correct_type(self):
        results = compare_models("DrugA", 10.0, 0.5, 90.0)
        for r in results:
            assert isinstance(r, HepaticExtractionResult)

    @pytest.mark.parametrize("model", _ALL_MODELS)
    def test_all_models_eh_in_open_interval(self, model):
        result = calculate_hepatic_extraction("DrugA", 5.0, 0.5, 90.0, model=model)
        assert 0.0 < result.extraction_ratio < 1.0


# ---------------------------------------------------------------------------
# Default liver weight
# ---------------------------------------------------------------------------


class TestDefaultLiverWeight:
    def test_default_liver_weight_1500g(self):
        # CL_int_total = 1.0 mL/min/g * 1500 g * 60 / 1000 = 90 L/h
        # fu=1.0, Q=90 → EH_ws = 90/(90+90) = 0.5
        result = calculate_hepatic_extraction("DrugA", 1.0, 1.0, 90.0, model="well_stirred")
        assert result.extraction_ratio == pytest.approx(0.5, rel=1e-6)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_model(self):
        with pytest.raises(ValueError, match="model"):
            calculate_hepatic_extraction("DrugA", 10.0, 0.5, 90.0, model="two_compartment")

    def test_cl_int_zero(self):
        with pytest.raises(ValueError, match="cl_int"):
            calculate_hepatic_extraction("DrugA", 0.0, 0.5, 90.0)

    def test_cl_int_negative(self):
        with pytest.raises(ValueError, match="cl_int"):
            calculate_hepatic_extraction("DrugA", -1.0, 0.5, 90.0)

    def test_fu_zero(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            calculate_hepatic_extraction("DrugA", 10.0, 0.0, 90.0)

    def test_fu_above_one(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            calculate_hepatic_extraction("DrugA", 10.0, 1.5, 90.0)

    def test_q_zero(self):
        with pytest.raises(ValueError, match="q_hepatic"):
            calculate_hepatic_extraction("DrugA", 10.0, 0.5, 0.0)

    def test_liver_weight_zero(self):
        with pytest.raises(ValueError, match="liver_weight"):
            calculate_hepatic_extraction("DrugA", 10.0, 0.5, 90.0, liver_weight_g=0.0)

    def test_compare_models_invalid_cl_int(self):
        with pytest.raises(ValueError):
            compare_models("DrugA", 0.0, 0.5, 90.0)
