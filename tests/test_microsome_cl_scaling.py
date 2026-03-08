"""Tests for Phase 580 - Hepatic Clearance Scaling from Microsomes."""

import pytest

from omega_pbpk.prediction.microsome_cl_scaling import (
    MicrosomeCLScalingResult,
    scale_microsomal_cl,
    screen_microsomal_cl,
)


def _default_result(**kwargs):
    defaults = dict(
        drug_name="TestDrug",
        clint_microsomal_mL_per_min_per_mg_protein=0.5,
    )
    defaults.update(kwargs)
    return scale_microsomal_cl(**defaults)


class TestReturnType:
    def test_returns_correct_type(self):
        r = _default_result()
        assert isinstance(r, MicrosomeCLScalingResult)

    def test_has_all_fields(self):
        r = _default_result()
        assert r.drug_name == "TestDrug"
        assert isinstance(r.clint_microsomal_mL_per_min_per_mg_protein, float)
        assert isinstance(r.cl_intrinsic_liver_mL_per_min, float)
        assert isinstance(r.cl_hepatic_L_per_h, float)
        assert isinstance(r.cl_hepatic_pt_L_per_h, float)
        assert isinstance(r.extraction_ratio_ws, float)
        assert isinstance(r.extraction_ratio_pt, float)
        assert isinstance(r.f_oral_ws, float)
        assert isinstance(r.f_oral_pt, float)
        assert isinstance(r.model_discrepancy_pct, float)
        assert isinstance(r.scaling_class, str)
        assert isinstance(r.notes, str)

    def test_frozen_dataclass(self):
        r = _default_result()
        with pytest.raises(AttributeError):
            r.drug_name = "Other"


class TestExtractionRatios:
    def test_e_ws_in_range(self):
        r = _default_result()
        assert 0.0 <= r.extraction_ratio_ws <= 1.0

    def test_e_pt_in_range(self):
        r = _default_result()
        assert 0.0 <= r.extraction_ratio_pt <= 1.0

    def test_high_clint_e_pt_greater_than_e_ws(self):
        # For high CLint, parallel-tube gives higher E
        r = _default_result(clint_microsomal_mL_per_min_per_mg_protein=50.0, fup=0.5)
        assert r.extraction_ratio_pt > r.extraction_ratio_ws

    def test_low_clint_models_similar(self):
        r = _default_result(clint_microsomal_mL_per_min_per_mg_protein=0.001)
        assert abs(r.extraction_ratio_pt - r.extraction_ratio_ws) < 0.01


class TestFOral:
    def test_f_oral_ws_equals_one_minus_e_ws(self):
        r = _default_result()
        assert abs(r.f_oral_ws - (1.0 - r.extraction_ratio_ws)) < 1e-12

    def test_f_oral_pt_equals_one_minus_e_pt(self):
        r = _default_result()
        assert abs(r.f_oral_pt - (1.0 - r.extraction_ratio_pt)) < 1e-12

    def test_f_oral_ws_in_range(self):
        r = _default_result()
        assert 0.0 <= r.f_oral_ws <= 1.0

    def test_f_oral_pt_in_range(self):
        r = _default_result()
        assert 0.0 <= r.f_oral_pt <= 1.0


class TestModelDiscrepancy:
    def test_discrepancy_non_negative(self):
        r = _default_result()
        assert r.model_discrepancy_pct >= 0.0

    def test_high_clint_higher_discrepancy(self):
        r_low = _default_result(clint_microsomal_mL_per_min_per_mg_protein=0.01)
        r_high = _default_result(clint_microsomal_mL_per_min_per_mg_protein=10.0)
        assert r_high.model_discrepancy_pct > r_low.model_discrepancy_pct


class TestCLintEffect:
    def test_higher_clint_higher_e(self):
        r_low = _default_result(clint_microsomal_mL_per_min_per_mg_protein=0.1)
        r_high = _default_result(clint_microsomal_mL_per_min_per_mg_protein=10.0)
        assert r_high.extraction_ratio_ws > r_low.extraction_ratio_ws

    def test_higher_clint_lower_f_oral(self):
        r_low = _default_result(clint_microsomal_mL_per_min_per_mg_protein=0.1)
        r_high = _default_result(clint_microsomal_mL_per_min_per_mg_protein=10.0)
        assert r_high.f_oral_ws < r_low.f_oral_ws


class TestFupEffect:
    def test_lower_fup_lower_e(self):
        r_high_fup = _default_result(fup=0.8)
        r_low_fup = _default_result(fup=0.05)
        assert r_low_fup.extraction_ratio_ws < r_high_fup.extraction_ratio_ws


class TestScalingClass:
    def test_low_e_class(self):
        r = _default_result(clint_microsomal_mL_per_min_per_mg_protein=0.01)
        assert r.scaling_class == "low_e"

    def test_high_e_class(self):
        r = _default_result(clint_microsomal_mL_per_min_per_mg_protein=100.0, fup=1.0)
        assert r.scaling_class == "high_e"

    def test_intermediate_e_class(self):
        # Need E_ws between 0.3 and 0.7
        # CL_ws = Q * fu*CLint / (Q + fu*CLint), E = CL/Q = fu*CLint/(Q+fu*CLint)
        # E=0.5 -> fu*CLint = Q -> CLint = Q/fu = 90/0.1 = 900 L/h
        # CLint_L_h = clint * 45 * 1500 * 60/1000 = clint * 4050
        # 900/4050 ~ 0.222
        r = _default_result(clint_microsomal_mL_per_min_per_mg_protein=0.222)
        assert r.scaling_class == "intermediate_e"


class TestScreenMicrosomalCL:
    def test_sorted_descending(self):
        compounds = [
            {"drug_name": "A", "clint_microsomal": 0.1},
            {"drug_name": "B", "clint_microsomal": 10.0},
            {"drug_name": "C", "clint_microsomal": 1.0},
        ]
        results = screen_microsomal_cl(compounds)
        assert len(results) == 3
        for i in range(1, len(results)):
            assert results[i - 1].cl_hepatic_L_per_h >= results[i].cl_hepatic_L_per_h

    def test_empty_compounds(self):
        assert screen_microsomal_cl([]) == []

    def test_with_fup_override(self):
        compounds = [
            {"drug_name": "A", "clint_microsomal": 1.0, "fup": 0.5},
        ]
        results = screen_microsomal_cl(compounds)
        assert len(results) == 1
        assert results[0].drug_name == "A"


class TestValidation:
    def test_clint_zero_raises(self):
        with pytest.raises(ValueError):
            _default_result(clint_microsomal_mL_per_min_per_mg_protein=0.0)

    def test_clint_negative_raises(self):
        with pytest.raises(ValueError):
            _default_result(clint_microsomal_mL_per_min_per_mg_protein=-1.0)

    def test_fup_zero_raises(self):
        with pytest.raises(ValueError):
            _default_result(fup=0.0)

    def test_fup_above_one_raises(self):
        with pytest.raises(ValueError):
            _default_result(fup=1.1)

    def test_liver_weight_zero_raises(self):
        with pytest.raises(ValueError):
            _default_result(liver_weight_g=0.0)
