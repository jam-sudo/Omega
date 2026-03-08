"""Tests for Phase 558 - Oral Bioavailability Sensitivity Analysis."""

import pytest

from omega_pbpk.clinical.bioavailability_sensitivity import (
    BioavailabilitySensitivityResult,
    analyze_bioavailability_sensitivity,
    sweep_fa_fg_fh,
)


class TestAnalyzeBioavailabilitySensitivity:
    """Tests for analyze_bioavailability_sensitivity."""

    def test_return_type(self):
        result = analyze_bioavailability_sensitivity("DrugA", 0.8, 0.6, 0.9)
        assert isinstance(result, BioavailabilitySensitivityResult)

    def test_f_oral_product(self):
        result = analyze_bioavailability_sensitivity("DrugA", 0.8, 0.6, 0.9)
        assert abs(result.f_oral - 0.8 * 0.6 * 0.9) < 1e-10

    def test_partial_derivative_fa(self):
        result = analyze_bioavailability_sensitivity("DrugA", 0.8, 0.6, 0.9)
        assert abs(result.d_f_d_fa - 0.6 * 0.9) < 1e-10

    def test_partial_derivative_fg(self):
        result = analyze_bioavailability_sensitivity("DrugA", 0.8, 0.6, 0.9)
        assert abs(result.d_f_d_fg - 0.8 * 0.9) < 1e-10

    def test_partial_derivative_fh(self):
        result = analyze_bioavailability_sensitivity("DrugA", 0.8, 0.6, 0.9)
        assert abs(result.d_f_d_fh - 0.8 * 0.6) < 1e-10

    def test_normalized_sensitivity_fa_is_one(self):
        result = analyze_bioavailability_sensitivity("DrugA", 0.5, 0.7, 0.3)
        assert result.sensitivity_fa == 1.0

    def test_normalized_sensitivity_fg_is_one(self):
        result = analyze_bioavailability_sensitivity("DrugA", 0.5, 0.7, 0.3)
        assert result.sensitivity_fg == 1.0

    def test_normalized_sensitivity_fh_is_one(self):
        result = analyze_bioavailability_sensitivity("DrugA", 0.5, 0.7, 0.3)
        assert result.sensitivity_fh == 1.0

    def test_limiting_factor_fa(self):
        result = analyze_bioavailability_sensitivity("DrugA", 0.2, 0.8, 0.9)
        assert result.limiting_factor == "fa"

    def test_limiting_factor_fg(self):
        result = analyze_bioavailability_sensitivity("DrugA", 0.8, 0.2, 0.9)
        assert result.limiting_factor == "fg"

    def test_limiting_factor_fh(self):
        result = analyze_bioavailability_sensitivity("DrugA", 0.8, 0.9, 0.2)
        assert result.limiting_factor == "fh"

    def test_f_class_high(self):
        result = analyze_bioavailability_sensitivity("DrugA", 0.95, 0.9, 0.85)
        assert result.f_class == "high"

    def test_f_class_moderate(self):
        result = analyze_bioavailability_sensitivity("DrugA", 0.7, 0.7, 0.7)
        assert result.f_class == "moderate"

    def test_f_class_low(self):
        result = analyze_bioavailability_sensitivity("DrugA", 0.3, 0.3, 0.9)
        assert result.f_class == "low"

    def test_frozen_dataclass(self):
        result = analyze_bioavailability_sensitivity("DrugA", 0.8, 0.6, 0.9)
        with pytest.raises(AttributeError):
            result.fa = 0.5

    def test_validation_fa_zero(self):
        with pytest.raises(ValueError, match="fa"):
            analyze_bioavailability_sensitivity("Bad", 0.0, 0.5, 0.5)

    def test_validation_fg_negative(self):
        with pytest.raises(ValueError, match="fg"):
            analyze_bioavailability_sensitivity("Bad", 0.5, -0.1, 0.5)

    def test_validation_fh_above_one(self):
        with pytest.raises(ValueError, match="fh"):
            analyze_bioavailability_sensitivity("Bad", 0.5, 0.5, 1.5)

    def test_boundary_value_one(self):
        result = analyze_bioavailability_sensitivity("DrugA", 1.0, 1.0, 1.0)
        assert abs(result.f_oral - 1.0) < 1e-10


class TestSweepFaFgFh:
    """Tests for sweep_fa_fg_fh."""

    def test_returns_n_cubed(self):
        results = sweep_fa_fg_fh("DrugA", n_points=3)
        assert len(results) == 27

    def test_returns_n_cubed_5(self):
        results = sweep_fa_fg_fh("DrugA", n_points=5)
        assert len(results) == 125

    def test_sorted_descending_by_f_oral(self):
        results = sweep_fa_fg_fh("DrugA", n_points=4)
        f_orals = [r.f_oral for r in results]
        for i in range(1, len(f_orals)):
            assert f_orals[i] <= f_orals[i - 1]

    def test_all_results_are_correct_type(self):
        results = sweep_fa_fg_fh("DrugA", n_points=3)
        for r in results:
            assert isinstance(r, BioavailabilitySensitivityResult)

    def test_first_result_is_max(self):
        results = sweep_fa_fg_fh("DrugA", n_points=5)
        assert abs(results[0].f_oral - 1.0) < 1e-10
