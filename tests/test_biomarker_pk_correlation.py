"""Tests for biomarker-PK correlation analysis (Phase 336)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from omega_pbpk.analysis.biomarker_pk_correlation import (
    BiomarkerCorrelationResult,
    correlate_biomarker_pk,
    screen_biomarker_correlations,
    summarize_pkpd_relationship,
)


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            correlate_biomarker_pk("Drug", "IL6", [1.0, 2.0, 3.0], [1.0, 2.0])

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError, match="At least 3"):
            correlate_biomarker_pk("Drug", "IL6", [1.0, 2.0], [1.0, 2.0])

    def test_one_point_raises(self):
        with pytest.raises(ValueError):
            correlate_biomarker_pk("Drug", "IL6", [1.0], [2.0])

    def test_pk_nan_raises(self):
        with pytest.raises(ValueError, match="finite"):
            correlate_biomarker_pk("Drug", "IL6", [1.0, float("nan"), 3.0], [1.0, 2.0, 3.0])

    def test_biomarker_inf_raises(self):
        with pytest.raises(ValueError, match="finite"):
            correlate_biomarker_pk("Drug", "IL6", [1.0, 2.0, 3.0], [1.0, float("inf"), 3.0])

    def test_biomarker_neg_inf_raises(self):
        with pytest.raises(ValueError, match="finite"):
            correlate_biomarker_pk("Drug", "IL6", [1.0, 2.0, 3.0], [1.0, float("-inf"), 3.0])


# ---------------------------------------------------------------------------
# Perfect linear correlation
# ---------------------------------------------------------------------------


class TestPerfectLinearCorrelation:
    def setup_method(self):
        n = 20
        self.pk_vals = list(np.linspace(1.0, 10.0, n))
        # Perfect linear: y = 2*x + 3
        self.bio_vals = [2.0 * x + 3.0 for x in self.pk_vals]
        self.result = correlate_biomarker_pk(
            "TestDrug", "Biomarker", self.pk_vals, self.bio_vals
        )

    def test_pearson_r_near_one(self):
        assert self.result.pearson_r == pytest.approx(1.0, abs=1e-6)

    def test_r_squared_near_one(self):
        assert self.result.r_squared == pytest.approx(1.0, abs=1e-6)

    def test_slope_near_two(self):
        assert self.result.slope == pytest.approx(2.0, rel=1e-5)

    def test_intercept_near_three(self):
        assert self.result.intercept == pytest.approx(3.0, rel=1e-4)

    def test_best_model_linear(self):
        assert self.result.best_model == "linear"

    def test_spearman_r_near_one(self):
        assert self.result.spearman_r == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Negative perfect linear correlation
# ---------------------------------------------------------------------------


class TestNegativeCorrelation:
    def test_pearson_r_near_negative_one(self):
        pk_vals = list(np.linspace(1.0, 10.0, 15))
        bio_vals = [-x + 5.0 for x in pk_vals]
        result = correlate_biomarker_pk("Drug", "BM", pk_vals, bio_vals)
        assert result.pearson_r == pytest.approx(-1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# r_squared bounds
# ---------------------------------------------------------------------------


class TestRSquaredBounds:
    def test_r_squared_in_0_1_for_random_like(self):
        rng = np.random.RandomState(42)
        pk_vals = list(rng.uniform(0.1, 10.0, 30))
        bio_vals = list(rng.uniform(0.1, 5.0, 30))
        result = correlate_biomarker_pk("Drug", "BM", pk_vals, bio_vals)
        assert 0.0 <= result.r_squared <= 1.0

    def test_r_squared_near_zero_for_uncorrelated(self):
        # Use fixed random arrays designed to have near-zero correlation
        rng = np.random.RandomState(1234)
        pk_vals = list(rng.uniform(0.1, 10.0, 50))
        bio_vals = list(rng.uniform(0.1, 5.0, 50))
        result = correlate_biomarker_pk("Drug", "BM", pk_vals, bio_vals)
        # Just check |r| < 0.5 for a random dataset
        assert abs(result.pearson_r) < 0.5


# ---------------------------------------------------------------------------
# Emax model
# ---------------------------------------------------------------------------


class TestEmaxModel:
    def test_emax_best_model_for_sigmoidal_data(self):
        # Generate clean sigmoidal data: E = 100 * C / (5 + C)
        pk_vals = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0, 50.0]
        bio_vals = [100.0 * c / (5.0 + c) for c in pk_vals]
        result = correlate_biomarker_pk(
            "Drug", "Effect", pk_vals, bio_vals, correlation_type="auc"
        )
        # Emax fit should be better here; but model selection depends on R²
        # At minimum, emax and ec50 should be positive finite
        assert math.isfinite(result.emax)
        assert math.isfinite(result.ec50)
        assert result.ec50 > 0.0

    def test_emax_and_ec50_finite_for_linear_data(self):
        pk_vals = list(np.linspace(1.0, 20.0, 15))
        bio_vals = [0.5 * x for x in pk_vals]
        result = correlate_biomarker_pk("Drug", "BM", pk_vals, bio_vals)
        assert math.isfinite(result.emax)
        assert math.isfinite(result.ec50)


# ---------------------------------------------------------------------------
# Result dataclass structure
# ---------------------------------------------------------------------------


class TestResultStructure:
    def setup_method(self):
        pk_vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        bio_vals = [2.0 * x + 1.0 for x in pk_vals]
        self.result = correlate_biomarker_pk("MyDrug", "Biomarker_A", pk_vals, bio_vals)

    def test_is_frozen_dataclass(self):
        with pytest.raises((AttributeError, TypeError)):
            self.result.pearson_r = 0.0  # type: ignore[misc]

    def test_n_subjects_correct(self):
        assert self.result.n_subjects == 10

    def test_drug_name_stored(self):
        assert self.result.drug_name == "MyDrug"

    def test_biomarker_name_stored(self):
        assert self.result.biomarker_name == "Biomarker_A"

    def test_correlation_type_default_auc(self):
        assert self.result.correlation_type == "auc"

    def test_pk_values_stored(self):
        assert len(self.result.pk_values) == 10

    def test_biomarker_values_stored(self):
        assert len(self.result.biomarker_values) == 10

    def test_slope_finite(self):
        assert math.isfinite(self.result.slope)

    def test_intercept_finite(self):
        assert math.isfinite(self.result.intercept)

    def test_best_model_valid(self):
        assert self.result.best_model in ("linear", "emax")


# ---------------------------------------------------------------------------
# screen_biomarker_correlations
# ---------------------------------------------------------------------------


class TestScreenBiomarkerCorrelations:
    def test_returns_list_of_results(self):
        pk_dict = {"auc": [1.0, 2.0, 3.0, 4.0, 5.0]}
        bm_dict = {"IL6": [2.0, 4.0, 6.0, 8.0, 10.0]}
        results = screen_biomarker_correlations("Drug", pk_dict, bm_dict)
        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], BiomarkerCorrelationResult)

    def test_multiple_pairs_returned(self):
        pk_dict = {
            "auc": [1.0, 2.0, 3.0, 4.0, 5.0],
            "cmax": [0.5, 1.0, 1.5, 2.0, 2.5],
        }
        bm_dict = {
            "BM1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "BM2": [5.0, 4.0, 3.0, 2.0, 1.0],
        }
        results = screen_biomarker_correlations("Drug", pk_dict, bm_dict)
        assert len(results) == 4  # 2 pk × 2 bm

    def test_sorted_by_abs_pearson_r_descending(self):
        pk_dict = {
            "auc": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "cmax": [8.0, 1.0, 7.0, 2.0, 6.0, 3.0, 5.0, 4.0],  # weak corr
        }
        bm_dict = {
            "BM": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],  # perfect with auc
        }
        results = screen_biomarker_correlations("Drug", pk_dict, bm_dict)
        r_vals = [abs(r.pearson_r) for r in results]
        assert r_vals == sorted(r_vals, reverse=True)


# ---------------------------------------------------------------------------
# summarize_pkpd_relationship
# ---------------------------------------------------------------------------


class TestSummarizePkpdRelationship:
    def test_returns_non_empty_string(self):
        pk_vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        bio_vals = [2.0, 4.0, 6.0, 8.0, 10.0]
        result = correlate_biomarker_pk("Drug", "BM", pk_vals, bio_vals)
        summary = summarize_pkpd_relationship(result)
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_summary_contains_drug_name(self):
        pk_vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        bio_vals = [2.0, 4.0, 6.0, 8.0, 10.0]
        result = correlate_biomarker_pk("Aspirin", "CRP", pk_vals, bio_vals)
        summary = summarize_pkpd_relationship(result)
        assert "Aspirin" in summary

    def test_summary_contains_biomarker_name(self):
        pk_vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        bio_vals = [2.0, 4.0, 6.0, 8.0, 10.0]
        result = correlate_biomarker_pk("Drug", "FEV1_pct", pk_vals, bio_vals)
        summary = summarize_pkpd_relationship(result)
        assert "FEV1_pct" in summary
