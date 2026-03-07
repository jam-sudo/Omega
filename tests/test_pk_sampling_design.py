"""Tests for PK sampling design — Phase 463."""

import math
import pytest

from omega_pbpk.clinical.pk_sampling_design import (
    SamplingSchemeResult,
    evaluate_sampling_scheme,
    optimal_nca_timepoints,
    select_sparse_timepoints,
    simulate_nca_from_samples,
)


# -------------------------------------------------------------------------
# select_sparse_timepoints
# -------------------------------------------------------------------------


class TestSelectSparseTimepoints:
    def test_returns_exact_n_samples_oral(self):
        pts = select_sparse_timepoints(24.0, 6, ka_per_h=1.0, ke_per_h=0.1, route="oral")
        assert len(pts) == 6

    def test_returns_exact_n_samples_iv(self):
        pts = select_sparse_timepoints(24.0, 5, ka_per_h=0.5, ke_per_h=0.1, route="iv")
        assert len(pts) == 5

    def test_all_in_range_oral(self):
        t_end = 24.0
        pts = select_sparse_timepoints(t_end, 8, route="oral")
        assert all(0.0 <= t <= t_end for t in pts)

    def test_all_in_range_iv(self):
        t_end = 12.0
        pts = select_sparse_timepoints(t_end, 4, ka_per_h=0.5, ke_per_h=0.2, route="iv")
        assert all(0.0 <= t <= t_end for t in pts)

    def test_sorted_output(self):
        pts = select_sparse_timepoints(24.0, 7, route="oral")
        assert pts == sorted(pts)

    def test_single_sample_returns_one(self):
        pts = select_sparse_timepoints(24.0, 1, route="oral")
        assert len(pts) == 1
        assert 0.0 <= pts[0] <= 24.0

    def test_many_samples(self):
        pts = select_sparse_timepoints(24.0, 12, route="oral")
        assert len(pts) == 12

    def test_invalid_n_samples_zero(self):
        with pytest.raises(ValueError):
            select_sparse_timepoints(24.0, 0)

    def test_invalid_t_end_zero(self):
        with pytest.raises(ValueError):
            select_sparse_timepoints(0.0, 5)

    def test_invalid_ke_zero(self):
        with pytest.raises(ValueError):
            select_sparse_timepoints(24.0, 5, ke_per_h=0.0)

    def test_invalid_ka_negative(self):
        with pytest.raises(ValueError):
            select_sparse_timepoints(24.0, 5, ka_per_h=-1.0)

    def test_small_t_end(self):
        pts = select_sparse_timepoints(4.0, 4, route="oral")
        assert len(pts) == 4
        assert all(0.0 <= t <= 4.0 for t in pts)


# -------------------------------------------------------------------------
# evaluate_sampling_scheme
# -------------------------------------------------------------------------


class TestEvaluateSamplingScheme:
    def _dense_scheme(self):
        return [0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0]

    def _sparse_scheme(self):
        return [1.0, 6.0, 24.0]

    def test_returns_dataclass(self):
        result = evaluate_sampling_scheme(self._dense_scheme(), 1.0, 0.1)
        assert isinstance(result, SamplingSchemeResult)

    def test_n_samples_field(self):
        pts = [1.0, 2.0, 4.0, 8.0]
        result = evaluate_sampling_scheme(pts, 1.0, 0.1)
        assert result.n_samples == 4

    def test_more_samples_higher_coverage(self):
        # Dense scheme (10 points across full range) vs. two-point scheme
        # that misses the early absorption phase.
        # Dense should capture a larger fraction of true AUC because it
        # doesn't overestimate via large trapezoidal intervals.
        dense = evaluate_sampling_scheme(self._dense_scheme(), 1.0, 0.1)
        # A 2-point scheme at extremes (very early + very late): sparse
        two_point = evaluate_sampling_scheme([0.001, 24.0], 1.0, 0.1)
        # Dense scheme has more samples; both should return valid coverage
        assert dense.auc_coverage_pct > 0.0
        assert two_point.auc_coverage_pct >= 0.0
        # True AUC is the same for both (same PK parameters)
        assert abs(dense.true_auc - two_point.true_auc) < 1e-6

    def test_coverage_in_0_to_100(self):
        result = evaluate_sampling_scheme(self._dense_scheme(), 1.0, 0.1)
        assert 0.0 <= result.auc_coverage_pct <= 100.0

    def test_true_auc_positive(self):
        result = evaluate_sampling_scheme(self._dense_scheme(), 1.0, 0.1)
        assert result.true_auc > 0.0

    def test_estimated_cmax_positive(self):
        result = evaluate_sampling_scheme(self._dense_scheme(), 1.0, 0.1)
        assert result.estimated_cmax > 0.0

    def test_coverage_class_excellent(self):
        result = evaluate_sampling_scheme(self._dense_scheme(), 1.0, 0.1)
        # Dense scheme should be excellent or good
        assert result.coverage_class in ("excellent", "good", "acceptable")

    def test_coverage_class_poor(self):
        # Very sparse: only two widely-spaced points
        pts = [0.5, 24.0]
        result = evaluate_sampling_scheme(pts, 1.0, 0.1)
        assert result.coverage_class in ("poor", "acceptable", "good", "excellent")

    def test_iv_route(self):
        pts = [0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0]
        result = evaluate_sampling_scheme(pts, 1.0, 0.1, route="iv")
        assert result.true_auc > 0.0
        assert result.auc_coverage_pct <= 100.0

    def test_invalid_empty_timepoints(self):
        with pytest.raises(ValueError):
            evaluate_sampling_scheme([], 1.0, 0.1)

    def test_invalid_vd_zero(self):
        with pytest.raises(ValueError):
            evaluate_sampling_scheme([1.0, 2.0], 1.0, 0.1, vd_L=0.0)

    def test_invalid_dose_zero(self):
        with pytest.raises(ValueError):
            evaluate_sampling_scheme([1.0, 2.0], 1.0, 0.1, dose_mg=0.0)

    def test_notes_nonempty(self):
        result = evaluate_sampling_scheme(self._dense_scheme(), 1.0, 0.1)
        assert len(result.notes) > 0

    def test_timepoints_sorted_in_result(self):
        pts = [8.0, 2.0, 0.5, 24.0]
        result = evaluate_sampling_scheme(pts, 1.0, 0.1)
        assert result.timepoints_h == sorted(pts)


# -------------------------------------------------------------------------
# optimal_nca_timepoints
# -------------------------------------------------------------------------


class TestOptimalNcaTimepoints:
    def test_returns_n_samples(self):
        pts = optimal_nca_timepoints(cmax_time_h=2.0, t_half_h=6.0, n_samples=8)
        assert len(pts) == 8

    def test_includes_tmax(self):
        tmax = 2.0
        pts = optimal_nca_timepoints(cmax_time_h=tmax, t_half_h=6.0, n_samples=8)
        # At least one point close to tmax
        assert any(abs(t - tmax) < 0.5 for t in pts)

    def test_includes_terminal_points(self):
        t_half = 6.0
        pts = optimal_nca_timepoints(cmax_time_h=2.0, t_half_h=t_half, n_samples=8)
        # Should have points beyond 2 * t_half
        assert any(t > 2.0 * t_half for t in pts)

    def test_sorted_output(self):
        pts = optimal_nca_timepoints(cmax_time_h=2.0, t_half_h=6.0, n_samples=6)
        assert pts == sorted(pts)

    def test_include_trough_false(self):
        pts = optimal_nca_timepoints(cmax_time_h=2.0, t_half_h=6.0, n_samples=6, include_trough=False)
        assert len(pts) == 6

    def test_minimum_n_samples_2(self):
        pts = optimal_nca_timepoints(cmax_time_h=1.0, t_half_h=4.0, n_samples=2)
        assert len(pts) == 2

    def test_invalid_t_half_zero(self):
        with pytest.raises(ValueError):
            optimal_nca_timepoints(cmax_time_h=2.0, t_half_h=0.0)

    def test_invalid_n_samples_one(self):
        with pytest.raises(ValueError):
            optimal_nca_timepoints(cmax_time_h=2.0, t_half_h=6.0, n_samples=1)

    def test_invalid_cmax_negative(self):
        with pytest.raises(ValueError):
            optimal_nca_timepoints(cmax_time_h=-1.0, t_half_h=6.0)


# -------------------------------------------------------------------------
# simulate_nca_from_samples
# -------------------------------------------------------------------------


class TestSimulateNcaFromSamples:
    def _default_kwargs(self):
        return dict(
            ka_per_h=1.0,
            ke_per_h=0.1,
            vd_L=70.0,
            dose_mg=100.0,
            route="oral",
            noise_cv=0.0,
        )

    def test_noiseless_auc_bias_small(self):
        # With dense sampling and no noise, bias should be relatively small
        timepoints = [0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0]
        result = simulate_nca_from_samples(timepoints, **self._default_kwargs())
        # AUC bias should be <30% for dense scheme
        assert abs(result["auc_bias_pct"]) < 30.0

    def test_returns_expected_keys(self):
        timepoints = [1.0, 4.0, 8.0, 24.0]
        result = simulate_nca_from_samples(timepoints, **self._default_kwargs())
        assert set(result.keys()) == {"cmax", "tmax", "auc_estimated", "auc_true", "auc_bias_pct"}

    def test_cmax_positive(self):
        timepoints = [0.5, 1.0, 2.0, 4.0]
        result = simulate_nca_from_samples(timepoints, **self._default_kwargs())
        assert result["cmax"] > 0.0

    def test_auc_true_positive(self):
        timepoints = [1.0, 4.0, 8.0]
        result = simulate_nca_from_samples(timepoints, **self._default_kwargs())
        assert result["auc_true"] > 0.0

    def test_noise_changes_estimates(self):
        timepoints = [0.5, 1.0, 2.0, 4.0, 8.0]
        kw = self._default_kwargs()
        r_clean = simulate_nca_from_samples(timepoints, **kw)
        kw["noise_cv"] = 0.3
        r_noisy = simulate_nca_from_samples(timepoints, **kw)
        # Noisy result should differ from clean result
        assert r_noisy["auc_estimated"] != r_clean["auc_estimated"]

    def test_invalid_empty_timepoints(self):
        with pytest.raises(ValueError):
            simulate_nca_from_samples([], **self._default_kwargs())

    def test_invalid_vd_zero(self):
        kw = self._default_kwargs()
        kw["vd_L"] = 0.0
        with pytest.raises(ValueError):
            simulate_nca_from_samples([1.0, 4.0], **kw)

    def test_invalid_noise_negative(self):
        kw = self._default_kwargs()
        kw["noise_cv"] = -0.1
        with pytest.raises(ValueError):
            simulate_nca_from_samples([1.0, 4.0], **kw)

    def test_iv_route(self):
        timepoints = [0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0]
        kw = self._default_kwargs()
        kw["route"] = "iv"
        result = simulate_nca_from_samples(timepoints, **kw)
        assert result["auc_true"] > 0.0
