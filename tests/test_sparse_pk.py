"""Tests for omega_pbpk.clinical.sparse_pk module."""

import pytest
from omega_pbpk.clinical.sparse_pk import (
    SamplingResult,
    optimize_sparse_sampling,
    recommend_sampling_times,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DEFAULT_PK = {"cl_l_h": 5.0, "vd_l": 50.0, "ka_h": 1.2, "t_half_h": 6.9}


def make_result(n_samples=3, drug_name="TestDrug", **kwargs):
    pk = {**DEFAULT_PK, **kwargs}
    return optimize_sparse_sampling(drug_name, n_samples, pk)


# ---------------------------------------------------------------------------
# SamplingResult type checks
# ---------------------------------------------------------------------------


class TestSamplingResultType:
    def test_returns_sampling_result_instance(self):
        result = make_result()
        assert isinstance(result, SamplingResult)

    def test_dataclass_has_all_fields(self):
        result = make_result()
        for field in (
            "drug_name",
            "n_subjects",
            "n_samples_per_subject",
            "optimal_times_h",
            "predicted_cv_cl_pct",
            "predicted_cv_auc_pct",
            "d_optimality_criterion",
            "sampling_strategy",
        ):
            assert hasattr(result, field), f"Missing field: {field}"

    def test_drug_name_preserved(self):
        result = make_result(drug_name="Warfarin")
        assert result.drug_name == "Warfarin"

    def test_n_samples_per_subject_matches_input(self):
        result = make_result(n_samples=4)
        assert result.n_samples_per_subject == 4


# ---------------------------------------------------------------------------
# optimal_times_h length and bounds
# ---------------------------------------------------------------------------


class TestOptimalTimes:
    @pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 10])
    def test_optimal_times_length_equals_n_samples(self, n):
        result = make_result(n_samples=n)
        assert len(result.optimal_times_h) == n

    @pytest.mark.parametrize("n", [1, 3, 5, 8])
    def test_all_times_within_window_default_24h(self, n):
        result = make_result(n_samples=n)
        for t in result.optimal_times_h:
            assert 0 < t <= 24.0, f"Time {t} out of (0, 24] range"

    def test_custom_window_respected(self):
        result = optimize_sparse_sampling("Drug", 3, DEFAULT_PK, t_window_h=12.0)
        for t in result.optimal_times_h:
            assert 0 < t <= 12.0

    def test_times_are_positive(self):
        result = make_result(n_samples=5)
        assert all(t > 0 for t in result.optimal_times_h)


# ---------------------------------------------------------------------------
# Sampling strategy classification
# ---------------------------------------------------------------------------


class TestSamplingStrategy:
    def test_strategy_ultra_sparse_for_1_sample(self):
        result = make_result(n_samples=1)
        assert result.sampling_strategy == "ultra_sparse"

    def test_strategy_ultra_sparse_for_2_samples(self):
        result = make_result(n_samples=2)
        assert result.sampling_strategy == "ultra_sparse"

    def test_strategy_sparse_for_5_samples(self):
        result = make_result(n_samples=5)
        assert result.sampling_strategy == "sparse"

    def test_strategy_intensive_for_8_samples(self):
        result = make_result(n_samples=8)
        assert result.sampling_strategy == "intensive"

    def test_strategy_values_in_valid_set(self):
        valid = {"intensive", "sparse", "ultra_sparse"}
        for n in [1, 3, 5, 8]:
            result = make_result(n_samples=n)
            assert result.sampling_strategy in valid


# ---------------------------------------------------------------------------
# Predicted CV and D-optimality
# ---------------------------------------------------------------------------


class TestPredictedMetrics:
    def test_predicted_cv_cl_positive(self):
        result = make_result()
        assert result.predicted_cv_cl_pct > 0

    def test_predicted_cv_auc_positive(self):
        result = make_result()
        assert result.predicted_cv_auc_pct > 0

    def test_d_optimality_positive(self):
        result = make_result()
        assert result.d_optimality_criterion > 0


# ---------------------------------------------------------------------------
# Reproducibility with seed
# ---------------------------------------------------------------------------


class TestSeedReproducibility:
    def test_same_seed_gives_same_times(self):
        r1 = optimize_sparse_sampling("Drug", 4, DEFAULT_PK, seed=42)
        r2 = optimize_sparse_sampling("Drug", 4, DEFAULT_PK, seed=42)
        assert r1.optimal_times_h == r2.optimal_times_h

    def test_different_seed_may_differ(self):
        r1 = optimize_sparse_sampling("Drug", 4, DEFAULT_PK, seed=0)
        r2 = optimize_sparse_sampling("Drug", 4, DEFAULT_PK, seed=999)
        # Not guaranteed to differ, but very likely for stochastic optimizers
        # At minimum both must be valid
        assert len(r1.optimal_times_h) == 4
        assert len(r2.optimal_times_h) == 4


# ---------------------------------------------------------------------------
# ValueError for invalid n_samples
# ---------------------------------------------------------------------------


class TestValueErrors:
    def test_optimize_raises_for_zero_samples(self):
        with pytest.raises(ValueError):
            optimize_sparse_sampling("Drug", 0, DEFAULT_PK)

    def test_optimize_raises_for_negative_samples(self):
        with pytest.raises(ValueError):
            optimize_sparse_sampling("Drug", -1, DEFAULT_PK)

    def test_recommend_raises_for_zero_samples(self):
        with pytest.raises(ValueError):
            recommend_sampling_times("Drug", 0, t_half_h=6.0)

    def test_recommend_raises_for_negative_samples(self):
        with pytest.raises(ValueError):
            recommend_sampling_times("Drug", -3, t_half_h=6.0)


# ---------------------------------------------------------------------------
# recommend_sampling_times
# ---------------------------------------------------------------------------


class TestRecommendSamplingTimes:
    @pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
    def test_returns_exactly_n_times(self, n):
        times = recommend_sampling_times("Drug", n, t_half_h=8.0)
        assert len(times) == n

    def test_times_sorted_ascending(self):
        times = recommend_sampling_times("Drug", 5, t_half_h=6.0)
        assert times == sorted(times)

    def test_all_times_positive(self):
        times = recommend_sampling_times("Drug", 4, t_half_h=6.0)
        assert all(t > 0 for t in times)

    def test_returns_list(self):
        times = recommend_sampling_times("Drug", 3, t_half_h=6.0)
        assert isinstance(times, list)

    def test_tmax_influences_times(self):
        times_early = recommend_sampling_times("Drug", 3, t_half_h=6.0, tmax_h=1.0)
        times_late = recommend_sampling_times("Drug", 3, t_half_h=6.0, tmax_h=4.0)
        # At least the first time should differ between early and late tmax
        assert times_early != times_late

    def test_single_sample_returns_one_time(self):
        times = recommend_sampling_times("Drug", 1, t_half_h=4.0)
        assert len(times) == 1
        assert times[0] > 0
