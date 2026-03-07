"""Tests for clinical/population_pk_sim.py — Phase 476."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.clinical.population_pk_sim import (
    PopulationPKResult,
    identify_outliers,
    simulate_population,
    summarize_population,
)

# ---------------------------------------------------------------------------
# Basic structure and reproducibility
# ---------------------------------------------------------------------------


class TestSimulatePopulationBasic:
    def test_returns_dataclass(self):
        result = simulate_population(10, 5.0, 50.0, 100.0, seed=42)
        assert isinstance(result, PopulationPKResult)

    def test_n_subjects_correct(self):
        result = simulate_population(100, 5.0, 50.0, 100.0, seed=1)
        assert result.n_subjects == 100
        assert len(result.cmax_values) == 100
        assert len(result.auc_values) == 100
        assert len(result.cl_values) == 100
        assert len(result.vd_values) == 100
        assert len(result.concentration_profiles) == 100

    def test_seeded_reproducibility(self):
        r1 = simulate_population(20, 5.0, 50.0, 100.0, seed=99)
        r2 = simulate_population(20, 5.0, 50.0, 100.0, seed=99)
        assert r1.cmax_values == r2.cmax_values
        assert r1.auc_values == r2.auc_values
        assert r1.cl_values == r2.cl_values

    def test_different_seeds_different_results(self):
        r1 = simulate_population(20, 5.0, 50.0, 100.0, seed=1)
        r2 = simulate_population(20, 5.0, 50.0, 100.0, seed=2)
        assert r1.cmax_values != r2.cmax_values

    def test_cmax_positive_for_all_subjects(self):
        result = simulate_population(50, 5.0, 50.0, 100.0, seed=7)
        for cmax in result.cmax_values:
            assert cmax > 0.0

    def test_auc_positive_for_all_subjects(self):
        result = simulate_population(50, 5.0, 50.0, 100.0, seed=7)
        for auc in result.auc_values:
            assert auc > 0.0

    def test_profile_length_matches_times(self):
        result = simulate_population(10, 5.0, 50.0, 100.0, seed=0)
        n_times = len(result.times_h)
        for profile in result.concentration_profiles:
            assert len(profile) == n_times

    def test_times_start_at_zero(self):
        result = simulate_population(10, 5.0, 50.0, 100.0, seed=0)
        assert result.times_h[0] == 0.0

    def test_times_end_at_t_end(self):
        result = simulate_population(10, 5.0, 50.0, 100.0, t_end_h=12.0, dt_h=0.5, seed=0)
        assert abs(result.times_h[-1] - 12.0) < 1e-9

    def test_notes_not_empty(self):
        result = simulate_population(10, 5.0, 50.0, 100.0, seed=0)
        assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# BSV / variability
# ---------------------------------------------------------------------------


class TestBSVVariability:
    def test_higher_omega_cl_wider_spread(self):
        """Higher omega_cl should yield higher coefficient of variation for CL."""
        r_low = simulate_population(200, 5.0, 50.0, 100.0, omega_cl=0.1, seed=11)
        r_high = simulate_population(200, 5.0, 50.0, 100.0, omega_cl=0.5, seed=11)
        cv_low = _gcv(r_low.cl_values)
        cv_high = _gcv(r_high.cl_values)
        assert cv_high > cv_low

    def test_zero_omega_cl_uniform_cl(self):
        """With omega=0, all CL values should equal the typical value."""
        r = simulate_population(20, 5.0, 50.0, 100.0, omega_cl=0.0, omega_vd=0.0, seed=5)
        for cl in r.cl_values:
            assert abs(cl - 5.0) < 1e-10
        for vd in r.vd_values:
            assert abs(vd - 50.0) < 1e-10

    def test_median_cmax_close_to_typical_iv(self):
        """IV: Cmax ≈ dose/Vd for typical patient."""
        dose = 100.0
        vd_typ = 50.0
        r = simulate_population(300, 5.0, vd_typ, dose, omega_cl=0.01, omega_vd=0.01, seed=42)
        expected_cmax = dose / vd_typ  # 2.0 mg/L
        assert abs(r.cmax_p50 / expected_cmax - 1.0) < 0.05  # within 5%

    def test_gcv_approximation(self):
        """Geometric CV ≈ omega for moderate omega values."""
        omega = 0.3
        r = simulate_population(500, 5.0, 50.0, 100.0, omega_cl=omega, omega_vd=0.0, seed=99)
        gcv = _gcv(r.cl_values)
        # GCV = sqrt(exp(omega²) - 1) ≈ omega for small omega
        expected_gcv = math.sqrt(math.exp(omega**2) - 1.0)
        assert abs(gcv - expected_gcv) < 0.05  # within 5 percentage points


# ---------------------------------------------------------------------------
# Dose linearity
# ---------------------------------------------------------------------------


class TestDoseLinearity:
    def test_double_dose_double_cmax_iv(self):
        """IV linear PK: 2× dose → 2× Cmax (at same seed, deterministic)."""
        r1 = simulate_population(100, 5.0, 50.0, 100.0, omega_cl=0.0, omega_vd=0.0, seed=1)
        r2 = simulate_population(100, 5.0, 50.0, 200.0, omega_cl=0.0, omega_vd=0.0, seed=1)
        for c1, c2 in zip(r1.cmax_values, r2.cmax_values):
            assert abs(c2 / c1 - 2.0) < 1e-9

    def test_double_dose_double_auc_iv(self):
        r1 = simulate_population(50, 5.0, 50.0, 100.0, omega_cl=0.0, omega_vd=0.0, seed=2)
        r2 = simulate_population(50, 5.0, 50.0, 200.0, omega_cl=0.0, omega_vd=0.0, seed=2)
        for a1, a2 in zip(r1.auc_values, r2.auc_values):
            assert abs(a2 / a1 - 2.0) < 1e-6

    def test_double_dose_double_median_cmax(self):
        """2× dose → ~2× median Cmax (linear PK)."""
        r1 = simulate_population(200, 5.0, 50.0, 100.0, seed=55)
        r2 = simulate_population(200, 5.0, 50.0, 200.0, seed=55)
        assert abs(r2.cmax_p50 / r1.cmax_p50 - 2.0) < 0.05


# ---------------------------------------------------------------------------
# Percentile ordering
# ---------------------------------------------------------------------------


class TestPercentileOrdering:
    def test_cmax_p5_lt_p50_lt_p95(self):
        r = simulate_population(100, 5.0, 50.0, 100.0, seed=3)
        assert r.cmax_p5 < r.cmax_p50 < r.cmax_p95

    def test_auc_p5_lt_p50_lt_p95(self):
        r = simulate_population(100, 5.0, 50.0, 100.0, seed=3)
        assert r.auc_p5 < r.auc_p50 < r.auc_p95


# ---------------------------------------------------------------------------
# Oral route
# ---------------------------------------------------------------------------


class TestOralRoute:
    def test_oral_gives_cmax_after_time_zero(self):
        """For oral dosing, Cmax should NOT occur at t=0 (absorption phase)."""
        r = simulate_population(
            5, 5.0, 50.0, 100.0, route="oral", omega_cl=0.0, omega_vd=0.0, omega_ka=0.0, seed=0
        )
        for i, profile in enumerate(r.concentration_profiles):
            cmax_idx = profile.index(max(profile))
            assert cmax_idx > 0, f"Subject {i}: Cmax at t=0 for oral route"

    def test_oral_cmax_positive(self):
        r = simulate_population(20, 5.0, 50.0, 100.0, route="oral", seed=4)
        for cmax in r.cmax_values:
            assert cmax > 0.0


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_n_subjects_less_than_2_raises(self):
        with pytest.raises(ValueError):
            simulate_population(1, 5.0, 50.0, 100.0)

    def test_zero_cl_raises(self):
        with pytest.raises(ValueError):
            simulate_population(10, 0.0, 50.0, 100.0)

    def test_negative_cl_raises(self):
        with pytest.raises(ValueError):
            simulate_population(10, -1.0, 50.0, 100.0)

    def test_zero_vd_raises(self):
        with pytest.raises(ValueError):
            simulate_population(10, 5.0, 0.0, 100.0)

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError):
            simulate_population(10, 5.0, 50.0, 0.0)

    def test_invalid_route_raises(self):
        with pytest.raises(ValueError):
            simulate_population(10, 5.0, 50.0, 100.0, route="im")


# ---------------------------------------------------------------------------
# summarize_population
# ---------------------------------------------------------------------------


class TestSummarizePopulation:
    def test_returns_dict(self):
        r = simulate_population(20, 5.0, 50.0, 100.0, seed=0)
        summary = summarize_population(r)
        assert isinstance(summary, dict)

    def test_required_keys_present(self):
        r = simulate_population(20, 5.0, 50.0, 100.0, seed=0)
        summary = summarize_population(r)
        for key in (
            "cmax_p5",
            "cmax_p50",
            "cmax_p95",
            "auc_p5",
            "auc_p50",
            "auc_p95",
            "n_subjects",
        ):
            assert key in summary

    def test_p5_lt_p50_lt_p95_in_summary(self):
        r = simulate_population(100, 5.0, 50.0, 100.0, seed=7)
        s = summarize_population(r)
        assert s["cmax_p5"] < s["cmax_p50"] < s["cmax_p95"]
        assert s["auc_p5"] < s["auc_p50"] < s["auc_p95"]

    def test_n_subjects_in_summary(self):
        r = simulate_population(30, 5.0, 50.0, 100.0, seed=0)
        s = summarize_population(r)
        assert s["n_subjects"] == 30


# ---------------------------------------------------------------------------
# identify_outliers
# ---------------------------------------------------------------------------


class TestIdentifyOutliers:
    def test_uniform_population_no_outliers(self):
        """Perfectly uniform population → 0 outliers."""
        r = simulate_population(20, 5.0, 50.0, 100.0, omega_cl=0.0, omega_vd=0.0, seed=0)
        outliers = identify_outliers(r, n_sd=2.0)
        assert outliers["cmax_outliers"] == []
        assert outliers["auc_outliers"] == []

    def test_returns_dict_with_correct_keys(self):
        r = simulate_population(20, 5.0, 50.0, 100.0, seed=0)
        outliers = identify_outliers(r)
        assert "cmax_outliers" in outliers
        assert "auc_outliers" in outliers

    def test_outlier_indices_in_range(self):
        r = simulate_population(50, 5.0, 50.0, 100.0, omega_cl=0.5, seed=8)
        outliers = identify_outliers(r, n_sd=1.5)
        n = r.n_subjects
        for idx in outliers["cmax_outliers"]:
            assert 0 <= idx < n
        for idx in outliers["auc_outliers"]:
            assert 0 <= idx < n

    def test_fewer_outliers_with_larger_n_sd(self):
        r = simulate_population(200, 5.0, 50.0, 100.0, omega_cl=0.4, seed=13)
        o1 = identify_outliers(r, n_sd=1.0)
        o2 = identify_outliers(r, n_sd=3.0)
        assert len(o1["cmax_outliers"]) >= len(o2["cmax_outliers"])

    def test_invalid_n_sd_raises(self):
        r = simulate_population(10, 5.0, 50.0, 100.0, seed=0)
        with pytest.raises(ValueError):
            identify_outliers(r, n_sd=0.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gcv(values: list[float]) -> float:
    """Geometric coefficient of variation: sqrt(exp(sigma_log²) - 1)."""
    log_vals = [math.log(v) for v in values if v > 0]
    n = len(log_vals)
    if n < 2:
        return 0.0
    mean_log = sum(log_vals) / n
    var_log = sum((x - mean_log) ** 2 for x in log_vals) / (n - 1)
    return math.sqrt(math.exp(var_log) - 1.0)
