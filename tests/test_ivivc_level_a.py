"""Tests for biopharmaceutics/ivivc_level_a.py (Phase 530)."""

import math

import pytest

from omega_pbpk.biopharmaceutics.ivivc_level_a import (
    assess_ivivc_quality,
    fit_ivivc_correlation,
    loo_riegelman_deconvolution,
    wagner_nelson_deconvolution,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_1cpt_oral_profile(
    ka: float,
    ke: float,
    dose_mg: float,
    vd_L: float,
    cl_L_per_h: float,
    times_h: list,
) -> list:
    """Analytical 1-cpt oral plasma concentration profile."""
    concs = []
    for t in times_h:
        if abs(ka - ke) < 1e-9:
            c = (ka * dose_mg / vd_L) * t * math.exp(-ke * t)
        else:
            c = (ka * dose_mg / (vd_L * (ka - ke))) * (math.exp(-ke * t) - math.exp(-ka * t))
        concs.append(max(0.0, c))
    return concs


# ---------------------------------------------------------------------------
# wagner_nelson_deconvolution
# ---------------------------------------------------------------------------


class TestWagnerNelsonDeconvolution:
    def test_returns_required_keys(self):
        times = [0.0, 1.0, 2.0, 4.0, 8.0]
        concs = [0.0, 2.0, 3.0, 2.5, 1.0]
        result = wagner_nelson_deconvolution(times, concs, dose_mg=100.0, cl_L_per_h=5.0, vd_L=20.0)
        assert "times_h" in result
        assert "fa" in result
        assert "max_fa" in result

    def test_fa_length_matches_times(self):
        times = [0.0, 1.0, 2.0, 4.0, 8.0]
        concs = [0.0, 2.0, 3.0, 2.5, 1.0]
        result = wagner_nelson_deconvolution(times, concs, dose_mg=100.0, cl_L_per_h=5.0, vd_L=20.0)
        assert len(result["fa"]) == len(times)

    def test_fa_non_negative(self):
        times = [0.0, 1.0, 2.0, 4.0, 8.0]
        concs = [0.0, 2.0, 3.0, 2.5, 1.0]
        result = wagner_nelson_deconvolution(times, concs, dose_mg=100.0, cl_L_per_h=5.0, vd_L=20.0)
        assert all(f >= 0 for f in result["fa"])

    def test_fa_starts_near_zero(self):
        times = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0]
        ka, ke = 1.5, 0.3
        cl, vd = 3.0, 10.0
        dose = 100.0
        concs = _generate_1cpt_oral_profile(ka, ke, dose, vd, cl, times)
        result = wagner_nelson_deconvolution(times, concs, dose_mg=dose, cl_L_per_h=cl, vd_L=vd)
        assert result["fa"][0] < 0.1

    def test_fa_at_late_time_close_to_one_complete_absorption(self):
        # Complete absorption scenario: fa at end should approach 1
        times = [float(t) for t in range(0, 49)]
        ka, ke = 1.0, 0.1
        cl, vd = 1.0, 10.0
        dose = 100.0
        concs = _generate_1cpt_oral_profile(ka, ke, dose, vd, cl, times)
        result = wagner_nelson_deconvolution(times, concs, dose_mg=dose, cl_L_per_h=cl, vd_L=vd)
        # Max fa should be close to 1.0 for complete absorption
        assert result["max_fa"] > 0.85

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            wagner_nelson_deconvolution(
                [0.0, 1.0, 2.0], [0.0, 1.0], dose_mg=100.0, cl_L_per_h=5.0, vd_L=20.0
            )

    def test_invalid_dose_raises(self):
        with pytest.raises(ValueError):
            wagner_nelson_deconvolution(
                [0.0, 1.0], [0.0, 1.0], dose_mg=0, cl_L_per_h=5.0, vd_L=20.0
            )

    def test_invalid_cl_raises(self):
        with pytest.raises(ValueError):
            wagner_nelson_deconvolution(
                [0.0, 1.0], [0.0, 1.0], dose_mg=100.0, cl_L_per_h=0, vd_L=20.0
            )

    def test_invalid_vd_raises(self):
        with pytest.raises(ValueError):
            wagner_nelson_deconvolution(
                [0.0, 1.0], [0.0, 1.0], dose_mg=100.0, cl_L_per_h=5.0, vd_L=0
            )


# ---------------------------------------------------------------------------
# loo_riegelman_deconvolution
# ---------------------------------------------------------------------------


class TestLooRiegelmanDeconvolution:
    def _run_basic(self):
        times = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0]
        concs = [0.0, 1.5, 2.8, 3.2, 2.1, 0.9, 0.3]
        return loo_riegelman_deconvolution(
            times_h=times,
            c_plasma_mg_L=concs,
            dose_mg=100.0,
            cl_L_per_h=2.0,
            vd_central_L=10.0,
            vd_peripheral_L=15.0,
            k12_per_h=0.3,
            k21_per_h=0.2,
        )

    def test_returns_times_and_fa(self):
        result = self._run_basic()
        assert "times_h" in result
        assert "fa" in result

    def test_fa_length_matches_times(self):
        result = self._run_basic()
        assert len(result["fa"]) == len(result["times_h"])

    def test_fa_non_negative(self):
        result = self._run_basic()
        assert all(f >= 0 for f in result["fa"])

    def test_fa_increases_over_time_initially(self):
        result = self._run_basic()
        # fa should increase from 0 initially
        assert result["fa"][-1] > result["fa"][0]

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            loo_riegelman_deconvolution(
                [0.0, 1.0, 2.0],
                [0.0, 1.0],
                dose_mg=100.0,
                cl_L_per_h=2.0,
                vd_central_L=10.0,
                vd_peripheral_L=15.0,
                k12_per_h=0.3,
                k21_per_h=0.2,
            )

    def test_invalid_dose_raises(self):
        with pytest.raises(ValueError):
            loo_riegelman_deconvolution(
                [0.0, 1.0],
                [0.0, 1.0],
                dose_mg=0,
                cl_L_per_h=2.0,
                vd_central_L=10.0,
                vd_peripheral_L=15.0,
                k12_per_h=0.3,
                k21_per_h=0.2,
            )


# ---------------------------------------------------------------------------
# fit_ivivc_correlation
# ---------------------------------------------------------------------------


class TestFitIvivcCorrelation:
    def test_perfect_linear_data_r2_one(self):
        x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        y = [0.1 * v + 0.05 for v in x]  # exact linear: slope=0.1, intercept=0.05
        result = fit_ivivc_correlation(x, y)
        assert abs(result["r_squared"] - 1.0) < 1e-8

    def test_perfect_data_slope_and_intercept(self):
        x = [0.2, 0.4, 0.6, 0.8, 1.0]
        slope_true = 0.9
        intercept_true = 0.05
        y = [slope_true * xi + intercept_true for xi in x]
        result = fit_ivivc_correlation(x, y)
        assert abs(result["slope"] - slope_true) < 1e-8
        assert abs(result["intercept"] - intercept_true) < 1e-8

    def test_noisy_data_r2_less_than_one(self):
        x = [0.1, 0.3, 0.5, 0.7, 0.9]
        y = [0.15, 0.28, 0.52, 0.68, 0.95]  # slightly noisy
        result = fit_ivivc_correlation(x, y)
        assert result["r_squared"] < 1.0
        assert result["r_squared"] > 0.0

    def test_returns_required_keys(self):
        x = [0.1, 0.5, 1.0]
        y = [0.1, 0.5, 1.0]
        result = fit_ivivc_correlation(x, y)
        assert "slope" in result
        assert "intercept" in result
        assert "r_squared" in result
        assert "predictive_error_pct" in result

    def test_perfect_data_zero_predictive_error(self):
        x = [0.2, 0.5, 0.8]
        y = [0.4, 1.0, 1.6]  # y = 2x exactly
        result = fit_ivivc_correlation(x, y)
        assert result["predictive_error_pct"] < 1e-6

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            fit_ivivc_correlation([0.1, 0.5], [0.1, 0.5, 0.9])

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError):
            fit_ivivc_correlation([0.5], [0.5])

    def test_identity_correlation(self):
        # fa_in_vivo = fa_in_vitro → slope=1, intercept=0
        x = [0.1, 0.3, 0.5, 0.7, 0.9]
        result = fit_ivivc_correlation(x, x)
        assert abs(result["slope"] - 1.0) < 1e-8
        assert abs(result["intercept"]) < 1e-8
        assert abs(result["r_squared"] - 1.0) < 1e-8


# ---------------------------------------------------------------------------
# assess_ivivc_quality
# ---------------------------------------------------------------------------


class TestAssessIvivcQuality:
    def test_accepted_high_r2_low_pe(self):
        result = assess_ivivc_quality(r_squared=0.95, predictive_error_pct=5.0)
        assert result["quality"] == "Level A accepted"

    def test_not_accepted_low_r2_high_pe(self):
        result = assess_ivivc_quality(r_squared=0.70, predictive_error_pct=20.0)
        assert "not accepted" in result["quality"]

    def test_borderline_high_r2_high_pe(self):
        result = assess_ivivc_quality(r_squared=0.92, predictive_error_pct=15.0)
        assert "borderline" in result["quality"] or "not accepted" in result["quality"]

    def test_borderline_low_r2_low_pe(self):
        result = assess_ivivc_quality(r_squared=0.85, predictive_error_pct=8.0)
        assert "borderline" in result["quality"] or "not accepted" in result["quality"]

    def test_returns_required_keys(self):
        result = assess_ivivc_quality(r_squared=0.95, predictive_error_pct=5.0)
        assert "quality" in result
        assert "r_squared" in result
        assert "predictive_error_pct" in result
        assert "recommendation" in result

    def test_recommendation_is_string(self):
        result = assess_ivivc_quality(r_squared=0.95, predictive_error_pct=5.0)
        assert isinstance(result["recommendation"], str)
        assert len(result["recommendation"]) > 10

    def test_exact_boundary_r2_accepted(self):
        # r²=0.90 and PE=10 → exactly on boundary, should be accepted
        result = assess_ivivc_quality(r_squared=0.90, predictive_error_pct=10.0)
        assert result["quality"] == "Level A accepted"

    def test_values_preserved_in_output(self):
        r2 = 0.93
        pe = 7.5
        result = assess_ivivc_quality(r_squared=r2, predictive_error_pct=pe)
        assert result["r_squared"] == r2
        assert result["predictive_error_pct"] == pe
