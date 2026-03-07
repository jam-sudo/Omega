"""Tests for Phase 560 — pk_data_summary.py (~25 tests)."""

import math

import pytest

from omega_pbpk.analysis.pk_data_summary import (
    accumulation_ratio_ss,
    coefficient_of_variation,
    geometric_mean,
    nca_summary,
    summarize_population_pk,
    time_above_threshold,
)

# ---------------------------------------------------------------------------
# Helpers for generating monoexponential PK data
# ---------------------------------------------------------------------------


def _mono_exp_profile(dose=100.0, cl=5.0, vd=50.0, n_points=20, t_end=24.0):
    """Generate iv monoexponential concentration-time profile."""
    ke = cl / vd
    c0 = dose / vd
    times = [t_end * i / (n_points - 1) for i in range(n_points)]
    concs = [c0 * math.exp(-ke * t) for t in times]
    return times, concs, ke, cl, vd


# ---------------------------------------------------------------------------
# nca_summary
# ---------------------------------------------------------------------------


class TestNcaSummary:
    def setup_method(self):
        self.times, self.concs, self.ke, self.cl, self.vd = _mono_exp_profile()

    def test_cmax_correct(self):
        result = nca_summary(self.times, self.concs, dose_mg=100.0)
        assert result["cmax"] == pytest.approx(self.concs[0], rel=0.01)

    def test_tmax_correct(self):
        result = nca_summary(self.times, self.concs, dose_mg=100.0)
        assert result["tmax_h"] == pytest.approx(0.0, abs=0.1)

    def test_ke_terminal_approx_correct(self):
        result = nca_summary(self.times, self.concs, dose_mg=100.0)
        assert result["ke_terminal"] == pytest.approx(self.ke, rel=0.05)

    def test_t_half_from_ke(self):
        result = nca_summary(self.times, self.concs, dose_mg=100.0)
        expected_t_half = math.log(2) / self.ke
        assert result["t_half_h"] == pytest.approx(expected_t_half, rel=0.05)

    def test_cl_approx_correct(self):
        result = nca_summary(self.times, self.concs, dose_mg=100.0)
        assert result["cl_L_per_h"] == pytest.approx(self.cl, rel=0.10)

    def test_vz_approx_correct(self):
        result = nca_summary(self.times, self.concs, dose_mg=100.0)
        assert result["vz_L"] == pytest.approx(self.vd, rel=0.15)

    def test_pct_extrap_low_for_well_sampled(self):
        # Well-sampled profile to t=24h (many half-lives) → low extrapolation
        result = nca_summary(self.times, self.concs, dose_mg=100.0)
        assert result["pct_extrap"] < 30.0

    def test_auc_0_inf_greater_than_auc_0_last(self):
        result = nca_summary(self.times, self.concs, dose_mg=100.0)
        assert result["auc_0_inf"] >= result["auc_0_last"]

    def test_result_keys(self):
        result = nca_summary(self.times, self.concs, dose_mg=100.0)
        for key in [
            "cmax",
            "tmax_h",
            "auc_0_last",
            "auc_0_inf",
            "ke_terminal",
            "t_half_h",
            "cl_L_per_h",
            "vz_L",
            "pct_extrap",
        ]:
            assert key in result

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            nca_summary([0, 1, 2], [1.0, 0.5], dose_mg=100.0)

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError):
            nca_summary([0, 1], [1.0, 0.5], dose_mg=100.0)

    def test_nonpositive_dose_raises(self):
        with pytest.raises(ValueError):
            nca_summary(self.times, self.concs, dose_mg=0.0)


# ---------------------------------------------------------------------------
# coefficient_of_variation
# ---------------------------------------------------------------------------


class TestCoefficientOfVariation:
    def test_identical_values_cv_zero(self):
        cv = coefficient_of_variation([100.0, 100.0, 100.0])
        assert cv == pytest.approx(0.0, abs=1e-9)

    def test_known_cv(self):
        # [50, 150]: mean=100, std = sqrt(((50-100)^2+(150-100)^2)/1) = 70.71
        cv = coefficient_of_variation([50.0, 150.0])
        assert cv == pytest.approx(70.71, rel=0.01)

    def test_single_value_raises(self):
        with pytest.raises(ValueError):
            coefficient_of_variation([5.0])

    def test_zero_mean_raises(self):
        with pytest.raises(ValueError):
            coefficient_of_variation([1.0, -1.0])


# ---------------------------------------------------------------------------
# geometric_mean
# ---------------------------------------------------------------------------


class TestGeometricMean:
    def test_two_values(self):
        gm = geometric_mean([1.0, 10.0])
        assert gm == pytest.approx(math.sqrt(10.0), rel=1e-6)

    def test_single_value(self):
        gm = geometric_mean([5.0])
        assert gm == pytest.approx(5.0)

    def test_negative_value_raises(self):
        with pytest.raises(ValueError):
            geometric_mean([-1.0, 2.0])

    def test_zero_value_raises(self):
        with pytest.raises(ValueError):
            geometric_mean([0.0, 2.0])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            geometric_mean([])

    def test_known_three_values(self):
        # GM(2,8,32) = (2*8*32)^(1/3) = 512^(1/3) = 8
        gm = geometric_mean([2.0, 8.0, 32.0])
        assert gm == pytest.approx(8.0, rel=1e-6)


# ---------------------------------------------------------------------------
# summarize_population_pk
# ---------------------------------------------------------------------------


class TestSummarizePopulationPk:
    def setup_method(self):
        self.individuals = [
            {"cmax": 100.0, "auc_0_inf": 500.0, "t_half_h": 8.0, "cl_L_per_h": 4.0},
            {"cmax": 120.0, "auc_0_inf": 600.0, "t_half_h": 9.0, "cl_L_per_h": 5.0},
            {"cmax": 80.0, "auc_0_inf": 400.0, "t_half_h": 7.0, "cl_L_per_h": 3.0},
        ]

    def test_returns_nested_dict(self):
        result = summarize_population_pk(self.individuals)
        assert isinstance(result, dict)
        for param in ["cmax", "auc_0_inf", "t_half_h", "cl_L_per_h"]:
            assert param in result

    def test_stat_keys_present(self):
        result = summarize_population_pk(self.individuals)
        for stat in ["mean", "median", "std", "cv_pct", "geo_mean", "min", "max"]:
            assert stat in result["cmax"]

    def test_mean_correct(self):
        result = summarize_population_pk(self.individuals)
        assert result["cmax"]["mean"] == pytest.approx(100.0)

    def test_min_max_correct(self):
        result = summarize_population_pk(self.individuals)
        assert result["cmax"]["min"] == 80.0
        assert result["cmax"]["max"] == 120.0

    def test_skips_missing_params(self):
        individuals_partial = [
            {"cmax": 100.0, "auc_0_inf": 500.0},
            {"cmax": 120.0},  # missing auc_0_inf
        ]
        result = summarize_population_pk(individuals_partial)
        assert "cmax" in result
        assert "auc_0_inf" not in result

    def test_custom_params(self):
        result = summarize_population_pk(self.individuals, pk_params=["cmax"])
        assert "cmax" in result
        assert "t_half_h" not in result

    def test_empty_individuals_returns_empty(self):
        result = summarize_population_pk([])
        assert result == {}


# ---------------------------------------------------------------------------
# accumulation_ratio_ss
# ---------------------------------------------------------------------------


class TestAccumulationRatioSs:
    def test_tau_equals_t_half_rac_approx_2(self):
        # When tau = t_half: Rac = 1/(1 - exp(-ln2)) = 1/(1 - 0.5) = 2
        rac = accumulation_ratio_ss(t_half_h=8.0, tau_h=8.0)
        assert rac == pytest.approx(2.0, rel=1e-6)

    def test_short_tau_high_accumulation(self):
        # tau << t_half → large accumulation
        rac = accumulation_ratio_ss(t_half_h=24.0, tau_h=4.0)
        assert rac > 5.0

    def test_long_tau_low_accumulation(self):
        # tau >> t_half → Rac ≈ 1
        rac = accumulation_ratio_ss(t_half_h=1.0, tau_h=100.0)
        assert rac < 1.01

    def test_zero_t_half_raises(self):
        with pytest.raises(ValueError):
            accumulation_ratio_ss(t_half_h=0.0, tau_h=12.0)

    def test_zero_tau_raises(self):
        with pytest.raises(ValueError):
            accumulation_ratio_ss(t_half_h=8.0, tau_h=0.0)


# ---------------------------------------------------------------------------
# time_above_threshold
# ---------------------------------------------------------------------------


class TestTimeAboveThreshold:
    def test_all_above(self):
        times = [0.0, 1.0, 2.0, 3.0]
        concs = [5.0, 5.0, 5.0, 5.0]
        result = time_above_threshold(times, concs, threshold=1.0)
        assert result["fraction_above"] == pytest.approx(1.0)
        assert result["time_above_h"] == pytest.approx(3.0)

    def test_none_above(self):
        times = [0.0, 1.0, 2.0]
        concs = [0.5, 0.3, 0.1]
        result = time_above_threshold(times, concs, threshold=1.0)
        assert result["fraction_above"] == pytest.approx(0.0)
        assert result["n_crossings"] == 0

    def test_crossing_once(self):
        # Goes below threshold halfway
        times = [0.0, 1.0, 2.0]
        concs = [2.0, 2.0, 0.0]  # crosses at t=1.5 (midpoint between t=1 and t=2)
        result = time_above_threshold(times, concs, threshold=1.0)
        assert result["n_crossings"] == 1
        assert 0 < result["fraction_above"] < 1

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            time_above_threshold([0, 1], [1.0], threshold=0.5)

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError):
            time_above_threshold([0.0], [1.0], threshold=0.5)
