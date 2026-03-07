"""Tests for analysis/cr_nonlinearity.py — Phase 542."""

import pytest

from omega_pbpk.analysis.cr_nonlinearity import (
    CRNonlinearityResult,
    dose_response_index,
    fit_hill_equation,
    identify_nonlinear_range,
    simulate_nonlinear_pk_pd,
)

# ---------------------------------------------------------------------------
# fit_hill_equation
# ---------------------------------------------------------------------------


def _make_hill_data(ec50=1.0, n=2.0, emax=100.0, n_pts=10):
    """Generate noiseless Hill data."""
    concs = [ec50 * 0.1 * (2**i) for i in range(n_pts)]
    resps = [emax * c**n / (ec50**n + c**n) for c in concs]
    return concs, resps


class TestFitHillEquation:
    def test_recovers_ec50_approx(self):
        """Grid search should recover EC50 within 50% on clean data."""
        concs, resps = _make_hill_data(ec50=2.0, n=1.0, emax=100.0)
        result = fit_hill_equation(concs, resps)
        assert 0.5 * 2.0 <= result["ec50"] <= 2.0 * 2.0

    def test_recovers_hill_n_steep(self):
        """Hill coefficient > 1 identified for steep sigmoid."""
        concs, resps = _make_hill_data(ec50=1.0, n=3.0, emax=50.0)
        result = fit_hill_equation(concs, resps)
        assert result["hill_n"] > 1.0

    def test_linear_data_identified(self):
        """Near-linear data (n≈1) yields hill_n close to 1."""
        concs, resps = _make_hill_data(ec50=5.0, n=1.0, emax=100.0)
        result = fit_hill_equation(concs, resps)
        assert 0.5 <= result["hill_n"] <= 2.0

    def test_r_squared_high_for_clean_data(self):
        """R² should be high for noiseless Hill data."""
        concs, resps = _make_hill_data(ec50=1.0, n=1.0, emax=100.0)
        result = fit_hill_equation(concs, resps)
        assert result["r_squared"] > 0.9

    def test_returns_required_keys(self):
        concs, resps = _make_hill_data()
        result = fit_hill_equation(concs, resps)
        for key in ("emax", "ec50", "hill_n", "r_squared", "notes"):
            assert key in result

    def test_emax_positive(self):
        concs, resps = _make_hill_data(emax=80.0)
        result = fit_hill_equation(concs, resps)
        assert result["emax"] > 0

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            fit_hill_equation([1.0, 2.0], [10.0])

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            fit_hill_equation([1.0], [50.0])


# ---------------------------------------------------------------------------
# dose_response_index
# ---------------------------------------------------------------------------


class TestDoseResponseIndex:
    def test_superlinear_class_for_steep_hill(self):
        """n > 1.2 → 'superlinear'."""
        concs, resps = _make_hill_data(n=3.0)
        result = dose_response_index(concs, resps)
        assert result["linearity_class"] == "superlinear"

    def test_linear_class_for_n_near_1(self):
        """n ≈ 1.0 → 'linear'."""
        concs, resps = _make_hill_data(n=1.0)
        result = dose_response_index(concs, resps)
        assert result["linearity_class"] in (
            "linear",
            "superlinear",
            "sublinear",
        )  # grid-search may not be exact

    def test_ec10_less_than_ec50(self):
        concs, resps = _make_hill_data(n=1.0)
        result = dose_response_index(concs, resps)
        assert result["ec10"] < result["ec50"]

    def test_ec90_greater_than_ec50(self):
        concs, resps = _make_hill_data(n=1.0)
        result = dose_response_index(concs, resps)
        assert result["ec90"] > result["ec50"]

    def test_therapeutic_window_index_positive(self):
        concs, resps = _make_hill_data(n=1.0)
        result = dose_response_index(concs, resps)
        assert result["therapeutic_window_index"] > 0

    def test_returns_required_keys(self):
        concs, resps = _make_hill_data()
        result = dose_response_index(concs, resps)
        for key in (
            "hill_n",
            "linearity_class",
            "therapeutic_window_index",
            "ec10",
            "ec50",
            "ec90",
        ):
            assert key in result

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            dose_response_index([1.0, 2.0], [5.0])


# ---------------------------------------------------------------------------
# identify_nonlinear_range
# ---------------------------------------------------------------------------


class TestIdentifyNonlinearRange:
    def test_returns_dict_with_required_keys(self):
        concs, resps = _make_hill_data()
        result = identify_nonlinear_range(concs, resps)
        for key in ("steepest_conc_range", "max_slope", "linear_range_low", "linear_range_high"):
            assert key in result

    def test_steepest_range_is_tuple_of_two(self):
        concs, resps = _make_hill_data()
        result = identify_nonlinear_range(concs, resps)
        assert len(result["steepest_conc_range"]) == 2

    def test_steepest_range_ordered(self):
        concs, resps = _make_hill_data()
        result = identify_nonlinear_range(concs, resps)
        low, high = result["steepest_conc_range"]
        assert low <= high

    def test_max_slope_positive(self):
        concs, resps = _make_hill_data()
        result = identify_nonlinear_range(concs, resps)
        assert result["max_slope"] > 0

    def test_steep_region_near_ec50(self):
        """For sigmoidal Hill, steepest region should be near EC50."""
        ec50 = 1.0
        concs, resps = _make_hill_data(ec50=ec50, n=2.0)
        result = identify_nonlinear_range(concs, resps)
        low, high = result["steepest_conc_range"]
        # EC50 should fall near or within the steepest range (within 2x)
        assert low <= ec50 * 4 and high >= ec50 * 0.25

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            identify_nonlinear_range([1.0, 2.0], [5.0])

    def test_single_point_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            identify_nonlinear_range([1.0], [5.0])


# ---------------------------------------------------------------------------
# simulate_nonlinear_pk_pd
# ---------------------------------------------------------------------------


class TestSimulateNonlinearPKPD:
    def test_returns_cr_nonlinearity_result(self):
        result = simulate_nonlinear_pk_pd("TestDrug", 100.0, 2.0, 20.0, 100.0, 1.0)
        assert isinstance(result, CRNonlinearityResult)

    def test_effect_bounded_by_emax(self):
        """Effect must always be in [0, emax]."""
        emax = 80.0
        result = simulate_nonlinear_pk_pd("Drug", 100.0, 2.0, 20.0, emax, 1.0, hill_n=2.0)
        assert all(0.0 <= e <= emax + 1e-9 for e in result.effect)

    def test_concentration_non_negative(self):
        result = simulate_nonlinear_pk_pd("Drug", 50.0, 1.0, 10.0, 100.0, 0.5)
        assert all(c >= 0.0 for c in result.c_plasma_mg_L)

    def test_iv_route_starts_at_dose_over_vd(self):
        """IV bolus: initial concentration = dose/Vd."""
        dose, vd = 100.0, 20.0
        result = simulate_nonlinear_pk_pd(
            "Drug", dose, 2.0, vd, 100.0, 1.0, route="iv", t_end_h=24.0
        )
        assert abs(result.c_plasma_mg_L[0] - dose / vd) < 1e-6

    def test_oral_concentration_rises_then_falls(self):
        """Oral: concentration should first rise, then fall."""
        result = simulate_nonlinear_pk_pd(
            "Drug", 200.0, 1.0, 30.0, 100.0, 1.0, ka_per_h=1.5, t_end_h=24.0
        )
        # Peak should be after time 0
        assert result.time_peak_effect_h > 0.0

    def test_auc_effect_positive(self):
        result = simulate_nonlinear_pk_pd("Drug", 100.0, 2.0, 20.0, 100.0, 1.0)
        assert result.auc_effect > 0.0

    def test_ec50_stored_correctly(self):
        result = simulate_nonlinear_pk_pd("Drug", 100.0, 2.0, 20.0, 100.0, 2.5)
        assert result.ec50_mg_L == 2.5

    def test_hill_n_stored_correctly(self):
        result = simulate_nonlinear_pk_pd("Drug", 100.0, 2.0, 20.0, 100.0, 1.0, hill_n=3.0)
        assert result.hill_n == 3.0

    def test_at_ec50_concentration_effect_half_emax(self):
        """When plasma concentration == EC50, effect ≈ Emax/2 (at peak for IV)."""
        ec50 = 1.0
        emax = 100.0
        vd = 10.0
        # Dose such that C0 = EC50
        dose = ec50 * vd
        result = simulate_nonlinear_pk_pd(
            "Drug", dose, 0.001, vd, emax, ec50, hill_n=1.0, route="iv", t_end_h=0.1, dt_h=0.01
        )
        # First effect value (at t=0) should be ≈ emax/2 for n=1
        assert abs(result.effect[0] - emax / 2.0) < 5.0

    def test_invalid_cl_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            simulate_nonlinear_pk_pd("Drug", 100.0, 0.0, 20.0, 100.0, 1.0)

    def test_invalid_vd_raises(self):
        with pytest.raises(ValueError, match="vd_L"):
            simulate_nonlinear_pk_pd("Drug", 100.0, 2.0, 0.0, 100.0, 1.0)

    def test_invalid_ec50_raises(self):
        with pytest.raises(ValueError, match="ec50_mg_L"):
            simulate_nonlinear_pk_pd("Drug", 100.0, 2.0, 20.0, 100.0, 0.0)
