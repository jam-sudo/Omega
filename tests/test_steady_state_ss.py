"""Tests for omega_pbpk.clinical.steady_state_ss."""

from __future__ import annotations

import numpy as np
import pytest

from omega_pbpk.clinical.steady_state_ss import (
    SteadyStateResult,
    simulate_to_steady_state,
    steady_state_metrics,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_KWARGS = dict(
    drug_name="TestDrug",
    dose_mg=100.0,
    interval_h=12.0,
    cl_L_per_h=5.0,
    vd_L=50.0,
)


def _run(**overrides):
    kw = {**_DEFAULT_KWARGS, **overrides}
    return simulate_to_steady_state(**kw)


# ---------------------------------------------------------------------------
# 1. Validation errors
# ---------------------------------------------------------------------------


class TestValidationErrors:
    def test_dose_zero_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _run(dose_mg=0.0)

    def test_dose_negative_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _run(dose_mg=-10.0)

    def test_interval_zero_raises(self):
        with pytest.raises(ValueError, match="interval_h"):
            _run(interval_h=0.0)

    def test_interval_negative_raises(self):
        with pytest.raises(ValueError, match="interval_h"):
            _run(interval_h=-6.0)

    def test_cl_zero_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            _run(cl_L_per_h=0.0)

    def test_vd_zero_raises(self):
        with pytest.raises(ValueError, match="vd_L"):
            _run(vd_L=0.0)

    def test_ka_zero_raises(self):
        with pytest.raises(ValueError, match="ka_per_h"):
            _run(ka_per_h=0.0)

    def test_max_cycles_too_small_raises(self):
        with pytest.raises(ValueError, match="max_cycles"):
            _run(max_cycles=2)


# ---------------------------------------------------------------------------
# 2. Return type and fields
# ---------------------------------------------------------------------------


class TestResultType:
    def test_returns_dataclass(self):
        result = _run()
        assert isinstance(result, SteadyStateResult)

    def test_has_all_expected_fields(self):
        result = _run()
        expected = {
            "drug_name",
            "dose_mg",
            "interval_h",
            "route",
            "n_cycles_to_ss",
            "css_max_mg_L",
            "css_min_mg_L",
            "css_avg_mg_L",
            "accumulation_ratio",
            "fluctuation_pct",
            "times_last3_cycles",
            "c_last3_cycles",
            "converged",
            "t_half_h",
            "notes",
        }
        for field in expected:
            assert hasattr(result, field), f"Missing field: {field}"

    def test_times_last3_is_ndarray(self):
        result = _run()
        assert isinstance(result.times_last3_cycles, np.ndarray)

    def test_c_last3_is_ndarray(self):
        result = _run()
        assert isinstance(result.c_last3_cycles, np.ndarray)

    def test_scalar_fields_are_float(self):
        result = _run()
        assert isinstance(result.css_max_mg_L, float)
        assert isinstance(result.css_min_mg_L, float)
        assert isinstance(result.css_avg_mg_L, float)
        assert isinstance(result.t_half_h, float)


# ---------------------------------------------------------------------------
# 3. Basic PK properties
# ---------------------------------------------------------------------------


class TestBasicPKProperties:
    def test_css_max_greater_than_css_min(self):
        result = _run()
        assert result.css_max_mg_L > result.css_min_mg_L

    def test_accumulation_ratio_geq_1(self):
        result = _run()
        assert result.accumulation_ratio >= 1.0

    def test_fluctuation_pct_geq_0(self):
        result = _run()
        assert result.fluctuation_pct >= 0.0

    def test_n_cycles_to_ss_geq_1(self):
        result = _run()
        assert result.n_cycles_to_ss >= 1

    def test_t_half_h_positive(self):
        result = _run()
        assert result.t_half_h > 0.0


# ---------------------------------------------------------------------------
# 4. Convergence behaviour
# ---------------------------------------------------------------------------


class TestConvergence:
    def test_short_half_life_converges_fast(self):
        # t_half = ln(2) / (CL/Vd) = 0.693 / (10/5) = ~0.35 h → very short
        result = simulate_to_steady_state(
            drug_name="FastDrug",
            dose_mg=50.0,
            interval_h=12.0,
            cl_L_per_h=10.0,
            vd_L=5.0,
        )
        assert result.converged is True
        assert result.n_cycles_to_ss < 10

    def test_converged_true_for_reasonable_inputs(self):
        result = _run()
        assert result.converged is True

    def test_long_half_life_takes_more_cycles(self):
        # t_half = ln(2) / (0.1/50) = ~347 h (very long)
        result = simulate_to_steady_state(
            drug_name="SlowDrug",
            dose_mg=100.0,
            interval_h=12.0,
            cl_L_per_h=0.1,
            vd_L=50.0,
            max_cycles=100,
        )
        # Should not converge in fewer cycles than short-half-life drug
        short_result = simulate_to_steady_state(
            drug_name="FastDrug",
            dose_mg=100.0,
            interval_h=12.0,
            cl_L_per_h=10.0,
            vd_L=5.0,
        )
        assert result.n_cycles_to_ss > short_result.n_cycles_to_ss


# ---------------------------------------------------------------------------
# 5. Dose proportionality
# ---------------------------------------------------------------------------


class TestDoseProportionality:
    def test_double_dose_doubles_css_max(self):
        r1 = _run(dose_mg=100.0)
        r2 = _run(dose_mg=200.0)
        assert abs(r2.css_max_mg_L / r1.css_max_mg_L - 2.0) < 0.1

    def test_double_dose_doubles_css_avg(self):
        r1 = _run(dose_mg=100.0)
        r2 = _run(dose_mg=200.0)
        assert abs(r2.css_avg_mg_L / r1.css_avg_mg_L - 2.0) < 0.1


# ---------------------------------------------------------------------------
# 6. Interval effects
# ---------------------------------------------------------------------------


class TestIntervalEffects:
    def test_longer_interval_lower_accumulation_ratio(self):
        r_short = _run(interval_h=6.0)
        r_long = _run(interval_h=24.0)
        assert r_short.accumulation_ratio >= r_long.accumulation_ratio


# ---------------------------------------------------------------------------
# 7. steady_state_metrics: analytical vs simulation
# ---------------------------------------------------------------------------


class TestSteadyStateMetrics:
    def test_returns_dict(self):
        metrics = steady_state_metrics(cl_L_per_h=5.0, vd_L=50.0, dose_mg=100.0, interval_h=12.0)
        assert isinstance(metrics, dict)

    def test_has_expected_keys(self):
        metrics = steady_state_metrics(cl_L_per_h=5.0, vd_L=50.0, dose_mg=100.0, interval_h=12.0)
        for key in ["Css_avg_mg_L", "t_half_h", "accumulation_ratio", "AUC_tau"]:
            assert key in metrics, f"Missing key: {key}"

    def test_css_avg_matches_simulation_within_20pct(self):
        cl = 5.0
        vd = 50.0
        dose = 100.0
        tau = 12.0
        metrics = steady_state_metrics(cl_L_per_h=cl, vd_L=vd, dose_mg=dose, interval_h=tau)
        sim_result = simulate_to_steady_state(
            drug_name="X", dose_mg=dose, interval_h=tau, cl_L_per_h=cl, vd_L=vd
        )
        rel_diff = abs(metrics["Css_avg_mg_L"] - sim_result.css_avg_mg_L) / max(
            metrics["Css_avg_mg_L"], 1e-12
        )
        assert rel_diff < 0.20, f"Css_avg relative diff = {rel_diff:.3f}"

    def test_accumulation_ratio_geq_1(self):
        metrics = steady_state_metrics(cl_L_per_h=5.0, vd_L=50.0, dose_mg=100.0, interval_h=12.0)
        assert metrics["accumulation_ratio"] >= 1.0

    def test_metrics_invalid_cl_raises(self):
        with pytest.raises(ValueError):
            steady_state_metrics(cl_L_per_h=0.0, vd_L=50.0, dose_mg=100.0, interval_h=12.0)

    def test_metrics_invalid_dose_raises(self):
        with pytest.raises(ValueError):
            steady_state_metrics(cl_L_per_h=5.0, vd_L=50.0, dose_mg=0.0, interval_h=12.0)
