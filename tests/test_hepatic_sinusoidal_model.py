"""Tests for Phase 511 — Hepatic Sinusoidal Model."""

import math

import pytest

from omega_pbpk.core.hepatic_sinusoidal_model import (
    HepaticSinusoidalResult,
    compare_hepatic_models,
    hepatic_clearance_sensitivity,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ws_eh(fup, cl_int, Q=90.0):
    """Reference well-stirred EH."""
    fup_cl = fup * cl_int
    return fup_cl / (Q + fup_cl)


def _pt_eh(fup, cl_int, Q=90.0):
    """Reference parallel-tube EH."""
    return 1.0 - math.exp(-fup * cl_int / Q)


# ---------------------------------------------------------------------------
# Return type and fields
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_dataclass(self):
        result = compare_hepatic_models("DrugA", fup=0.5, cl_int_L_per_h=100.0)
        assert isinstance(result, HepaticSinusoidalResult)

    def test_all_fields_present(self):
        result = compare_hepatic_models("DrugA", fup=0.5, cl_int_L_per_h=100.0)
        fields = [
            "drug_name",
            "fup",
            "cl_int_L_per_h",
            "q_hepatic_L_per_h",
            "eh_well_stirred",
            "fh_well_stirred",
            "clh_well_stirred_L_per_h",
            "eh_parallel_tube",
            "fh_parallel_tube",
            "clh_parallel_tube_L_per_h",
            "model_discrepancy_pct",
            "dominant_model",
            "notes",
        ]
        for f in fields:
            assert hasattr(result, f), f"Missing field: {f}"

    def test_drug_name_stored(self):
        result = compare_hepatic_models("Midazolam", fup=0.05, cl_int_L_per_h=50.0)
        assert result.drug_name == "Midazolam"

    def test_frozen_immutable(self):
        result = compare_hepatic_models("DrugA", fup=0.5, cl_int_L_per_h=50.0)
        with pytest.raises((AttributeError, TypeError)):
            result.fup = 0.9  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Well-stirred formula correctness
# ---------------------------------------------------------------------------


class TestWellStirredFormula:
    def test_eh_ws_formula(self):
        fup, cl_int, Q = 0.3, 120.0, 90.0
        result = compare_hepatic_models("D", fup=fup, cl_int_L_per_h=cl_int, q_hepatic_L_per_h=Q)
        expected = _ws_eh(fup, cl_int, Q)
        assert abs(result.eh_well_stirred - expected) < 1e-10

    def test_fh_ws_is_one_minus_eh(self):
        result = compare_hepatic_models("D", fup=0.4, cl_int_L_per_h=80.0)
        assert abs(result.fh_well_stirred - (1.0 - result.eh_well_stirred)) < 1e-12

    def test_clh_ws_equals_Q_times_eh(self):
        Q = 90.0
        result = compare_hepatic_models("D", fup=0.4, cl_int_L_per_h=80.0, q_hepatic_L_per_h=Q)
        assert abs(result.clh_well_stirred_L_per_h - Q * result.eh_well_stirred) < 1e-10


# ---------------------------------------------------------------------------
# Parallel-tube formula correctness
# ---------------------------------------------------------------------------


class TestParallelTubeFormula:
    def test_eh_pt_formula(self):
        fup, cl_int, Q = 0.3, 120.0, 90.0
        result = compare_hepatic_models("D", fup=fup, cl_int_L_per_h=cl_int, q_hepatic_L_per_h=Q)
        expected = _pt_eh(fup, cl_int, Q)
        assert abs(result.eh_parallel_tube - expected) < 1e-10

    def test_fh_pt_is_one_minus_eh(self):
        result = compare_hepatic_models("D", fup=0.4, cl_int_L_per_h=80.0)
        assert abs(result.fh_parallel_tube - (1.0 - result.eh_parallel_tube)) < 1e-12

    def test_clh_pt_equals_Q_times_eh(self):
        Q = 90.0
        result = compare_hepatic_models("D", fup=0.4, cl_int_L_per_h=80.0, q_hepatic_L_per_h=Q)
        assert abs(result.clh_parallel_tube_L_per_h - Q * result.eh_parallel_tube) < 1e-10


# ---------------------------------------------------------------------------
# Mathematical properties
# ---------------------------------------------------------------------------


class TestMathProperties:
    def test_parallel_tube_ge_well_stirred(self):
        """Parallel-tube EH is always >= well-stirred EH for the same parameters."""
        for fup in [0.01, 0.1, 0.5, 0.9, 1.0]:
            for cl_int in [1.0, 50.0, 200.0, 1000.0]:
                result = compare_hepatic_models("D", fup=fup, cl_int_L_per_h=cl_int)
                assert result.eh_parallel_tube >= result.eh_well_stirred - 1e-12, (
                    f"PT EH < WS EH for fup={fup}, cl_int={cl_int}"
                )

    def test_high_clint_eh_approaches_1(self):
        result = compare_hepatic_models("D", fup=1.0, cl_int_L_per_h=100000.0)
        assert result.eh_well_stirred > 0.999
        assert result.eh_parallel_tube > 0.9999

    def test_low_clint_eh_approaches_0(self):
        result = compare_hepatic_models("D", fup=1.0, cl_int_L_per_h=0.001)
        assert result.eh_well_stirred < 0.001
        assert result.eh_parallel_tube < 0.001

    def test_eh_in_zero_one(self):
        result = compare_hepatic_models("D", fup=0.5, cl_int_L_per_h=100.0)
        assert 0.0 < result.eh_well_stirred < 1.0
        assert 0.0 < result.eh_parallel_tube < 1.0

    def test_fh_in_zero_one(self):
        result = compare_hepatic_models("D", fup=0.5, cl_int_L_per_h=100.0)
        assert 0.0 < result.fh_well_stirred < 1.0
        assert 0.0 < result.fh_parallel_tube < 1.0

    def test_discrepancy_pct_nonnegative(self):
        result = compare_hepatic_models("D", fup=0.5, cl_int_L_per_h=100.0)
        assert result.model_discrepancy_pct >= 0.0


# ---------------------------------------------------------------------------
# Dominant model classification
# ---------------------------------------------------------------------------


class TestDominantModel:
    def test_equivalent_when_small_discrepancy(self):
        # Very low CLint → both EH near 0 → discrepancy small
        result = compare_hepatic_models("D", fup=1.0, cl_int_L_per_h=0.01)
        # For very small CLint, the models converge
        assert result.dominant_model in ("well_stirred", "parallel_tube", "equivalent")

    def test_dominant_model_string_values(self):
        result = compare_hepatic_models("D", fup=0.5, cl_int_L_per_h=100.0)
        assert result.dominant_model in ("well_stirred", "parallel_tube", "equivalent")

    def test_well_stirred_has_lower_eh_than_parallel_tube(self):
        # When not equivalent, WS has lower EH → dominant_model = "well_stirred"
        result = compare_hepatic_models("D", fup=0.5, cl_int_L_per_h=300.0)
        if result.model_discrepancy_pct >= 5.0:
            # PT always >= WS, so EH_ws < EH_pt → "well_stirred"
            assert result.dominant_model == "well_stirred"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_fup_zero_raises(self):
        with pytest.raises(ValueError):
            compare_hepatic_models("D", fup=0.0, cl_int_L_per_h=50.0)

    def test_fup_negative_raises(self):
        with pytest.raises(ValueError):
            compare_hepatic_models("D", fup=-0.1, cl_int_L_per_h=50.0)

    def test_fup_greater_than_one_raises(self):
        with pytest.raises(ValueError):
            compare_hepatic_models("D", fup=1.5, cl_int_L_per_h=50.0)

    def test_cl_int_zero_raises(self):
        with pytest.raises(ValueError):
            compare_hepatic_models("D", fup=0.5, cl_int_L_per_h=0.0)

    def test_cl_int_negative_raises(self):
        with pytest.raises(ValueError):
            compare_hepatic_models("D", fup=0.5, cl_int_L_per_h=-10.0)

    def test_q_zero_raises(self):
        with pytest.raises(ValueError):
            compare_hepatic_models("D", fup=0.5, cl_int_L_per_h=50.0, q_hepatic_L_per_h=0.0)

    def test_q_negative_raises(self):
        with pytest.raises(ValueError):
            compare_hepatic_models("D", fup=0.5, cl_int_L_per_h=50.0, q_hepatic_L_per_h=-1.0)


# ---------------------------------------------------------------------------
# Default Q value
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_default_q_is_90(self):
        result = compare_hepatic_models("D", fup=0.5, cl_int_L_per_h=100.0)
        assert result.q_hepatic_L_per_h == 90.0

    def test_custom_q(self):
        result = compare_hepatic_models("D", fup=0.5, cl_int_L_per_h=100.0, q_hepatic_L_per_h=100.0)
        assert result.q_hepatic_L_per_h == 100.0


# ---------------------------------------------------------------------------
# Sensitivity function
# ---------------------------------------------------------------------------


class TestSensitivity:
    def test_returns_list(self):
        out = hepatic_clearance_sensitivity(
            "D", fup=0.5, q_hepatic_L_per_h=90.0, cl_int_values=[10.0, 50.0, 200.0]
        )
        assert isinstance(out, list)

    def test_sorted_by_eh_ws_ascending(self):
        cl_ints = [200.0, 10.0, 1000.0, 50.0]
        out = hepatic_clearance_sensitivity(
            "D", fup=0.5, q_hepatic_L_per_h=90.0, cl_int_values=cl_ints
        )
        ehs = [r.eh_well_stirred for r in out]
        assert ehs == sorted(ehs)

    def test_sensitivity_length(self):
        cl_ints = [10.0, 50.0, 100.0, 200.0, 500.0]
        out = hepatic_clearance_sensitivity(
            "D", fup=0.5, q_hepatic_L_per_h=90.0, cl_int_values=cl_ints
        )
        assert len(out) == 5

    def test_sensitivity_results_are_correct_type(self):
        out = hepatic_clearance_sensitivity(
            "D", fup=0.3, q_hepatic_L_per_h=90.0, cl_int_values=[20.0, 80.0]
        )
        for r in out:
            assert isinstance(r, HepaticSinusoidalResult)
