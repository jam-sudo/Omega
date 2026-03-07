"""Tests for OAT PK parameter sensitivity — Phase 482."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.analysis.pk_parameter_sensitivity import (
    OATSensitivityResult,
    normalized_sensitivity_index,
    oat_sensitivity,
    plot_sensitivity_table,
    rank_parameters_by_sensitivity,
)

# ---------------------------------------------------------------------------
# normalized_sensitivity_index
# ---------------------------------------------------------------------------


class TestNormalizedSensitivityIndex:
    def test_proportional_relationship_plus_one(self):
        # Y = P → dY/dP = 1, S = 1 * (P/Y) = 1
        si = normalized_sensitivity_index(
            param_ref=2.0, output_ref=2.0, delta_output=0.02, delta_param=0.02
        )
        assert math.isclose(si, 1.0, rel_tol=1e-6)

    def test_inverse_relationship_minus_one(self):
        # AUC = Dose/CL → dAUC/dCL = -Dose/CL^2
        # S = (-Dose/CL^2) * (CL/AUC) = -1 analytically.
        # Finite difference converges; use a very small perturbation (0.001%).
        dose = 100.0
        cl_ref = 5.0
        auc_ref = dose / cl_ref  # 20
        delta = cl_ref * 1e-6
        cl_pert = cl_ref + delta
        auc_pert = dose / cl_pert
        si = normalized_sensitivity_index(
            param_ref=cl_ref,
            output_ref=auc_ref,
            delta_output=auc_pert - auc_ref,
            delta_param=delta,
        )
        assert math.isclose(si, -1.0, rel_tol=1e-6)

    def test_zero_delta_param_returns_zero(self):
        si = normalized_sensitivity_index(
            param_ref=1.0, output_ref=5.0, delta_output=1.0, delta_param=0.0
        )
        assert si == 0.0

    def test_zero_output_ref_returns_zero(self):
        si = normalized_sensitivity_index(
            param_ref=1.0, output_ref=0.0, delta_output=1.0, delta_param=0.1
        )
        assert si == 0.0

    def test_zero_param_ref_returns_zero(self):
        si = normalized_sensitivity_index(
            param_ref=0.0, output_ref=5.0, delta_output=1.0, delta_param=0.1
        )
        assert si == 0.0

    def test_scaling_invariance(self):
        # Double both output and param → same S
        si1 = normalized_sensitivity_index(1.0, 10.0, 0.1, 0.1)
        si2 = normalized_sensitivity_index(2.0, 20.0, 0.2, 0.2)
        assert math.isclose(si1, si2, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# oat_sensitivity
# ---------------------------------------------------------------------------


class TestOATSensitivity:
    def _default(self, **kwargs) -> OATSensitivityResult:
        params = dict(cl_ref=5.0, vd_ref=50.0, dose_mg=100.0, ka_ref=1.0, route="iv")
        params.update(kwargs)
        return oat_sensitivity(**params)

    def test_returns_dataclass(self):
        result = self._default()
        assert isinstance(result, OATSensitivityResult)

    def test_three_parameters_returned(self):
        result = self._default()
        assert len(result.parameter_names) == 3
        assert "CL" in result.parameter_names
        assert "Vd" in result.parameter_names
        assert "ka" in result.parameter_names

    def test_s_cl_is_minus_one_for_iv(self):
        # AUC_iv = Dose/CL → S_CL = -1 exactly
        result = self._default(route="iv")
        idx = result.parameter_names.index("CL")
        assert math.isclose(result.sensitivity_indices[idx], -1.0, rel_tol=1e-4)

    def test_s_vd_is_zero_for_iv(self):
        # AUC_iv = Dose/CL, independent of Vd → S_Vd = 0
        result = self._default(route="iv")
        idx = result.parameter_names.index("Vd")
        assert math.isclose(result.sensitivity_indices[idx], 0.0, abs_tol=1e-10)

    def test_s_ka_is_zero_for_iv(self):
        # AUC_iv = Dose/CL, independent of ka → S_ka = 0
        result = self._default(route="iv")
        idx = result.parameter_names.index("ka")
        assert math.isclose(result.sensitivity_indices[idx], 0.0, abs_tol=1e-10)

    def test_output_name_is_auc(self):
        result = self._default()
        assert result.output_name == "AUC"

    def test_output_values_per_param_shape(self):
        result = self._default(n_points=11)
        assert len(result.output_values_per_param) == 3
        for row in result.output_values_per_param:
            assert len(row) == 11

    def test_param_value_ranges_shape(self):
        result = self._default(n_points=7)
        assert len(result.param_value_ranges) == 3
        for row in result.param_value_ranges:
            assert len(row) == 7

    def test_higher_cl_lower_auc(self):
        # As CL increases, AUC decreases
        result = self._default(route="iv")
        idx = result.parameter_names.index("CL")
        cl_auc_pairs = list(
            zip(result.param_value_ranges[idx], result.output_values_per_param[idx])
        )
        cl_sorted = sorted(cl_auc_pairs, key=lambda x: x[0])
        aucs = [a for _, a in cl_sorted]
        # AUC should be monotonically decreasing as CL increases
        for i in range(len(aucs) - 1):
            assert aucs[i] >= aucs[i + 1]

    def test_notes_non_empty(self):
        result = self._default()
        assert len(result.notes) > 0

    def test_param_ref_values_match_inputs(self):
        result = oat_sensitivity(cl_ref=3.0, vd_ref=30.0, dose_mg=50.0, ka_ref=2.0)
        assert math.isclose(result.param_ref_values[0], 3.0)
        assert math.isclose(result.param_ref_values[1], 30.0)
        assert math.isclose(result.param_ref_values[2], 2.0)

    def test_oral_route_runs(self):
        result = oat_sensitivity(cl_ref=5.0, vd_ref=50.0, dose_mg=100.0, route="oral")
        assert isinstance(result, OATSensitivityResult)

    def test_custom_n_points(self):
        result = self._default(n_points=5)
        assert all(len(row) == 5 for row in result.output_values_per_param)

    def test_invalid_cl_ref_zero(self):
        with pytest.raises(ValueError, match="cl_ref"):
            oat_sensitivity(cl_ref=0.0, vd_ref=50.0, dose_mg=100.0)

    def test_invalid_vd_ref_negative(self):
        with pytest.raises(ValueError, match="vd_ref"):
            oat_sensitivity(cl_ref=5.0, vd_ref=-1.0, dose_mg=100.0)

    def test_invalid_dose_zero(self):
        with pytest.raises(ValueError, match="dose_mg"):
            oat_sensitivity(cl_ref=5.0, vd_ref=50.0, dose_mg=0.0)

    def test_invalid_route(self):
        with pytest.raises(ValueError, match="route"):
            oat_sensitivity(cl_ref=5.0, vd_ref=50.0, dose_mg=100.0, route="unknown")

    def test_invalid_n_points_one(self):
        with pytest.raises(ValueError, match="n_points"):
            oat_sensitivity(cl_ref=5.0, vd_ref=50.0, dose_mg=100.0, n_points=1)


# ---------------------------------------------------------------------------
# rank_parameters_by_sensitivity
# ---------------------------------------------------------------------------


class TestRankParametersBySensitivity:
    def test_sorted_by_abs_descending(self):
        result = oat_sensitivity(cl_ref=5.0, vd_ref=50.0, dose_mg=100.0, route="iv")
        ranked = rank_parameters_by_sensitivity(result)
        abs_vals = [abs(r["sensitivity_index"]) for r in ranked]
        assert abs_vals == sorted(abs_vals, reverse=True)

    def test_rank_field_starts_at_one(self):
        result = oat_sensitivity(cl_ref=5.0, vd_ref=50.0, dose_mg=100.0, route="iv")
        ranked = rank_parameters_by_sensitivity(result)
        assert ranked[0]["rank"] == 1

    def test_cl_ranked_first_iv(self):
        # CL has |S|=1, Vd and ka have S=0
        result = oat_sensitivity(cl_ref=5.0, vd_ref=50.0, dose_mg=100.0, route="iv")
        ranked = rank_parameters_by_sensitivity(result)
        assert ranked[0]["parameter"] == "CL"

    def test_returns_three_items(self):
        result = oat_sensitivity(cl_ref=5.0, vd_ref=50.0, dose_mg=100.0, route="iv")
        ranked = rank_parameters_by_sensitivity(result)
        assert len(ranked) == 3

    def test_each_item_has_expected_keys(self):
        result = oat_sensitivity(cl_ref=5.0, vd_ref=50.0, dose_mg=100.0, route="iv")
        ranked = rank_parameters_by_sensitivity(result)
        for item in ranked:
            assert "parameter" in item
            assert "sensitivity_index" in item
            assert "rank" in item


# ---------------------------------------------------------------------------
# plot_sensitivity_table
# ---------------------------------------------------------------------------


class TestPlotSensitivityTable:
    def test_returns_non_empty_string(self):
        result = oat_sensitivity(cl_ref=5.0, vd_ref=50.0, dose_mg=100.0, route="iv")
        chart = plot_sensitivity_table(result)
        assert isinstance(chart, str)
        assert len(chart) > 0

    def test_contains_parameter_names(self):
        result = oat_sensitivity(cl_ref=5.0, vd_ref=50.0, dose_mg=100.0, route="iv")
        chart = plot_sensitivity_table(result)
        assert "CL" in chart
        assert "Vd" in chart
        assert "ka" in chart

    def test_contains_output_name(self):
        result = oat_sensitivity(cl_ref=5.0, vd_ref=50.0, dose_mg=100.0, route="iv")
        chart = plot_sensitivity_table(result)
        assert "AUC" in chart

    def test_multiline(self):
        result = oat_sensitivity(cl_ref=5.0, vd_ref=50.0, dose_mg=100.0, route="iv")
        chart = plot_sensitivity_table(result)
        assert "\n" in chart
