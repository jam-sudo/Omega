"""Tests for sensitivity_ranking module (Phase 376)."""

from __future__ import annotations

import pytest

from omega_pbpk.analysis.sensitivity_ranking import (
    SensitivityRankResult,
    rank_parameters,
    tornado_chart_data,
)

# ── Simple test PK functions ────────────────────────────────────────────────


def _auc_function(params: dict) -> dict:
    """AUC = dose * F / CL (one-compartment)."""
    auc = params["dose"] * params["F"] / params["CL"]
    return {"AUC": auc, "Cmax": params["dose"] / params["Vd"]}


def _linear_function(params: dict) -> dict:
    """Output = a * x + b."""
    return {"output": params["a"] * params["x"] + params["b"]}


def _constant_function(params: dict) -> dict:
    """Output is constant (zero sensitivity)."""
    return {"metric": 42.0}


# ── Input validation for rank_parameters ──────────────────────────────────


class TestRankParametersValidation:
    BASE = {"dose": 100.0, "F": 0.8, "CL": 5.0, "Vd": 20.0}

    def test_empty_base_params_raises(self):
        with pytest.raises(ValueError, match="base_params"):
            rank_parameters({}, {}, _auc_function, "AUC")

    def test_non_callable_pk_function_raises(self):
        with pytest.raises(ValueError, match="callable"):
            rank_parameters(self.BASE, {}, "not_a_function", "AUC")

    def test_empty_output_key_raises(self):
        with pytest.raises(ValueError, match="output_key"):
            rank_parameters(self.BASE, {}, _auc_function, "")

    def test_n_samples_less_than_2_raises(self):
        with pytest.raises(ValueError, match="n_samples"):
            rank_parameters(self.BASE, {}, _auc_function, "AUC", n_samples=1)

    def test_unknown_param_range_key_raises(self):
        with pytest.raises(ValueError, match="not found in base_params"):
            rank_parameters(self.BASE, {"unknown_param": (0.5, 1.5)}, _auc_function, "AUC")

    def test_non_dict_param_ranges_raises(self):
        with pytest.raises(ValueError, match="param_ranges"):
            rank_parameters(self.BASE, None, _auc_function, "AUC")  # type: ignore[arg-type]

    def test_lo_ge_hi_range_raises(self):
        with pytest.raises(ValueError):
            rank_parameters(
                self.BASE,
                {"dose": (200.0, 50.0)},  # inverted range
                _auc_function,
                "AUC",
            )


# ── Basic return structure ───────────────────────────────────────────────────


class TestRankParametersStructure:
    BASE = {"dose": 100.0, "F": 0.8, "CL": 5.0, "Vd": 20.0}

    def test_returns_sensitivity_rank_result(self):
        result = rank_parameters(self.BASE, {}, _auc_function, "AUC")
        assert isinstance(result, SensitivityRankResult)

    def test_param_names_length(self):
        result = rank_parameters(self.BASE, {}, _auc_function, "AUC")
        assert len(result.param_names) == len(self.BASE)

    def test_sensitivity_indices_same_length(self):
        result = rank_parameters(self.BASE, {}, _auc_function, "AUC")
        assert len(result.sensitivity_indices) == len(self.BASE)

    def test_ranked_params_same_length(self):
        result = rank_parameters(self.BASE, {}, _auc_function, "AUC")
        assert len(result.ranked_params) == len(self.BASE)

    def test_ranked_indices_same_length(self):
        result = rank_parameters(self.BASE, {}, _auc_function, "AUC")
        assert len(result.ranked_indices) == len(self.BASE)

    def test_output_key_stored(self):
        result = rank_parameters(self.BASE, {}, _auc_function, "AUC")
        assert result.output_key == "AUC"

    def test_base_output_correct(self):
        base = {"dose": 100.0, "F": 0.8, "CL": 5.0, "Vd": 20.0}
        expected_auc = 100.0 * 0.8 / 5.0
        result = rank_parameters(base, {}, _auc_function, "AUC")
        assert result.base_output == pytest.approx(expected_auc)

    def test_all_params_appear_in_ranked_params(self):
        result = rank_parameters(self.BASE, {}, _auc_function, "AUC")
        assert set(result.ranked_params) == set(self.BASE.keys())

    def test_ranked_indices_descending(self):
        result = rank_parameters(self.BASE, {}, _auc_function, "AUC")
        for i in range(len(result.ranked_indices) - 1):
            assert result.ranked_indices[i] >= result.ranked_indices[i + 1]


# ── Sensitivity correctness ──────────────────────────────────────────────────


class TestRankParametersSensitivity:
    def test_cl_most_sensitive_for_auc(self):
        """AUC = dose*F/CL, so CL and F dominate; Vd only affects Cmax."""
        base = {"CL": 5.0, "F": 0.8, "dose": 100.0, "Vd": 20.0}
        result = rank_parameters(base, {}, _auc_function, "AUC")
        # CL appears in denominator → ±20% change has large fractional impact
        assert "CL" in result.ranked_params[:2]

    def test_sensitivity_non_negative(self):
        base = {"dose": 100.0, "F": 0.8, "CL": 5.0, "Vd": 20.0}
        result = rank_parameters(base, {}, _auc_function, "AUC")
        assert all(s >= 0.0 for s in result.sensitivity_indices)

    def test_constant_output_zero_sensitivity(self):
        """If output is constant, all sensitivity indices should be ~0."""
        # _constant_function always returns 42.0 regardless of params.
        # We need base_output != 0, which is satisfied (42.0).
        base = {"a": 1.0, "b": 2.0}
        result = rank_parameters(base, {}, _constant_function, "metric")
        assert all(s == pytest.approx(0.0, abs=1e-9) for s in result.sensitivity_indices)

    def test_linear_function_slope_sensitivity(self):
        """For output = a*x + b, sensitivity to 'x' depends on 'a'; 'b' additive."""
        # Use b=1.0 so that its ±20% range (0.8-1.2) is well-defined and
        # output_base = 10*5 + 1 = 51.0 != 0.  b's delta_output vs x's is small.
        base = {"a": 10.0, "x": 5.0, "b": 1.0}
        result = rank_parameters(base, {}, _linear_function, "output")
        # param 'a' and 'x' dominate; 'b' has the smallest impact because
        # delta_output for b is only 0.4 whereas for a and x it's much larger
        b_idx = result.param_names.index("b")
        a_idx = result.param_names.index("a")
        assert result.sensitivity_indices[a_idx] > result.sensitivity_indices[b_idx]

    def test_custom_param_ranges_used(self):
        """Providing a tighter range for one param reduces its apparent SI."""
        base = {"dose": 100.0, "F": 0.8, "CL": 5.0, "Vd": 20.0}
        ranges_narrow = {"CL": (4.9, 5.1)}  # very narrow vs ±20%
        result_narrow = rank_parameters(base, ranges_narrow, _auc_function, "AUC")
        result_wide = rank_parameters(base, {}, _auc_function, "AUC")
        cl_idx_narrow = result_narrow.param_names.index("CL")
        cl_idx_wide = result_wide.param_names.index("CL")
        # Narrow range → smaller delta_param → lower SI
        assert (
            result_narrow.sensitivity_indices[cl_idx_narrow]
            < (result_wide.sensitivity_indices[cl_idx_wide])
        )

    def test_n_samples_parameter_accepted(self):
        base = {"dose": 100.0, "F": 0.8, "CL": 5.0, "Vd": 20.0}
        result = rank_parameters(base, {}, _auc_function, "AUC", n_samples=5)
        assert isinstance(result, SensitivityRankResult)
        assert len(result.param_names) == 4

    def test_notes_field_is_string(self):
        base = {"dose": 100.0, "F": 0.8, "CL": 5.0, "Vd": 20.0}
        result = rank_parameters(base, {}, _auc_function, "AUC")
        assert isinstance(result.notes, str)
        assert len(result.notes) > 0


# ── tornado_chart_data ───────────────────────────────────────────────────────


class TestTornadoChartData:
    def _make_result(self) -> SensitivityRankResult:
        base = {"dose": 100.0, "F": 0.8, "CL": 5.0, "Vd": 20.0}
        return rank_parameters(base, {}, _auc_function, "AUC")

    def test_returns_list_of_dicts(self):
        result = self._make_result()
        data = tornado_chart_data(result)
        assert isinstance(data, list)
        assert all(isinstance(d, dict) for d in data)

    def test_length_matches_params(self):
        result = self._make_result()
        data = tornado_chart_data(result)
        assert len(data) == len(result.ranked_params)

    def test_required_keys_present(self):
        result = self._make_result()
        data = tornado_chart_data(result)
        for d in data:
            assert "param" in d
            assert "sensitivity" in d
            assert "rank" in d

    def test_first_entry_is_most_sensitive(self):
        result = self._make_result()
        data = tornado_chart_data(result)
        assert data[0]["param"] == result.ranked_params[0]
        assert data[0]["rank"] == 1

    def test_ranks_are_sequential(self):
        result = self._make_result()
        data = tornado_chart_data(result)
        for i, d in enumerate(data):
            assert d["rank"] == i + 1

    def test_sensitivity_values_match_ranked_indices(self):
        result = self._make_result()
        data = tornado_chart_data(result)
        for d, expected_si in zip(data, result.ranked_indices, strict=True):
            assert d["sensitivity"] == pytest.approx(expected_si, rel=1e-9)

    def test_non_result_raises(self):
        with pytest.raises(ValueError):
            tornado_chart_data({"not": "a result"})  # type: ignore[arg-type]

    def test_empty_ranked_params_raises(self):
        empty = SensitivityRankResult(
            param_names=[],
            sensitivity_indices=[],
            ranked_params=[],
            ranked_indices=[],
            output_key="X",
            base_output=1.0,
            notes="",
        )
        with pytest.raises(ValueError, match="no ranked parameters"):
            tornado_chart_data(empty)

    def test_param_names_in_order(self):
        result = self._make_result()
        data = tornado_chart_data(result)
        for d, name in zip(data, result.ranked_params, strict=True):
            assert d["param"] == name
