"""Tests for omega_pbpk.core.sensitivity module."""

import pytest
from omega_pbpk.core.sensitivity import (
    SensitivityResult,
    local_sensitivity,
    normalized_sensitivity_matrix,
    tornado_plot_data,
)


# Simple linear model: output = CL * Vd (product)
def sim(p):
    return p["CL"] * p["Vd"]


NOMINAL_PARAMS = {"CL": 2.0, "Vd": 10.0}


# ---------------------------------------------------------------------------
# SensitivityResult dataclass
# ---------------------------------------------------------------------------

class TestSensitivityResultDataclass:
    def test_has_required_fields(self):
        r = SensitivityResult(
            parameter_name="CL",
            nominal_value=2.0,
            perturbed_up_value=2.2,
            perturbed_down_value=1.8,
            output_metric="auc",
            nominal_output=20.0,
            output_up=22.0,
            output_down=18.0,
            sensitivity_index=1.0,
            relative_sensitivity=1.0,
            rank=1,
        )
        assert r.parameter_name == "CL"
        assert r.nominal_value == 2.0
        assert r.perturbed_up_value == 2.2
        assert r.perturbed_down_value == 1.8
        assert r.output_metric == "auc"
        assert r.nominal_output == 20.0
        assert r.output_up == 22.0
        assert r.output_down == 18.0
        assert r.sensitivity_index == 1.0
        assert r.relative_sensitivity == 1.0
        assert r.rank == 1


# ---------------------------------------------------------------------------
# local_sensitivity — basic structure
# ---------------------------------------------------------------------------

class TestLocalSensitivityStructure:
    def test_returns_list(self):
        results = local_sensitivity(sim, NOMINAL_PARAMS)
        assert isinstance(results, list)

    def test_list_not_empty(self):
        results = local_sensitivity(sim, NOMINAL_PARAMS)
        assert len(results) > 0

    def test_length_equals_number_of_params(self):
        results = local_sensitivity(sim, NOMINAL_PARAMS)
        assert len(results) == len(NOMINAL_PARAMS)

    def test_each_element_is_sensitivity_result(self):
        results = local_sensitivity(sim, NOMINAL_PARAMS)
        for r in results:
            assert isinstance(r, SensitivityResult)

    def test_sensitivity_index_is_float(self):
        results = local_sensitivity(sim, NOMINAL_PARAMS)
        for r in results:
            assert isinstance(r.sensitivity_index, float)

    def test_nominal_output_matches_simulate_fn(self):
        results = local_sensitivity(sim, NOMINAL_PARAMS)
        expected = sim(NOMINAL_PARAMS)
        for r in results:
            assert r.nominal_output == pytest.approx(expected)

    def test_parameter_names_present(self):
        results = local_sensitivity(sim, NOMINAL_PARAMS)
        names = {r.parameter_name for r in results}
        assert names == set(NOMINAL_PARAMS.keys())

    def test_output_metric_default_is_auc(self):
        results = local_sensitivity(sim, NOMINAL_PARAMS)
        for r in results:
            assert r.output_metric == "auc"

    def test_custom_output_metric_stored(self):
        results = local_sensitivity(sim, NOMINAL_PARAMS, output_metric="cmax")
        for r in results:
            assert r.output_metric == "cmax"


# ---------------------------------------------------------------------------
# local_sensitivity — ranking
# ---------------------------------------------------------------------------

class TestLocalSensitivityRanking:
    def test_ranks_are_unique_integers(self):
        results = local_sensitivity(sim, NOMINAL_PARAMS)
        ranks = [r.rank for r in results]
        assert len(ranks) == len(set(ranks))
        for rank in ranks:
            assert isinstance(rank, int)

    def test_rank_1_has_highest_abs_sensitivity(self):
        results = local_sensitivity(sim, NOMINAL_PARAMS)
        rank1 = next(r for r in results if r.rank == 1)
        max_abs = max(abs(r.sensitivity_index) for r in results)
        assert abs(rank1.sensitivity_index) == pytest.approx(max_abs)

    def test_sorted_by_abs_sensitivity_descending(self):
        results = local_sensitivity(sim, NOMINAL_PARAMS)
        abs_vals = [abs(r.sensitivity_index) for r in results]
        assert abs_vals == sorted(abs_vals, reverse=True)

    def test_ranks_start_at_1(self):
        results = local_sensitivity(sim, NOMINAL_PARAMS)
        assert min(r.rank for r in results) == 1

    def test_ranks_contiguous(self):
        results = local_sensitivity(sim, NOMINAL_PARAMS)
        ranks = sorted(r.rank for r in results)
        assert ranks == list(range(1, len(results) + 1))


# ---------------------------------------------------------------------------
# local_sensitivity — unit elasticity (product model)
# ---------------------------------------------------------------------------

class TestLocalSensitivityElasticity:
    """For output = CL * Vd, doubling either param doubles output.
    Normalized (relative) sensitivity index should be ~1.0."""

    def test_cl_sensitivity_index_approx_one(self):
        results = local_sensitivity(sim, NOMINAL_PARAMS)
        cl_result = next(r for r in results if r.parameter_name == "CL")
        # Relative/normalized sensitivity: (dY/Y) / (dX/X) = 1 for linear product
        assert abs(cl_result.sensitivity_index) == pytest.approx(1.0, abs=0.05)

    def test_vd_sensitivity_index_approx_one(self):
        results = local_sensitivity(sim, NOMINAL_PARAMS)
        vd_result = next(r for r in results if r.parameter_name == "Vd")
        assert abs(vd_result.sensitivity_index) == pytest.approx(1.0, abs=0.05)

    def test_equal_sensitivity_equal_rank_or_consistent(self):
        """CL and Vd have equal sensitivity; both should have abs SI ~ equal."""
        results = local_sensitivity(sim, NOMINAL_PARAMS)
        cl = next(r for r in results if r.parameter_name == "CL")
        vd = next(r for r in results if r.parameter_name == "Vd")
        assert abs(cl.sensitivity_index) == pytest.approx(abs(vd.sensitivity_index), abs=0.05)

    def test_perturbed_up_value_consistent_with_h(self):
        h = 0.1
        results = local_sensitivity(sim, NOMINAL_PARAMS, h=h)
        cl = next(r for r in results if r.parameter_name == "CL")
        assert cl.perturbed_up_value == pytest.approx(NOMINAL_PARAMS["CL"] * (1 + h))

    def test_perturbed_down_value_consistent_with_h(self):
        h = 0.1
        results = local_sensitivity(sim, NOMINAL_PARAMS, h=h)
        cl = next(r for r in results if r.parameter_name == "CL")
        assert cl.perturbed_down_value == pytest.approx(NOMINAL_PARAMS["CL"] * (1 - h))

    def test_output_up_greater_than_output_down_for_positive_param(self):
        results = local_sensitivity(sim, NOMINAL_PARAMS)
        for r in results:
            assert r.output_up > r.output_down


# ---------------------------------------------------------------------------
# local_sensitivity — ValueError for invalid h
# ---------------------------------------------------------------------------

class TestLocalSensitivityValidation:
    def test_h_zero_raises_value_error(self):
        with pytest.raises(ValueError):
            local_sensitivity(sim, NOMINAL_PARAMS, h=0.0)

    def test_h_negative_raises_value_error(self):
        with pytest.raises(ValueError):
            local_sensitivity(sim, NOMINAL_PARAMS, h=-0.1)

    def test_h_one_raises_value_error(self):
        with pytest.raises(ValueError):
            local_sensitivity(sim, NOMINAL_PARAMS, h=1.0)

    def test_h_greater_than_one_raises_value_error(self):
        with pytest.raises(ValueError):
            local_sensitivity(sim, NOMINAL_PARAMS, h=1.5)

    def test_valid_small_h_works(self):
        results = local_sensitivity(sim, NOMINAL_PARAMS, h=0.01)
        assert len(results) == 2

    def test_valid_large_h_just_below_one_works(self):
        results = local_sensitivity(sim, NOMINAL_PARAMS, h=0.99)
        assert len(results) == 2


# ---------------------------------------------------------------------------
# normalized_sensitivity_matrix
# ---------------------------------------------------------------------------

class TestNormalizedSensitivityMatrix:
    def test_returns_dict(self):
        matrix = normalized_sensitivity_matrix(
            sim, NOMINAL_PARAMS, output_metrics=["auc"]
        )
        assert isinstance(matrix, dict)

    def test_dict_has_correct_metric_keys(self):
        metrics = ["auc", "cmax"]
        # Each metric uses the same sim function; the function is called per metric
        matrix = normalized_sensitivity_matrix(sim, NOMINAL_PARAMS, output_metrics=metrics)
        for m in metrics:
            assert m in matrix

    def test_single_metric_key_present(self):
        matrix = normalized_sensitivity_matrix(
            sim, NOMINAL_PARAMS, output_metrics=["auc"]
        )
        assert "auc" in matrix

    def test_matrix_values_contain_param_sensitivities(self):
        matrix = normalized_sensitivity_matrix(
            sim, NOMINAL_PARAMS, output_metrics=["auc"]
        )
        auc_row = matrix["auc"]
        assert isinstance(auc_row, list)
        param_names = [r.parameter_name for r in auc_row]
        for param in NOMINAL_PARAMS:
            assert param in param_names

    def test_invalid_h_raises_value_error(self):
        with pytest.raises(ValueError):
            normalized_sensitivity_matrix(sim, NOMINAL_PARAMS, output_metrics=["auc"], h=0.0)


# ---------------------------------------------------------------------------
# tornado_plot_data
# ---------------------------------------------------------------------------

class TestTornadoPlotData:
    def setup_method(self):
        self.results = local_sensitivity(sim, NOMINAL_PARAMS)

    def test_returns_list(self):
        data = tornado_plot_data(self.results)
        assert isinstance(data, list)

    def test_each_element_is_dict(self):
        data = tornado_plot_data(self.results)
        for item in data:
            assert isinstance(item, dict)

    def test_required_keys_present(self):
        data = tornado_plot_data(self.results)
        required = {"param_name", "low_output", "high_output", "delta"}
        for item in data:
            assert required.issubset(item.keys())

    def test_param_names_match_results(self):
        data = tornado_plot_data(self.results)
        result_names = {r.parameter_name for r in self.results}
        data_names = {d["param_name"] for d in data}
        assert data_names == result_names

    def test_sorted_by_abs_delta_descending(self):
        data = tornado_plot_data(self.results)
        deltas = [abs(d["delta"]) for d in data]
        assert deltas == sorted(deltas, reverse=True)

    def test_delta_equals_high_minus_low(self):
        data = tornado_plot_data(self.results)
        for item in data:
            expected_delta = item["high_output"] - item["low_output"]
            assert item["delta"] == pytest.approx(expected_delta)

    def test_high_output_greater_than_low_output(self):
        data = tornado_plot_data(self.results)
        for item in data:
            # For our positive linear model perturbing up should give higher output
            assert item["high_output"] >= item["low_output"]

    def test_length_matches_results(self):
        data = tornado_plot_data(self.results)
        assert len(data) == len(self.results)
