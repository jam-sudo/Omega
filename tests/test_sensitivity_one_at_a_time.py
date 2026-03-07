"""
Tests for Phase 583: analysis/sensitivity_one_at_a_time.py
"""

from omega_pbpk.analysis.sensitivity_one_at_a_time import (
    local_elasticity,
    oat_sensitivity,
    sensitivity_ranking,
    tornado_chart_data,
)

# ---------------------------------------------------------------------------
# Simple model fixtures
# ---------------------------------------------------------------------------


def linear_model(**params):
    """output = 2 * x  (linear in x, insensitive to y)."""
    return {"output": params["x"] * 2.0}


def multi_model(**params):
    """output = a * x + b * y  (sensitive to both)."""
    return {"output": params["a"] * params["x"] + params["b"] * params["y"]}


def constant_model(**params):
    """output = 5 always."""
    return {"output": 5.0}


def quadratic_model(**params):
    """output = x^2."""
    return {"output": params["x"] ** 2}


# ---------------------------------------------------------------------------
# local_elasticity tests
# ---------------------------------------------------------------------------


class TestLocalElasticity:
    def test_linear_relationship_elasticity_is_one(self):
        # output = 2*x → elasticity = 1 at any point
        base_param = 10.0
        base_output = 2.0 * base_param
        new_param = 12.0
        new_output = 2.0 * new_param
        e = local_elasticity(new_param, base_param, new_output, base_output)
        assert abs(e - 1.0) < 1e-9

    def test_zero_base_param_returns_zero(self):
        e = local_elasticity(1.0, 0.0, 10.0, 5.0)
        assert e == 0.0

    def test_zero_base_output_returns_zero(self):
        e = local_elasticity(2.0, 1.0, 5.0, 0.0)
        assert e == 0.0

    def test_no_change_in_param_returns_zero(self):
        # param_value == base_param
        e = local_elasticity(5.0, 5.0, 20.0, 10.0)
        assert e == 0.0

    def test_inverse_relationship_elasticity_approaches_minus_one(self):
        # output = C / x → true elasticity = -1 (continuous limit)
        # With a small step the finite-difference approximation is close to -1
        C = 100.0
        base_x = 10.0
        base_out = C / base_x
        # Use a tiny perturbation to stay near the continuous limit
        new_x = 10.001
        new_out = C / new_x
        e = local_elasticity(new_x, base_x, new_out, base_out)
        assert abs(e - (-1.0)) < 1e-3

    def test_double_output_per_double_param_gives_one(self):
        e = local_elasticity(20.0, 10.0, 40.0, 20.0)
        assert abs(e - 1.0) < 1e-9

    def test_output_unchanged_gives_zero_elasticity(self):
        e = local_elasticity(15.0, 10.0, 20.0, 20.0)
        assert e == 0.0


# ---------------------------------------------------------------------------
# oat_sensitivity tests
# ---------------------------------------------------------------------------


class TestOatSensitivity:
    def test_returns_n_levels_per_parameter(self):
        base = {"x": 5.0}
        ranges = {"x": (1.0, 10.0)}
        results = oat_sensitivity(base, ranges, linear_model, "output", n_levels=5)
        assert len(results) == 5

    def test_returns_n_levels_for_multiple_params(self):
        base = {"x": 5.0, "a": 2.0, "b": 1.0, "y": 3.0}
        ranges = {"x": (1.0, 10.0), "a": (1.0, 5.0)}
        results = oat_sensitivity(base, ranges, multi_model, "output", n_levels=4)
        assert len(results) == 8  # 4 levels * 2 params

    def test_linear_model_normalized_sensitivity_is_one(self):
        base = {"x": 5.0}
        ranges = {"x": (2.0, 10.0)}
        results = oat_sensitivity(base, ranges, linear_model, "output", n_levels=5)
        # At all non-base levels, normalized sensitivity should be ~1
        non_base = [r for r in results if r["value"] != base["x"]]
        for r in non_base:
            assert abs(r["normalized_sensitivity"] - 1.0) < 1e-9

    def test_base_output_included_in_results(self):
        base = {"x": 5.0}
        ranges = {"x": (1.0, 10.0)}
        results = oat_sensitivity(base, ranges, linear_model, "output", n_levels=5)
        # Each result should carry base_output
        for r in results:
            assert "base_output" in r
            assert abs(r["base_output"] - 10.0) < 1e-9  # 2 * 5

    def test_param_name_present_in_result(self):
        base = {"x": 5.0}
        ranges = {"x": (1.0, 10.0)}
        results = oat_sensitivity(base, ranges, linear_model, "output")
        for r in results:
            assert r["param_name"] == "x"

    def test_result_keys(self):
        base = {"x": 5.0}
        ranges = {"x": (1.0, 10.0)}
        results = oat_sensitivity(base, ranges, linear_model, "output")
        required_keys = {"param_name", "value", "output", "normalized_sensitivity"}
        for r in results:
            assert required_keys.issubset(r.keys())

    def test_output_matches_model(self):
        base = {"x": 5.0}
        ranges = {"x": (1.0, 9.0)}
        results = oat_sensitivity(base, ranges, linear_model, "output", n_levels=3)
        # Levels: 1, 5, 9 → outputs: 2, 10, 18
        outputs = sorted([r["output"] for r in results])
        assert abs(outputs[0] - 2.0) < 1e-9
        assert abs(outputs[1] - 10.0) < 1e-9
        assert abs(outputs[2] - 18.0) < 1e-9

    def test_constant_model_sensitivity_is_zero(self):
        base = {"x": 5.0}
        ranges = {"x": (1.0, 10.0)}
        results = oat_sensitivity(base, ranges, constant_model, "output", n_levels=5)
        non_base = [r for r in results if r["value"] != base["x"]]
        for r in non_base:
            assert r["normalized_sensitivity"] == 0.0

    def test_n_levels_one(self):
        base = {"x": 5.0}
        ranges = {"x": (3.0, 3.0)}
        results = oat_sensitivity(base, ranges, linear_model, "output", n_levels=1)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# sensitivity_ranking tests
# ---------------------------------------------------------------------------


class TestSensitivityRanking:
    def _get_oat_results(self):
        base = {"x": 5.0, "a": 2.0, "b": 1.0, "y": 3.0}
        ranges = {"x": (1.0, 10.0), "a": (0.5, 4.0)}
        return oat_sensitivity(base, ranges, multi_model, "output", n_levels=5)

    def test_ranking_returns_list(self):
        oat = self._get_oat_results()
        ranking = sensitivity_ranking(oat)
        assert isinstance(ranking, list)

    def test_ranking_contains_required_keys(self):
        oat = self._get_oat_results()
        ranking = sensitivity_ranking(oat)
        for entry in ranking:
            assert "param_name" in entry
            assert "max_sensitivity" in entry
            assert "rank" in entry

    def test_rank_one_has_highest_sensitivity(self):
        oat = self._get_oat_results()
        ranking = sensitivity_ranking(oat)
        rank1 = next(r for r in ranking if r["rank"] == 1)
        max_sens = max(r["max_sensitivity"] for r in ranking)
        assert abs(rank1["max_sensitivity"] - max_sens) < 1e-9

    def test_ranks_are_sequential_starting_at_one(self):
        oat = self._get_oat_results()
        ranking = sensitivity_ranking(oat)
        ranks = sorted(r["rank"] for r in ranking)
        assert ranks == list(range(1, len(ranks) + 1))

    def test_each_param_appears_once(self):
        oat = self._get_oat_results()
        ranking = sensitivity_ranking(oat)
        names = [r["param_name"] for r in ranking]
        assert len(names) == len(set(names))


# ---------------------------------------------------------------------------
# tornado_chart_data tests
# ---------------------------------------------------------------------------


class TestTornadoChartData:
    def _base_oat(self):
        base = {"x": 5.0}
        ranges = {"x": (1.0, 9.0)}
        return oat_sensitivity(base, ranges, linear_model, "output", n_levels=3), 10.0

    def test_tornado_returns_required_keys(self):
        oat, base_out = self._base_oat()
        chart = tornado_chart_data(oat, base_out)
        for entry in chart:
            assert "param_name" in entry
            assert "output_low" in entry
            assert "output_high" in entry
            assert "swing" in entry

    def test_swing_is_output_high_minus_output_low(self):
        oat, base_out = self._base_oat()
        chart = tornado_chart_data(oat, base_out)
        for entry in chart:
            assert abs(entry["swing"] - (entry["output_high"] - entry["output_low"])) < 1e-9

    def test_sorted_by_swing_descending(self):
        base = {"x": 5.0, "a": 2.0, "b": 1.0, "y": 3.0}
        ranges = {"x": (1.0, 10.0), "a": (0.1, 5.0)}
        oat = oat_sensitivity(base, ranges, multi_model, "output", n_levels=5)
        chart = tornado_chart_data(oat, 10.0)
        swings = [abs(c["swing"]) for c in chart]
        assert swings == sorted(swings, reverse=True)

    def test_linear_model_output_low_less_than_output_high(self):
        oat, base_out = self._base_oat()
        chart = tornado_chart_data(oat, base_out)
        assert chart[0]["output_low"] < chart[0]["output_high"]
