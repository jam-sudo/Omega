"""Tests for Phase 659 — PK Sensitivity Ranking (OAT)."""

from __future__ import annotations

import pytest

from omega_pbpk.analysis.pk_sensitivity_ranking import (
    SensitivityIndex,
    compute_sensitivity_index,
    rank_pk_sensitivity,
    sensitivity_heatmap_data,
)

# ---------------------------------------------------------------------------
# compute_sensitivity_index tests
# ---------------------------------------------------------------------------


def test_compute_returns_sensitivity_index():
    """compute_sensitivity_index must return a SensitivityIndex instance."""
    si = compute_sensitivity_index("CL", 10.0, 5.0, 15.0, 2.0, 1.0, 1.5)
    assert isinstance(si, SensitivityIndex)


def test_abs_s_index_non_negative():
    """abs_s_index must always be >= 0."""
    si = compute_sensitivity_index("CL", 10.0, 5.0, 15.0, 2.0, 1.0, 1.5)
    assert si.abs_s_index >= 0.0


def test_direction_positive_when_outcome_increases():
    """direction == 'positive' when higher param value → higher outcome."""
    # outcome_low < outcome_high → positive relationship
    si = compute_sensitivity_index("ka", 1.0, 0.5, 1.5, 50.0, 100.0, 75.0)
    assert si.direction == "positive"


def test_direction_negative_when_outcome_decreases():
    """direction == 'negative' when higher param value → lower outcome."""
    # AUC = D/CL: higher CL → lower AUC → outcome_low > outcome_high
    si = compute_sensitivity_index("CL", 10.0, 5.0, 15.0, 200.0, 66.67, 100.0)
    assert si.direction == "negative"


def test_perturbed_values_stored_correctly():
    """perturbed_low and perturbed_high must equal the supplied values."""
    si = compute_sensitivity_index("Vd", 50.0, 25.0, 75.0, 10.0, 5.0, 7.0)
    assert si.perturbed_low == 25.0
    assert si.perturbed_high == 75.0


def test_outcome_values_stored_correctly():
    """outcome_at_low, nominal, high must be stored as supplied."""
    si = compute_sensitivity_index("F", 0.5, 0.3, 0.7, 30.0, 70.0, 50.0)
    assert si.outcome_at_low == 30.0
    assert si.outcome_at_high == 70.0
    assert si.outcome_at_nominal == 50.0


def test_linear_sensitivity_equals_one():
    """For y = k * x, S_index should equal 1.0."""
    k = 5.0
    nominal = 10.0
    low = 5.0
    high = 15.0
    si = compute_sensitivity_index("x", nominal, low, high, k * low, k * high, k * nominal)
    assert abs(si.s_index - 1.0) < 1e-6


def test_inverse_relationship_sensitivity_negative_one():
    """For y = 1/x, S_index should be approximately -1.0."""
    nominal = 10.0
    low = 5.0
    high = 15.0
    si = compute_sensitivity_index("x", nominal, low, high, 1.0 / low, 1.0 / high, 1.0 / nominal)
    assert abs(si.s_index - (-1.0)) < 0.1  # OAT finite difference approx


def test_zero_sensitivity_when_outcome_unchanged():
    """s_index == 0 when the outcome is the same at low and high."""
    si = compute_sensitivity_index("noise", 10.0, 5.0, 15.0, 42.0, 42.0, 42.0)
    assert si.s_index == 0.0
    assert si.abs_s_index == 0.0


def test_notes_nonempty():
    """notes field must be a nonempty string."""
    si = compute_sensitivity_index("k", 1.0, 0.5, 1.5, 1.0, 2.0, 1.5)
    assert isinstance(si.notes, str)
    assert len(si.notes) > 0


def test_validation_low_ge_nominal_raises():
    """low_value >= nominal_value must raise ValueError."""
    with pytest.raises(ValueError):
        compute_sensitivity_index("p", 10.0, 10.0, 15.0, 1.0, 2.0, 1.5)


def test_validation_nominal_ge_high_raises():
    """nominal_value >= high_value must raise ValueError."""
    with pytest.raises(ValueError):
        compute_sensitivity_index("p", 10.0, 5.0, 10.0, 1.0, 2.0, 1.5)


# ---------------------------------------------------------------------------
# rank_pk_sensitivity tests
# ---------------------------------------------------------------------------


def _make_linear_outcome(coefficients: dict[str, float]):
    """Return a function: outcome = sum(coeff[k] * params[k])."""

    def f(params: dict[str, float]) -> float:
        return sum(coefficients[k] * v for k, v in params.items())

    return f


def test_rank_returns_correct_length():
    """rank_pk_sensitivity returns a list of the same length as param_ranges."""
    params = {
        "CL": (5.0, 10.0, 15.0),
        "Vd": (25.0, 50.0, 75.0),
        "ka": (0.5, 1.0, 1.5),
    }
    outcome_fn = _make_linear_outcome({"CL": 1.0, "Vd": 0.5, "ka": 0.2})
    results = rank_pk_sensitivity(params, outcome_fn)
    assert len(results) == 3


def test_rank_one_is_most_influential():
    """The parameter with rank 1 should have the highest abs_s_index."""
    params = {
        "CL": (5.0, 10.0, 15.0),
        "Vd": (25.0, 50.0, 75.0),
        "ka": (0.5, 1.0, 1.5),
    }
    # CL coefficient is much larger so it should dominate
    outcome_fn = _make_linear_outcome({"CL": 100.0, "Vd": 1.0, "ka": 0.1})
    results = rank_pk_sensitivity(params, outcome_fn)
    rank1 = [r for r in results if r.rank == 1][0]
    assert rank1.parameter_name == "CL"


def test_ranks_are_unique():
    """No two parameters should share the same rank number."""
    params = {
        "CL": (5.0, 10.0, 15.0),
        "Vd": (25.0, 50.0, 75.0),
        "ka": (0.5, 1.0, 1.5),
    }
    outcome_fn = _make_linear_outcome({"CL": 5.0, "Vd": 3.0, "ka": 1.0})
    results = rank_pk_sensitivity(params, outcome_fn)
    ranks = [r.rank for r in results]
    assert len(ranks) == len(set(ranks))


def test_sorted_by_abs_s_index_descending():
    """List must be sorted by abs_s_index descending."""
    params = {
        "CL": (5.0, 10.0, 15.0),
        "Vd": (25.0, 50.0, 75.0),
        "ka": (0.5, 1.0, 1.5),
    }
    outcome_fn = _make_linear_outcome({"CL": 10.0, "Vd": 2.0, "ka": 0.5})
    results = rank_pk_sensitivity(params, outcome_fn)
    for i in range(len(results) - 1):
        assert results[i].abs_s_index >= results[i + 1].abs_s_index


def test_cl_on_auc_sensitivity_approximately_minus_one():
    """AUC = D/CL → S(CL, AUC) ≈ -1.0."""
    dose = 100.0

    def auc(params: dict[str, float]) -> float:
        return dose / params["CL"]

    params = {"CL": (5.0, 10.0, 20.0)}
    results = rank_pk_sensitivity(params, auc, outcome_name="AUC")
    si = results[0]
    assert si.parameter_name == "CL"
    assert abs(si.s_index - (-1.0)) < 0.1


def test_three_params_highest_abs_s_is_rank1():
    """With 3 params, the highest |S| must be rank 1."""
    params = {
        "CL": (5.0, 10.0, 15.0),
        "Vd": (25.0, 50.0, 75.0),
        "F": (0.3, 0.6, 0.9),
    }

    def auc(p: dict[str, float]) -> float:
        return 100.0 * p["F"] / p["CL"]

    results = rank_pk_sensitivity(params, auc)
    rank1 = results[0]
    max_abs_s = max(r.abs_s_index for r in results)
    assert abs(rank1.abs_s_index - max_abs_s) < 1e-9


def test_empty_param_ranges_raises():
    """Empty param_ranges must raise ValueError."""
    with pytest.raises(ValueError):
        rank_pk_sensitivity({}, lambda p: 1.0)


def test_direction_positive_for_logP_cl_relationship():
    """Increasing logP increases CL (positive direction)."""

    def cl_from_logp(params: dict[str, float]) -> float:
        return params["logP"] * 10.0 + 1.0

    params = {"logP": (1.0, 3.0, 5.0)}
    results = rank_pk_sensitivity(params, cl_from_logp, outcome_name="CL")
    assert results[0].direction == "positive"


# ---------------------------------------------------------------------------
# sensitivity_heatmap_data tests
# ---------------------------------------------------------------------------


def test_heatmap_data_returns_correct_keys():
    """sensitivity_heatmap_data must return dict with same keys as outcome_functions."""
    params = {"CL": (5.0, 10.0, 15.0), "Vd": (25.0, 50.0, 75.0)}
    outcomes = {
        "Cmax": lambda p: 100.0 / p["Vd"],
        "AUC": lambda p: 100.0 / p["CL"],
    }
    result = sensitivity_heatmap_data(params, outcomes)
    assert set(result.keys()) == {"Cmax", "AUC"}


def test_heatmap_data_each_value_is_list():
    """Each value in heatmap data must be a list of SensitivityIndex objects."""
    params = {"CL": (5.0, 10.0, 15.0)}
    outcomes = {"AUC": lambda p: 100.0 / p["CL"]}
    result = sensitivity_heatmap_data(params, outcomes)
    assert isinstance(result["AUC"], list)
    assert all(isinstance(si, SensitivityIndex) for si in result["AUC"])
