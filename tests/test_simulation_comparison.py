"""Tests for analysis/simulation_comparison.py — Phase 420."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.analysis.simulation_comparison import (
    SimulationComparisonResult,
    compare_pk_simulations,
    fold_change_matrix,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _monoexp_profile(
    dose_mg: float = 100.0,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    ka_per_h: float = 1.0,
    n_points: int = 12,
    t_end_h: float = 24.0,
) -> tuple[list[float], list[float]]:
    """One-compartment oral absorption PK profile."""
    times = [i * t_end_h / (n_points - 1) for i in range(n_points)]
    ke = cl_L_per_h / vd_L
    conc = []
    for t in times:
        if abs(ka_per_h - ke) < 1e-9:
            ka_per_h += 1e-6
        c = (dose_mg / vd_L) * (ka_per_h / (ka_per_h - ke)) * (
            math.exp(-ke * t) - math.exp(-ka_per_h * t)
        )
        conc.append(max(c, 0.0))
    return times, conc


def _make_sim(
    name: str,
    cl: float = 5.0,
    vd: float = 50.0,
    dose: float = 100.0,
    ka: float = 1.0,
) -> dict:
    times, conc = _monoexp_profile(dose_mg=dose, cl_L_per_h=cl, vd_L=vd, ka_per_h=ka)
    return {"name": name, "times": times, "concentrations": conc, "dose_mg": dose}


SIMS_TWO = [
    _make_sim("DrugA", cl=5.0, vd=50.0, dose=100.0),
    _make_sim("DrugB", cl=10.0, vd=50.0, dose=100.0),  # higher CL → lower AUC
]

SIMS_THREE = [
    _make_sim("Low", cl=10.0, dose=100.0),
    _make_sim("Mid", cl=5.0, dose=100.0),
    _make_sim("High", cl=2.0, dose=100.0),
]


# ---------------------------------------------------------------------------
# compare_pk_simulations tests
# ---------------------------------------------------------------------------


class TestComparePKSimulations:
    def test_returns_dataclass(self):
        result = compare_pk_simulations(SIMS_TWO, "auc")
        assert isinstance(result, SimulationComparisonResult)

    def test_metric_stored(self):
        result = compare_pk_simulations(SIMS_TWO, "cmax")
        assert result.metric == "cmax"

    def test_n_simulations(self):
        result = compare_pk_simulations(SIMS_THREE, "auc")
        assert result.n_simulations == 3

    def test_simulation_names(self):
        result = compare_pk_simulations(SIMS_TWO, "auc")
        assert result.simulation_names == ["DrugA", "DrugB"]

    def test_metric_values_length(self):
        result = compare_pk_simulations(SIMS_THREE, "auc")
        assert len(result.metric_values) == 3

    def test_ranking_length(self):
        result = compare_pk_simulations(SIMS_THREE, "auc")
        assert len(result.ranking) == 3

    def test_ranking_is_permutation(self):
        result = compare_pk_simulations(SIMS_THREE, "auc")
        assert sorted(result.ranking) == list(range(3))

    def test_auc_ranking_descending(self):
        # Higher AUC (lower CL) should rank first
        result = compare_pk_simulations(SIMS_THREE, "auc")
        ordered_aucs = [result.metric_values[i] for i in result.ranking]
        assert ordered_aucs == sorted(ordered_aucs, reverse=True)

    def test_best_simulation_has_highest_auc(self):
        result = compare_pk_simulations(SIMS_THREE, "auc")
        assert result.best_simulation_name == "High"  # lowest CL → highest AUC

    def test_worst_simulation_has_lowest_auc(self):
        result = compare_pk_simulations(SIMS_THREE, "auc")
        assert result.worst_simulation_name == "Low"  # highest CL → lowest AUC

    def test_cmax_ranking_descending(self):
        result = compare_pk_simulations(SIMS_THREE, "cmax")
        ordered = [result.metric_values[i] for i in result.ranking]
        assert ordered == sorted(ordered, reverse=True)

    def test_tmax_metric(self):
        result = compare_pk_simulations(SIMS_TWO, "tmax")
        assert result.metric == "tmax"
        assert all(v >= 0 for v in result.metric_values)

    def test_t_half_metric(self):
        result = compare_pk_simulations(SIMS_TWO, "t_half")
        assert result.metric == "t_half"
        assert all(v > 0 for v in result.metric_values if not math.isnan(v))

    def test_auc_normalized_metric(self):
        result = compare_pk_simulations(SIMS_TWO, "auc_normalized")
        assert result.metric == "auc_normalized"
        assert len(result.metric_values) == 2

    def test_metric_case_insensitive(self):
        result = compare_pk_simulations(SIMS_TWO, "AUC")
        assert result.metric == "auc"

    def test_auc_values_positive(self):
        result = compare_pk_simulations(SIMS_TWO, "auc")
        assert all(v > 0 for v in result.metric_values)

    def test_notes_non_empty(self):
        result = compare_pk_simulations(SIMS_TWO, "auc")
        assert isinstance(result.notes, str)

    # Validation tests
    def test_too_few_simulations(self):
        with pytest.raises(ValueError, match="At least 2"):
            compare_pk_simulations([_make_sim("Only")], "auc")

    def test_invalid_metric(self):
        with pytest.raises(ValueError, match="metric"):
            compare_pk_simulations(SIMS_TWO, "unknown_metric")

    def test_too_few_time_points(self):
        bad_sim = {
            "name": "Bad", "times": [0.0, 1.0], "concentrations": [1.0, 0.5],
            "dose_mg": 100.0,
        }
        with pytest.raises(ValueError):
            compare_pk_simulations([_make_sim("Good"), bad_sim], "auc")

    def test_mismatched_lengths(self):
        bad_sim = {
            "name": "Bad",
            "times": [0.0, 1.0, 2.0, 4.0],
            "concentrations": [1.0, 0.5],
            "dose_mg": 100.0,
        }
        with pytest.raises(ValueError):
            compare_pk_simulations([_make_sim("Good"), bad_sim], "auc")

    def test_non_monotonic_times(self):
        t, c = _monoexp_profile()
        t[3] = t[2]  # duplicate time → non-monotonic
        bad_sim = {"name": "Bad", "times": t, "concentrations": c, "dose_mg": 100.0}
        with pytest.raises(ValueError):
            compare_pk_simulations([_make_sim("Good"), bad_sim], "auc")

    def test_negative_concentration(self):
        t, c = _monoexp_profile()
        c[2] = -0.1
        bad_sim = {"name": "Bad", "times": t, "concentrations": c, "dose_mg": 100.0}
        with pytest.raises(ValueError):
            compare_pk_simulations([_make_sim("Good"), bad_sim], "auc")

    def test_two_sims_best_worst_differ(self):
        result = compare_pk_simulations(SIMS_TWO, "auc")
        assert result.best_simulation_name != result.worst_simulation_name


# ---------------------------------------------------------------------------
# fold_change_matrix tests
# ---------------------------------------------------------------------------


class TestFoldChangeMatrix:
    def test_returns_list_of_lists(self):
        mat = fold_change_matrix(SIMS_TWO)
        assert isinstance(mat, list)
        assert all(isinstance(row, list) for row in mat)

    def test_matrix_shape(self):
        mat = fold_change_matrix(SIMS_THREE)
        assert len(mat) == 3
        assert all(len(row) == 3 for row in mat)

    def test_diagonal_is_one(self):
        mat = fold_change_matrix(SIMS_THREE)
        for i in range(3):
            assert mat[i][i] == pytest.approx(1.0, rel=1e-6)

    def test_reciprocal_relationship(self):
        mat = fold_change_matrix(SIMS_TWO)
        # mat[0][1] * mat[1][0] should == 1.0
        if not (math.isnan(mat[0][1]) or math.isnan(mat[1][0])):
            assert mat[0][1] * mat[1][0] == pytest.approx(1.0, rel=1e-4)

    def test_fold_changes_positive(self):
        mat = fold_change_matrix(SIMS_THREE)
        for row in mat:
            for v in row:
                if not math.isnan(v):
                    assert v > 0

    def test_higher_auc_sim_has_fc_gt_1_vs_lower(self):
        # "High" (low CL) has higher AUC than "Low" (high CL)
        # fold_change[High_idx][Low_idx] should be > 1
        mat = fold_change_matrix(SIMS_THREE)
        # SIMS_THREE: indices 0=Low, 1=Mid, 2=High
        # AUC_High > AUC_Low => mat[2][0] > 1
        assert mat[2][0] > 1.0

    def test_too_few_simulations(self):
        with pytest.raises(ValueError, match="At least 2"):
            fold_change_matrix([_make_sim("Only")])

    def test_malformed_simulation(self):
        bad_sim = {
            "name": "Bad", "times": [0.0, 1.0], "concentrations": [1.0, 0.5],
            "dose_mg": 100.0,
        }
        with pytest.raises(ValueError):
            fold_change_matrix([_make_sim("Good"), bad_sim])
