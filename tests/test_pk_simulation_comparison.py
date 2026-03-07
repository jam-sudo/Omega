"""Tests for PK simulation comparison module."""

import pytest

from omega_pbpk.analysis.pk_simulation_comparison import (
    ComparisonResult,
    PKScenario,
    compare_scenarios,
    rank_scenarios_by_auc,
    simulate_scenario,
)


def make_scenario(name="ScenA", dose_mg=100.0, cl=5.0, vd=50.0, route="oral"):
    return simulate_scenario(name=name, dose_mg=dose_mg, cl_L_per_h=cl, vd_L=vd, route=route)


def test_simulate_scenario_returns_result():
    result = make_scenario()
    assert isinstance(result, PKScenario)


def test_scenario_cmax_positive():
    result = make_scenario()
    assert result.cmax_mg_L > 0.0


def test_scenario_auc_positive():
    result = make_scenario()
    assert result.auc_mg_h_L > 0.0


def test_scenario_tmax_positive_for_oral():
    result = make_scenario(route="oral")
    assert result.tmax_h >= 0.0


def test_scenario_t_half_positive():
    result = make_scenario()
    assert result.t_half_h > 0.0


def test_double_dose_double_cmax():
    s1 = simulate_scenario("S1", dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0)
    s2 = simulate_scenario("S2", dose_mg=200.0, cl_L_per_h=5.0, vd_L=50.0)
    ratio = s2.cmax_mg_L / s1.cmax_mg_L
    assert abs(ratio - 2.0) < 0.01


def test_double_dose_double_auc():
    s1 = simulate_scenario("S1", dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0)
    s2 = simulate_scenario("S2", dose_mg=200.0, cl_L_per_h=5.0, vd_L=50.0)
    ratio = s2.auc_mg_h_L / s1.auc_mg_h_L
    assert abs(ratio - 2.0) < 0.01


def test_f_oral_reduces_cmax():
    s_full = simulate_scenario("Full", dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0, f_oral=1.0)
    s_half = simulate_scenario("Half", dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0, f_oral=0.5)
    assert s_half.cmax_mg_L < s_full.cmax_mg_L


def test_compare_returns_result():
    s1 = make_scenario("A")
    s2 = make_scenario("B", dose_mg=200.0)
    result = compare_scenarios([s1, s2])
    assert isinstance(result, ComparisonResult)


def test_compare_n_scenarios():
    scenarios = [make_scenario(f"S{i}", dose_mg=100.0 * (i + 1)) for i in range(3)]
    result = compare_scenarios(scenarios)
    assert result.n_scenarios == 3


def test_compare_reference_ratio_is_1():
    s1 = make_scenario("Ref")
    s2 = make_scenario("Other", dose_mg=200.0)
    result = compare_scenarios([s1, s2], reference_name="Ref")
    assert abs(result.cmax_ratios["Ref"] - 1.0) < 1e-9


def test_compare_auc_ratios_positive():
    s1 = make_scenario("A")
    s2 = make_scenario("B", dose_mg=150.0)
    result = compare_scenarios([s1, s2])
    for ratio in result.auc_ratios.values():
        assert ratio > 0.0


def test_compare_best_scenario():
    s1 = simulate_scenario("Low", dose_mg=50.0, cl_L_per_h=5.0, vd_L=50.0)
    s2 = simulate_scenario("High", dose_mg=200.0, cl_L_per_h=5.0, vd_L=50.0)
    s3 = simulate_scenario("Mid", dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0)
    result = compare_scenarios([s1, s2, s3])
    assert result.best_scenario == "High"


def test_compare_worst_scenario():
    s1 = simulate_scenario("Low", dose_mg=50.0, cl_L_per_h=5.0, vd_L=50.0)
    s2 = simulate_scenario("High", dose_mg=200.0, cl_L_per_h=5.0, vd_L=50.0)
    s3 = simulate_scenario("Mid", dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0)
    result = compare_scenarios([s1, s2, s3])
    assert result.worst_scenario == "Low"


def test_auc_cv_nonnegative():
    s1 = make_scenario("A")
    s2 = make_scenario("B", dose_mg=200.0)
    result = compare_scenarios([s1, s2])
    assert result.auc_cv_pct >= 0.0


def test_identical_scenarios_zero_cv():
    s1 = simulate_scenario("S1", dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0)
    s2 = simulate_scenario("S2", dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0)
    result = compare_scenarios([s1, s2])
    assert result.auc_cv_pct < 1e-6


def test_rank_scenarios_descending():
    s1 = simulate_scenario("Low", dose_mg=50.0, cl_L_per_h=5.0, vd_L=50.0)
    s2 = simulate_scenario("High", dose_mg=200.0, cl_L_per_h=5.0, vd_L=50.0)
    s3 = simulate_scenario("Mid", dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0)
    ranked = rank_scenarios_by_auc([s1, s2, s3])
    aucs = [s.auc_mg_h_L for s in ranked]
    assert aucs == sorted(aucs, reverse=True)


def test_rank_scenarios_returns_list():
    s1 = make_scenario("A")
    s2 = make_scenario("B", dose_mg=200.0)
    result = rank_scenarios_by_auc([s1, s2])
    assert isinstance(result, list)


def test_compare_tmax_difference_ref_is_zero():
    s1 = make_scenario("Ref")
    s2 = make_scenario("Other", dose_mg=200.0)
    result = compare_scenarios([s1, s2], reference_name="Ref")
    assert result.tmax_differences_h["Ref"] == 0.0


def test_invalid_single_scenario_raises():
    s1 = make_scenario("Only")
    with pytest.raises(ValueError):
        compare_scenarios([s1])


def test_invalid_reference_name_raises():
    s1 = make_scenario("A")
    s2 = make_scenario("B")
    with pytest.raises(ValueError):
        compare_scenarios([s1, s2], reference_name="NonExistent")


def test_notes_nonempty():
    s1 = make_scenario("A")
    s2 = make_scenario("B", dose_mg=150.0)
    result = compare_scenarios([s1, s2])
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0
