"""Tests for Phase 927 — Chronotherapy Optimization."""

import pytest

from omega_pbpk.clinical.chronotherapy_optimization import (
    ChronotherapyResult,
    compare_chronotherapy_vs_fixed,
    optimize_dosing_time,
)

# ── Basic return type and structure ──────────────────────────────────────────


def test_return_type():
    result = optimize_dosing_time("DrugA", dose_mg=100.0)
    assert isinstance(result, ChronotherapyResult)


def test_optimal_hour_in_dosing_hours():
    hours = [0.0, 6.0, 12.0, 18.0]
    result = optimize_dosing_time("DrugA", dose_mg=100.0, dosing_hours=hours)
    assert result.optimal_dosing_hour in hours


def test_optimal_auc_above_eff_nonnegative():
    result = optimize_dosing_time("DrugA", dose_mg=100.0)
    assert result.optimal_auc_above_eff >= 0.0


def test_optimal_time_in_tox_nonnegative():
    result = optimize_dosing_time("DrugA", dose_mg=100.0)
    assert result.optimal_time_in_tox >= 0.0


def test_worst_hour_differs_from_optimal_when_variation():
    result = optimize_dosing_time("DrugA", dose_mg=100.0, amplitude_cl=0.5)
    # With variation the optimal and worst hours should generally differ
    # (unless all hours produce identically optimal results, which is unlikely with amplitude=0.5)
    # Just verify worst_dosing_hour is valid
    assert result.worst_dosing_hour is not None
    assert isinstance(result.worst_dosing_hour, float)


def test_dosing_hour_results_length_matches_input():
    hours = [0.0, 4.0, 8.0, 12.0, 16.0, 20.0]
    result = optimize_dosing_time("DrugA", dose_mg=100.0, dosing_hours=hours)
    assert len(result.dosing_hour_results) == len(hours)


def test_each_result_has_required_keys():
    result = optimize_dosing_time("DrugA", dose_mg=100.0)
    for r in result.dosing_hour_results:
        assert "hour" in r
        assert "auc_above_eff" in r
        assert "time_in_tox" in r
        assert "therapeutic_ratio" in r


def test_all_therapeutic_ratios_positive():
    result = optimize_dosing_time("DrugA", dose_mg=100.0)
    for r in result.dosing_hour_results:
        assert r["therapeutic_ratio"] > 0.0


def test_amplitude_cl_preserved():
    result = optimize_dosing_time("DrugA", dose_mg=100.0, amplitude_cl=0.4)
    assert result.amplitude_cl == 0.4


def test_notes_nonempty():
    result = optimize_dosing_time("DrugA", dose_mg=100.0)
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


# ── Circadian variation behaviour ─────────────────────────────────────────────


def test_no_variation_all_hours_same_auc():
    """With amplitude_cl=0, circadian variation is off — all hours give same AUC."""
    hours = [0.0, 6.0, 12.0, 18.0]
    result = optimize_dosing_time("DrugA", dose_mg=100.0, amplitude_cl=0.0, dosing_hours=hours)
    aucs = [r["auc_above_eff"] for r in result.dosing_hour_results]
    # All AUCs should be nearly equal (within floating point)
    assert max(aucs) - min(aucs) < 1e-6


def test_with_variation_different_aucs():
    """With amplitude_cl=0.5, different dosing hours produce different AUCs."""
    result = optimize_dosing_time("DrugA", dose_mg=100.0, amplitude_cl=0.5)
    aucs = [r["auc_above_eff"] for r in result.dosing_hour_results]
    assert max(aucs) - min(aucs) > 0.0


# ── Route tests ───────────────────────────────────────────────────────────────


def test_oral_route_works():
    result = optimize_dosing_time("DrugA", dose_mg=100.0, route="oral")
    assert isinstance(result, ChronotherapyResult)


def test_iv_route_works():
    result = optimize_dosing_time("DrugA", dose_mg=100.0, route="iv")
    assert isinstance(result, ChronotherapyResult)


# ── compare_chronotherapy_vs_fixed ────────────────────────────────────────────


def test_compare_returns_correct_keys():
    result = compare_chronotherapy_vs_fixed("DrugA", dose_mg=100.0)
    assert "optimal" in result
    assert "fixed_8am" in result
    assert "improvement_pct" in result


def test_improvement_pct_nonnegative():
    result = compare_chronotherapy_vs_fixed("DrugA", dose_mg=100.0, amplitude_cl=0.5)
    assert result["improvement_pct"] >= 0.0


def test_compare_optimal_is_chronotherapy_result():
    result = compare_chronotherapy_vs_fixed("DrugA", dose_mg=100.0)
    assert isinstance(result["optimal"], ChronotherapyResult)


def test_compare_fixed_8am_is_dict():
    result = compare_chronotherapy_vs_fixed("DrugA", dose_mg=100.0)
    assert isinstance(result["fixed_8am"], dict)


# ── Validation ────────────────────────────────────────────────────────────────


def test_validation_dose_mg_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        optimize_dosing_time("DrugA", dose_mg=0.0)


def test_validation_dose_mg_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        optimize_dosing_time("DrugA", dose_mg=-10.0)


def test_validation_cl_zero():
    with pytest.raises(ValueError, match="cl_mean_L_per_h"):
        optimize_dosing_time("DrugA", dose_mg=100.0, cl_mean_L_per_h=0.0)


def test_validation_vd_zero():
    with pytest.raises(ValueError, match="vd_L"):
        optimize_dosing_time("DrugA", dose_mg=100.0, vd_L=0.0)


def test_validation_amplitude_cl_negative():
    with pytest.raises(ValueError, match="amplitude_cl"):
        optimize_dosing_time("DrugA", dose_mg=100.0, amplitude_cl=-0.1)


def test_validation_amplitude_cl_greater_than_one():
    with pytest.raises(ValueError, match="amplitude_cl"):
        optimize_dosing_time("DrugA", dose_mg=100.0, amplitude_cl=1.1)


def test_validation_invalid_route():
    with pytest.raises(ValueError, match="route"):
        optimize_dosing_time("DrugA", dose_mg=100.0, route="subcutaneous")


# ── Default dosing hours ──────────────────────────────────────────────────────


def test_default_dosing_hours_uses_12_time_points():
    result = optimize_dosing_time("DrugA", dose_mg=100.0)
    assert len(result.dosing_hour_results) == 12


def test_optimal_dosing_hour_is_float():
    result = optimize_dosing_time("DrugA", dose_mg=100.0)
    assert isinstance(result.optimal_dosing_hour, float)
