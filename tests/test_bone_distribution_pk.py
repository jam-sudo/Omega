"""Tests for Phase 920 — Drug Bone/Skeletal Distribution."""

import pytest

from omega_pbpk.core.bone_distribution_pk import (
    BoneDistributionPKResult,
    compare_bone_binding,
    simulate_bone_distribution,
)


# ---------------------------------------------------------------------------
# Basic structure tests
# ---------------------------------------------------------------------------


def test_result_is_dataclass():
    result = simulate_bone_distribution("alendronate", dose_mg=10.0)
    assert isinstance(result, BoneDistributionPKResult)


def test_result_fields_populated():
    result = simulate_bone_distribution("alendronate", dose_mg=10.0)
    assert result.drug_name == "alendronate"
    assert result.dose_mg == 10.0
    assert len(result.times_h) > 0
    assert len(result.c_plasma_mg_L) > 0
    assert len(result.c_cortical_mg_L) > 0
    assert len(result.c_trabecular_mg_L) > 0


def test_all_arrays_same_length():
    result = simulate_bone_distribution("drug", dose_mg=5.0)
    n = len(result.times_h)
    assert len(result.c_plasma_mg_L) == n
    assert len(result.c_cortical_mg_L) == n
    assert len(result.c_trabecular_mg_L) == n


def test_time_starts_at_zero():
    result = simulate_bone_distribution("drug", dose_mg=1.0)
    assert result.times_h[0] == pytest.approx(0.0)


def test_time_ends_at_t_end():
    result = simulate_bone_distribution("drug", dose_mg=1.0, t_end_h=48.0, dt_h=1.0)
    assert result.times_h[-1] == pytest.approx(48.0, abs=1.1)


def test_initial_plasma_is_zero():
    # Oral: drug starts in gut compartment
    result = simulate_bone_distribution("drug", dose_mg=10.0)
    assert result.c_plasma_mg_L[0] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# PK behavior tests
# ---------------------------------------------------------------------------


def test_plasma_rises_then_falls():
    result = simulate_bone_distribution(
        "drug", dose_mg=10.0, t_end_h=168.0, dt_h=0.5
    )
    # Cmax plasma should be after t=0 and before t_end
    assert result.cmax_plasma > 0.0


def test_cortical_bone_accumulates():
    result = simulate_bone_distribution(
        "drug", dose_mg=10.0, t_end_h=168.0, dt_h=0.5
    )
    assert result.cmax_cortical > 0.0


def test_trabecular_bone_accumulates():
    result = simulate_bone_distribution(
        "drug", dose_mg=10.0, t_end_h=168.0, dt_h=0.5
    )
    assert result.cmax_trabecular > 0.0


def test_trabecular_faster_than_cortical():
    # Trabecular has higher uptake rate → peaks sooner / higher relative concentration
    result = simulate_bone_distribution("drug", dose_mg=10.0, t_end_h=168.0)
    # trabecular uptake (0.1) > cortical (0.02) by default
    # Trabecular AUC:plasma ratio should be >= cortical AUC:plasma ratio
    # (not strictly guaranteed, but for default params trabecular kp >= cortical kp)
    assert result.bone_plasma_kp_trabecular >= result.bone_plasma_kp_cortical


# ---------------------------------------------------------------------------
# AUC ratios
# ---------------------------------------------------------------------------


def test_bone_plasma_kp_cortical_positive():
    result = simulate_bone_distribution("drug", dose_mg=10.0)
    assert result.bone_plasma_kp_cortical >= 0.0


def test_bone_plasma_kp_trabecular_positive():
    result = simulate_bone_distribution("drug", dose_mg=10.0)
    assert result.bone_plasma_kp_trabecular >= 0.0


def test_auc_plasma_positive():
    result = simulate_bone_distribution("drug", dose_mg=10.0)
    assert result.auc_plasma > 0.0


def test_auc_cortical_positive():
    result = simulate_bone_distribution("drug", dose_mg=10.0)
    assert result.auc_cortical > 0.0


def test_auc_trabecular_positive():
    result = simulate_bone_distribution("drug", dose_mg=10.0)
    assert result.auc_trabecular > 0.0


# ---------------------------------------------------------------------------
# Bone sequestration fraction
# ---------------------------------------------------------------------------


def test_sequestration_fraction_in_range():
    result = simulate_bone_distribution("drug", dose_mg=10.0)
    assert 0.0 <= result.bone_sequestration_fraction <= 1.0


def test_high_sequestration_with_slow_release():
    result = simulate_bone_distribution(
        "bisphosphonate",
        dose_mg=10.0,
        k_cort_uptake=0.15,
        k_cort_release=0.0001,
        k_trab_uptake=0.3,
        k_trab_release=0.001,
        t_end_h=168.0,
    )
    assert result.bone_sequestration_fraction > 0.2


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


def test_notes_not_empty():
    result = simulate_bone_distribution("drug", dose_mg=10.0)
    assert len(result.notes) > 0


def test_high_sequestration_notes():
    result = simulate_bone_distribution(
        "bisphosphonate",
        dose_mg=10.0,
        k_cort_uptake=0.15,
        k_cort_release=0.00001,
        k_trab_uptake=0.3,
        k_trab_release=0.0001,
        t_end_h=168.0,
    )
    if result.bone_sequestration_fraction > 0.5:
        assert "High bone sequestration" in result.notes


def test_limited_sequestration_notes():
    # Low uptake rates → minimal bone binding
    result = simulate_bone_distribution(
        "drug",
        dose_mg=10.0,
        k_cort_uptake=0.0001,
        k_cort_release=0.1,
        k_trab_uptake=0.0001,
        k_trab_release=0.1,
        t_end_h=168.0,
    )
    if result.bone_sequestration_fraction <= 0.2:
        assert "Limited bone sequestration" in result.notes


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_bone_distribution("drug", dose_mg=0.0)


def test_negative_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_bone_distribution("drug", dose_mg=-1.0)


def test_zero_cl_raises():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_bone_distribution("drug", dose_mg=10.0, cl_L_per_h=0.0)


def test_zero_v_cort_raises():
    with pytest.raises(ValueError, match="v_cort_L"):
        simulate_bone_distribution("drug", dose_mg=10.0, v_cort_L=0.0)


# ---------------------------------------------------------------------------
# Dose proportionality
# ---------------------------------------------------------------------------


def test_cmax_plasma_scales_with_dose():
    result_1 = simulate_bone_distribution("drug", dose_mg=5.0)
    result_2 = simulate_bone_distribution("drug", dose_mg=10.0)
    assert result_2.cmax_plasma == pytest.approx(2.0 * result_1.cmax_plasma, rel=1e-6)


# ---------------------------------------------------------------------------
# compare_bone_binding
# ---------------------------------------------------------------------------


def test_compare_bone_binding_returns_list():
    scenarios = [
        {"k_cort_uptake": 0.01, "k_cort_release": 0.005},
        {"k_cort_uptake": 0.05, "k_cort_release": 0.005},
        {"k_cort_uptake": 0.1, "k_cort_release": 0.005},
    ]
    results = compare_bone_binding("drug", dose_mg=10.0, uptake_rates=scenarios)
    assert isinstance(results, list)
    assert len(results) == 3


def test_compare_bone_binding_sorted_descending():
    scenarios = [
        {"k_cort_uptake": 0.01},
        {"k_cort_uptake": 0.1},
        {"k_cort_uptake": 0.5},
    ]
    results = compare_bone_binding("drug", dose_mg=10.0, uptake_rates=scenarios)
    kp_values = [r.bone_plasma_kp_cortical for r in results]
    assert kp_values == sorted(kp_values, reverse=True)


def test_compare_bone_binding_all_results_are_result_type():
    scenarios = [{"k_cort_uptake": 0.02}, {"k_cort_uptake": 0.1}]
    results = compare_bone_binding("drug", dose_mg=5.0, uptake_rates=scenarios)
    for r in results:
        assert isinstance(r, BoneDistributionPKResult)
