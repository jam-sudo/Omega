"""Tests for Phase 948 — Drug-target binding kinetics."""
import math
import pytest
from omega_pbpk.prediction.drug_target_binding import (
    DrugTargetBindingResult,
    simulate_target_binding,
    compare_binding_kinetics,
)


# --- Basic return type ---

def test_return_type():
    result = simulate_target_binding("Drug", dose_mg=100.0, mw_Da=300.0, ic50_nM=10.0)
    assert isinstance(result, DrugTargetBindingResult)


def test_kd_equals_ic50():
    ic50 = 50.0
    result = simulate_target_binding("Drug", dose_mg=100.0, mw_Da=300.0, ic50_nM=ic50)
    assert abs(result.kd_nM - ic50) < 1e-9


def test_kon_positive():
    result = simulate_target_binding("Drug", dose_mg=100.0, mw_Da=300.0, ic50_nM=10.0)
    assert result.kon_per_nM_per_h > 0


def test_koff_positive():
    result = simulate_target_binding("Drug", dose_mg=100.0, mw_Da=300.0, ic50_nM=10.0)
    assert result.koff_per_h > 0


def test_residence_time_positive():
    result = simulate_target_binding("Drug", dose_mg=100.0, mw_Da=300.0, ic50_nM=10.0)
    assert result.residence_time_h > 0


def test_residence_time_equals_one_over_koff():
    result = simulate_target_binding("Drug", dose_mg=100.0, mw_Da=300.0, ic50_nM=10.0)
    expected_rt = 1.0 / result.koff_per_h
    assert abs(result.residence_time_h - expected_rt) / expected_rt < 1e-6


def test_peak_occupancy_in_range():
    result = simulate_target_binding("Drug", dose_mg=100.0, mw_Da=300.0, ic50_nM=10.0)
    assert 0.0 <= result.peak_occupancy <= 1.0


def test_time_to_peak_occupancy_non_negative():
    result = simulate_target_binding("Drug", dose_mg=100.0, mw_Da=300.0, ic50_nM=10.0)
    assert result.time_to_peak_occupancy_h >= 0


def test_time_above_50pct_non_negative():
    result = simulate_target_binding("Drug", dose_mg=100.0, mw_Da=300.0, ic50_nM=10.0)
    assert result.time_above_50pct_occupancy_h >= 0


def test_auc_occupancy_non_negative():
    result = simulate_target_binding("Drug", dose_mg=100.0, mw_Da=300.0, ic50_nM=10.0)
    assert result.auc_occupancy >= 0


def test_kinetic_selectivity_index_non_negative():
    result = simulate_target_binding("Drug", dose_mg=100.0, mw_Da=300.0, ic50_nM=10.0)
    assert result.kinetic_selectivity_index >= 0


def test_notes_non_empty():
    result = simulate_target_binding("Drug", dose_mg=100.0, mw_Da=300.0, ic50_nM=10.0)
    assert isinstance(result.notes, str) and len(result.notes) > 0


def test_cmax_plasma_positive():
    result = simulate_target_binding("Drug", dose_mg=100.0, mw_Da=300.0, ic50_nM=10.0)
    assert result.cmax_plasma_mg_L > 0


def test_cmax_equals_dose_over_vd():
    dose = 200.0
    vd = 80.0
    result = simulate_target_binding("Drug", dose_mg=dose, mw_Da=300.0, ic50_nM=10.0,
                                      vd_L=vd)
    assert abs(result.cmax_plasma_mg_L - dose / vd) < 1e-9


def test_receptor_occupancy_starts_at_zero():
    result = simulate_target_binding("Drug", dose_mg=100.0, mw_Da=300.0, ic50_nM=10.0)
    assert result.receptor_occupancy[0] == pytest.approx(0.0, abs=1e-9)


def test_receptor_occupancy_is_list_of_floats():
    result = simulate_target_binding("Drug", dose_mg=100.0, mw_Da=300.0, ic50_nM=10.0)
    assert isinstance(result.receptor_occupancy, list)
    assert all(isinstance(v, float) for v in result.receptor_occupancy)


# --- Pharmacology checks ---

def test_low_ic50_higher_peak_occupancy():
    """Lower IC50 (higher potency) → higher peak occupancy.

    Use a small dose so drug concentration is sub-saturating for the weak binder.
    Cmax = dose/Vd / MW * 1e6 nM = 0.01/50 / 300 * 1e6 ≈ 0.67 nM << IC50=1000 nM.
    """
    low_ic50 = simulate_target_binding("Potent", dose_mg=0.01, mw_Da=300.0,
                                        ic50_nM=0.1, vd_L=50.0, cl_L_per_h=1.0,
                                        t_end_h=48.0)
    high_ic50 = simulate_target_binding("Weak", dose_mg=0.01, mw_Da=300.0,
                                         ic50_nM=100.0, vd_L=50.0, cl_L_per_h=1.0,
                                         t_end_h=48.0)
    assert low_ic50.peak_occupancy > high_ic50.peak_occupancy


def test_large_mw_lower_kon():
    """Larger MW → lower kon (slower association)."""
    small_mw = simulate_target_binding("Small", dose_mg=100.0, mw_Da=200.0, ic50_nM=10.0)
    large_mw = simulate_target_binding("Large", dose_mg=100.0, mw_Da=1000.0, ic50_nM=10.0)
    assert small_mw.kon_per_nM_per_h > large_mw.kon_per_nM_per_h


# --- compare_binding_kinetics ---

def test_compare_binding_kinetics_correct_length():
    compounds = [
        {"drug_name": "A", "dose_mg": 100.0, "mw_Da": 300.0, "ic50_nM": 10.0},
        {"drug_name": "B", "dose_mg": 100.0, "mw_Da": 400.0, "ic50_nM": 50.0},
        {"drug_name": "C", "dose_mg": 100.0, "mw_Da": 250.0, "ic50_nM": 5.0},
    ]
    results = compare_binding_kinetics(compounds)
    assert len(results) == 3


def test_compare_binding_kinetics_sorted_descending():
    compounds = [
        {"drug_name": "A", "dose_mg": 100.0, "mw_Da": 300.0, "ic50_nM": 100.0},
        {"drug_name": "B", "dose_mg": 100.0, "mw_Da": 300.0, "ic50_nM": 1.0},
        {"drug_name": "C", "dose_mg": 100.0, "mw_Da": 300.0, "ic50_nM": 10.0},
    ]
    results = compare_binding_kinetics(compounds)
    ksi_values = [r.kinetic_selectivity_index for r in results]
    assert ksi_values == sorted(ksi_values, reverse=True)


# --- Validation ---

def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_target_binding("Drug", dose_mg=0.0, mw_Da=300.0, ic50_nM=10.0)


def test_validation_mw_zero():
    with pytest.raises(ValueError, match="mw_Da"):
        simulate_target_binding("Drug", dose_mg=100.0, mw_Da=0.0, ic50_nM=10.0)


def test_validation_ic50_zero():
    with pytest.raises(ValueError, match="ic50_nM"):
        simulate_target_binding("Drug", dose_mg=100.0, mw_Da=300.0, ic50_nM=0.0)


def test_validation_vd_zero():
    with pytest.raises(ValueError, match="vd_L"):
        simulate_target_binding("Drug", dose_mg=100.0, mw_Da=300.0, ic50_nM=10.0,
                                  vd_L=0.0)


def test_validation_cl_zero():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_target_binding("Drug", dose_mg=100.0, mw_Da=300.0, ic50_nM=10.0,
                                  cl_L_per_h=0.0)


def test_validation_r_total_zero():
    with pytest.raises(ValueError, match="r_total_nM"):
        simulate_target_binding("Drug", dose_mg=100.0, mw_Da=300.0, ic50_nM=10.0,
                                  r_total_nM=0.0)
