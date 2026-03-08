"""Tests for Phase 467 — Muscle Compartment PK."""

import pytest

from omega_pbpk.core.muscle_compartment_pk import (
    MuscleCompartmentResult,
    compare_drugs_muscle,
    simulate_muscle_pk,
)


def default_result() -> MuscleCompartmentResult:
    return simulate_muscle_pk(
        drug_name="DrugA",
        dose_mg=100.0,
        logP=2.0,
        cl_sys_L_per_h=10.0,
    )


# --- Return type ---


def test_return_type():
    result = default_result()
    assert isinstance(result, MuscleCompartmentResult)


def test_result_has_drug_name():
    result = default_result()
    assert result.drug_name == "DrugA"


# --- Initial conditions ---


def test_plasma_initial_concentration():
    """C_plasma[0] = dose / Vd_plasma (IV bolus)."""
    result = simulate_muscle_pk("X", 100.0, 2.0, 10.0, vd_plasma_L=5.0)
    assert abs(result.c_plasma_mg_L[0] - 100.0 / 5.0) < 1e-9


def test_muscle_starts_near_zero():
    result = default_result()
    assert result.c_muscle_mg_L[0] == pytest.approx(0.0, abs=1e-9)


# --- Time series length ---


def test_times_length_matches_concentrations():
    result = default_result()
    assert len(result.times_h) == len(result.c_plasma_mg_L)
    assert len(result.times_h) == len(result.c_muscle_mg_L)


def test_first_time_is_zero():
    result = default_result()
    assert result.times_h[0] == pytest.approx(0.0)


def test_last_time_matches_t_end():
    result = simulate_muscle_pk("X", 100.0, 2.0, 10.0, t_end_h=12.0, dt_h=0.1)
    assert result.times_h[-1] == pytest.approx(12.0, rel=1e-4)


# --- Kp from logP ---


def test_kp_muscle_formula():
    """kp = 0.5 + 0.3 * logP for logP in reasonable range."""
    result = simulate_muscle_pk("X", 100.0, 2.0, 10.0)
    assert result.kp_muscle == pytest.approx(0.5 + 0.3 * 2.0)


def test_higher_logp_higher_kp():
    r_low = simulate_muscle_pk("X", 100.0, 1.0, 10.0)
    r_high = simulate_muscle_pk("X", 100.0, 4.0, 10.0)
    assert r_high.kp_muscle > r_low.kp_muscle


def test_kp_clamped_at_minimum():
    """logP very negative: kp clamped to 0.1."""
    result = simulate_muscle_pk("X", 100.0, -10.0, 10.0)
    assert result.kp_muscle == pytest.approx(0.1)


def test_kp_clamped_at_maximum():
    """logP very high: kp clamped to 10.0."""
    result = simulate_muscle_pk("X", 100.0, 50.0, 10.0)
    assert result.kp_muscle == pytest.approx(10.0)


# --- cmax and tmax muscle ---


def test_cmax_muscle_positive():
    result = default_result()
    assert result.cmax_muscle_mg_L > 0.0


def test_tmax_muscle_positive():
    """Drug distribution takes time — muscle peak is after t=0."""
    result = default_result()
    assert result.tmax_muscle_h > 0.0


def test_cmax_muscle_equals_max_of_list():
    result = default_result()
    assert result.cmax_muscle_mg_L == pytest.approx(max(result.c_muscle_mg_L))


# --- AUC ---


def test_auc_plasma_positive():
    result = default_result()
    assert result.auc_plasma_mg_h_per_L > 0.0


def test_auc_muscle_positive():
    result = default_result()
    assert result.auc_muscle_mg_h_per_L > 0.0


def test_auc_larger_dose_larger_auc():
    r1 = simulate_muscle_pk("X", 100.0, 2.0, 10.0)
    r2 = simulate_muscle_pk("X", 200.0, 2.0, 10.0)
    assert r2.auc_plasma_mg_h_per_L > r1.auc_plasma_mg_h_per_L


# --- Muscle-to-plasma ratio ---


def test_muscle_to_plasma_ratio_positive():
    result = default_result()
    assert result.muscle_to_plasma_ratio >= 0.0


# --- compare_drugs_muscle ---


def test_compare_drugs_muscle_returns_list():
    drugs = [
        {"drug_name": "A", "dose_mg": 100.0, "logP": 1.0, "cl_sys_L_per_h": 10.0},
        {"drug_name": "B", "dose_mg": 100.0, "logP": 3.0, "cl_sys_L_per_h": 10.0},
        {"drug_name": "C", "dose_mg": 100.0, "logP": 2.0, "cl_sys_L_per_h": 10.0},
    ]
    results = compare_drugs_muscle(drugs)
    assert isinstance(results, list)
    assert len(results) == 3


def test_compare_drugs_muscle_sorted_by_kp_descending():
    drugs = [
        {"drug_name": "A", "dose_mg": 100.0, "logP": 1.0, "cl_sys_L_per_h": 10.0},
        {"drug_name": "B", "dose_mg": 100.0, "logP": 3.0, "cl_sys_L_per_h": 10.0},
        {"drug_name": "C", "dose_mg": 100.0, "logP": 2.0, "cl_sys_L_per_h": 10.0},
    ]
    results = compare_drugs_muscle(drugs)
    kps = [r.kp_muscle for r in results]
    assert kps == sorted(kps, reverse=True)


# --- Notes ---


def test_notes_is_string():
    result = default_result()
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


# --- Validation errors ---


def test_validation_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_muscle_pk("X", -10.0, 2.0, 10.0)


def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_muscle_pk("X", 0.0, 2.0, 10.0)


def test_validation_cl_zero():
    with pytest.raises(ValueError, match="cl_sys"):
        simulate_muscle_pk("X", 100.0, 2.0, 0.0)


def test_validation_vd_plasma_zero():
    with pytest.raises(ValueError, match="vd_plasma"):
        simulate_muscle_pk("X", 100.0, 2.0, 10.0, vd_plasma_L=0.0)


def test_validation_vd_muscle_zero():
    with pytest.raises(ValueError, match="vd_muscle"):
        simulate_muscle_pk("X", 100.0, 2.0, 10.0, vd_muscle_L=0.0)


def test_validation_q_muscle_zero():
    with pytest.raises(ValueError, match="Q_muscle"):
        simulate_muscle_pk("X", 100.0, 2.0, 10.0, Q_muscle_L_per_h=0.0)
