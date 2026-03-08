"""Tests for Phase 609 — Receptor Desensitization PD Model."""

import pytest

from omega_pbpk.core.receptor_desensitization_p609 import (
    ReceptorDesensitizationResult,
    simulate_receptor_desensitization,
)

# Default parameters for quick tests
DEFAULTS = dict(
    drug_name="TestDrug",
    dose_mg=100.0,
    n_doses=3,
    tau_h=12.0,
    vd_L=50.0,
    cl_L_per_h=5.0,
    emax=100.0,
    ec50_mg_L=1.0,
    k_desens_per_h=0.5,
    k_resens_per_h=0.1,
)


def run_default(**overrides):
    params = {**DEFAULTS, **overrides}
    return simulate_receptor_desensitization(**params)


# --- Return type and structure ---


def test_returns_correct_type():
    result = run_default()
    assert isinstance(result, ReceptorDesensitizationResult)


def test_drug_name_stored():
    result = run_default(drug_name="Morphine")
    assert result.drug_name == "Morphine"


def test_dose_stored():
    result = run_default(dose_mg=50.0)
    assert result.dose_mg == 50.0


def test_n_doses_stored():
    result = run_default(n_doses=5)
    assert result.n_doses == 5


def test_list_fields_are_lists():
    result = run_default()
    assert isinstance(result.times_h, list)
    assert isinstance(result.c_drug_mg_L, list)
    assert isinstance(result.r_active, list)
    assert isinstance(result.effect, list)


def test_list_lengths_consistent():
    result = run_default()
    n = len(result.times_h)
    assert len(result.c_drug_mg_L) == n
    assert len(result.r_active) == n
    assert len(result.effect) == n


def test_times_start_at_zero():
    result = run_default()
    assert result.times_h[0] == pytest.approx(0.0, abs=1e-9)


def test_times_end_near_t_end():
    # Default t_end = n_doses * tau_h + 24 = 3*12+24 = 60 h
    result = run_default()
    assert result.times_h[-1] == pytest.approx(60.0, abs=0.2)


def test_custom_t_end():
    result = run_default(t_end_h=48.0)
    assert result.times_h[-1] == pytest.approx(48.0, abs=0.2)


def test_c_drug_nonnegative():
    result = run_default()
    assert all(c >= 0.0 for c in result.c_drug_mg_L)


def test_r_active_in_unit_interval():
    result = run_default()
    assert all(0.0 <= r <= 1.0 for r in result.r_active)


def test_effect_nonnegative():
    result = run_default()
    assert all(e >= 0.0 for e in result.effect)


def test_r_active_starts_at_one():
    result = run_default()
    assert result.r_active[0] == pytest.approx(1.0, abs=1e-6)


# --- Tolerance development ---


def test_tolerance_develops_after_multiple_doses():
    """After many doses with fast desensitization, tolerance_index < 1."""
    result = run_default(
        n_doses=5,
        k_desens_per_h=2.0,
        k_resens_per_h=0.05,
        tau_h=8.0,
    )
    assert result.tolerance_index < 1.0


def test_single_dose_tolerance_index_is_one():
    """With one dose, tolerance_index should be 1.0 (equal to itself)."""
    result = run_default(n_doses=1, tau_h=24.0)
    assert result.tolerance_index == pytest.approx(1.0, abs=1e-6)


def test_peak_effect_dose1_positive():
    result = run_default()
    assert result.peak_effect_dose1 > 0.0


def test_peak_effect_last_dose_positive():
    result = run_default()
    assert result.peak_effect_last_dose > 0.0


def test_tolerance_index_float():
    result = run_default()
    assert isinstance(result.tolerance_index, float)


def test_tolerance_index_decreases_with_stronger_desensitization():
    """Stronger desensitization should give a lower tolerance index."""
    weak = run_default(k_desens_per_h=0.1, n_doses=5)
    strong = run_default(k_desens_per_h=2.0, n_doses=5)
    assert strong.tolerance_index <= weak.tolerance_index


# --- Receptor recovery ---


def test_receptor_recovery_time_is_float():
    result = run_default()
    assert isinstance(result.receptor_recovery_time_h, float)


def test_receptor_recovery_is_positive_or_inf():
    result = run_default()
    assert result.receptor_recovery_time_h > 0.0


def test_receptor_recovers_faster_with_high_resensitization():
    """Higher k_resens means faster receptor recovery."""
    slow = run_default(k_resens_per_h=0.01, t_end_h=200.0)
    fast = run_default(k_resens_per_h=1.0, t_end_h=200.0)
    assert fast.receptor_recovery_time_h <= slow.receptor_recovery_time_h


# --- Notes ---


def test_notes_is_string():
    result = run_default()
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


# --- Validation ---


def test_raises_on_n_doses_zero():
    with pytest.raises(ValueError):
        run_default(n_doses=0)


def test_raises_on_negative_tau():
    with pytest.raises(ValueError):
        run_default(tau_h=-1.0)


def test_raises_on_zero_dose():
    with pytest.raises(ValueError):
        run_default(dose_mg=0.0)


def test_raises_on_zero_k_desens():
    with pytest.raises(ValueError):
        run_default(k_desens_per_h=0.0)


def test_raises_on_zero_k_resens():
    with pytest.raises(ValueError):
        run_default(k_resens_per_h=0.0)


def test_raises_on_nonpositive_vd():
    with pytest.raises(ValueError):
        run_default(vd_L=0.0)
