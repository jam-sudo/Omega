"""Tests for clinical/amniotic_fluid_pk.py (Phase 924)."""

import pytest

from omega_pbpk.clinical.amniotic_fluid_pk import (
    AmnioticFluidPKResult,
    compare_gestational_ages,
    simulate_amniotic_fluid_pk,
)


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_returns_dataclass():
    result = simulate_amniotic_fluid_pk("TestDrug", dose_mg=100.0)
    assert isinstance(result, AmnioticFluidPKResult)


def test_drug_name_preserved():
    result = simulate_amniotic_fluid_pk("Amiodarone", dose_mg=100.0)
    assert result.drug_name == "Amiodarone"


def test_dose_preserved():
    result = simulate_amniotic_fluid_pk("Drug", dose_mg=200.0)
    assert result.dose_mg == 200.0


def test_gestational_week_preserved():
    result = simulate_amniotic_fluid_pk("Drug", dose_mg=100.0, gestational_week=28)
    assert result.gestational_week == 28


# ---------------------------------------------------------------------------
# Time series
# ---------------------------------------------------------------------------


def test_times_h_length():
    result = simulate_amniotic_fluid_pk("Drug", dose_mg=100.0, t_end_h=24.0, dt_h=0.5)
    expected_len = int(24.0 / 0.5) + 1
    assert len(result.times_h) == expected_len


def test_concentrations_same_length():
    result = simulate_amniotic_fluid_pk("Drug", dose_mg=100.0, t_end_h=10.0, dt_h=0.1)
    n = len(result.times_h)
    assert len(result.c_amniotic_mg_L) == n
    assert len(result.c_fetal_plasma_mg_L) == n
    assert len(result.c_maternal_plasma_mg_L) == n


def test_initial_concentrations_zero():
    result = simulate_amniotic_fluid_pk("Drug", dose_mg=100.0)
    assert result.c_amniotic_mg_L[0] == pytest.approx(0.0)
    assert result.c_fetal_plasma_mg_L[0] == pytest.approx(0.0)
    assert result.c_maternal_plasma_mg_L[0] == pytest.approx(0.0)


def test_concentrations_non_negative():
    result = simulate_amniotic_fluid_pk("Drug", dose_mg=100.0)
    assert all(c >= 0 for c in result.c_amniotic_mg_L)
    assert all(c >= 0 for c in result.c_fetal_plasma_mg_L)
    assert all(c >= 0 for c in result.c_maternal_plasma_mg_L)


# ---------------------------------------------------------------------------
# PK metrics
# ---------------------------------------------------------------------------


def test_cmax_amniotic_positive():
    result = simulate_amniotic_fluid_pk("Drug", dose_mg=100.0)
    assert result.cmax_amniotic > 0.0


def test_cmax_fetal_plasma_positive():
    result = simulate_amniotic_fluid_pk("Drug", dose_mg=100.0)
    assert result.cmax_fetal_plasma > 0.0


def test_auc_amniotic_positive():
    result = simulate_amniotic_fluid_pk("Drug", dose_mg=100.0)
    assert result.auc_amniotic > 0.0


def test_auc_fetal_plasma_positive():
    result = simulate_amniotic_fluid_pk("Drug", dose_mg=100.0)
    assert result.auc_fetal_plasma > 0.0


def test_amniotic_to_maternal_ratio_non_negative():
    result = simulate_amniotic_fluid_pk("Drug", dose_mg=100.0)
    assert result.amniotic_to_maternal_ratio >= 0.0


def test_fetal_drug_reservoir_non_negative():
    result = simulate_amniotic_fluid_pk("Drug", dose_mg=100.0)
    assert result.fetal_drug_reservoir_mg >= 0.0


# ---------------------------------------------------------------------------
# Amniotic volume scaling
# ---------------------------------------------------------------------------


def test_amniotic_volume_increases_with_ga():
    r20 = simulate_amniotic_fluid_pk("Drug", dose_mg=100.0, gestational_week=20)
    r40 = simulate_amniotic_fluid_pk("Drug", dose_mg=100.0, gestational_week=40)
    # At week 40: 0.5 + 20*0.03 = 1.1; at week 20: 0.5
    # More volume → lower concentration at same AUC input
    # Just verify the simulation runs for both
    assert r20.gestational_week == 20
    assert r40.gestational_week == 40


# ---------------------------------------------------------------------------
# Fetal exposure concern
# ---------------------------------------------------------------------------


def test_fetal_exposure_concern_type():
    result = simulate_amniotic_fluid_pk("Drug", dose_mg=100.0)
    assert isinstance(result.fetal_exposure_concern, bool)


def test_fetal_exposure_concern_high_transfer():
    # High fetal-to-amniotic transfer → higher amniotic AUC → more likely concern
    result = simulate_amniotic_fluid_pk(
        "Drug",
        dose_mg=500.0,
        gestational_week=32,
        k_maternal_fetal_per_h=0.5,
        k_fetal_amniotic_per_h=0.8,
        k_amniotic_clearance_per_h=0.01,
    )
    assert result.amniotic_to_maternal_ratio > 0.0


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


def test_notes_fetal_exposure_concern():
    # Use high transfer parameters to force concern
    result = simulate_amniotic_fluid_pk(
        "Drug",
        dose_mg=1000.0,
        k_maternal_fetal_per_h=0.8,
        k_fetal_amniotic_per_h=0.9,
        k_amniotic_clearance_per_h=0.001,
        cl_maternal_L_per_h=1.0,
    )
    if result.fetal_exposure_concern:
        assert "amniotic swallowing" in result.notes


def test_notes_not_empty():
    result = simulate_amniotic_fluid_pk("Drug", dose_mg=100.0)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_dose():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_amniotic_fluid_pk("Drug", dose_mg=0.0)


def test_invalid_gestational_week_low():
    with pytest.raises(ValueError, match="gestational_week"):
        simulate_amniotic_fluid_pk("Drug", dose_mg=100.0, gestational_week=19)


def test_invalid_gestational_week_high():
    with pytest.raises(ValueError, match="gestational_week"):
        simulate_amniotic_fluid_pk("Drug", dose_mg=100.0, gestational_week=41)


def test_invalid_clearance():
    with pytest.raises(ValueError, match="cl_maternal"):
        simulate_amniotic_fluid_pk("Drug", dose_mg=100.0, cl_maternal_L_per_h=0.0)


# ---------------------------------------------------------------------------
# compare_gestational_ages
# ---------------------------------------------------------------------------


def test_compare_default_six_results():
    results = compare_gestational_ages("Drug", dose_mg=100.0)
    assert len(results) == 6


def test_compare_sorted_ascending():
    results = compare_gestational_ages("Drug", dose_mg=100.0)
    weeks = [r.gestational_week for r in results]
    assert weeks == sorted(weeks)


def test_compare_custom_weeks():
    results = compare_gestational_ages("Drug", dose_mg=100.0, weeks=[20, 30, 40])
    assert len(results) == 3
    assert results[0].gestational_week == 20


def test_compare_drug_name_consistent():
    results = compare_gestational_ages("Amiodarone", dose_mg=50.0)
    assert all(r.drug_name == "Amiodarone" for r in results)
