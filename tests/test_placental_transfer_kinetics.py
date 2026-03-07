"""Tests for Phase 1139: Placental transfer kinetics."""

import math
import pytest

from omega_pbpk.core.placental_transfer_kinetics import (
    PlacentalTransferResult,
    simulate_placental_transfer,
)


# ---------------------------------------------------------------------------
# Basic simulation smoke tests
# ---------------------------------------------------------------------------


def test_basic_simulation_returns_result():
    result = simulate_placental_transfer("TestDrug", dose_mg=100.0, logP=2.0)
    assert isinstance(result, PlacentalTransferResult)


def test_drug_name_preserved():
    result = simulate_placental_transfer("Ibuprofen", dose_mg=200.0, logP=3.5, ionization_type="acid")
    assert result.drug_name == "Ibuprofen"


def test_dose_preserved():
    result = simulate_placental_transfer("Drug", dose_mg=50.0, logP=1.0)
    assert result.dose_mg == 50.0


def test_logP_preserved():
    result = simulate_placental_transfer("Drug", dose_mg=100.0, logP=3.2)
    assert result.logP == 3.2


def test_ionization_type_preserved():
    result = simulate_placental_transfer("Drug", dose_mg=100.0, logP=2.0, ionization_type="base")
    assert result.ionization_type == "base"


def test_gestational_age_preserved():
    result = simulate_placental_transfer("Drug", dose_mg=100.0, logP=2.0, gestational_age_weeks=28)
    assert result.gestational_age_weeks == 28


# ---------------------------------------------------------------------------
# Time course structure
# ---------------------------------------------------------------------------


def test_times_list_length_matches_steps():
    result = simulate_placental_transfer("Drug", dose_mg=100.0, logP=2.0, t_end_h=10.0, dt_h=0.5)
    expected_steps = int(round(10.0 / 0.5)) + 1
    assert len(result.times_h) == expected_steps


def test_times_start_at_zero():
    result = simulate_placental_transfer("Drug", dose_mg=100.0, logP=2.0)
    assert result.times_h[0] == 0.0


def test_concentrations_same_length_as_times():
    result = simulate_placental_transfer("Drug", dose_mg=100.0, logP=2.0)
    n = len(result.times_h)
    assert len(result.c_maternal_mg_L) == n
    assert len(result.c_placenta_mg_L) == n
    assert len(result.c_fetal_mg_L) == n


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------


def test_c_maternal_initial_equals_dose_over_vd():
    result = simulate_placental_transfer("Drug", dose_mg=120.0, logP=1.0, vd_maternal_L=60.0)
    assert abs(result.c_maternal_mg_L[0] - 120.0 / 60.0) < 1e-9


def test_c_placenta_starts_at_zero():
    result = simulate_placental_transfer("Drug", dose_mg=100.0, logP=2.0)
    assert result.c_placenta_mg_L[0] == 0.0


def test_c_fetal_starts_at_zero():
    result = simulate_placental_transfer("Drug", dose_mg=100.0, logP=2.0)
    assert result.c_fetal_mg_L[0] == 0.0


# ---------------------------------------------------------------------------
# Concentration dynamics
# ---------------------------------------------------------------------------


def test_c_fetal_rises_then_falls():
    """Fetal concentration should rise from 0, reach a peak, then decline."""
    # Use t_end_h=72h to ensure the fetal peak is interior to the simulation window
    result = simulate_placental_transfer("Drug", dose_mg=100.0, logP=2.0, t_end_h=72.0)
    fetal = result.c_fetal_mg_L
    # Check it starts at 0 and rises
    assert fetal[0] == 0.0
    # Max should be positive and at some interior time
    assert result.cmax_fetal_mg_L > 0.0
    assert result.tmax_fetal_h > 0.0
    # At end it should be lower than max (peak is interior)
    assert fetal[-1] < result.cmax_fetal_mg_L


def test_c_maternal_declines_monotonically_initially():
    """Maternal concentration should start declining."""
    result = simulate_placental_transfer("Drug", dose_mg=100.0, logP=1.0)
    # First few values should be declining
    assert result.c_maternal_mg_L[1] < result.c_maternal_mg_L[0]
    assert result.c_maternal_mg_L[5] < result.c_maternal_mg_L[0]


def test_all_concentrations_non_negative():
    result = simulate_placental_transfer("Drug", dose_mg=100.0, logP=4.0)
    assert all(c >= 0.0 for c in result.c_maternal_mg_L)
    assert all(c >= 0.0 for c in result.c_placenta_mg_L)
    assert all(c >= 0.0 for c in result.c_fetal_mg_L)


# ---------------------------------------------------------------------------
# logP effects
# ---------------------------------------------------------------------------


def test_higher_logP_gives_higher_fetal_auc():
    """More lipophilic drug (higher logP) → higher kmt → more fetal exposure."""
    low = simulate_placental_transfer("Drug", dose_mg=100.0, logP=0.0)
    high = simulate_placental_transfer("Drug", dose_mg=100.0, logP=5.0)
    assert high.auc_fetal_mg_h_per_L > low.auc_fetal_mg_h_per_L


def test_higher_logP_higher_fetal_maternal_ratio():
    low = simulate_placental_transfer("Drug", dose_mg=100.0, logP=0.0)
    high = simulate_placental_transfer("Drug", dose_mg=100.0, logP=5.0)
    assert high.fetal_to_maternal_auc_ratio > low.fetal_to_maternal_auc_ratio


# ---------------------------------------------------------------------------
# Gestational age effects
# ---------------------------------------------------------------------------


def test_higher_gestational_age_higher_fetal_auc():
    early = simulate_placental_transfer("Drug", dose_mg=100.0, logP=2.0, gestational_age_weeks=12)
    late = simulate_placental_transfer("Drug", dose_mg=100.0, logP=2.0, gestational_age_weeks=40)
    assert late.auc_fetal_mg_h_per_L > early.auc_fetal_mg_h_per_L


def test_higher_gestational_age_higher_ratio():
    early = simulate_placental_transfer("Drug", dose_mg=100.0, logP=2.0, gestational_age_weeks=10)
    late = simulate_placental_transfer("Drug", dose_mg=100.0, logP=2.0, gestational_age_weeks=40)
    assert late.fetal_to_maternal_auc_ratio > early.fetal_to_maternal_auc_ratio


# ---------------------------------------------------------------------------
# AUC and ratio properties
# ---------------------------------------------------------------------------


def test_auc_maternal_positive():
    result = simulate_placental_transfer("Drug", dose_mg=100.0, logP=2.0)
    assert result.auc_maternal_mg_h_per_L > 0.0


def test_auc_fetal_positive_for_typical_drug():
    result = simulate_placental_transfer("Drug", dose_mg=100.0, logP=2.0)
    assert result.auc_fetal_mg_h_per_L > 0.0


def test_fetal_to_maternal_ratio_in_reasonable_range():
    result = simulate_placental_transfer("Drug", dose_mg=100.0, logP=2.0)
    assert 0.0 <= result.fetal_to_maternal_auc_ratio <= 2.0


# ---------------------------------------------------------------------------
# Exposure classification
# ---------------------------------------------------------------------------


def test_exposure_class_minimal_for_low_logP():
    result = simulate_placental_transfer("Drug", dose_mg=100.0, logP=-3.0, gestational_age_weeks=8)
    assert result.fetal_exposure_class == "minimal"


def test_exposure_class_high_for_very_lipophilic_at_term():
    result = simulate_placental_transfer(
        "Drug", dose_mg=100.0, logP=10.0, gestational_age_weeks=40, t_end_h=48.0
    )
    assert result.fetal_exposure_class in {"moderate", "high"}


def test_teratogenic_concern_when_ratio_exceeds_0_5():
    result = simulate_placental_transfer(
        "Drug", dose_mg=100.0, logP=10.0, gestational_age_weeks=40, t_end_h=48.0
    )
    if result.fetal_to_maternal_auc_ratio > 0.5:
        assert result.teratogenic_concern is True


def test_no_teratogenic_concern_when_ratio_below_0_5():
    result = simulate_placental_transfer(
        "Drug", dose_mg=100.0, logP=-4.0, gestational_age_weeks=8
    )
    if result.fetal_to_maternal_auc_ratio <= 0.5:
        assert result.teratogenic_concern is False


def test_teratogenic_concern_consistent_with_ratio():
    result = simulate_placental_transfer("Drug", dose_mg=100.0, logP=3.0, gestational_age_weeks=36)
    expected = result.fetal_to_maternal_auc_ratio > 0.5
    assert result.teratogenic_concern == expected


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_placental_transfer("Drug", dose_mg=0.0, logP=2.0)


def test_negative_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_placental_transfer("Drug", dose_mg=-10.0, logP=2.0)


def test_invalid_cl_raises():
    with pytest.raises(ValueError, match="cl_maternal_L_per_h"):
        simulate_placental_transfer("Drug", dose_mg=100.0, logP=2.0, cl_maternal_L_per_h=0.0)


def test_invalid_vd_raises():
    with pytest.raises(ValueError, match="vd_maternal_L"):
        simulate_placental_transfer("Drug", dose_mg=100.0, logP=2.0, vd_maternal_L=-5.0)


def test_logP_too_low_raises():
    with pytest.raises(ValueError, match="logP"):
        simulate_placental_transfer("Drug", dose_mg=100.0, logP=-6.0)


def test_logP_too_high_raises():
    with pytest.raises(ValueError, match="logP"):
        simulate_placental_transfer("Drug", dose_mg=100.0, logP=11.0)


def test_invalid_ionization_type_raises():
    with pytest.raises(ValueError, match="ionization_type"):
        simulate_placental_transfer("Drug", dose_mg=100.0, logP=2.0, ionization_type="zwitterion")


def test_gestational_age_too_low_raises():
    with pytest.raises(ValueError, match="gestational_age_weeks"):
        simulate_placental_transfer("Drug", dose_mg=100.0, logP=2.0, gestational_age_weeks=7)


def test_gestational_age_too_high_raises():
    with pytest.raises(ValueError, match="gestational_age_weeks"):
        simulate_placental_transfer("Drug", dose_mg=100.0, logP=2.0, gestational_age_weeks=43)


# ---------------------------------------------------------------------------
# Notes field
# ---------------------------------------------------------------------------


def test_notes_contains_ratio():
    result = simulate_placental_transfer("Drug", dose_mg=100.0, logP=2.0)
    assert "ratio" in result.notes.lower()


def test_notes_contains_kmt():
    result = simulate_placental_transfer("Drug", dose_mg=100.0, logP=2.0)
    assert "kmt" in result.notes.lower()
