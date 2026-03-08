"""Tests for cyclosporine PK model (Phase 979)."""

import math

import pytest

from omega_pbpk.clinical.cyclosporine_pk import (
    CyclosporinePKResult,
    simulate_cyclosporine_pk,
)

# ---------------------------------------------------------------------------
# Basic return type and fields
# ---------------------------------------------------------------------------


def test_returns_result_type():
    result = simulate_cyclosporine_pk(200.0)
    assert isinstance(result, CyclosporinePKResult)


def test_drug_name():
    result = simulate_cyclosporine_pk(200.0)
    assert result.drug_name == "Cyclosporine"


def test_dose_stored():
    result = simulate_cyclosporine_pk(150.0)
    assert result.dose_mg == 150.0


def test_weight_stored():
    result = simulate_cyclosporine_pk(200.0, patient_weight_kg=80.0)
    assert result.patient_weight_kg == 80.0


# ---------------------------------------------------------------------------
# Positive concentration checks
# ---------------------------------------------------------------------------


def test_cmax_positive():
    result = simulate_cyclosporine_pk(200.0)
    assert result.cmax_ng_mL > 0


def test_c2_positive():
    result = simulate_cyclosporine_pk(200.0)
    assert result.c2_ng_mL > 0


def test_auc_positive():
    result = simulate_cyclosporine_pk(200.0)
    assert result.auc_0_12h_ng_h_per_mL > 0


def test_tmax_in_range():
    result = simulate_cyclosporine_pk(200.0, t_end_h=12.0)
    assert 0 < result.tmax_h < 12.0


# ---------------------------------------------------------------------------
# Therapeutic range and recommendation
# ---------------------------------------------------------------------------


def test_within_therapeutic_range_is_bool():
    result = simulate_cyclosporine_pk(200.0)
    assert isinstance(result.within_therapeutic_range, bool)


def test_dose_recommendation_valid():
    result = simulate_cyclosporine_pk(200.0)
    assert result.dose_recommendation in {"decrease", "maintain", "increase"}


def test_high_dose_may_trigger_decrease():
    # Very high dose should push C0 above 400 ng/mL
    result = simulate_cyclosporine_pk(5000.0)
    assert result.dose_recommendation == "decrease"
    assert result.within_therapeutic_range is False


def test_low_dose_triggers_increase():
    # Very low dose — C0 should be < 100 ng/mL
    result = simulate_cyclosporine_pk(1.0)
    assert result.dose_recommendation == "increase"
    assert result.within_therapeutic_range is False


# ---------------------------------------------------------------------------
# Initial concentration is zero
# ---------------------------------------------------------------------------


def test_initial_blood_concentration_zero():
    result = simulate_cyclosporine_pk(200.0)
    assert math.isclose(result.c_whole_blood_ng_mL[0], 0.0, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# Weight effect: heavier patient → lower C0 per mg
# ---------------------------------------------------------------------------


def test_heavier_patient_lower_c0_per_mg():
    dose = 200.0
    r50 = simulate_cyclosporine_pk(dose, patient_weight_kg=50.0)
    r100 = simulate_cyclosporine_pk(dose, patient_weight_kg=100.0)
    # C0/dose should be lower for heavier patient (larger Vd and CL)
    assert r50.c0_trough_ng_mL / dose > r100.c0_trough_ng_mL / dose


# ---------------------------------------------------------------------------
# Creatinine effect: elevated creatinine → higher C0
# ---------------------------------------------------------------------------


def test_elevated_creatinine_higher_c0():
    r_normal = simulate_cyclosporine_pk(200.0, creatinine_umol_L=100.0)
    r_impaired = simulate_cyclosporine_pk(200.0, creatinine_umol_L=300.0)
    assert r_impaired.c0_trough_ng_mL > r_normal.c0_trough_ng_mL


# ---------------------------------------------------------------------------
# Hematocrit effect: high hematocrit → higher whole blood concentration
# ---------------------------------------------------------------------------


def test_high_hematocrit_higher_blood_conc():
    r_low = simulate_cyclosporine_pk(200.0, hematocrit=0.25)
    r_high = simulate_cyclosporine_pk(200.0, hematocrit=0.65)
    assert r_high.cmax_ng_mL > r_low.cmax_ng_mL


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_cyclosporine_pk(0.0)


def test_negative_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_cyclosporine_pk(-100.0)


def test_invalid_weight_raises():
    with pytest.raises(ValueError, match="patient_weight_kg"):
        simulate_cyclosporine_pk(200.0, patient_weight_kg=0.0)


def test_invalid_hematocrit_zero_raises():
    with pytest.raises(ValueError, match="hematocrit"):
        simulate_cyclosporine_pk(200.0, hematocrit=0.0)


def test_invalid_hematocrit_one_raises():
    with pytest.raises(ValueError, match="hematocrit"):
        simulate_cyclosporine_pk(200.0, hematocrit=1.0)


# ---------------------------------------------------------------------------
# Times list
# ---------------------------------------------------------------------------


def test_times_start_at_zero():
    result = simulate_cyclosporine_pk(200.0)
    assert result.times_h[0] == 0.0


def test_times_length_matches_concentrations():
    result = simulate_cyclosporine_pk(200.0)
    assert len(result.times_h) == len(result.c_whole_blood_ng_mL)
