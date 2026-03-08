"""Tests for Phase 993: Fentanyl PK model."""

import pytest

from omega_pbpk.clinical.fentanyl_pk import FentanylPKResult, simulate_fentanyl_pk

# ---------------------------------------------------------------------------
# Basic return type and field presence
# ---------------------------------------------------------------------------


def test_returns_fentanyl_pk_result():
    result = simulate_fentanyl_pk()
    assert isinstance(result, FentanylPKResult)


def test_drug_name_is_fentanyl():
    result = simulate_fentanyl_pk()
    assert result.drug_name == "Fentanyl"


def test_dose_stored():
    result = simulate_fentanyl_pk(dose_mcg=50.0)
    assert result.dose_mcg == pytest.approx(50.0)


def test_route_stored():
    result = simulate_fentanyl_pk(route="iv_bolus")
    assert result.route == "iv_bolus"


# ---------------------------------------------------------------------------
# Positive metrics
# ---------------------------------------------------------------------------


def test_cmax_central_positive():
    result = simulate_fentanyl_pk()
    assert result.cmax_central_ng_mL > 0


def test_cmax_effect_positive():
    result = simulate_fentanyl_pk()
    assert result.cmax_effect_ng_mL > 0


def test_tmax_effect_positive():
    result = simulate_fentanyl_pk()
    assert result.tmax_effect_min > 0


def test_auc_central_positive():
    result = simulate_fentanyl_pk()
    assert result.auc_central_ng_min_per_mL > 0


def test_time_above_analgesia_nonnegative():
    result = simulate_fentanyl_pk()
    assert result.time_above_analgesia_min >= 0


def test_time_above_apnea_nonnegative():
    result = simulate_fentanyl_pk()
    assert result.time_above_apnea_min >= 0


def test_recovery_time_positive():
    result = simulate_fentanyl_pk()
    assert result.recovery_time_min > 0


# ---------------------------------------------------------------------------
# Standard dose sanity check
# ---------------------------------------------------------------------------


def test_standard_dose_cmax_central_above_2_ng_mL():
    """100 mcg IV bolus in 70 kg patient: Cmax central > 2 ng/mL."""
    result = simulate_fentanyl_pk(dose_mcg=100.0, weight_kg=70.0, route="iv_bolus")
    assert result.cmax_central_ng_mL > 2.0


# ---------------------------------------------------------------------------
# Dose-response: higher dose → longer analgesia
# ---------------------------------------------------------------------------


def test_higher_dose_longer_analgesia():
    low = simulate_fentanyl_pk(dose_mcg=50.0, t_end_min=120.0)
    high = simulate_fentanyl_pk(dose_mcg=200.0, t_end_min=120.0)
    assert high.time_above_analgesia_min > low.time_above_analgesia_min


# ---------------------------------------------------------------------------
# IV bolus vs IV infusion initial condition
# ---------------------------------------------------------------------------


def test_iv_bolus_c_central_t0_nonzero():
    """IV bolus: first concentration value should be >> 0."""
    result = simulate_fentanyl_pk(route="iv_bolus")
    assert result.c_central_ng_mL[0] > 0.0


def test_iv_infusion_c_central_t0_approx_zero():
    """IV infusion: concentration at t=0 should be 0 (infusion starts but first step before ODE update)."""
    result = simulate_fentanyl_pk(route="iv_infusion")
    assert result.c_central_ng_mL[0] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Weight effect on Cmax (allometric CL scaling)
# ---------------------------------------------------------------------------


def test_heavier_patient_lower_auc():
    """Heavier patient has higher allometric CL → lower central AUC (same dose)."""
    light = simulate_fentanyl_pk(dose_mcg=100.0, weight_kg=50.0, t_end_min=120.0)
    heavy = simulate_fentanyl_pk(dose_mcg=100.0, weight_kg=100.0, t_end_min=120.0)
    # Lighter patient should have higher AUC because CL is lower
    assert light.auc_central_ng_min_per_mL > heavy.auc_central_ng_min_per_mL


# ---------------------------------------------------------------------------
# List fields match times length
# ---------------------------------------------------------------------------


def test_output_list_lengths_match():
    result = simulate_fentanyl_pk(t_end_min=10.0, dt_min=0.1)
    n = len(result.times_min)
    assert len(result.c_central_ng_mL) == n
    assert len(result.c_effect_ng_mL) == n


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_invalid_dose_raises_value_error():
    with pytest.raises(ValueError, match="dose_mcg"):
        simulate_fentanyl_pk(dose_mcg=0.0)


def test_negative_dose_raises_value_error():
    with pytest.raises(ValueError, match="dose_mcg"):
        simulate_fentanyl_pk(dose_mcg=-10.0)


def test_invalid_route_raises_value_error():
    with pytest.raises(ValueError, match="route"):
        simulate_fentanyl_pk(route="oral")


def test_invalid_weight_raises_value_error():
    with pytest.raises(ValueError, match="weight_kg"):
        simulate_fentanyl_pk(weight_kg=0.0)


# ---------------------------------------------------------------------------
# Notes field is non-empty string
# ---------------------------------------------------------------------------


def test_notes_nonempty():
    result = simulate_fentanyl_pk()
    assert isinstance(result.notes, str) and len(result.notes) > 0
