"""Tests for Phase 987 — aminoglycoside PK model."""

import math
import pytest

from omega_pbpk.clinical.aminoglycoside_pk import (
    AminoglycosidePKResult,
    simulate_aminoglycoside_pk,
)


# ---------------------------------------------------------------------------
# Basic return type and structure
# ---------------------------------------------------------------------------


def test_returns_result_type():
    result = simulate_aminoglycoside_pk()
    assert isinstance(result, AminoglycosidePKResult)


def test_times_and_concentrations_same_length():
    result = simulate_aminoglycoside_pk()
    assert len(result.times_h) == len(result.c_central_mg_L)


def test_times_start_at_zero():
    result = simulate_aminoglycoside_pk()
    assert result.times_h[0] == 0.0


def test_cpeak_positive():
    result = simulate_aminoglycoside_pk()
    assert result.cpeak_mg_L > 0.0


def test_ctrough_non_negative():
    result = simulate_aminoglycoside_pk()
    assert result.ctrough_mg_L >= 0.0


def test_auc_positive():
    result = simulate_aminoglycoside_pk()
    assert result.auc_24h_mg_h_per_L > 0.0


def test_cmax_mic_ratio_positive():
    result = simulate_aminoglycoside_pk()
    assert result.cmax_mic_ratio > 0.0


def test_auc_mic_ratio_positive():
    result = simulate_aminoglycoside_pk()
    assert result.auc_mic_ratio > 0.0


def test_pk_pd_target_is_bool():
    result = simulate_aminoglycoside_pk()
    assert isinstance(result.pk_pd_target_achieved, bool)


def test_nephrotoxicity_risk_valid():
    result = simulate_aminoglycoside_pk()
    assert result.nephrotoxicity_risk in {"low", "moderate", "high"}


def test_dose_recommendation_valid():
    result = simulate_aminoglycoside_pk()
    assert result.dose_recommendation in {"increase", "maintain", "decrease"}


def test_drug_name_preserved():
    result = simulate_aminoglycoside_pk(drug_name="Tobramycin")
    assert result.drug_name == "Tobramycin"


def test_dose_preserved():
    result = simulate_aminoglycoside_pk(dose_mg=240.0)
    assert result.dose_mg == 240.0


# ---------------------------------------------------------------------------
# Physiological correctness
# ---------------------------------------------------------------------------


def test_reduced_gfr_higher_ctrough():
    """Lower GFR → slower clearance → higher trough."""
    normal = simulate_aminoglycoside_pk(gfr_mL_min=90.0, dose_mg=240.0)
    ckd = simulate_aminoglycoside_pk(gfr_mL_min=20.0, dose_mg=240.0)
    assert ckd.ctrough_mg_L > normal.ctrough_mg_L


def test_higher_dose_higher_cpeak():
    """Doubling dose should approximately double peak."""
    low_dose = simulate_aminoglycoside_pk(dose_mg=200.0)
    high_dose = simulate_aminoglycoside_pk(dose_mg=400.0)
    assert high_dose.cpeak_mg_L > low_dose.cpeak_mg_L


def test_higher_dose_higher_cmax_mic_ratio():
    low_dose = simulate_aminoglycoside_pk(dose_mg=200.0, mic_mg_L=1.0)
    high_dose = simulate_aminoglycoside_pk(dose_mg=400.0, mic_mg_L=1.0)
    assert high_dose.cmax_mic_ratio > low_dose.cmax_mic_ratio


def test_standard_once_daily_achieves_cmax_mic_target():
    """5 mg/kg once-daily (350 mg for 70 kg) should achieve Cmax/MIC >= 8 for MIC=1."""
    result = simulate_aminoglycoside_pk(
        dose_mg=350.0,
        patient_weight_kg=70.0,
        gfr_mL_min=90.0,
        mic_mg_L=1.0,
    )
    assert result.cmax_mic_ratio >= 8.0


def test_nephrotoxicity_high_for_elevated_trough():
    """Very low GFR causes high trough → high nephrotoxicity risk."""
    result = simulate_aminoglycoside_pk(
        dose_mg=480.0,
        gfr_mL_min=10.0,
    )
    assert result.nephrotoxicity_risk == "high"


def test_low_dose_recommendation_increase():
    """Sub-therapeutic dose should trigger 'increase' recommendation."""
    result = simulate_aminoglycoside_pk(dose_mg=50.0, mic_mg_L=1.0)
    assert result.dose_recommendation == "increase"


def test_high_trough_recommendation_decrease():
    """Very low GFR + high dose → high trough → 'decrease' recommendation."""
    result = simulate_aminoglycoside_pk(dose_mg=480.0, gfr_mL_min=10.0)
    assert result.dose_recommendation == "decrease"


def test_notes_is_string():
    result = simulate_aminoglycoside_pk()
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_invalid_dose_raises():
    with pytest.raises(ValueError):
        simulate_aminoglycoside_pk(dose_mg=-10.0)


def test_zero_dose_raises():
    with pytest.raises(ValueError):
        simulate_aminoglycoside_pk(dose_mg=0.0)


def test_invalid_gfr_raises():
    with pytest.raises(ValueError):
        simulate_aminoglycoside_pk(gfr_mL_min=0.0)


def test_invalid_weight_raises():
    with pytest.raises(ValueError):
        simulate_aminoglycoside_pk(patient_weight_kg=-5.0)


def test_cmax_mic_equals_cpeak_over_mic():
    mic = 2.0
    result = simulate_aminoglycoside_pk(mic_mg_L=mic)
    assert math.isclose(result.cmax_mic_ratio, result.cpeak_mg_L / mic, rel_tol=1e-9)
