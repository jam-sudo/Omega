"""Tests for Phase 1005 — Tacrolimus PK Model."""

import pytest

from omega_pbpk.clinical.tacrolimus_pk import (
    TacrolimusPKResult,
    simulate_tacrolimus_pk,
)

# --- Basic return type ---


def test_returns_tacrolimus_pk_result():
    result = simulate_tacrolimus_pk()
    assert isinstance(result, TacrolimusPKResult)


def test_drug_name():
    result = simulate_tacrolimus_pk()
    assert result.drug_name == "Tacrolimus"


# --- Positive PK metrics ---


def test_cmax_positive():
    result = simulate_tacrolimus_pk(dose_mg=5.0)
    assert result.cmax_ng_mL > 0.0


def test_auc_positive():
    result = simulate_tacrolimus_pk(dose_mg=5.0)
    assert result.auc_0_12h_ng_h_per_mL > 0.0


def test_c0_trough_non_negative():
    result = simulate_tacrolimus_pk(dose_mg=5.0)
    assert result.c0_trough_ng_mL >= 0.0


def test_within_therapeutic_range_is_bool():
    result = simulate_tacrolimus_pk()
    assert isinstance(result.within_therapeutic_range, bool)


def test_dose_recommendation_valid():
    result = simulate_tacrolimus_pk()
    assert result.dose_recommendation in {"increase", "maintain", "decrease"}


def test_cyp3a5_phenotype_valid():
    result = simulate_tacrolimus_pk()
    assert result.cyp3a5_phenotype in {"poor_metabolizer", "extensive_metabolizer"}


# --- CYP3A5 phenotype assignment ---


def test_cyp3a5_poor_metabolizer_default():
    result = simulate_tacrolimus_pk(cyp3a5_star1=False)
    assert result.cyp3a5_phenotype == "poor_metabolizer"


def test_cyp3a5_extensive_metabolizer():
    result = simulate_tacrolimus_pk(cyp3a5_star1=True)
    assert result.cyp3a5_phenotype == "extensive_metabolizer"


# --- CYP3A5 effect on C0 ---


def test_extensive_metabolizer_lower_c0_than_poor():
    """Extensive metabolizer clears faster → lower trough."""
    pm = simulate_tacrolimus_pk(cyp3a5_star1=False)
    em = simulate_tacrolimus_pk(cyp3a5_star1=True)
    assert em.c0_trough_ng_mL < pm.c0_trough_ng_mL


# --- Time-course checks ---


def test_initial_concentration_near_zero():
    result = simulate_tacrolimus_pk()
    assert result.c_whole_blood_ng_mL[0] == pytest.approx(0.0, abs=1e-6)


def test_tmax_positive():
    result = simulate_tacrolimus_pk()
    assert result.tmax_h > 0.0


def test_tmax_within_dosing_interval():
    result = simulate_tacrolimus_pk()
    assert result.tmax_h < 12.0


# --- Hematocrit effect ---


def test_higher_hematocrit_higher_whole_blood_conc():
    """Higher hematocrit → higher blood:plasma ratio → higher C_blood."""
    low_hct = simulate_tacrolimus_pk(hematocrit=0.30)
    high_hct = simulate_tacrolimus_pk(hematocrit=0.55)
    assert high_hct.cmax_ng_mL > low_hct.cmax_ng_mL


# --- Dose dependence ---


def test_higher_dose_higher_cmax():
    low_dose = simulate_tacrolimus_pk(dose_mg=2.0)
    high_dose = simulate_tacrolimus_pk(dose_mg=10.0)
    assert high_dose.cmax_ng_mL > low_dose.cmax_ng_mL


# --- Validation errors ---


def test_invalid_dose_raises():
    with pytest.raises(ValueError):
        simulate_tacrolimus_pk(dose_mg=0.0)


def test_negative_dose_raises():
    with pytest.raises(ValueError):
        simulate_tacrolimus_pk(dose_mg=-1.0)


def test_invalid_weight_raises():
    with pytest.raises(ValueError):
        simulate_tacrolimus_pk(weight_kg=0.0)


def test_invalid_hematocrit_zero_raises():
    with pytest.raises(ValueError):
        simulate_tacrolimus_pk(hematocrit=0.0)


def test_invalid_hematocrit_one_raises():
    with pytest.raises(ValueError):
        simulate_tacrolimus_pk(hematocrit=1.0)
