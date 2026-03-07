"""Tests for Phase 963 — Vancomycin PK model."""
import pytest
from omega_pbpk.clinical.vancomycin_pk import (
    VancomycinPKResult,
    simulate_vancomycin_pk,
    recommend_vancomycin_dose,
)


# ── Basic return-type and value tests ───────────────────────────────────────

def test_return_type():
    result = simulate_vancomycin_pk(dose_mg=1500.0)
    assert isinstance(result, VancomycinPKResult)


def test_cmax_positive():
    result = simulate_vancomycin_pk(dose_mg=1500.0)
    assert result.cmax_mg_L > 0


def test_cmin_nonnegative():
    result = simulate_vancomycin_pk(dose_mg=1500.0)
    assert result.cmin_mg_L >= 0


def test_auc24_positive():
    result = simulate_vancomycin_pk(dose_mg=1500.0)
    assert result.auc24_mg_h_per_L > 0


def test_auc_mic_ratio_positive():
    result = simulate_vancomycin_pk(dose_mg=1500.0)
    assert result.auc_mic_ratio > 0


def test_target_attained_is_bool():
    result = simulate_vancomycin_pk(dose_mg=1500.0)
    assert isinstance(result.target_attained, bool)


def test_recommended_dose_positive():
    result = simulate_vancomycin_pk(dose_mg=1500.0)
    assert result.recommended_dose_mg > 0


def test_crcl_positive():
    result = simulate_vancomycin_pk(dose_mg=1500.0)
    assert result.crcl_ml_min > 0


def test_cl_positive():
    result = simulate_vancomycin_pk(dose_mg=1500.0)
    assert result.cl_L_per_h > 0


def test_vd_central_positive():
    result = simulate_vancomycin_pk(dose_mg=1500.0)
    assert result.vd_central_L > 0


def test_times_nonempty():
    result = simulate_vancomycin_pk(dose_mg=1500.0)
    assert len(result.times_h) > 0


def test_notes_nonempty():
    result = simulate_vancomycin_pk(dose_mg=1500.0)
    assert result.notes != ""


# ── Pharmacological relationships ───────────────────────────────────────────

def test_higher_scr_lower_crcl():
    r1 = simulate_vancomycin_pk(dose_mg=1500.0, scr_mg_dL=1.0)
    r2 = simulate_vancomycin_pk(dose_mg=1500.0, scr_mg_dL=3.0)
    assert r2.crcl_ml_min < r1.crcl_ml_min


def test_higher_scr_lower_cl():
    r1 = simulate_vancomycin_pk(dose_mg=1500.0, scr_mg_dL=1.0)
    r2 = simulate_vancomycin_pk(dose_mg=1500.0, scr_mg_dL=3.0)
    assert r2.cl_L_per_h < r1.cl_L_per_h


def test_higher_scr_higher_auc():
    r1 = simulate_vancomycin_pk(dose_mg=1500.0, scr_mg_dL=1.0)
    r2 = simulate_vancomycin_pk(dose_mg=1500.0, scr_mg_dL=3.0)
    assert r2.auc24_mg_h_per_L > r1.auc24_mg_h_per_L


def test_female_lower_crcl_than_male():
    r_male = simulate_vancomycin_pk(dose_mg=1500.0, sex="male")
    r_female = simulate_vancomycin_pk(dose_mg=1500.0, sex="female")
    assert r_female.crcl_ml_min < r_male.crcl_ml_min


# ── recommend_vancomycin_dose ───────────────────────────────────────────────

def test_recommend_returns_dict():
    result = recommend_vancomycin_dose(weight_kg=70.0, age_years=50.0, sex="male", scr_mg_dL=1.0)
    assert isinstance(result, dict)


def test_recommend_has_required_keys():
    result = recommend_vancomycin_dose(weight_kg=70.0, age_years=50.0, sex="male", scr_mg_dL=1.0)
    for key in ("recommended_dose_mg", "dosing_interval_h", "predicted_auc24", "target_attained"):
        assert key in result, f"Missing key: {key}"


def test_recommend_dose_positive():
    result = recommend_vancomycin_dose(weight_kg=70.0, age_years=50.0, sex="male", scr_mg_dL=1.0)
    assert result["recommended_dose_mg"] > 0


def test_recommend_predicted_auc24_positive():
    result = recommend_vancomycin_dose(weight_kg=70.0, age_years=50.0, sex="male", scr_mg_dL=1.0)
    assert result["predicted_auc24"] > 0


def test_recommend_target_attained_is_bool():
    result = recommend_vancomycin_dose(weight_kg=70.0, age_years=50.0, sex="male", scr_mg_dL=1.0)
    assert isinstance(result["target_attained"], bool)


# ── Validation ───────────────────────────────────────────────────────────────

def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_vancomycin_pk(dose_mg=0.0)


def test_validation_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_vancomycin_pk(dose_mg=-100.0)


def test_validation_weight_zero():
    with pytest.raises(ValueError, match="weight_kg"):
        simulate_vancomycin_pk(dose_mg=1500.0, weight_kg=0.0)


def test_validation_age_zero():
    with pytest.raises(ValueError, match="age_years"):
        simulate_vancomycin_pk(dose_mg=1500.0, age_years=0.0)


def test_validation_scr_zero():
    with pytest.raises(ValueError, match="scr_mg_dL"):
        simulate_vancomycin_pk(dose_mg=1500.0, scr_mg_dL=0.0)


def test_validation_invalid_sex():
    with pytest.raises(ValueError, match="sex"):
        simulate_vancomycin_pk(dose_mg=1500.0, sex="other")


def test_validation_n_doses_zero():
    with pytest.raises(ValueError, match="n_doses"):
        simulate_vancomycin_pk(dose_mg=1500.0, n_doses=0)


def test_n_doses_one_works():
    result = simulate_vancomycin_pk(dose_mg=1500.0, n_doses=1, t_end_h=12.0)
    assert isinstance(result, VancomycinPKResult)
    assert result.cmax_mg_L > 0
