"""Tests for Phase 617 — critical_care_pk."""

import pytest

from omega_pbpk.clinical.critical_care_pk import (
    CriticalCarePKResult,
    adjust_for_critical_illness,
    sepsis_pk_adjustment,
)

# Default values used across tests
DRUG = "vancomycin"
CL = 5.0  # L/h
VD = 50.0  # L
DOSE = 1000.0  # mg
INTERVAL = 12.0  # h


def make_result(**kwargs):
    defaults = dict(
        drug_name=DRUG,
        baseline_cl_L_per_h=CL,
        baseline_vd_L=VD,
        baseline_dose_mg=DOSE,
        baseline_interval_h=INTERVAL,
        condition="sepsis",
        patient_weight_kg=70.0,
        gfr_mL_per_min=90.0,
        albumin_g_dL=4.0,
        fe_renal=0.5,
    )
    defaults.update(kwargs)
    return adjust_for_critical_illness(**defaults)


def test_returns_result():
    r = make_result()
    assert isinstance(r, CriticalCarePKResult)


def test_sepsis_higher_vd():
    r = make_result(condition="sepsis")
    assert r.adjusted_vd_L > VD


def test_hepatic_failure_lower_cl():
    r = make_result(condition="hepatic_failure")
    assert r.adjusted_cl_L_per_h < CL


def test_ards_lower_cl():
    r = make_result(condition="ards")
    assert r.adjusted_cl_L_per_h <= CL


def test_icu_general_moderate_changes():
    r = make_result(condition="icu_general", albumin_g_dL=4.0, gfr_mL_per_min=90.0)
    # cl_factor should be close to 1 (0.9 for icu_general without albumin/ARC adjustment)
    assert 0.7 < r.cl_scaling_factor < 1.2


def test_arc_detected():
    r = make_result(gfr_mL_per_min=150)
    assert r.augmented_renal_clearance is True


def test_no_arc_normal_gfr():
    r = make_result(gfr_mL_per_min=90)
    assert r.augmented_renal_clearance is False


def test_loading_dose_positive():
    r = make_result()
    assert r.loading_dose_mg > 0


def test_recommended_dose_positive():
    r = make_result()
    assert r.recommended_dose_mg > 0


def test_adjusted_t_half_positive():
    r = make_result()
    assert r.adjusted_t_half_h > 0


def test_sepsis_pk_adjustment_returns_dict():
    result = sepsis_pk_adjustment(CL, VD)
    assert isinstance(result, dict)


def test_sepsis_pk_adjustment_keys():
    result = sepsis_pk_adjustment(CL, VD)
    for key in ("cl_factor", "vd_factor", "arc", "albumin_effect"):
        assert key in result


def test_low_albumin_increases_cl():
    r_low = make_result(albumin_g_dL=2.0)
    r_normal = make_result(albumin_g_dL=4.0)
    assert r_low.cl_scaling_factor > r_normal.cl_scaling_factor


def test_recommended_dose_lower_for_hepatic():
    r_hepatic = make_result(condition="hepatic_failure", albumin_g_dL=4.0)
    r_icu = make_result(condition="icu_general", albumin_g_dL=4.0)
    assert r_hepatic.recommended_dose_mg < r_icu.recommended_dose_mg


def test_loading_dose_higher_for_sepsis():
    r = make_result(condition="sepsis")
    # vd_factor >= 1.4 for sepsis → loading > baseline
    assert r.loading_dose_mg > DOSE


def test_monitoring_recommendation_nonempty():
    r = make_result()
    assert len(r.monitoring_recommendation) > 0


def test_invalid_condition_raises():
    with pytest.raises(ValueError):
        make_result(condition="unknown_condition")


def test_invalid_weight_raises():
    with pytest.raises(ValueError):
        make_result(patient_weight_kg=-10)


def test_invalid_gfr_negative_raises():
    with pytest.raises(ValueError):
        make_result(gfr_mL_per_min=-5)


def test_invalid_albumin_raises():
    with pytest.raises(ValueError):
        make_result(albumin_g_dL=0)


def test_invalid_fe_renal_raises():
    with pytest.raises(ValueError):
        make_result(fe_renal=1.5)


def test_notes_nonempty():
    r = make_result()
    assert len(r.notes) > 0


def test_condition_stored_in_result():
    r = make_result(condition="ards")
    assert r.condition == "ards"
