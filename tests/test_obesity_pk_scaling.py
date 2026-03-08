"""Tests for Phase 180 — Obesity PK scaling."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.obesity_pk_scaling import (
    ObesityPKResult,
    calculate_lean_body_weight,
    scale_pk_for_obesity,
)

_BASE = dict(
    drug_name="TestDrug",
    sex="male",
    weight_kg=120.0,
    height_cm=175.0,
    logP=2.5,
    cl_hepatic_L_per_h=10.0,
    vd_normal_L=50.0,
    fu_plasma=0.2,
)


def test_returns_correct_type():
    result = scale_pk_for_obesity(**_BASE)
    assert isinstance(result, ObesityPKResult)


def test_drug_name_preserved():
    result = scale_pk_for_obesity(**_BASE)
    assert result.drug_name == "TestDrug"


def test_sex_preserved_lowercase():
    result = scale_pk_for_obesity(**_BASE)
    assert result.sex == "male"


def test_weight_preserved():
    result = scale_pk_for_obesity(**_BASE)
    assert result.actual_weight_kg == pytest.approx(120.0)


def test_bmi_computed():
    result = scale_pk_for_obesity(**_BASE)
    expected_bmi = 120.0 / (1.75**2)
    assert result.bmi == pytest.approx(expected_bmi, rel=1e-3)


def test_obesity_class_obese():
    result = scale_pk_for_obesity(**_BASE)
    assert result.obesity_class == "obese"


def test_normal_bmi_class():
    result = scale_pk_for_obesity(**{**_BASE, "weight_kg": 70.0, "height_cm": 175.0})
    assert result.obesity_class == "normal"


def test_morbidly_obese_class():
    result = scale_pk_for_obesity(**{**_BASE, "weight_kg": 180.0})
    assert result.obesity_class == "morbidly_obese"


def test_lbw_less_than_actual_weight():
    result = scale_pk_for_obesity(**_BASE)
    assert result.lean_body_weight_kg < result.actual_weight_kg


def test_fat_mass_positive():
    result = scale_pk_for_obesity(**_BASE)
    assert result.fat_mass_kg > 0.0


def test_fat_mass_plus_lbw_equals_actual():
    result = scale_pk_for_obesity(**_BASE)
    assert result.fat_mass_kg + result.lean_body_weight_kg == pytest.approx(
        result.actual_weight_kg, rel=1e-3
    )


def test_vd_obese_positive():
    result = scale_pk_for_obesity(**_BASE)
    assert result.vd_obese_L > 0.0


def test_lipophilic_increases_vd():
    r_lip = scale_pk_for_obesity(**{**_BASE, "logP": 4.0})
    r_hyd = scale_pk_for_obesity(**{**_BASE, "logP": -1.0})
    assert r_lip.vd_obese_L > r_hyd.vd_obese_L


def test_t_half_normal_positive():
    result = scale_pk_for_obesity(**_BASE)
    assert result.t_half_normal_h > 0.0


def test_t_half_obese_positive():
    result = scale_pk_for_obesity(**_BASE)
    assert result.t_half_obese_h > 0.0


def test_lipophilic_recommendation_adjbw():
    result = scale_pk_for_obesity(**{**_BASE, "logP": 3.0})
    assert result.dose_recommendation == "use_adjbw"


def test_hydrophilic_recommendation_lbw():
    result = scale_pk_for_obesity(**{**_BASE, "logP": -1.0})
    assert result.dose_recommendation == "use_lbw"


def test_dosing_weight_lbw_method():
    result = scale_pk_for_obesity(**{**_BASE, "dosing_weight_method": "lean_body_weight"})
    assert result.dosing_weight_kg == pytest.approx(result.lean_body_weight_kg)


def test_dosing_weight_abw_method():
    result = scale_pk_for_obesity(**{**_BASE, "dosing_weight_method": "actual_body_weight"})
    assert result.dosing_weight_kg == pytest.approx(result.actual_weight_kg)


def test_notes_nonempty():
    result = scale_pk_for_obesity(**_BASE)
    assert len(result.notes) > 0


def test_calculate_lbw_male():
    lbw = calculate_lean_body_weight("male", 80.0, 180.0)
    assert lbw > 0.0
    assert lbw < 80.0


def test_calculate_lbw_female():
    lbw = calculate_lean_body_weight("female", 70.0, 165.0)
    assert lbw > 0.0
    assert lbw < 70.0


def test_calculate_lbw_clamped_min():
    # Very small person: LBW cannot be below 10% of actual weight
    lbw = calculate_lean_body_weight("male", 150.0, 100.0)
    assert lbw >= 0.1 * 150.0


# Validation errors
def test_zero_weight_raises():
    with pytest.raises(ValueError, match="weight_kg"):
        scale_pk_for_obesity(**{**_BASE, "weight_kg": 0.0})


def test_invalid_sex_raises():
    with pytest.raises(ValueError, match="sex"):
        scale_pk_for_obesity(**{**_BASE, "sex": "other"})


def test_zero_cl_raises():
    with pytest.raises(ValueError, match="cl_hepatic"):
        scale_pk_for_obesity(**{**_BASE, "cl_hepatic_L_per_h": 0.0})


def test_invalid_logp_raises():
    with pytest.raises(ValueError, match="logP"):
        scale_pk_for_obesity(**{**_BASE, "logP": 15.0})


def test_invalid_dosing_method_raises():
    with pytest.raises(ValueError, match="dosing_weight_method"):
        scale_pk_for_obesity(**{**_BASE, "dosing_weight_method": "ideal_body_weight"})
