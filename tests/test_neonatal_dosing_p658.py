"""Tests for neonatal drug dosing prediction (Phase 658)."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.clinical.neonatal_dosing_p658 import (
    NeonatalDosingResult,
    predict_neonatal_dosing,
)

# --- Frozen dataclass tests ---


def test_result_is_frozen():
    r = predict_neonatal_dosing("DrugA", 10.0, 50.0, 5.0, age_days=7, weight_kg=3.5)
    with pytest.raises(AttributeError):
        r.drug_name = "other"


def test_result_type():
    r = predict_neonatal_dosing("DrugA", 10.0, 50.0, 5.0, age_days=7, weight_kg=3.5)
    assert isinstance(r, NeonatalDosingResult)


# --- Maturation scaling ---


def test_term_neonate_scaling_less_than_one():
    """Term neonate (40w, day 0) should have scaling < 1."""
    r = predict_neonatal_dosing(
        "DrugA", 10.0, 50.0, 5.0, age_days=0, weight_kg=3.5, gestational_age_weeks=40
    )
    assert r.total_cl_scaling_factor < 1.0


def test_premature_lower_scaling_than_term():
    """Premature neonate should have lower scaling than term."""
    r_term = predict_neonatal_dosing(
        "DrugA", 10.0, 50.0, 5.0, age_days=0, weight_kg=3.5, gestational_age_weeks=40
    )
    r_prem = predict_neonatal_dosing(
        "DrugA", 10.0, 50.0, 5.0, age_days=0, weight_kg=1.5, gestational_age_weeks=28
    )
    assert r_prem.total_cl_scaling_factor < r_term.total_cl_scaling_factor


def test_older_age_higher_maturation():
    """Older postnatal age should increase maturation."""
    r_young = predict_neonatal_dosing("DrugA", 10.0, 50.0, 5.0, age_days=0, weight_kg=3.5)
    r_older = predict_neonatal_dosing("DrugA", 10.0, 50.0, 5.0, age_days=90, weight_kg=5.0)
    assert r_older.renal_maturation_factor > r_young.renal_maturation_factor
    assert r_older.hepatic_maturation_factor > r_young.hepatic_maturation_factor


def test_renal_maturation_between_0_and_1():
    r = predict_neonatal_dosing("DrugA", 10.0, 50.0, 5.0, age_days=7, weight_kg=3.5)
    assert 0.0 < r.renal_maturation_factor < 1.0


def test_hepatic_maturation_between_0_and_1():
    r = predict_neonatal_dosing("DrugA", 10.0, 50.0, 5.0, age_days=7, weight_kg=3.5)
    assert 0.0 < r.hepatic_maturation_factor < 1.0


# --- PK parameters ---


def test_cl_neonate_less_than_adult():
    """For immature neonate, CL should be much less than adult."""
    r = predict_neonatal_dosing("DrugA", 10.0, 50.0, 5.0, age_days=0, weight_kg=3.5)
    assert r.cl_neonate_L_per_h < r.cl_adult_L_per_h


def test_t_half_positive():
    r = predict_neonatal_dosing("DrugA", 10.0, 50.0, 5.0, age_days=7, weight_kg=3.5)
    assert r.t_half_neonate_h > 0


def test_vd_neonate_proportional_to_weight():
    """Vd scales linearly with weight."""
    r1 = predict_neonatal_dosing("DrugA", 10.0, 50.0, 5.0, age_days=7, weight_kg=3.0)
    r2 = predict_neonatal_dosing("DrugA", 10.0, 50.0, 5.0, age_days=7, weight_kg=6.0)
    assert math.isclose(r2.vd_neonate_L / r1.vd_neonate_L, 2.0, rel_tol=1e-6)


# --- Dose calculations ---


def test_dose_total_equals_mg_kg_times_weight():
    r = predict_neonatal_dosing("DrugA", 10.0, 50.0, 5.0, age_days=7, weight_kg=3.5)
    assert math.isclose(r.dose_neonate_total_mg, r.dose_neonate_mg_kg * r.weight_kg, rel_tol=1e-6)


def test_dose_neonate_less_than_adult_dose():
    """Dose per kg should be reduced for neonates."""
    r = predict_neonatal_dosing("DrugA", 10.0, 50.0, 5.0, age_days=0, weight_kg=3.5)
    assert r.dose_neonate_mg_kg < r.dose_adult_mg_kg


def test_dose_neonate_positive():
    r = predict_neonatal_dosing("DrugA", 10.0, 50.0, 5.0, age_days=7, weight_kg=3.5)
    assert r.dose_neonate_mg_kg > 0
    assert r.dose_neonate_total_mg > 0


# --- Dosing interval ---


def test_dosing_interval_valid():
    r = predict_neonatal_dosing("DrugA", 10.0, 50.0, 5.0, age_days=7, weight_kg=3.5)
    assert r.dosing_interval_h in (6.0, 12.0, 24.0)


def test_long_half_life_gives_24h_interval():
    """Very low CL → long half-life → 24h interval."""
    r = predict_neonatal_dosing("DrugA", 0.1, 50.0, 5.0, age_days=0, weight_kg=3.5)
    assert r.dosing_interval_h == 24.0


# --- Safety flag ---


def test_safety_flag_caution_for_premature():
    """Very premature neonate should get 'caution' flag."""
    r = predict_neonatal_dosing(
        "DrugA", 10.0, 50.0, 5.0, age_days=0, weight_kg=0.8, gestational_age_weeks=24
    )
    assert r.safety_flag == "caution"


def test_safety_flag_caution_low_weight():
    r = predict_neonatal_dosing(
        "DrugA", 10.0, 50.0, 5.0, age_days=7, weight_kg=0.9, gestational_age_weeks=28
    )
    assert r.safety_flag == "caution"


def test_safety_flag_standard_term():
    r = predict_neonatal_dosing("DrugA", 10.0, 50.0, 5.0, age_days=7, weight_kg=3.5)
    assert r.safety_flag == "standard"


# --- Notes ---


def test_notes_contain_pma():
    r = predict_neonatal_dosing("DrugA", 10.0, 50.0, 5.0, age_days=7, weight_kg=3.5)
    assert "PMA" in r.notes


def test_notes_immature_metabolism():
    r = predict_neonatal_dosing(
        "DrugA", 10.0, 50.0, 5.0, age_days=0, weight_kg=0.8, gestational_age_weeks=24
    )
    assert "immature" in r.notes


# --- Validation ---


def test_cl_zero_raises():
    with pytest.raises(ValueError, match="cl_adult"):
        predict_neonatal_dosing("DrugA", 0.0, 50.0, 5.0, age_days=7, weight_kg=3.5)


def test_vd_negative_raises():
    with pytest.raises(ValueError, match="vd_adult"):
        predict_neonatal_dosing("DrugA", 10.0, -1.0, 5.0, age_days=7, weight_kg=3.5)


def test_age_negative_raises():
    with pytest.raises(ValueError, match="age_days"):
        predict_neonatal_dosing("DrugA", 10.0, 50.0, 5.0, age_days=-1, weight_kg=3.5)


def test_weight_zero_raises():
    with pytest.raises(ValueError, match="weight_kg"):
        predict_neonatal_dosing("DrugA", 10.0, 50.0, 5.0, age_days=7, weight_kg=0.0)


def test_gestational_age_too_low():
    with pytest.raises(ValueError, match="gestational_age_weeks"):
        predict_neonatal_dosing(
            "DrugA", 10.0, 50.0, 5.0, age_days=7, weight_kg=3.5, gestational_age_weeks=20
        )


def test_gestational_age_too_high():
    with pytest.raises(ValueError, match="gestational_age_weeks"):
        predict_neonatal_dosing(
            "DrugA", 10.0, 50.0, 5.0, age_days=7, weight_kg=3.5, gestational_age_weeks=46
        )


def test_scaling_clamped_minimum():
    """Scaling factor should never go below 0.01."""
    r = predict_neonatal_dosing(
        "DrugA", 10.0, 50.0, 5.0, age_days=0, weight_kg=0.5, gestational_age_weeks=22
    )
    assert r.total_cl_scaling_factor >= 0.01


def test_short_half_life_gives_6h_interval():
    """High CL relative to Vd → short half-life → 6h interval."""
    r = predict_neonatal_dosing(
        "DrugA", 100.0, 10.0, 5.0, age_days=90, weight_kg=5.0, gestational_age_weeks=40
    )
    assert r.dosing_interval_h == 6.0
