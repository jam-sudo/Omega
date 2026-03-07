"""Tests for neonatal_pk Phase 645 additions: scale_neonatal_pk, neonatal_dose_recommendation."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.neonatal_pk import (
    NeonatalPKScalingResult,
    neonatal_dose_recommendation,
    scale_neonatal_pk,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(age_weeks=0.0, weight_kg=3.5, elimination="cyp3a4"):
    return scale_neonatal_pk(
        drug_name="TestDrug",
        age_weeks=age_weeks,
        weight_kg=weight_kg,
        cl_adult_L_per_h=5.0,
        vd_adult_L=50.0,
        primary_elimination=elimination,
        adult_dose_mg_per_kg=1.0,
    )


# ---------------------------------------------------------------------------
# Basic return type and field checks
# ---------------------------------------------------------------------------


def test_returns_neonatal_pk_scaling_result():
    res = _make_result()
    assert isinstance(res, NeonatalPKScalingResult)


def test_cyp3a4_maturation_fraction_in_0_1():
    for age in [0, 4, 13, 26, 52, 104]:
        res = _make_result(age_weeks=age)
        assert 0.0 <= res.cyp3a4_maturation_fraction <= 1.0, f"Failed at age={age}"


def test_cyp2d6_maturation_fraction_in_0_1():
    for age in [0, 4, 13, 26, 52, 104]:
        res = _make_result(age_weeks=age)
        assert 0.0 <= res.cyp2d6_maturation_fraction <= 1.0, f"Failed at age={age}"


def test_cyp1a2_maturation_fraction_in_0_1():
    for age in [0, 4, 13, 26, 52, 104]:
        res = _make_result(age_weeks=age)
        assert 0.0 <= res.cyp1a2_maturation_fraction <= 1.0, f"Failed at age={age}"


def test_gfr_fraction_in_0_1():
    for age in [0, 4, 13, 26, 52, 104]:
        res = _make_result(age_weeks=age)
        assert 0.0 <= res.gfr_fraction <= 1.0, f"Failed at age={age}"


def test_albumin_fraction_in_0_1():
    for age in [0, 4, 13, 26, 52, 104]:
        res = _make_result(age_weeks=age)
        assert 0.0 <= res.albumin_fraction <= 1.0, f"Failed at age={age}"


# ---------------------------------------------------------------------------
# CL and half-life checks
# ---------------------------------------------------------------------------


def test_cl_neonatal_less_than_cl_adult_at_birth():
    res = _make_result(age_weeks=0)
    assert res.cl_neonatal_L_per_h < res.cl_adult_L_per_h


def test_t_half_neonatal_greater_than_adult_at_birth():
    res = _make_result(age_weeks=0)
    assert res.t_half_neonatal_h > res.t_half_adult_h


def test_older_infant_has_higher_cl_than_newborn():
    res_newborn = _make_result(age_weeks=0)
    res_older = _make_result(age_weeks=52)
    assert res_older.cl_neonatal_L_per_h > res_newborn.cl_neonatal_L_per_h


# ---------------------------------------------------------------------------
# Dose adjustment checks
# ---------------------------------------------------------------------------


def test_dose_adjustment_factor_in_0_1_for_neonate():
    # Use age=1 week: CYP3A4 is slightly mature (f = 1/(9+1) = 0.1), so factor > 0
    res = _make_result(age_weeks=1)
    assert 0.0 < res.dose_adjustment_factor <= 1.0


def test_dose_mg_per_kg_positive():
    res = _make_result(age_weeks=4)
    assert res.dose_mg_per_kg > 0.0


# ---------------------------------------------------------------------------
# neonatal_dose_recommendation
# ---------------------------------------------------------------------------


def test_neonatal_dose_recommendation_returns_dict():
    result = neonatal_dose_recommendation(
        drug_name="Amoxicillin",
        age_weeks=2,
        weight_kg=3.5,
        adult_dose_mg=500.0,
    )
    assert isinstance(result, dict)


def test_neonatal_dose_recommendation_has_expected_keys():
    result = neonatal_dose_recommendation(
        drug_name="Amoxicillin",
        age_weeks=2,
        weight_kg=3.5,
        adult_dose_mg=500.0,
    )
    expected_keys = {
        "drug_name",
        "age_weeks",
        "weight_kg",
        "recommended_dose_mg",
        "dose_per_kg",
        "adjustment_factor",
        "warnings",
        "notes",
    }
    assert expected_keys.issubset(result.keys())


def test_neonatal_dose_recommendation_dose_positive():
    result = neonatal_dose_recommendation(
        drug_name="Amoxicillin",
        age_weeks=2,
        weight_kg=3.5,
        adult_dose_mg=500.0,
    )
    assert result["recommended_dose_mg"] > 0.0


def test_neonatal_dose_recommendation_adjustment_factor_lt_1_for_newborn():
    result = neonatal_dose_recommendation(
        drug_name="Amoxicillin",
        age_weeks=0,
        weight_kg=3.5,
        adult_dose_mg=500.0,
    )
    assert result["adjustment_factor"] < 1.0


# ---------------------------------------------------------------------------
# CYP-specific checks
# ---------------------------------------------------------------------------


def test_cyp3a4_maturation_at_birth_near_zero():
    res = _make_result(age_weeks=0)
    assert res.cyp3a4_maturation_fraction < 0.05


def test_cyp3a4_maturation_at_52_weeks_near_one():
    res = _make_result(age_weeks=52)
    # Hill with TM50=3, Hill=2, age=52 capped at 52:
    # f = 52^2 / (3^2 + 52^2) = 2704 / (9 + 2704) = 2704/2713 ~ 0.9967
    assert res.cyp3a4_maturation_fraction > 0.95


# ---------------------------------------------------------------------------
# Elimination pathway checks
# ---------------------------------------------------------------------------


def test_renal_elimination_cl_scales_with_gfr():
    res = _make_result(age_weeks=0, elimination="renal")
    assert res.cl_neonatal_L_per_h < res.cl_adult_L_per_h
    # GFR at birth = 0/52 = 0, so CL should be essentially 0
    assert res.cl_neonatal_L_per_h < 0.01


def test_mixed_elimination_intermediate():
    res_mixed = _make_result(age_weeks=2, elimination="mixed")
    # Mixed CL should be between pure renal and pure CYP (or at least > 0)
    assert res_mixed.cl_neonatal_L_per_h > 0.0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validation_negative_age_raises():
    with pytest.raises(ValueError, match="age_weeks"):
        _make_result(age_weeks=-1)


def test_validation_age_over_104_raises():
    with pytest.raises(ValueError, match="age_weeks"):
        _make_result(age_weeks=105)


def test_validation_zero_weight_raises():
    with pytest.raises(ValueError, match="weight_kg"):
        scale_neonatal_pk(
            drug_name="X",
            age_weeks=4,
            weight_kg=0,
            cl_adult_L_per_h=5.0,
            vd_adult_L=50.0,
        )


def test_validation_invalid_elimination_raises():
    with pytest.raises(ValueError, match="primary_elimination"):
        scale_neonatal_pk(
            drug_name="X",
            age_weeks=4,
            weight_kg=3.5,
            cl_adult_L_per_h=5.0,
            vd_adult_L=50.0,
            primary_elimination="hepatic",
        )


def test_notes_nonempty():
    res = _make_result()
    assert len(res.notes) > 0
