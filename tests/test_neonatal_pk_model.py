"""Tests for Phase 520 — Neonatal PK Model."""

import math

import pytest

from omega_pbpk.clinical.neonatal_pk_model import (
    NeonatalPKResult,
    _f_alb,
    _f_cyp,
    _f_gfr,
    neonatal_age_progression,
    predict_neonatal_pk,
)

# ---------------------------------------------------------------------------
# Common test parameters
# ---------------------------------------------------------------------------

DRUG = "TestAntibiotic"
WEIGHT_KG = 3.0  # typical neonate weight
CL_ADULT = 2.0  # L/h adult metabolic CL
VD_ADULT = 30.0  # L
FUP_ADULT = 0.1


# ---------------------------------------------------------------------------
# Return type and basic fields
# ---------------------------------------------------------------------------


def test_return_type():
    r = predict_neonatal_pk(DRUG, 7.0, WEIGHT_KG, "CYP3A4", CL_ADULT, VD_ADULT)
    assert isinstance(r, NeonatalPKResult)


def test_drug_name_stored():
    r = predict_neonatal_pk(DRUG, 7.0, WEIGHT_KG, "CYP3A4", CL_ADULT, VD_ADULT)
    assert r.drug_name == DRUG


def test_postnatal_age_stored():
    r = predict_neonatal_pk(DRUG, 14.0, WEIGHT_KG, "CYP3A4", CL_ADULT, VD_ADULT)
    assert r.postnatal_age_days == 14.0


def test_weight_stored():
    r = predict_neonatal_pk(DRUG, 7.0, WEIGHT_KG, "CYP3A4", CL_ADULT, VD_ADULT)
    assert r.weight_kg == WEIGHT_KG


def test_dominant_cyp_stored():
    r = predict_neonatal_pk(DRUG, 7.0, WEIGHT_KG, "CYP3A4", CL_ADULT, VD_ADULT)
    assert r.dominant_cyp == "CYP3A4"


# ---------------------------------------------------------------------------
# Maturation factors
# ---------------------------------------------------------------------------


def test_f_gfr_at_day0():
    """GFR maturation at birth = 10%."""
    assert abs(_f_gfr(0) - 0.1) < 1e-9


def test_f_gfr_at_day28():
    """GFR maturation at day 28 = 100%."""
    assert abs(_f_gfr(28) - 1.0) < 1e-9


def test_f_cyp2d6_at_day0_is_zero():
    """CYP2D6 is absent at birth."""
    assert abs(_f_cyp("CYP2D6", 0) - 0.0) < 1e-9


def test_f_cyp2d6_at_day28_approaches_adult():
    assert abs(_f_cyp("CYP2D6", 28) - 1.0) < 1e-9


def test_f_cyp3a4_at_day0():
    assert abs(_f_cyp("CYP3A4", 0) - 0.05) < 1e-9


def test_f_cyp1a2_constant():
    """CYP1A2 is fixed at 0.05 regardless of age."""
    assert abs(_f_cyp("CYP1A2", 0) - 0.05) < 1e-9
    assert abs(_f_cyp("CYP1A2", 28) - 0.05) < 1e-9


def test_f_alb_at_day0():
    assert abs(_f_alb(0) - 0.6) < 1e-9


def test_f_alb_at_day28():
    assert abs(_f_alb(28) - 1.0) < 1e-9


def test_f_tbw_fixed_at_1_3():
    r = predict_neonatal_pk(DRUG, 7.0, WEIGHT_KG, "CYP3A4", CL_ADULT, VD_ADULT)
    assert abs(r.f_tbw - 1.3) < 1e-9


# ---------------------------------------------------------------------------
# GFR increases with age
# ---------------------------------------------------------------------------


def test_gfr_increases_with_postnatal_age():
    r0 = predict_neonatal_pk(DRUG, 0.0, WEIGHT_KG, "CYP3A4", CL_ADULT, VD_ADULT)
    r14 = predict_neonatal_pk(DRUG, 14.0, WEIGHT_KG, "CYP3A4", CL_ADULT, VD_ADULT)
    r28 = predict_neonatal_pk(DRUG, 28.0, WEIGHT_KG, "CYP3A4", CL_ADULT, VD_ADULT)
    assert r0.gfr_neonatal_L_per_h < r14.gfr_neonatal_L_per_h < r28.gfr_neonatal_L_per_h


# ---------------------------------------------------------------------------
# Vd_neo = Vd_adult * 1.3
# ---------------------------------------------------------------------------


def test_vd_neonatal_equals_vd_adult_times_f_tbw():
    r = predict_neonatal_pk(DRUG, 7.0, WEIGHT_KG, "CYP3A4", CL_ADULT, VD_ADULT)
    assert abs(r.vd_neonatal_L - VD_ADULT * 1.3) < 1e-9


# ---------------------------------------------------------------------------
# Half-life calculation: t1/2 = ln(2) * Vd / CL
# ---------------------------------------------------------------------------


def test_t_half_formula():
    r = predict_neonatal_pk(DRUG, 14.0, WEIGHT_KG, "CYP3A4", CL_ADULT, VD_ADULT)
    expected = math.log(2) * r.vd_neonatal_L / r.cl_total_neonatal_L_per_h
    assert abs(r.t_half_neonatal_h - expected) < 1e-9


# ---------------------------------------------------------------------------
# CYP2D6 at day 0: zero metabolic CL
# ---------------------------------------------------------------------------


def test_cyp2d6_day0_zero_metabolic_cl():
    r = predict_neonatal_pk(DRUG, 0.0, WEIGHT_KG, "CYP2D6", CL_ADULT, VD_ADULT)
    assert abs(r.cl_metabolic_neonatal_L_per_h) < 1e-12


# ---------------------------------------------------------------------------
# monitoring_required for very low dose_adjustment_factor
# ---------------------------------------------------------------------------


def test_monitoring_required_day0_cyp2d6():
    """At day 0 with CYP2D6, metabolic CL is ~0 → low dose adjustment → monitoring."""
    # Renal-only cleared drug (small renal contribution)
    r = predict_neonatal_pk(DRUG, 0.0, WEIGHT_KG, "CYP2D6", CL_ADULT, VD_ADULT, fup_adult=0.05)
    assert isinstance(r.monitoring_required, bool)
    # With CL_adult_metabolic=2 L/h and near-zero metabolic at day 0, factor should be low
    assert r.dose_adjustment_factor < 1.0


def test_monitoring_required_flag():
    r = predict_neonatal_pk(DRUG, 0.0, WEIGHT_KG, "CYP2D6", CL_ADULT * 10, VD_ADULT)
    # With very high adult CL reference, neonatal adjustment factor is tiny
    assert r.monitoring_required is True


# ---------------------------------------------------------------------------
# Age progression
# ---------------------------------------------------------------------------


def test_age_progression_returns_5_results():
    results = neonatal_age_progression(DRUG, WEIGHT_KG, "CYP3A4", CL_ADULT, VD_ADULT)
    assert len(results) == 5


def test_age_progression_sorted_ascending():
    results = neonatal_age_progression(DRUG, WEIGHT_KG, "CYP3A4", CL_ADULT, VD_ADULT)
    ages = [r.postnatal_age_days for r in results]
    assert ages == sorted(ages)


def test_age_progression_days_are_0_7_14_21_28():
    results = neonatal_age_progression(DRUG, WEIGHT_KG, "CYP3A4", CL_ADULT, VD_ADULT)
    ages = [r.postnatal_age_days for r in results]
    assert ages == [0.0, 7.0, 14.0, 21.0, 28.0]


def test_age_progression_all_neonatal_pk_result():
    results = neonatal_age_progression(DRUG, WEIGHT_KG, "CYP3A4", CL_ADULT, VD_ADULT)
    for r in results:
        assert isinstance(r, NeonatalPKResult)


def test_age_progression_cl_increases_with_age():
    results = neonatal_age_progression(DRUG, WEIGHT_KG, "CYP3A4", CL_ADULT, VD_ADULT)
    cls = [r.cl_total_neonatal_L_per_h for r in results]
    # CL should increase (both renal and CYP3A4 mature)
    assert cls[-1] > cls[0]


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_age_out_of_range_negative():
    with pytest.raises(ValueError, match="postnatal_age_days"):
        predict_neonatal_pk(DRUG, -1.0, WEIGHT_KG, "CYP3A4", CL_ADULT, VD_ADULT)


def test_age_out_of_range_too_large():
    with pytest.raises(ValueError, match="postnatal_age_days"):
        predict_neonatal_pk(DRUG, 29.0, WEIGHT_KG, "CYP3A4", CL_ADULT, VD_ADULT)


def test_invalid_weight():
    with pytest.raises(ValueError, match="weight_kg"):
        predict_neonatal_pk(DRUG, 7.0, 0.0, "CYP3A4", CL_ADULT, VD_ADULT)


def test_invalid_dominant_cyp():
    with pytest.raises(ValueError, match="dominant_cyp"):
        predict_neonatal_pk(DRUG, 7.0, WEIGHT_KG, "CYP99", CL_ADULT, VD_ADULT)
