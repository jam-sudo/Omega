"""Tests for Phase 441 — renal tubular secretion model."""

import pytest

from omega_pbpk.clinical.renal_tubular_secretion import (
    TRANSPORTER_CL,
    TubularSecretionResult,
    assess_tubular_secretion,
    compare_transporter_impact,
)


def make_result(**kwargs):
    defaults = dict(
        drug_name="TestDrug",
        transporter="OAT1",
        substrate_fraction=0.5,
        fu_plasma=0.3,
        logp=1.0,
    )
    defaults.update(kwargs)
    return assess_tubular_secretion(**defaults)


# --- Return type ---
def test_return_type():
    result = make_result()
    assert isinstance(result, TubularSecretionResult)


# --- Field preservation ---
def test_drug_name_preserved():
    result = make_result(drug_name="Penicillin")
    assert result.drug_name == "Penicillin"


def test_transporter_preserved():
    for t in TRANSPORTER_CL:
        result = make_result(transporter=t)
        assert result.transporter == t


# --- Filtration CL ---
def test_cl_filtration_equals_gfr_times_fu():
    gfr = 7.5
    fu = 0.4
    result = assess_tubular_secretion("D", "OAT1", 0.5, fu, 0.5, gfr_L_per_h=gfr)
    assert abs(result.cl_filtration_L_per_h - gfr * fu) < 1e-9


def test_cl_filtration_custom_gfr():
    gfr = 5.0
    fu = 0.5
    result = assess_tubular_secretion("D", "OAT3", 0.5, fu, 0.0, gfr_L_per_h=gfr)
    assert abs(result.cl_filtration_L_per_h - gfr * fu) < 1e-9


# --- Secretion CL ---
def test_cl_secretion_equals_cl_max_times_fraction():
    sf = 0.4
    result = assess_tubular_secretion("D", "OAT1", sf, 0.3, 0.0)
    expected = TRANSPORTER_CL["OAT1"]["cl_max_L_per_h"] * sf
    assert abs(result.cl_secretion_L_per_h - expected) < 1e-9


def test_cl_secretion_oat3():
    sf = 0.7
    result = assess_tubular_secretion("D", "OAT3", sf, 0.3, 0.0)
    expected = TRANSPORTER_CL["OAT3"]["cl_max_L_per_h"] * sf
    assert abs(result.cl_secretion_L_per_h - expected) < 1e-9


# --- Substrate class ---
def test_high_substrate_fraction_gives_high_class():
    result = make_result(substrate_fraction=0.8)
    assert result.substrate_class == "high"


def test_moderate_substrate_class():
    result = make_result(substrate_fraction=0.45)
    assert result.substrate_class == "moderate"


def test_low_substrate_class():
    result = make_result(substrate_fraction=0.2)
    assert result.substrate_class == "low"


def test_non_substrate_class():
    result = make_result(substrate_fraction=0.05)
    assert result.substrate_class == "non_substrate"


# --- Renal CL positivity ---
def test_cl_renal_total_positive():
    result = make_result(substrate_fraction=0.5, fu_plasma=0.3, logp=0.0)
    assert result.cl_renal_total_L_per_h >= 0.0


def test_secretion_adds_to_filtration_for_hydrophilic():
    # logP <= 1 → no reabsorption, so renal total >= filtration
    result = assess_tubular_secretion("D", "OAT1", 0.5, 0.3, 0.5)
    # reabsorption for logP=0.5 (<=1) is 0
    assert result.cl_renal_total_L_per_h >= result.cl_filtration_L_per_h


# --- Reabsorption ---
def test_high_logp_higher_reabsorption():
    r_low = assess_tubular_secretion("D", "OAT1", 0.5, 0.3, 0.0)
    r_high = assess_tubular_secretion("D", "OAT1", 0.5, 0.3, 5.0)
    assert r_high.cl_reabsorption_offset_L_per_h > r_low.cl_reabsorption_offset_L_per_h


def test_reabsorption_capped_at_50pct_filtration():
    # Very high logP → reabsorption should not exceed 50% of filtration
    result = assess_tubular_secretion("D", "OAT1", 0.5, 0.5, 100.0)
    assert result.cl_reabsorption_offset_L_per_h <= result.cl_filtration_L_per_h * 0.5 + 1e-9


# --- fe_urine fraction ---
def test_fe_urine_between_0_and_1():
    result = make_result(substrate_fraction=0.5, fu_plasma=0.3, logp=1.0)
    assert 0.0 <= result.fe_urine_fraction <= 1.0


def test_high_fe_gives_major_ckd_adjustment():
    # High substrate_fraction, high fu, low logP, low nonrenal CL
    result = assess_tubular_secretion("D", "combined", 1.0, 1.0, 0.0, cl_nonrenal_L_per_h=1.0)
    assert result.dose_adjustment_with_ckd == "major"


def test_low_fe_gives_none_ckd_adjustment():
    # Very low substrate_fraction, very high non-renal CL
    result = assess_tubular_secretion("D", "OAT1", 0.01, 0.1, 0.0, cl_nonrenal_L_per_h=200.0)
    assert result.dose_adjustment_with_ckd == "none"


# --- compare_transporter_impact ---
def test_compare_transporter_impact_returns_all():
    results = compare_transporter_impact("D", 0.5, 0.3, 0.5)
    assert len(results) == 4  # OAT1, OAT3, OCT2, combined


def test_compare_transporter_impact_sorted_descending():
    results = compare_transporter_impact("D", 0.5, 0.3, 0.5)
    for i in range(len(results) - 1):
        assert results[i].cl_secretion_L_per_h >= results[i + 1].cl_secretion_L_per_h


def test_compare_transporter_subset():
    results = compare_transporter_impact("D", 0.5, 0.3, 0.5, transporters_list=["OAT1", "OCT2"])
    assert len(results) == 2
    transporters = {r.transporter for r in results}
    assert transporters == {"OAT1", "OCT2"}


# --- Validation errors ---
def test_invalid_transporter_raises():
    with pytest.raises(ValueError, match="Unknown transporter"):
        assess_tubular_secretion("D", "UNKNOWN", 0.5, 0.3, 0.5)


def test_substrate_fraction_out_of_range_raises():
    with pytest.raises(ValueError):
        assess_tubular_secretion("D", "OAT1", 1.5, 0.3, 0.5)


def test_substrate_fraction_negative_raises():
    with pytest.raises(ValueError):
        assess_tubular_secretion("D", "OAT1", -0.1, 0.3, 0.5)


def test_fu_plasma_zero_raises():
    with pytest.raises(ValueError):
        assess_tubular_secretion("D", "OAT1", 0.5, 0.0, 0.5)


def test_fu_plasma_greater_than_1_raises():
    with pytest.raises(ValueError):
        assess_tubular_secretion("D", "OAT1", 0.5, 1.5, 0.5)
