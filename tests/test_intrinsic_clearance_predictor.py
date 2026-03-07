"""Tests for Phase 603 — Intrinsic Clearance Predictor."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.intrinsic_clearance_predictor import (
    IntrinsicClearanceResult,
    predict_intrinsic_clearance,
    scale_microsomal_to_in_vivo,
)

VALID_KWARGS = dict(
    drug_name="TestDrug",
    logP=2.0,
    mw=350.0,
    psa=80.0,
    n_aromatic_rings=1,
    n_hbd=2,
    fu_plasma=0.5,
    vd_L=50.0,
)


def default_result() -> IntrinsicClearanceResult:
    return predict_intrinsic_clearance(**VALID_KWARGS)


# --- basic return type ---


def test_returns_result():
    r = default_result()
    assert isinstance(r, IntrinsicClearanceResult)


def test_clint_positive():
    r = default_result()
    assert r.clint_mL_per_min_per_mg > 0


def test_scaled_clint_positive():
    r = default_result()
    assert r.clint_scaled_mL_per_min > 0


def test_hepatic_cl_positive():
    r = default_result()
    assert r.hepatic_cl_L_per_h > 0


def test_fu_mic_in_range():
    r = default_result()
    assert 0 < r.fu_mic <= 1.0


def test_er_in_0_1():
    r = default_result()
    assert 0.0 <= r.extraction_ratio <= 1.0


def test_er_le_1():
    r = default_result()
    assert r.extraction_ratio <= 1.0


def test_t_half_positive():
    r = default_result()
    assert r.predicted_t_half_h > 0


# --- directional tests ---


def test_high_psa_lower_clint():
    """Higher PSA decreases CLint (coefficient -0.01*psa)."""
    r_low_psa = predict_intrinsic_clearance(**{**VALID_KWARGS, "psa": 10.0})
    r_high_psa = predict_intrinsic_clearance(**{**VALID_KWARGS, "psa": 150.0})
    assert r_low_psa.clint_mL_per_min_per_mg > r_high_psa.clint_mL_per_min_per_mg


def test_basic_drug_higher_clint():
    """Basic drug (pka_base > 7) gets bonus → higher CLint."""
    r_no_base = predict_intrinsic_clearance(**VALID_KWARGS)
    r_basic = predict_intrinsic_clearance(**{**VALID_KWARGS, "pka_base": 9.0})
    assert r_basic.clint_mL_per_min_per_mg > r_no_base.clint_mL_per_min_per_mg


# --- classification ---


def test_metabolism_class_valid():
    r = default_result()
    assert r.metabolism_class in {"low", "moderate", "high", "very_high"}


def test_high_clearance_flag():
    """Force very high CLint to get ER > 0.7."""
    # n_aromatic_rings=5, low psa, large mw, basic drug, logP=0
    r = predict_intrinsic_clearance(
        drug_name="HighCL",
        logP=0.0,
        mw=600.0,
        psa=10.0,
        n_aromatic_rings=5,
        n_hbd=0,
        pka_base=9.0,
        fu_plasma=0.9,
        vd_L=100.0,
    )
    assert r.high_clearance is True
    assert r.extraction_ratio > 0.7


def test_low_clearance_flag():
    """Force very low CLint → ER < 0.7."""
    r = predict_intrinsic_clearance(
        drug_name="LowCL",
        logP=5.0,
        mw=100.0,
        psa=200.0,
        n_aromatic_rings=0,
        n_hbd=5,
        fu_plasma=0.01,
        vd_L=50.0,
    )
    assert r.high_clearance is False


# --- primary enzyme ---

VALID_CYPS = {"CYP3A4", "CYP2D6", "CYP2C19"}


def test_primary_enzyme_valid():
    r = default_result()
    assert r.primary_enzyme in VALID_CYPS


def test_primary_enzyme_cyp3a4_lipophilic_large():
    r = predict_intrinsic_clearance(
        drug_name="LipophilicLarge",
        logP=4.0,
        mw=500.0,
        psa=80.0,
        fu_plasma=0.5,
        vd_L=50.0,
    )
    assert r.primary_enzyme == "CYP3A4"


def test_primary_enzyme_cyp2d6_basic():
    r = predict_intrinsic_clearance(
        drug_name="BasicDrug",
        logP=1.0,
        mw=250.0,
        psa=80.0,
        pka_base=8.5,
        fu_plasma=0.5,
        vd_L=50.0,
    )
    assert r.primary_enzyme == "CYP2D6"


def test_primary_enzyme_cyp2c19_small_polar():
    r = predict_intrinsic_clearance(
        drug_name="SmallPolar",
        logP=1.0,
        mw=200.0,
        psa=40.0,
        fu_plasma=0.5,
        vd_L=50.0,
    )
    assert r.primary_enzyme == "CYP2C19"


# --- scale_microsomal_to_in_vivo ---


def test_scale_microsomal_formula():
    result = scale_microsomal_to_in_vivo(1.0, 45.0, 1500.0)
    assert abs(result - 67.5) < 1e-9


def test_scale_microsomal_proportional():
    r1 = scale_microsomal_to_in_vivo(1.0)
    r2 = scale_microsomal_to_in_vivo(2.0)
    assert abs(r2 - 2 * r1) < 1e-9


# --- validation errors ---


def test_invalid_mw_raises():
    with pytest.raises(ValueError, match="mw"):
        predict_intrinsic_clearance(drug_name="X", logP=2.0, mw=0.0, psa=80.0)


def test_invalid_fu_plasma_raises():
    with pytest.raises(ValueError, match="fu_plasma"):
        predict_intrinsic_clearance(drug_name="X", logP=2.0, mw=300.0, psa=80.0, fu_plasma=0.0)


def test_invalid_vd_raises():
    with pytest.raises(ValueError, match="vd_L"):
        predict_intrinsic_clearance(drug_name="X", logP=2.0, mw=300.0, psa=80.0, vd_L=-1.0)


def test_invalid_n_aromatic_raises():
    with pytest.raises(ValueError, match="n_aromatic_rings"):
        predict_intrinsic_clearance(
            drug_name="X", logP=2.0, mw=300.0, psa=80.0, n_aromatic_rings=-1
        )


# --- notes ---


def test_notes_nonempty():
    r = default_result()
    assert len(r.notes) > 0


# --- hepatic CL vs blood flow ---


def test_hepatic_cl_le_blood_flow():
    """Hepatic CL must not exceed hepatic blood flow (90 L/h = 1500 mL/min)."""
    r = default_result()
    QH_L_per_h = 90.0
    assert r.hepatic_cl_L_per_h <= QH_L_per_h + 1e-6
