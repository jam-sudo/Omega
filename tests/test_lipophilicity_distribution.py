"""Tests for lipophilicity_distribution module (Phase 248)."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.prediction.lipophilicity_distribution import (
    VssPredictionResult,
    logD_at_pH,
    predict_tissue_Kp,
    predict_vss,
)

# ---------------------------------------------------------------------------
# logD_at_pH tests
# ---------------------------------------------------------------------------


def test_logd_neutral_equals_logP():
    """For a neutral compound, logD = logP regardless of pH."""
    logP = 2.5
    assert logD_at_pH(logP, pka=7.0, ph=7.4, compound_type="neutral") == pytest.approx(logP)


def test_logd_acid_at_high_ph_less_than_logP():
    """Acid at pH >> pKa: mostly ionized → logD << logP."""
    logD = logD_at_pH(logP=3.0, pka=4.0, ph=7.4, compound_type="acid")
    assert logD < 3.0


def test_logd_acid_at_low_ph_approaches_logP():
    """Acid at pH << pKa: mostly neutral → logD ≈ logP."""
    logD = logD_at_pH(logP=3.0, pka=7.0, ph=1.0, compound_type="acid")
    assert logD > 2.9  # very close to logP=3.0


def test_logd_base_at_low_ph_less_than_logP():
    """Base at pH << pKa: mostly ionized → logD << logP."""
    logD = logD_at_pH(logP=3.0, pka=9.0, ph=4.0, compound_type="base")
    assert logD < 3.0


def test_logd_base_at_high_ph_approaches_logP():
    """Base at pH >> pKa: mostly neutral → logD ≈ logP."""
    logD = logD_at_pH(logP=3.0, pka=5.0, ph=9.0, compound_type="base")
    assert logD > 2.9


def test_logd_formula_acid():
    """Verify acid formula: logD = logP - log10(1 + 10^(pH - pKa))."""
    logP, pka, ph = 2.0, 5.0, 7.4
    expected = logP - math.log10(1 + 10 ** (ph - pka))
    result = logD_at_pH(logP, pka, ph, compound_type="acid")
    assert result == pytest.approx(expected, rel=1e-6)


def test_logd_formula_base():
    """Verify base formula: logD = logP - log10(1 + 10^(pKa - pH))."""
    logP, pka, ph = 2.0, 9.0, 7.4
    expected = logP - math.log10(1 + 10 ** (pka - ph))
    result = logD_at_pH(logP, pka, ph, compound_type="base")
    assert result == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# predict_tissue_Kp tests
# ---------------------------------------------------------------------------


def test_kp_fat_highest_for_lipophilic():
    """Fat Kp should be highest for a highly lipophilic drug."""
    logP, fup = 4.0, 0.1
    kps = {
        t: predict_tissue_Kp(logP, fup, t)
        for t in ["liver", "kidney", "muscle", "fat", "brain", "lung"]
    }
    assert kps["fat"] == max(kps.values())


def test_kp_muscle_lowest_for_lipophilic():
    """Muscle should have a low Kp for any drug."""
    logP, fup = 2.0, 0.5
    kp_muscle = predict_tissue_Kp(logP, fup, "muscle")
    kp_fat = predict_tissue_Kp(logP, fup, "fat")
    kp_liver = predict_tissue_Kp(logP, fup, "liver")
    assert kp_muscle < kp_fat
    assert kp_muscle < kp_liver


def test_kp_fat_greater_than_muscle():
    """Fat Kp > muscle Kp for logP > 0."""
    assert predict_tissue_Kp(2.0, 0.5, "fat") > predict_tissue_Kp(2.0, 0.5, "muscle")


def test_kp_all_tissues_positive():
    for tissue in ["liver", "kidney", "muscle", "fat", "brain", "lung"]:
        assert predict_tissue_Kp(2.0, 0.5, tissue) > 0.0


def test_kp_invalid_tissue_raises():
    with pytest.raises(ValueError, match="tissue"):
        predict_tissue_Kp(2.0, 0.5, "spleen")


def test_kp_increases_with_logP_fat():
    """Higher logP → higher fat Kp."""
    kp_low = predict_tissue_Kp(1.0, 0.5, "fat")
    kp_high = predict_tissue_Kp(4.0, 0.5, "fat")
    assert kp_high > kp_low


# ---------------------------------------------------------------------------
# predict_vss tests
# ---------------------------------------------------------------------------


def test_predict_vss_returns_result():
    r = predict_vss(logP=2.5, fup=0.3, mw=350.0)
    assert isinstance(r, VssPredictionResult)


def test_vss_70kg_equals_vss_per_kg_times_70():
    r = predict_vss(logP=2.5, fup=0.3, mw=350.0)
    assert r.vss_70kg_L == pytest.approx(r.vss_L_per_kg * 70.0, rel=1e-6)


def test_high_logP_high_vss():
    """High logP → high Vss."""
    r_low = predict_vss(logP=0.5, fup=0.5, mw=300.0)
    r_high = predict_vss(logP=5.0, fup=0.5, mw=300.0)
    assert r_high.vss_L_per_kg > r_low.vss_L_per_kg


def test_low_fup_higher_vss():
    """Lower fup → higher tissue distribution → higher Vss."""
    r_bound = predict_vss(logP=2.0, fup=0.05, mw=300.0)
    r_free = predict_vss(logP=2.0, fup=0.9, mw=300.0)
    assert r_bound.vss_L_per_kg > r_free.vss_L_per_kg


def test_distribution_category_low():
    """Low logP, high fup → low Vss category."""
    r = predict_vss(logP=-1.0, fup=1.0, mw=200.0)
    assert r.distribution_category == "low"


def test_distribution_category_high():
    """High logP, low fup → high Vss category."""
    r = predict_vss(logP=6.5, fup=0.01, mw=400.0)
    assert r.distribution_category == "high"


def test_distribution_category_moderate():
    """Moderate conditions → moderate Vss category."""
    r = predict_vss(logP=2.0, fup=0.3, mw=300.0)
    if 0.6 <= r.vss_L_per_kg <= 3.0:
        assert r.distribution_category == "moderate"
    elif r.vss_L_per_kg < 0.6:
        assert r.distribution_category == "low"
    else:
        assert r.distribution_category == "high"


def test_acid_vss_less_than_base():
    """Acid distributes less than base (same logP/fup)."""
    r_acid = predict_vss(logP=2.5, fup=0.3, mw=300.0, compound_type="acid")
    r_base = predict_vss(logP=2.5, fup=0.3, mw=300.0, compound_type="base")
    assert r_acid.vss_L_per_kg < r_base.vss_L_per_kg


def test_vss_minimum_floor():
    """Vss should never fall below 0.04 L/kg (plasma volume minimum)."""
    r = predict_vss(logP=-5.0, fup=1.0, mw=100.0)
    assert r.vss_L_per_kg >= 0.04


def test_field_types():
    r = predict_vss(logP=2.0, fup=0.5, mw=300.0)
    assert isinstance(r.logP, float)
    assert isinstance(r.fup, float)
    assert isinstance(r.mw, float)
    assert isinstance(r.vss_L_per_kg, float)
    assert isinstance(r.vss_70kg_L, float)
    assert isinstance(r.kp_fat, float)
    assert isinstance(r.kp_brain, float)
    assert isinstance(r.distribution_category, str)
    assert isinstance(r.notes, str)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_fup_zero_raises():
    with pytest.raises(ValueError, match="fup"):
        predict_vss(logP=2.0, fup=0.0, mw=300.0)


def test_invalid_fup_greater_than_one_raises():
    with pytest.raises(ValueError, match="fup"):
        predict_vss(logP=2.0, fup=1.5, mw=300.0)


def test_invalid_mw_raises():
    with pytest.raises(ValueError, match="mw"):
        predict_vss(logP=2.0, fup=0.3, mw=0.0)


def test_invalid_compound_type_raises():
    with pytest.raises(ValueError, match="compound_type"):
        predict_vss(logP=2.0, fup=0.3, mw=300.0, compound_type="zwitterion")


def test_logd_invalid_compound_type_raises():
    with pytest.raises(ValueError, match="compound_type"):
        logD_at_pH(2.0, 7.0, 7.4, compound_type="amphiphile")
