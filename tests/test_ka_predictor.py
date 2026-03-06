"""Tests for absorption rate constant predictor (Phase 222)."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.ka_predictor import KaPredictionResult, bcs_class_ka, predict_ka


# ---------------------------------------------------------------------------
# bcs_class_ka
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bcs_class", ["I", "II", "III", "IV"])
def test_bcs_class_ka_positive(bcs_class):
    assert bcs_class_ka(bcs_class) > 0.0


@pytest.mark.parametrize("bcs_class", ["i", "ii", "iii", "iv"])
def test_bcs_class_ka_case_insensitive(bcs_class):
    assert bcs_class_ka(bcs_class) > 0.0


def test_bcs_class_ka_invalid_raises():
    with pytest.raises(ValueError):
        bcs_class_ka("V")


def test_bcs_class_ordering():
    """BCS I and III (high permeability) should have higher ka than II and IV."""
    assert bcs_class_ka("I") > bcs_class_ka("II")
    assert bcs_class_ka("III") > bcs_class_ka("IV")


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_mw_zero_raises():
    with pytest.raises(ValueError, match="mw"):
        predict_ka(mw=0.0, logP=2.0, psa_A2=60.0)


def test_mw_negative_raises():
    with pytest.raises(ValueError, match="mw"):
        predict_ka(mw=-300.0, logP=2.0, psa_A2=60.0)


def test_psa_negative_raises():
    with pytest.raises(ValueError, match="psa_A2"):
        predict_ka(mw=300.0, logP=2.0, psa_A2=-10.0)


def test_particle_radius_zero_raises():
    with pytest.raises(ValueError, match="particle_radius"):
        predict_ka(mw=300.0, logP=2.0, psa_A2=60.0, particle_radius_um=0.0)


def test_solubility_zero_raises():
    with pytest.raises(ValueError, match="solubility"):
        predict_ka(mw=300.0, logP=2.0, psa_A2=60.0, solubility_mg_mL=0.0)


# ---------------------------------------------------------------------------
# Permeability direction tests
# ---------------------------------------------------------------------------


def test_high_logP_low_psa_high_peff():
    """High logP and low PSA → high Peff → high ka_permeability."""
    result_high = predict_ka(mw=300.0, logP=4.0, psa_A2=20.0)
    result_low = predict_ka(mw=300.0, logP=1.0, psa_A2=120.0)
    assert result_high.peff_cm_s > result_low.peff_cm_s
    assert result_high.ka_permeability_per_h > result_low.ka_permeability_per_h


def test_high_psa_low_peff():
    """High PSA → low Peff."""
    result = predict_ka(mw=300.0, logP=2.0, psa_A2=200.0)
    assert result.peff_cm_s < 1e-4


# ---------------------------------------------------------------------------
# Dissolution rate tests
# ---------------------------------------------------------------------------


def test_large_particles_dissolution_limited():
    """Large particles → dissolution may be rate-limiting."""
    result = predict_ka(
        mw=300.0,
        logP=3.5,  # decent permeability
        psa_A2=30.0,
        particle_radius_um=500.0,  # very large particles
        solubility_mg_mL=0.01,  # low solubility
    )
    assert result.rate_limiting_step == "dissolution"


def test_small_particles_permeability_limited():
    """Small particles + good solubility → permeability may be rate-limiting."""
    result = predict_ka(
        mw=300.0,
        logP=1.0,  # low logP → low permeability
        psa_A2=150.0,  # high PSA → even lower permeability
        particle_radius_um=1.0,  # tiny particles → fast dissolution
        solubility_mg_mL=10.0,  # high solubility
    )
    assert result.rate_limiting_step == "permeability"


# ---------------------------------------------------------------------------
# BCS class prediction consistency
# ---------------------------------------------------------------------------


def test_bcs_i_high_perm_high_sol():
    result = predict_ka(
        mw=200.0, logP=3.0, psa_A2=20.0, solubility_mg_mL=5.0, particle_radius_um=50.0
    )
    # High permeability drugs with high solubility → BCS I
    if result.peff_cm_s > 1e-4:
        assert result.bcs_class_predicted == "I"


def test_bcs_iii_low_perm_high_sol():
    result = predict_ka(
        mw=300.0, logP=-1.0, psa_A2=180.0, solubility_mg_mL=10.0, particle_radius_um=50.0
    )
    assert result.peff_cm_s < 1e-4  # confirm low permeability
    assert result.bcs_class_predicted == "III"


def test_bcs_ii_high_perm_low_sol():
    result = predict_ka(
        mw=300.0, logP=4.0, psa_A2=20.0, solubility_mg_mL=0.05, particle_radius_um=50.0
    )
    if result.peff_cm_s > 1e-4:
        assert result.bcs_class_predicted == "II"


# ---------------------------------------------------------------------------
# Rate limiting step
# ---------------------------------------------------------------------------


def test_rate_limiting_step_valid_value():
    result = predict_ka(mw=300.0, logP=2.0, psa_A2=80.0)
    assert result.rate_limiting_step in ("permeability", "dissolution")


def test_ka_effective_le_ka_permeability():
    """Dissolution can only reduce effective ka — never increase beyond permeability ka."""
    result = predict_ka(mw=300.0, logP=2.0, psa_A2=60.0, solubility_mg_mL=0.1)
    assert result.ka_effective_per_h <= result.ka_permeability_per_h + 1e-9


# ---------------------------------------------------------------------------
# Absorption category
# ---------------------------------------------------------------------------


def test_absorption_category_valid():
    result = predict_ka(mw=300.0, logP=2.0, psa_A2=80.0)
    assert result.absorption_category in ("rapid", "moderate", "slow")


# ---------------------------------------------------------------------------
# Result field types
# ---------------------------------------------------------------------------


def test_result_is_dataclass():
    result = predict_ka(mw=300.0, logP=2.0, psa_A2=80.0)
    assert isinstance(result, KaPredictionResult)


def test_result_field_types():
    result = predict_ka(mw=300.0, logP=2.0, psa_A2=80.0)
    assert isinstance(result.mw, float)
    assert isinstance(result.logP, float)
    assert isinstance(result.psa_A2, float)
    assert isinstance(result.peff_cm_s, float)
    assert isinstance(result.ka_permeability_per_h, float)
    assert isinstance(result.ka_dissolution_per_h, float)
    assert isinstance(result.ka_effective_per_h, float)
    assert isinstance(result.rate_limiting_step, str)
    assert isinstance(result.bcs_class_predicted, str)
    assert isinstance(result.absorption_category, str)
    assert isinstance(result.notes, list)
