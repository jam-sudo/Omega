"""Tests for prediction/absorption_rate_predictor.py — Phase 634."""

import pytest

from omega_pbpk.prediction.absorption_rate_predictor import (
    AbsorptionRatePrediction,
    predict_absorption_rate,
    screen_absorption,
)

# ---------------------------------------------------------------------------
# Helper fixtures / constants
# ---------------------------------------------------------------------------

# BCS I: high logP, low PSA, low MW, high solubility
BCS_I_KWARGS = dict(
    compound_name="bcs1_drug",
    logP=2.5,
    mw=250.0,
    psa=60.0,
    solubility_mg_mL=5.0,  # 5 * 250 = 1250 mg >> dose 100 mg
    dose_mg=100.0,
)

# BCS II: high logP, low PSA, low MW, low solubility
BCS_II_KWARGS = dict(
    compound_name="bcs2_drug",
    logP=3.5,
    mw=350.0,
    psa=70.0,
    solubility_mg_mL=0.01,  # 0.01 * 250 = 2.5 mg << dose 100 mg
    dose_mg=100.0,
)

# BCS IV: low logP, high PSA, low solubility
BCS_IV_KWARGS = dict(
    compound_name="bcs4_drug",
    logP=-1.0,
    mw=600.0,
    psa=200.0,
    solubility_mg_mL=0.001,
    dose_mg=100.0,
)


# ---------------------------------------------------------------------------
# Basic return type tests
# ---------------------------------------------------------------------------


def test_returns_absorption_rate_prediction():
    result = predict_absorption_rate(**BCS_I_KWARGS)
    assert isinstance(result, AbsorptionRatePrediction)


def test_ka_positive():
    result = predict_absorption_rate(**BCS_I_KWARGS)
    assert result.ka_per_h > 0


def test_lag_time_nonnegative():
    result = predict_absorption_rate(**BCS_I_KWARGS)
    assert result.lag_time_h >= 0


def test_predicted_tmax_positive():
    result = predict_absorption_rate(**BCS_I_KWARGS)
    assert result.predicted_tmax_h > 0


def test_bcs_class_valid():
    for kwargs in [BCS_I_KWARGS, BCS_II_KWARGS, BCS_IV_KWARGS]:
        result = predict_absorption_rate(**kwargs)
        assert result.bcs_class in ("I", "II", "III", "IV")


def test_f_abs_in_range():
    result = predict_absorption_rate(**BCS_I_KWARGS)
    assert 0 < result.f_abs_predicted <= 1.0


def test_notes_nonempty():
    result = predict_absorption_rate(**BCS_I_KWARGS)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Scientific sensitivity tests
# ---------------------------------------------------------------------------


def test_high_logp_low_psa_higher_ka():
    """High logP + low PSA → higher ka than low logP + high PSA."""
    res_high = predict_absorption_rate(
        compound_name="high_perm",
        logP=3.0,
        mw=300.0,
        psa=40.0,
        solubility_mg_mL=2.0,
    )
    res_low = predict_absorption_rate(
        compound_name="low_perm",
        logP=-1.0,
        mw=300.0,
        psa=160.0,
        solubility_mg_mL=2.0,
    )
    assert res_high.ka_per_h > res_low.ka_per_h


def test_small_particles_higher_ka():
    """Smaller particles → faster dissolution → higher ka."""
    res_small = predict_absorption_rate(
        compound_name="small_part",
        logP=2.0,
        mw=300.0,
        psa=80.0,
        solubility_mg_mL=1.0,
        particle_size_um=10.0,
    )
    res_large = predict_absorption_rate(
        compound_name="large_part",
        logP=2.0,
        mw=300.0,
        psa=80.0,
        solubility_mg_mL=1.0,
        particle_size_um=500.0,
    )
    assert res_small.ka_per_h > res_large.ka_per_h


def test_bcs_i_classification():
    result = predict_absorption_rate(**BCS_I_KWARGS)
    assert result.bcs_class == "I"
    # Should be neither permeability- nor dissolution-limited for BCS I with small particles
    # (dissolution_limited may still be True for 100 um particles per spec)


def test_bcs_ii_classification():
    result = predict_absorption_rate(**BCS_II_KWARGS)
    assert result.bcs_class == "II"


def test_bcs_iv_classification():
    result = predict_absorption_rate(**BCS_IV_KWARGS)
    assert result.bcs_class == "IV"


def test_dissolution_limited_flag_large_particles():
    """Particles > 50 um should trigger dissolution_limited=True."""
    result = predict_absorption_rate(
        compound_name="drug",
        logP=2.0,
        mw=300.0,
        psa=60.0,
        solubility_mg_mL=5.0,
        particle_size_um=200.0,
    )
    assert result.dissolution_limited is True


# ---------------------------------------------------------------------------
# screen_absorption tests
# ---------------------------------------------------------------------------


def test_screen_returns_sorted_by_ka():
    compounds = [
        dict(compound_name="slow", logP=-1.0, mw=600.0, psa=180.0, solubility_mg_mL=5.0),
        dict(compound_name="fast", logP=3.0, mw=250.0, psa=40.0, solubility_mg_mL=5.0),
        dict(compound_name="medium", logP=1.5, mw=350.0, psa=90.0, solubility_mg_mL=2.0),
    ]
    results = screen_absorption(compounds)
    assert len(results) == 3
    # Sorted descending by ka
    for i in range(len(results) - 1):
        assert results[i].ka_per_h >= results[i + 1].ka_per_h


def test_screen_empty_list():
    results = screen_absorption([])
    assert results == []


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------


def test_validation_mw_zero_raises():
    with pytest.raises(ValueError, match="mw"):
        predict_absorption_rate(
            compound_name="drug", logP=2.0, mw=0.0, psa=80.0, solubility_mg_mL=1.0
        )


def test_validation_psa_negative_raises():
    with pytest.raises(ValueError, match="psa"):
        predict_absorption_rate(
            compound_name="drug", logP=2.0, mw=300.0, psa=-5.0, solubility_mg_mL=1.0
        )


def test_validation_solubility_negative_raises():
    with pytest.raises(ValueError, match="solubility"):
        predict_absorption_rate(
            compound_name="drug", logP=2.0, mw=300.0, psa=80.0, solubility_mg_mL=-0.1
        )


def test_validation_particle_size_zero_raises():
    with pytest.raises(ValueError, match="particle_size"):
        predict_absorption_rate(
            compound_name="drug",
            logP=2.0,
            mw=300.0,
            psa=80.0,
            solubility_mg_mL=1.0,
            particle_size_um=0.0,
        )
