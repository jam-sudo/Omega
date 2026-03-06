"""Tests for high-concentration mAb formulation viscosity prediction (Phase 158)."""

import pytest

from omega_pbpk.prediction.formulation_viscosity import (
    ViscosityResult,
    predict_viscosity,
    screen_concentrations,
)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_zero_concentration_raises():
    with pytest.raises(ValueError, match="concentration_mg_mL"):
        predict_viscosity("TestAb", concentration_mg_mL=0.0)


def test_negative_concentration_raises():
    with pytest.raises(ValueError, match="concentration_mg_mL"):
        predict_viscosity("TestAb", concentration_mg_mL=-10.0)


def test_zero_mw_raises():
    with pytest.raises(ValueError, match="mw_kDa"):
        predict_viscosity("TestAb", concentration_mg_mL=50.0, mw_kDa=0.0)


def test_negative_temperature_raises():
    with pytest.raises(ValueError, match="temperature_C"):
        predict_viscosity("TestAb", concentration_mg_mL=50.0, temperature_C=-10.0)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

def test_returns_result():
    r = predict_viscosity("TestAb", concentration_mg_mL=50.0)
    assert isinstance(r, ViscosityResult)


def test_result_fields():
    r = predict_viscosity("TestAb", concentration_mg_mL=50.0)
    assert r.protein_name == "TestAb"
    assert r.concentration_mg_mL == 50.0
    assert r.viscosity_cP > 0
    assert r.viscosity_category in ("low", "moderate", "high")
    assert isinstance(r.sc_injectable, bool)
    assert r.relative_viscosity >= 1.0
    assert 0.0 <= r.excipient_effect <= 0.40


# ---------------------------------------------------------------------------
# Concentration effect
# ---------------------------------------------------------------------------

def test_viscosity_increases_with_concentration():
    low = predict_viscosity("TestAb", concentration_mg_mL=10.0)
    high = predict_viscosity("TestAb", concentration_mg_mL=150.0)
    assert high.viscosity_cP > low.viscosity_cP


def test_low_conc_sc_injectable():
    r = predict_viscosity("TestAb", concentration_mg_mL=10.0)
    assert r.sc_injectable is True


def test_very_high_conc_higher_than_low_conc():
    low = predict_viscosity("TestAb", concentration_mg_mL=5.0)
    high = predict_viscosity("TestAb", concentration_mg_mL=200.0)
    assert high.viscosity_cP > low.viscosity_cP


# ---------------------------------------------------------------------------
# Temperature effect
# ---------------------------------------------------------------------------

def test_higher_temp_lower_viscosity():
    cold = predict_viscosity("TestAb", concentration_mg_mL=100.0, temperature_C=4.0)
    warm = predict_viscosity("TestAb", concentration_mg_mL=100.0, temperature_C=37.0)
    assert warm.viscosity_cP < cold.viscosity_cP


# ---------------------------------------------------------------------------
# Excipient effects
# ---------------------------------------------------------------------------

def test_arginine_reduces_viscosity():
    no_arg = predict_viscosity("TestAb", concentration_mg_mL=120.0, arginine_mM=0.0)
    with_arg = predict_viscosity("TestAb", concentration_mg_mL=120.0, arginine_mM=150.0)
    assert with_arg.viscosity_cP < no_arg.viscosity_cP


def test_nacl_reduces_viscosity():
    no_nacl = predict_viscosity("TestAb", concentration_mg_mL=100.0, nacl_mM=0.0)
    with_nacl = predict_viscosity("TestAb", concentration_mg_mL=100.0, nacl_mM=150.0)
    assert with_nacl.viscosity_cP < no_nacl.viscosity_cP


def test_excipient_effect_bounded():
    r = predict_viscosity("TestAb", concentration_mg_mL=100.0,
                          arginine_mM=300.0, nacl_mM=300.0)
    assert 0.0 <= r.excipient_effect <= 0.40


# ---------------------------------------------------------------------------
# Self-association index
# ---------------------------------------------------------------------------

def test_self_association_nonnegative():
    r = predict_viscosity("TestAb", concentration_mg_mL=50.0)
    assert r.self_association_index >= 0.0


def test_higher_hydrophobicity_increases_kd():
    low_lp = predict_viscosity("TestAb", concentration_mg_mL=50.0, logP=0.0)
    high_lp = predict_viscosity("TestAb", concentration_mg_mL=50.0, logP=3.0)
    assert high_lp.self_association_index > low_lp.self_association_index


# ---------------------------------------------------------------------------
# SC injectability threshold
# ---------------------------------------------------------------------------

def test_sc_injectable_below_20cP():
    r = predict_viscosity("TestAb", concentration_mg_mL=1.0)
    assert r.viscosity_cP < 20.0
    assert r.sc_injectable is True


def test_concentration_for_20cP_positive():
    r = predict_viscosity("TestAb", concentration_mg_mL=50.0)
    assert r.concentration_for_20cP > 0


# ---------------------------------------------------------------------------
# Category values
# ---------------------------------------------------------------------------

def test_viscosity_category_values():
    r = predict_viscosity("TestAb", concentration_mg_mL=50.0)
    assert r.viscosity_category in ("low", "moderate", "high")


def test_low_category_sc_injectable():
    r = predict_viscosity("TestAb", concentration_mg_mL=5.0)
    if r.viscosity_category == "low":
        assert r.sc_injectable is True


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------

def test_notes_not_empty():
    r = predict_viscosity("TestAb", concentration_mg_mL=100.0)
    assert len(r.notes) > 0
    assert "TestAb" in r.notes


# ---------------------------------------------------------------------------
# screen_concentrations
# ---------------------------------------------------------------------------

def test_screen_returns_correct_count():
    concs = [10.0, 50.0, 100.0, 150.0, 200.0]
    results = screen_concentrations("TestAb", concs)
    assert len(results) == len(concs)


def test_screen_results_monotone():
    concs = [10.0, 50.0, 100.0]
    results = screen_concentrations("TestAb", concs)
    visc = [r.viscosity_cP for r in results]
    assert visc[0] < visc[1] < visc[2]


def test_screen_kwargs_passed():
    concs = [50.0]
    results = screen_concentrations("TestAb", concs, temperature_C=37.0)
    assert results[0].temperature_C == 37.0
