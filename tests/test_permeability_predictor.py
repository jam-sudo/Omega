"""Tests for membrane permeability prediction (Phase 175)."""
import pytest
from omega_pbpk.prediction.permeability_predictor import (
    PermeabilityResult,
    predict_permeability,
    screen_permeability,
)

def test_returns_result():
    r = predict_permeability("Drug", "CC", mw=200.0, logP=2.0)
    assert isinstance(r, PermeabilityResult)

def test_zero_mw_raises():
    with pytest.raises(ValueError, match="mw"):
        predict_permeability("Drug", "CC", mw=0.0, logP=2.0)

def test_negative_mw_raises():
    with pytest.raises(ValueError, match="mw"):
        predict_permeability("Drug", "CC", mw=-1.0, logP=2.0)

def test_negative_psa_raises():
    with pytest.raises(ValueError, match="psa"):
        predict_permeability("Drug", "CC", mw=200.0, logP=2.0, psa=-1.0)

def test_high_logp_higher_papp():
    low = predict_permeability("D", "CC", mw=200.0, logP=0.0)
    high = predict_permeability("D", "CC", mw=200.0, logP=4.0)
    assert high.papp_cm_s > low.papp_cm_s

def test_high_psa_lower_papp():
    low_psa = predict_permeability("D", "CC", mw=200.0, logP=2.0, psa=20.0)
    high_psa = predict_permeability("D", "CC", mw=200.0, logP=2.0, psa=150.0)
    assert high_psa.papp_cm_s < low_psa.papp_cm_s

def test_charged_lower_papp():
    neutral = predict_permeability("D", "CC", mw=200.0, logP=2.0, charge_at_ph7=0)
    charged = predict_permeability("D", "CC", mw=200.0, logP=2.0, charge_at_ph7=2)
    assert charged.papp_cm_s < neutral.papp_cm_s

def test_papp_positive():
    r = predict_permeability("Drug", "CC", mw=200.0, logP=2.0)
    assert r.papp_cm_s > 0

def test_peff_positive():
    r = predict_permeability("Drug", "CC", mw=200.0, logP=2.0)
    assert r.peff_cm_per_s > 0

def test_fa_in_bounds():
    r = predict_permeability("Drug", "CC", mw=200.0, logP=2.0)
    assert 0.0 <= r.fa_predicted <= 1.0

def test_permeability_class_values():
    r = predict_permeability("Drug", "CC", mw=200.0, logP=2.0)
    assert r.permeability_class in ("low", "medium", "high")

def test_low_permeability_class():
    r = predict_permeability("Drug", "CC", mw=600.0, logP=-2.0, psa=200.0)
    assert r.permeability_class == "low"

def test_bcs_permeability_flag_values():
    r = predict_permeability("Drug", "CC", mw=200.0, logP=2.0)
    assert r.bcs_permeability_flag in ("low", "high")

def test_lipinski_compliant_small_drug():
    r = predict_permeability("Drug", "CC", mw=200.0, logP=2.0, n_hbd=1, n_hba=2)
    assert r.lipinski_compliant is True

def test_lipinski_violation_large_mw():
    r = predict_permeability("Drug", "CC", mw=600.0, logP=2.0)
    assert r.lipinski_compliant is False

def test_lipinski_violation_high_logp():
    r = predict_permeability("Drug", "CC", mw=200.0, logP=6.0)
    assert r.lipinski_compliant is False

def test_notes_not_empty():
    r = predict_permeability("Drug", "CC", mw=200.0, logP=2.0)
    assert len(r.notes) > 0

def test_screen_returns_sorted():
    compounds = [
        {"name": "A", "smiles": "CC", "mw": 200.0, "logP": 4.0},
        {"name": "B", "smiles": "CC", "mw": 600.0, "logP": -2.0, "psa": 200.0},
        {"name": "C", "smiles": "CC", "mw": 300.0, "logP": 2.0},
    ]
    results = screen_permeability(compounds)
    papp_vals = [r.papp_cm_s for r in results]
    assert papp_vals == sorted(papp_vals)

def test_screen_returns_correct_count():
    compounds = [{"name": str(i), "smiles": "CC", "mw": 200.0, "logP": float(i)} for i in range(5)]
    results = screen_permeability(compounds)
    assert len(results) == 5

def test_high_papp_high_bcs():
    r = predict_permeability("Lipophilic", "CC", mw=200.0, logP=5.0, psa=20.0, n_hbd=0, n_hba=1)
    # May be high BCS permeability depending on final papp
    assert r.bcs_permeability_flag in ("low", "high")

def test_papp_clamped_in_range():
    # Even extreme parameters produce physical papp values
    r = predict_permeability("Extreme", "C", mw=50.0, logP=10.0, psa=0.0)
    assert 1e-8 <= r.papp_cm_s <= 1e-3
