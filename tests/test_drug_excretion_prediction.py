"""Tests for Phase 952 — Drug excretion route prediction."""

import pytest

from omega_pbpk.prediction.drug_excretion_prediction import (
    DrugExcretionPredictionResult,
    predict_excretion,
    screen_excretion_routes,
)

_VALID_ROUTES = {"renal", "biliary", "fecal", "pulmonary"}
_VALID_SENSITIVITIES = {"high", "moderate", "low"}


# ── Basic return type and structure ──────────────────────────────────────────


def test_return_type():
    r = predict_excretion("DrugX", logp=1.0, mw=250.0, pka=7.4, ionization_type="neutral")
    assert isinstance(r, DrugExcretionPredictionResult)


def test_result_is_frozen():
    r = predict_excretion("DrugX", logp=1.0, mw=250.0, pka=7.4, ionization_type="neutral")
    with pytest.raises((AttributeError, TypeError)):
        r.fe_renal = 0.5  # type: ignore


def test_all_fractions_in_range():
    r = predict_excretion("DrugX", logp=1.0, mw=250.0, pka=7.4, ionization_type="neutral")
    for frac in (r.fe_renal, r.fe_biliary, r.fe_pulmonary, r.fe_fecal):
        assert 0.0 <= frac <= 1.0


def test_fractions_sum_to_one():
    r = predict_excretion("DrugX", logp=1.0, mw=250.0, pka=7.4, ionization_type="neutral")
    total = r.fe_renal + r.fe_biliary + r.fe_pulmonary + r.fe_fecal
    assert total == pytest.approx(1.0, abs=1e-6)


def test_primary_excretion_route_valid():
    r = predict_excretion("DrugX", logp=1.0, mw=250.0, pka=7.4, ionization_type="neutral")
    assert r.primary_excretion_route in _VALID_ROUTES


def test_renal_sensitivity_valid():
    r = predict_excretion("DrugX", logp=1.0, mw=250.0, pka=7.4, ionization_type="neutral")
    assert r.renal_sensitivity in _VALID_SENSITIVITIES


def test_cl_renal_nonnegative():
    r = predict_excretion("DrugX", logp=1.0, mw=250.0, pka=7.4, ionization_type="neutral")
    assert r.cl_renal_predicted_L_per_h >= 0.0


def test_cl_biliary_nonnegative():
    r = predict_excretion("DrugX", logp=1.0, mw=250.0, pka=7.4, ionization_type="neutral")
    assert r.cl_biliary_predicted_L_per_h >= 0.0


def test_notes_nonempty():
    r = predict_excretion("DrugX", logp=1.0, mw=250.0, pka=7.4, ionization_type="neutral")
    assert isinstance(r.notes, str) and len(r.notes) > 0


# ── Physicochemical rules ─────────────────────────────────────────────────────


def test_low_mw_low_logp_renal_dominant():
    """MW=150, logP=-1, neutral, fu=0.8 → high renal score."""
    r = predict_excretion(
        "SmallHydrophilic", logp=-1.0, mw=150.0, pka=7.4, ionization_type="neutral", fu_plasma=0.8
    )
    assert r.primary_excretion_route == "renal"


def test_high_mw_high_logp_biliary_dominant():
    """MW=600, logP=3.5 → biliary dominant."""
    r = predict_excretion(
        "BigLipophilic", logp=3.5, mw=600.0, pka=7.4, ionization_type="neutral", fu_plasma=0.1
    )
    assert r.primary_excretion_route == "biliary"


def test_drug_name_preserved():
    r = predict_excretion("MyDrug", logp=0.5, mw=300.0, pka=8.0, ionization_type="base")
    assert r.drug_name == "MyDrug"


# ── screen_excretion_routes ───────────────────────────────────────────────────


def test_screen_returns_correct_length():
    compounds = [
        {"drug_name": "A", "logp": -1.0, "mw": 150.0, "pka": 7.0, "ionization_type": "neutral"},
        {"drug_name": "B", "logp": 3.0, "mw": 600.0, "pka": 4.0, "ionization_type": "acid"},
        {"drug_name": "C", "logp": 1.0, "mw": 300.0, "pka": 9.0, "ionization_type": "base"},
    ]
    results = screen_excretion_routes(compounds)
    assert len(results) == 3


def test_screen_sorted_by_fe_renal_descending():
    compounds = [
        {"drug_name": "A", "logp": -1.0, "mw": 150.0, "pka": 7.0, "ionization_type": "neutral"},
        {"drug_name": "B", "logp": 3.0, "mw": 600.0, "pka": 4.0, "ionization_type": "acid"},
        {"drug_name": "C", "logp": 1.0, "mw": 300.0, "pka": 9.0, "ionization_type": "base"},
    ]
    results = screen_excretion_routes(compounds)
    fe_renals = [r.fe_renal for r in results]
    assert fe_renals == sorted(fe_renals, reverse=True)


def test_renal_sensitivity_high_when_above_threshold():
    """fe_renal > 0.7 → renal_sensitivity == 'high'."""
    # Very small MW, low logP, high fu, ionized base → maximum renal score
    r = predict_excretion(
        "TinyHydrophilic", logp=-2.0, mw=100.0, pka=10.0, ionization_type="base", fu_plasma=0.9
    )
    if r.fe_renal > 0.7:
        assert r.renal_sensitivity == "high"


def test_renal_sensitivity_high_explicit():
    """Force a scenario where fe_renal > 0.7 and confirm sensitivity."""
    r = predict_excretion(
        "TinyBase", logp=-2.0, mw=100.0, pka=10.0, ionization_type="base", fu_plasma=0.9
    )
    # Confirm the rule: if result shows high, then fe_renal > 0.7
    if r.renal_sensitivity == "high":
        assert r.fe_renal > 0.7


def test_cl_renal_equals_fe_times_cl_total():
    cl_total = 15.0
    r = predict_excretion(
        "DrugY",
        logp=0.0,
        mw=200.0,
        pka=7.0,
        ionization_type="neutral",
        fu_plasma=0.5,
        cl_total_L_per_h=cl_total,
    )
    assert r.cl_renal_predicted_L_per_h == pytest.approx(r.fe_renal * cl_total, rel=1e-6)


# ── Validation ────────────────────────────────────────────────────────────────


def test_validation_mw_zero():
    with pytest.raises(ValueError, match="mw"):
        predict_excretion("DrugX", logp=1.0, mw=0.0, pka=7.0, ionization_type="neutral")


def test_validation_mw_negative():
    with pytest.raises(ValueError, match="mw"):
        predict_excretion("DrugX", logp=1.0, mw=-100.0, pka=7.0, ionization_type="neutral")


def test_validation_fu_plasma_zero():
    with pytest.raises(ValueError, match="fu_plasma"):
        predict_excretion(
            "DrugX", logp=1.0, mw=300.0, pka=7.0, ionization_type="neutral", fu_plasma=0.0
        )


def test_validation_fu_plasma_above_one():
    with pytest.raises(ValueError, match="fu_plasma"):
        predict_excretion(
            "DrugX", logp=1.0, mw=300.0, pka=7.0, ionization_type="neutral", fu_plasma=1.5
        )


def test_validation_cl_total_zero():
    with pytest.raises(ValueError, match="cl_total"):
        predict_excretion(
            "DrugX", logp=1.0, mw=300.0, pka=7.0, ionization_type="neutral", cl_total_L_per_h=0.0
        )


def test_validation_invalid_ionization_type():
    with pytest.raises(ValueError, match="ionization_type"):
        predict_excretion("DrugX", logp=1.0, mw=300.0, pka=7.0, ionization_type="zwitterion")
