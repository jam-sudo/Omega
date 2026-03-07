"""Tests for Phase 1106 — Drug Plasma Half-Life Prediction."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.prediction.half_life_prediction import (
    HalfLifePredictionResult,
    predict_half_life,
    screen_compounds,
)


def test_return_type():
    r = predict_half_life("DrugA", mw_Da=350.0, logP=2.0, psa_A2=80.0)
    assert isinstance(r, HalfLifePredictionResult)


def test_drug_name_preserved():
    r = predict_half_life("Aspirin", mw_Da=180.0, logP=1.2, psa_A2=63.0)
    assert r.drug_name == "Aspirin"


def test_t_half_positive():
    r = predict_half_life("DrugA", mw_Da=350.0, logP=2.0, psa_A2=80.0)
    assert r.t_half_predicted_h > 0.0


def test_cl_positive():
    r = predict_half_life("DrugA", mw_Da=350.0, logP=2.0, psa_A2=80.0)
    assert r.cl_predicted_L_per_h > 0.0


def test_vd_assumed_fixed():
    r = predict_half_life("DrugA", mw_Da=350.0, logP=2.0, psa_A2=80.0)
    assert r.vd_assumed_L == pytest.approx(60.0)


def test_larger_mw_longer_half_life():
    r_small = predict_half_life("Small", mw_Da=200.0, logP=2.0, psa_A2=80.0)
    r_large = predict_half_life("Large", mw_Da=600.0, logP=2.0, psa_A2=80.0)
    assert r_large.t_half_predicted_h > r_small.t_half_predicted_h


def test_high_logp_increases_half_life():
    r_low = predict_half_life("A", mw_Da=350.0, logP=0.0, psa_A2=80.0)
    r_high = predict_half_life("B", mw_Da=350.0, logP=6.0, psa_A2=80.0)
    assert r_high.t_half_predicted_h > r_low.t_half_predicted_h


def test_negative_logp_reduces_half_life():
    r_neg = predict_half_life("A", mw_Da=350.0, logP=-1.0, psa_A2=80.0)
    r_mid = predict_half_life("B", mw_Da=350.0, logP=2.0, psa_A2=80.0)
    assert r_neg.t_half_predicted_h < r_mid.t_half_predicted_h


def test_acid_longer_than_neutral():
    r_neutral = predict_half_life("A", mw_Da=350.0, logP=2.0, psa_A2=80.0,
                                   ionization_type="neutral")
    r_acid = predict_half_life("B", mw_Da=350.0, logP=2.0, psa_A2=80.0,
                                ionization_type="acid")
    assert r_acid.t_half_predicted_h > r_neutral.t_half_predicted_h


def test_ester_reduces_half_life():
    r_no_ester = predict_half_life("A", mw_Da=350.0, logP=2.0, psa_A2=80.0,
                                    has_ester=False)
    r_ester = predict_half_life("B", mw_Da=350.0, logP=2.0, psa_A2=80.0,
                                 has_ester=True)
    assert r_ester.t_half_predicted_h < r_no_ester.t_half_predicted_h


def test_high_psa_shorter_half_life():
    r_low = predict_half_life("A", mw_Da=350.0, logP=2.0, psa_A2=20.0)
    r_high = predict_half_life("B", mw_Da=350.0, logP=2.0, psa_A2=200.0)
    assert r_high.t_half_predicted_h < r_low.t_half_predicted_h


def test_long_classification():
    r = predict_half_life("Big", mw_Da=25000.0, logP=6.0, psa_A2=20.0,
                           ionization_type="neutral", n_rotatable_bonds=0)
    assert r.half_life_class == "long"


def test_valid_classification_values():
    valid = {"ultra_short", "short", "intermediate", "long"}
    r = predict_half_life("X", mw_Da=350.0, logP=2.0, psa_A2=80.0)
    assert r.half_life_class in valid


def test_cl_consistent_with_t_half():
    r = predict_half_life("X", mw_Da=350.0, logP=2.0, psa_A2=80.0)
    expected_cl = math.log(2) * r.vd_assumed_L / r.t_half_predicted_h
    assert abs(r.cl_predicted_L_per_h - expected_cl) < 1e-9


def test_notes_nonempty():
    r = predict_half_life("X", mw_Da=350.0, logP=2.0, psa_A2=80.0)
    assert len(r.notes) > 0


def test_invalid_mw_zero_raises():
    with pytest.raises(ValueError, match="mw_Da"):
        predict_half_life("X", mw_Da=0.0, logP=2.0, psa_A2=80.0)


def test_invalid_mw_negative_raises():
    with pytest.raises(ValueError, match="mw_Da"):
        predict_half_life("X", mw_Da=-10.0, logP=2.0, psa_A2=80.0)


def test_invalid_logp_raises():
    with pytest.raises(ValueError, match="logP"):
        predict_half_life("X", mw_Da=350.0, logP=15.0, psa_A2=80.0)


def test_invalid_psa_negative_raises():
    with pytest.raises(ValueError, match="psa_A2"):
        predict_half_life("X", mw_Da=350.0, logP=2.0, psa_A2=-5.0)


def test_invalid_rot_bonds_raises():
    with pytest.raises(ValueError, match="n_rotatable_bonds"):
        predict_half_life("X", mw_Da=350.0, logP=2.0, psa_A2=80.0,
                           n_rotatable_bonds=-1)


def test_invalid_ionization_type_raises():
    with pytest.raises(ValueError, match="ionization_type"):
        predict_half_life("X", mw_Da=350.0, logP=2.0, psa_A2=80.0,
                           ionization_type="zwitterion")


def test_screen_returns_list():
    comps = [
        {"drug_name": "A", "mw_Da": 200.0, "logP": 1.0, "psa_A2": 60.0},
        {"drug_name": "B", "mw_Da": 500.0, "logP": 3.0, "psa_A2": 100.0},
    ]
    results = screen_compounds(comps)
    assert isinstance(results, list)
    assert len(results) == 2


def test_screen_sorted_ascending():
    comps = [
        {"drug_name": "A", "mw_Da": 800.0, "logP": 5.0, "psa_A2": 40.0},
        {"drug_name": "B", "mw_Da": 150.0, "logP": -1.0, "psa_A2": 100.0},
        {"drug_name": "C", "mw_Da": 350.0, "logP": 2.0, "psa_A2": 80.0},
    ]
    results = screen_compounds(comps)
    t_halves = [r.t_half_predicted_h for r in results]
    assert t_halves == sorted(t_halves)


def test_screen_empty_raises():
    with pytest.raises(ValueError):
        screen_compounds([])
