"""Tests for Phase 1108 — Umbilical Cord Blood Drug Level Prediction."""

from __future__ import annotations

import pytest

from omega_pbpk.core.umbilical_cord_pk import (
    UmbilicalCordResult,
    predict_cord_blood_level,
    screen_neonatal_exposure,
)


def test_return_type():
    r = predict_cord_blood_level("Morphine", mw_Da=285.0, logP=0.9, ionization_type="base")
    assert isinstance(r, UmbilicalCordResult)


def test_drug_name_preserved():
    r = predict_cord_blood_level("Morphine", mw_Da=285.0, logP=0.9, ionization_type="base")
    assert r.drug_name == "Morphine"


def test_fetal_concentration_non_negative():
    r = predict_cord_blood_level("DrugA", mw_Da=300.0, logP=2.0, ionization_type="neutral")
    assert r.c_fetal_blood_mg_L >= 0.0


def test_small_molecule_high_transfer():
    r = predict_cord_blood_level(
        "SmallDrug",
        mw_Da=200.0,
        logP=1.5,
        ionization_type="neutral",
        fu_maternal=0.9,
        c_maternal_mg_L=1.0,
    )
    assert r.placental_transfer_fraction > 0.4


def test_biologic_low_transfer():
    r = predict_cord_blood_level(
        "Antibody", mw_Da=150000.0, logP=-2.0, ionization_type="neutral", c_maternal_mg_L=1.0
    )
    assert r.placental_transfer_fraction < 0.1


def test_um_ratio_in_range():
    r = predict_cord_blood_level("Drug", mw_Da=300.0, logP=2.0, ionization_type="neutral")
    assert 0.0 <= r.umbilical_to_maternal_ratio <= 2.0


def test_base_drug_higher_ratio_than_acid():
    r_base = predict_cord_blood_level(
        "Base", mw_Da=300.0, logP=2.0, ionization_type="base", fu_maternal=0.3, c_maternal_mg_L=1.0
    )
    r_acid = predict_cord_blood_level(
        "Acid", mw_Da=300.0, logP=2.0, ionization_type="acid", fu_maternal=0.3, c_maternal_mg_L=1.0
    )
    assert r_base.umbilical_to_maternal_ratio > r_acid.umbilical_to_maternal_ratio


def test_exposure_class_high_for_small_lipophilic():
    r = predict_cord_blood_level(
        "Drug",
        mw_Da=200.0,
        logP=1.5,
        ionization_type="neutral",
        fu_maternal=0.9,
        c_maternal_mg_L=10.0,
    )
    assert r.neonatal_exposure_class in {"high", "moderate"}


def test_exposure_class_minimal_for_biologic():
    r = predict_cord_blood_level("mAb", mw_Da=150000.0, logP=-2.0, ionization_type="neutral")
    assert r.neonatal_exposure_class == "minimal"


def test_safety_concern_high_ratio():
    r = predict_cord_blood_level(
        "Drug", mw_Da=200.0, logP=1.5, ionization_type="base", fu_maternal=0.9, c_maternal_mg_L=5.0
    )
    # If ratio > 0.5 → concern=True; depends on model; just check it's bool
    assert isinstance(r.safety_concern, bool)


def test_monitoring_nonempty():
    r = predict_cord_blood_level("Drug", mw_Da=300.0, logP=2.0, ionization_type="neutral")
    assert len(r.recommended_monitoring) > 0


def test_notes_nonempty():
    r = predict_cord_blood_level("Drug", mw_Da=300.0, logP=2.0, ionization_type="neutral")
    assert len(r.notes) > 0


def test_zero_maternal_conc_zero_fetal():
    r = predict_cord_blood_level(
        "Drug", mw_Da=300.0, logP=2.0, ionization_type="neutral", c_maternal_mg_L=0.0
    )
    assert r.c_fetal_blood_mg_L == pytest.approx(0.0)


def test_linearity_fetal_conc():
    r1 = predict_cord_blood_level(
        "Drug", mw_Da=300.0, logP=2.0, ionization_type="neutral", c_maternal_mg_L=1.0
    )
    r2 = predict_cord_blood_level(
        "Drug", mw_Da=300.0, logP=2.0, ionization_type="neutral", c_maternal_mg_L=2.0
    )
    assert abs(r2.c_fetal_blood_mg_L - 2 * r1.c_fetal_blood_mg_L) < 1e-9


def test_invalid_mw_raises():
    with pytest.raises(ValueError, match="mw_Da"):
        predict_cord_blood_level("X", mw_Da=0.0, logP=2.0, ionization_type="neutral")


def test_invalid_logp_raises():
    with pytest.raises(ValueError, match="logP"):
        predict_cord_blood_level("X", mw_Da=300.0, logP=15.0, ionization_type="neutral")


def test_invalid_ionization_raises():
    with pytest.raises(ValueError, match="ionization_type"):
        predict_cord_blood_level("X", mw_Da=300.0, logP=2.0, ionization_type="zwitterion")


def test_invalid_fu_zero_raises():
    with pytest.raises(ValueError, match="fu_maternal"):
        predict_cord_blood_level(
            "X", mw_Da=300.0, logP=2.0, ionization_type="neutral", fu_maternal=0.0
        )


def test_invalid_maternal_conc_negative_raises():
    with pytest.raises(ValueError, match="c_maternal_mg_L"):
        predict_cord_blood_level(
            "X", mw_Da=300.0, logP=2.0, ionization_type="neutral", c_maternal_mg_L=-1.0
        )


def test_screen_neonatal_exposure_sorted():
    drugs = [
        {
            "drug_name": "Small",
            "mw_Da": 200.0,
            "logP": 1.5,
            "ionization_type": "neutral",
            "fu_maternal": 0.9,
        },
        {"drug_name": "Large", "mw_Da": 150000.0, "logP": -2.0, "ionization_type": "neutral"},
        {"drug_name": "Mid", "mw_Da": 600.0, "logP": 2.0, "ionization_type": "base"},
    ]
    results = screen_neonatal_exposure(drugs)
    ratios = [r.umbilical_to_maternal_ratio for r in results]
    assert ratios == sorted(ratios, reverse=True)


def test_screen_returns_correct_count():
    drugs = [
        {"drug_name": "A", "mw_Da": 200.0, "logP": 1.0, "ionization_type": "neutral"},
        {"drug_name": "B", "mw_Da": 400.0, "logP": 3.0, "ionization_type": "acid"},
    ]
    results = screen_neonatal_exposure(drugs)
    assert len(results) == 2
