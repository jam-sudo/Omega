"""Tests for Phase 986 — drug_efflux_transporter_prediction_p986.py"""

import pytest
from omega_pbpk.prediction.drug_efflux_transporter_prediction_p986 import (
    EffluxTransporterResult,
    predict_efflux_transporters,
    screen_efflux_transporters,
)


def test_returns_result_type():
    result = predict_efflux_transporters("Drug_A", logp=2.0, mw=450.0)
    assert isinstance(result, EffluxTransporterResult)


def test_all_probs_in_range():
    result = predict_efflux_transporters("Drug_A", logp=2.0, mw=450.0)
    for prob in [
        result.pgp_substrate_prob,
        result.pgp_inhibitor_prob,
        result.bcrp_substrate_prob,
        result.bcrp_inhibitor_prob,
        result.mrp2_substrate_prob,
        result.mate_substrate_prob,
    ]:
        assert 0.0 <= prob <= 1.0


def test_high_mw_logp_pgp_substrate():
    # mw > 400, logP in 1-4 → high P-gp substrate
    result = predict_efflux_transporters("HighMW", logp=2.5, mw=500.0)
    assert result.pgp_substrate_prob >= 0.5


def test_acid_drug_high_bcrp_substrate():
    # acidic drug, pKa in 3-7 → high BCRP substrate
    result = predict_efflux_transporters(
        "AcidDrug", logp=0.5, mw=350.0, pka=5.0, ionization_type="acid"
    )
    assert result.bcrp_substrate_prob >= 0.5


def test_basic_drug_high_mate_substrate():
    # basic drug, pKa > 7, mw < 400, logP < 2 → high MATE substrate
    result = predict_efflux_transporters(
        "BaseDrug", logp=1.0, mw=300.0, pka=9.0, ionization_type="base"
    )
    assert result.mate_substrate_prob >= 0.5


def test_dominant_transporter_is_string():
    result = predict_efflux_transporters("Drug", logp=2.0, mw=450.0)
    assert isinstance(result.dominant_efflux_transporter, str)
    assert result.dominant_efflux_transporter in {"P-gp", "BCRP", "MRP2", "MATE"}


def test_bbb_efflux_risk_valid():
    result = predict_efflux_transporters("Drug", logp=2.0, mw=450.0)
    assert result.bbb_efflux_risk in {"low", "moderate", "high"}


def test_oral_efflux_risk_valid():
    result = predict_efflux_transporters("Drug", logp=2.0, mw=450.0)
    assert result.oral_efflux_risk in {"low", "moderate", "high"}


def test_high_pgp_substrate_high_bbb_risk():
    # mw > 400, logP in 1-4, hbd > 2, hba > 4 → pgp_substrate near 0.7+
    result = predict_efflux_transporters(
        "HighPgp", logp=2.0, mw=500.0, hbd=3, hba=5
    )
    assert result.bbb_efflux_risk in {"moderate", "high"}


def test_low_mw_low_logp_low_pgp():
    # Small, hydrophilic drug → low P-gp substrate
    result = predict_efflux_transporters("SmallHydro", logp=-1.0, mw=200.0)
    assert result.pgp_substrate_prob < 0.5


def test_mrp2_acid_substrate():
    result = predict_efflux_transporters(
        "Acid", logp=0.0, mw=500.0, pka=4.0, ionization_type="acid", hba=6
    )
    assert result.mrp2_substrate_prob >= 0.4


def test_invalid_ionization_type_raises():
    with pytest.raises(ValueError, match="ionization_type"):
        predict_efflux_transporters("Drug", logp=2.0, mw=400.0, ionization_type="cation")


def test_screen_returns_list():
    drugs = [
        {"drug_name": "A", "logp": 2.0, "mw": 500.0},
        {"drug_name": "B", "logp": -1.0, "mw": 200.0},
    ]
    results = screen_efflux_transporters(drugs)
    assert isinstance(results, list)
    assert len(results) == 2


def test_screen_sorted_by_pgp_descending():
    drugs = [
        {"drug_name": "Low", "logp": -1.0, "mw": 200.0},
        {"drug_name": "High", "logp": 2.5, "mw": 500.0, "hbd": 3, "hba": 5},
    ]
    results = screen_efflux_transporters(drugs)
    assert results[0].pgp_substrate_prob >= results[1].pgp_substrate_prob


def test_screen_empty_list():
    results = screen_efflux_transporters([])
    assert results == []


def test_pgp_inhibitor_high_logp_base():
    result = predict_efflux_transporters(
        "BaseHigh", logp=4.0, mw=550.0, ionization_type="base"
    )
    assert result.pgp_inhibitor_prob >= 0.5


def test_bcrp_inhibitor_high_logp():
    result = predict_efflux_transporters("HiLogP", logp=5.0, mw=450.0)
    assert result.bcrp_inhibitor_prob >= 0.3


def test_notes_is_string():
    result = predict_efflux_transporters("Drug", logp=2.0, mw=400.0)
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


def test_result_is_frozen():
    result = predict_efflux_transporters("Drug", logp=2.0, mw=400.0)
    with pytest.raises(Exception):
        result.pgp_substrate_prob = 0.5  # type: ignore[misc]


def test_neutral_drug_low_mate():
    result = predict_efflux_transporters(
        "Neutral", logp=2.0, mw=400.0, ionization_type="neutral"
    )
    # Neutral drug, pKa default 7.0: base condition fails (not "base")
    # base prob 0.1 + 0.2 (mw<400? no, mw==400) + 0.15 (logp<2? no)
    # Actually mw=400 is not < 400, logp=2 is not < 2
    assert result.mate_substrate_prob == pytest.approx(0.1, abs=0.01)


def test_screen_with_optional_params():
    drugs = [
        {
            "drug_name": "Full",
            "logp": 3.0,
            "mw": 480.0,
            "pka": 8.5,
            "ionization_type": "base",
            "hbd": 3,
            "hba": 6,
        }
    ]
    results = screen_efflux_transporters(drugs)
    assert len(results) == 1
    assert results[0].drug_name == "Full"
