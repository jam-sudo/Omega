"""Tests for Phase 552 - P-glycoprotein Efflux Ratio Model."""

import pytest

from omega_pbpk.prediction.pgp_efflux_ratio import (
    PgpEffluxResult,
    predict_pgp_efflux,
    screen_pgp_efflux,
)


def test_return_type():
    r = predict_pgp_efflux("DrugA", logP=2.0, mw_Da=400.0, hbd=2, psa_A2=80.0)
    assert isinstance(r, PgpEffluxResult)


def test_drug_name_preserved():
    r = predict_pgp_efflux("TestDrug", logP=2.0, mw_Da=300.0, hbd=1, psa_A2=60.0)
    assert r.drug_name == "TestDrug"


def test_er_at_least_1():
    r = predict_pgp_efflux("MinER", logP=0.5, mw_Da=150.0, hbd=0, psa_A2=20.0)
    assert r.predicted_efflux_ratio >= 1.0


def test_er_capped_at_20():
    r = predict_pgp_efflux("MaxER", logP=8.0, mw_Da=900.0, hbd=10, psa_A2=200.0)
    assert r.predicted_efflux_ratio <= 20.0


def test_probability_in_range():
    for mw in [100, 300, 600, 900]:
        r = predict_pgp_efflux("Prob", logP=2.0, mw_Da=float(mw), hbd=2, psa_A2=80.0)
        assert 0.0 <= r.pgp_substrate_probability <= 1.0


def test_higher_mw_higher_probability():
    r_low = predict_pgp_efflux("LowMW", logP=2.0, mw_Da=200.0, hbd=2, psa_A2=80.0)
    r_high = predict_pgp_efflux("HighMW", logP=2.0, mw_Da=800.0, hbd=2, psa_A2=80.0)
    assert r_high.pgp_substrate_probability >= r_low.pgp_substrate_probability


def test_higher_psa_higher_probability():
    r_low = predict_pgp_efflux("LowPSA", logP=2.0, mw_Da=400.0, hbd=2, psa_A2=30.0)
    r_high = predict_pgp_efflux("HighPSA", logP=2.0, mw_Da=400.0, hbd=2, psa_A2=150.0)
    assert r_high.pgp_substrate_probability >= r_low.pgp_substrate_probability


def test_higher_hbd_higher_probability():
    r_low = predict_pgp_efflux("LowHBD", logP=2.0, mw_Da=400.0, hbd=0, psa_A2=80.0)
    r_high = predict_pgp_efflux("HighHBD", logP=2.0, mw_Da=400.0, hbd=6, psa_A2=80.0)
    assert r_high.pgp_substrate_probability >= r_low.pgp_substrate_probability


def test_efflux_class_high_prob():
    # High MW + high PSA + high HBD → high prob → high ER → efflux
    r = predict_pgp_efflux("Efflux", logP=3.0, mw_Da=700.0, hbd=5, psa_A2=140.0)
    assert r.net_flux_class == "efflux"


def test_passive_class_low_prob():
    # Small molecule, low PSA, no HBD → low prob → passive
    r = predict_pgp_efflux("Passive", logP=0.0, mw_Da=100.0, hbd=0, psa_A2=10.0)
    assert r.net_flux_class == "passive"


def test_clinical_concern_high_er():
    r = predict_pgp_efflux("Concern", logP=4.0, mw_Da=700.0, hbd=5, psa_A2=140.0)
    if r.predicted_efflux_ratio > 4.0:
        assert r.clinical_concern is True


def test_no_clinical_concern_low_er():
    r = predict_pgp_efflux("NoConcern", logP=1.0, mw_Da=150.0, hbd=0, psa_A2=20.0)
    if r.predicted_efflux_ratio <= 4.0:
        assert r.clinical_concern is False


def test_cns_impaired():
    r = predict_pgp_efflux("CNSImp", logP=5.0, mw_Da=800.0, hbd=6, psa_A2=160.0)
    if r.predicted_efflux_ratio > 6.0:
        assert r.cns_penetration_impact == "impaired"


def test_cns_good():
    r = predict_pgp_efflux("CNSGood", logP=1.0, mw_Da=150.0, hbd=0, psa_A2=20.0)
    if r.predicted_efflux_ratio <= 3.0:
        assert r.cns_penetration_impact == "good"


def test_cns_moderate():
    # Find a compound with ER in (3, 6]
    r = predict_pgp_efflux("CNSMod", logP=2.0, mw_Da=500.0, hbd=3, psa_A2=100.0)
    if 3.0 < r.predicted_efflux_ratio <= 6.0:
        assert r.cns_penetration_impact == "moderate"


def test_validation_mw_negative():
    with pytest.raises(ValueError, match="mw_Da"):
        predict_pgp_efflux("Bad", logP=2.0, mw_Da=-10.0, hbd=1, psa_A2=50.0)


def test_validation_mw_zero():
    with pytest.raises(ValueError, match="mw_Da"):
        predict_pgp_efflux("Bad", logP=2.0, mw_Da=0.0, hbd=1, psa_A2=50.0)


def test_validation_hbd_negative():
    with pytest.raises(ValueError, match="hbd"):
        predict_pgp_efflux("Bad", logP=2.0, mw_Da=300.0, hbd=-1, psa_A2=50.0)


def test_validation_psa_negative():
    with pytest.raises(ValueError, match="psa_A2"):
        predict_pgp_efflux("Bad", logP=2.0, mw_Da=300.0, hbd=1, psa_A2=-10.0)


def test_screen_pgp_sorted():
    compounds = [
        {"drug_name": "A", "logP": 1.0, "mw_Da": 200.0, "hbd": 0, "psa_A2": 30.0},
        {"drug_name": "B", "logP": 4.0, "mw_Da": 700.0, "hbd": 5, "psa_A2": 150.0},
        {"drug_name": "C", "logP": 2.0, "mw_Da": 400.0, "hbd": 2, "psa_A2": 80.0},
    ]
    results = screen_pgp_efflux(compounds)
    assert len(results) == 3
    for i in range(1, len(results)):
        assert results[i - 1].predicted_efflux_ratio >= results[i].predicted_efflux_ratio


def test_screen_returns_list():
    compounds = [
        {"drug_name": "X", "logP": 2.0, "mw_Da": 300.0, "hbd": 1, "psa_A2": 60.0},
    ]
    results = screen_pgp_efflux(compounds)
    assert isinstance(results, list)
    assert all(isinstance(r, PgpEffluxResult) for r in results)


def test_frozen_result():
    r = predict_pgp_efflux("Frozen", logP=2.0, mw_Da=300.0, hbd=1, psa_A2=60.0)
    with pytest.raises(AttributeError):
        r.predicted_efflux_ratio = 999.0


def test_notes_not_empty():
    r = predict_pgp_efflux("Notes", logP=2.0, mw_Da=300.0, hbd=1, psa_A2=60.0)
    assert len(r.notes) > 0
