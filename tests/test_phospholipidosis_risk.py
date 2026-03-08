"""Tests for Phase 498: Drug Induced Phospholipidosis Risk."""

from __future__ import annotations

import pytest

from omega_pbpk.risk.phospholipidosis_risk import (
    PhospholipidosisRiskResult,
    predict_phospholipidosis_risk,
    screen_phospholipidosis,
)


def _predict(**kw) -> PhospholipidosisRiskResult:
    defaults = dict(
        drug_name="TestDrug",
        logP=3.0,
        pKa=8.0,
        mw_Da=400.0,
        ring_count=2,
        tertiary_amine=False,
        positive_charge=False,
    )
    defaults.update(kw)
    return predict_phospholipidosis_risk(**defaults)


def test_return_type():
    assert isinstance(_predict(), PhospholipidosisRiskResult)


def test_risk_score_in_range():
    r = _predict()
    assert 0.0 <= r.risk_score <= 100.0


def test_low_risk_profile():
    r = _predict(logP=-1.0, pKa=4.0, mw_Da=200.0, ring_count=0)
    assert r.risk_class == "low"


def test_high_risk_profile():
    r = _predict(
        logP=5.0, pKa=9.0, mw_Da=500.0, ring_count=3, tertiary_amine=True, positive_charge=True
    )
    assert r.risk_class == "high"


def test_risk_classes_valid():
    assert _predict().risk_class in {"low", "moderate", "high"}


def test_cad_pattern_high_pka_logp_ring():
    r = _predict(pKa=8.5, logP=3.5, ring_count=2)
    assert r.cad_pattern is True


def test_no_cad_pattern_neutral():
    r = _predict(pKa=5.0, logP=1.0, ring_count=0)
    assert r.cad_pattern is False


def test_lysosomal_trapping_high_pka():
    r = _predict(pKa=9.0)
    assert r.lysosomal_trapping_factor > 10.0


def test_lysosomal_trapping_low_pka_equals_one():
    r = _predict(pKa=5.0)
    assert r.lysosomal_trapping_factor == pytest.approx(1.0)


def test_lysosomal_trapping_clamped_max():
    r = _predict(pKa=12.0)
    assert r.lysosomal_trapping_factor <= 1000.0


def test_preclinical_flag_high_score():
    r = _predict(
        logP=5.0, pKa=9.0, mw_Da=500.0, ring_count=3, tertiary_amine=True, positive_charge=True
    )
    assert r.preclinical_flag is True


def test_no_preclinical_flag_low_score():
    r = _predict(logP=-1.0, pKa=4.0, mw_Da=200.0, ring_count=0)
    assert r.preclinical_flag is False


def test_mitigation_none_for_low():
    r = _predict(logP=-1.0, pKa=4.0, mw_Da=200.0, ring_count=0)
    assert r.mitigation_strategy == "none"


def test_mitigation_avoid_for_high():
    r = _predict(
        logP=5.0, pKa=9.0, mw_Da=500.0, ring_count=3, tertiary_amine=True, positive_charge=True
    )
    assert r.mitigation_strategy == "avoid_or_redesign"


def test_drug_name_stored():
    r = _predict(drug_name="Amiodarone")
    assert r.drug_name == "Amiodarone"


def test_invalid_mw_zero():
    with pytest.raises(ValueError, match="mw_Da"):
        _predict(mw_Da=0.0)


def test_invalid_mw_negative():
    with pytest.raises(ValueError, match="mw_Da"):
        _predict(mw_Da=-100.0)


def test_invalid_ring_count_negative():
    with pytest.raises(ValueError, match="ring_count"):
        _predict(ring_count=-1)


def test_screen_returns_list():
    compounds = [
        {"drug_name": "A", "logP": 5.0, "pKa": 9.0, "mw_Da": 500.0},
        {"drug_name": "B", "logP": 1.0, "pKa": 4.0, "mw_Da": 200.0},
    ]
    results = screen_phospholipidosis(compounds)
    assert len(results) == 2


def test_screen_sorted_descending():
    compounds = [
        {"drug_name": "Low", "logP": -1.0, "pKa": 4.0, "mw_Da": 200.0},
        {"drug_name": "High", "logP": 5.0, "pKa": 9.0, "mw_Da": 500.0},
    ]
    results = screen_phospholipidosis(compounds)
    assert results[0].risk_score >= results[1].risk_score


def test_screen_single_compound():
    compounds = [{"drug_name": "X", "logP": 3.0, "pKa": 7.5, "mw_Da": 350.0}]
    results = screen_phospholipidosis(compounds)
    assert len(results) == 1
