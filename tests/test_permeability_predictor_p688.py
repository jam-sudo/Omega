"""Tests for permeability predictor Phase 688."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.permeability_predictor_p688 import (
    PermeabilityResult,
    classify_permeability,
    predict_caco2_papp,
    predict_mdck_papp,
    predict_pampa_papp,
    predict_permeability,
    screen_permeability,
)

# ---------------------------------------------------------------------------
# predict_caco2_papp
# ---------------------------------------------------------------------------


def test_predict_caco2_papp_returns_float():
    result = predict_caco2_papp(logP=2.0, mw=350.0, psa=70.0)
    assert isinstance(result, float)


def test_predict_caco2_papp_positive():
    result = predict_caco2_papp(logP=2.0, mw=350.0, psa=70.0)
    assert result > 0.0


def test_high_logp_gives_high_papp():
    papp_high = predict_caco2_papp(logP=5.0, mw=350.0, psa=60.0)
    papp_low = predict_caco2_papp(logP=0.0, mw=350.0, psa=60.0)
    assert papp_high > papp_low


def test_high_psa_reduces_papp():
    papp_low_psa = predict_caco2_papp(logP=2.0, mw=350.0, psa=30.0)
    papp_high_psa = predict_caco2_papp(logP=2.0, mw=350.0, psa=150.0)
    assert papp_high_psa < papp_low_psa


def test_high_mw_reduces_papp():
    papp_low_mw = predict_caco2_papp(logP=2.0, mw=300.0, psa=60.0)
    papp_high_mw = predict_caco2_papp(logP=2.0, mw=800.0, psa=60.0)
    assert papp_high_mw < papp_low_mw


def test_charge_reduces_papp():
    papp_neutral = predict_caco2_papp(logP=2.0, mw=350.0, psa=60.0, charge=0)
    papp_charged = predict_caco2_papp(logP=2.0, mw=350.0, psa=60.0, charge=1)
    assert papp_charged < papp_neutral


# ---------------------------------------------------------------------------
# classify_permeability
# ---------------------------------------------------------------------------


def test_classify_high():
    assert classify_permeability(15.0) == "high"


def test_classify_moderate():
    assert classify_permeability(5.0) == "moderate"


def test_classify_low():
    assert classify_permeability(0.5) == "low"


def test_classify_boundary_high():
    assert classify_permeability(10.0) == "high"


def test_classify_boundary_low():
    # papp == 1.0 is "moderate" (>= 1.0)
    assert classify_permeability(1.0) == "moderate"


# ---------------------------------------------------------------------------
# predict_mdck_papp
# ---------------------------------------------------------------------------


def test_mdck_greater_than_caco2():
    papp_caco2 = predict_caco2_papp(logP=2.0, mw=350.0, psa=60.0)
    papp_mdck = predict_mdck_papp(logP=2.0, mw=350.0, psa=60.0)
    assert abs(papp_mdck - papp_caco2 * 2.0) < 1e-9


def test_mdck_positive():
    result = predict_mdck_papp(logP=2.0, mw=350.0, psa=60.0)
    assert result > 0.0


# ---------------------------------------------------------------------------
# predict_pampa_papp
# ---------------------------------------------------------------------------


def test_pampa_positive():
    result = predict_pampa_papp(logP=2.0, mw=350.0, psa=60.0)
    assert result > 0.0


def test_pampa_no_charge_penalty():
    """PAMPA should give higher Papp than Caco-2 for charged molecules
    because there is no charge penalty in PAMPA."""
    papp_caco2_charged = predict_caco2_papp(logP=2.0, mw=350.0, psa=60.0, charge=1)
    papp_pampa = predict_pampa_papp(logP=2.0, mw=350.0, psa=60.0)
    assert papp_pampa >= papp_caco2_charged


# ---------------------------------------------------------------------------
# predict_permeability
# ---------------------------------------------------------------------------


def test_predict_permeability_returns_result():
    result = predict_permeability("DrugA", logP=2.0, mw=350.0, psa=70.0)
    assert isinstance(result, PermeabilityResult)


def test_predict_permeability_fa_in_range():
    result = predict_permeability("DrugB", logP=2.0, mw=350.0, psa=70.0)
    assert 0.0 <= result.fa_predicted <= 1.0


def test_predict_permeability_high_logp_high_fa():
    result_high = predict_permeability("DrugC", logP=5.0, mw=350.0, psa=40.0)
    result_low = predict_permeability("DrugD", logP=-1.0, mw=350.0, psa=120.0)
    assert result_high.fa_predicted > result_low.fa_predicted


def test_predict_permeability_class_matches_papp():
    result = predict_permeability("DrugE", logP=5.0, mw=300.0, psa=30.0)
    expected_class = classify_permeability(result.papp_caco2)
    assert result.permeability_class == expected_class


def test_predict_permeability_bcs_class_matches():
    result = predict_permeability("DrugF", logP=2.0, mw=350.0, psa=70.0)
    assert result.bcs_permeability_class == result.permeability_class


def test_predict_permeability_notes_nonempty():
    result = predict_permeability("DrugG", logP=2.0, mw=350.0, psa=70.0)
    assert len(result.notes) > 0


def test_predict_permeability_raises_negative_mw():
    with pytest.raises(ValueError):
        predict_permeability("DrugH", logP=2.0, mw=-100.0, psa=70.0)


def test_predict_permeability_raises_zero_mw():
    with pytest.raises(ValueError):
        predict_permeability("DrugI", logP=2.0, mw=0.0, psa=70.0)


def test_predict_permeability_raises_negative_psa():
    with pytest.raises(ValueError):
        predict_permeability("DrugJ", logP=2.0, mw=350.0, psa=-10.0)


# ---------------------------------------------------------------------------
# screen_permeability
# ---------------------------------------------------------------------------


def test_screen_permeability_empty_returns_empty():
    result = screen_permeability([])
    assert result == []


def test_screen_permeability_returns_list():
    compounds = [
        {"compound_name": "A", "logP": 2.0, "mw": 300.0, "psa": 60.0},
        {"compound_name": "B", "logP": 4.0, "mw": 400.0, "psa": 80.0},
    ]
    result = screen_permeability(compounds)
    assert isinstance(result, list)
    assert len(result) == 2


def test_screen_permeability_sorted_descending():
    compounds = [
        {"compound_name": "LowPerm", "logP": -1.0, "mw": 500.0, "psa": 140.0},
        {"compound_name": "HighPerm", "logP": 5.0, "mw": 250.0, "psa": 30.0},
        {"compound_name": "MidPerm", "logP": 2.0, "mw": 350.0, "psa": 70.0},
    ]
    result = screen_permeability(compounds)
    assert result[0].papp_caco2 >= result[1].papp_caco2 >= result[2].papp_caco2


def test_screen_permeability_single_compound():
    compounds = [{"compound_name": "DrugX", "logP": 3.0, "mw": 300.0}]
    result = screen_permeability(compounds)
    assert len(result) == 1
    assert isinstance(result[0], PermeabilityResult)
