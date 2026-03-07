"""Tests for membrane permeability prediction — Phases 175 & 315."""
from __future__ import annotations

import math
import pytest

from omega_pbpk.prediction.permeability_predictor import (
    PermeabilityResult,
    predict_papp_caco2,
    predict_peff_intestinal,
    mdck_to_caco2,
    permeability_classification,
    predict_cnss_permeability,
    predict_permeability,
    screen_permeability,
)


# ---------------------------------------------------------------------------
# predict_papp_caco2 — basic returns
# ---------------------------------------------------------------------------

def test_predict_papp_caco2_returns_result():
    r = predict_papp_caco2(logP=2.0, mw=300.0, hbd=2, hba=4, psa=80.0)
    assert isinstance(r, PermeabilityResult)


def test_predict_papp_caco2_papp_positive():
    r = predict_papp_caco2(logP=2.0, mw=300.0, hbd=2, hba=4, psa=80.0)
    assert r.papp_caco2_cm_s > 0


def test_predict_papp_caco2_papp_lower_bound():
    # Even with very low logP / high MW / high PSA, papp >= 1e-9
    r = predict_papp_caco2(logP=-5.0, mw=900.0, hbd=10, hba=2, psa=300.0)
    assert r.papp_caco2_cm_s >= 1e-9


def test_predict_papp_caco2_papp_upper_bound():
    # Very high logP, zero HBD/PSA, small MW => papp <= 1e-4
    r = predict_papp_caco2(logP=10.0, mw=100.0, hbd=0, hba=0, psa=0.0)
    assert r.papp_caco2_cm_s <= 1e-4


def test_predict_papp_caco2_higher_logp_increases_papp():
    low = predict_papp_caco2(logP=0.0, mw=300.0, hbd=2, hba=4, psa=80.0)
    high = predict_papp_caco2(logP=4.0, mw=300.0, hbd=2, hba=4, psa=80.0)
    assert high.papp_caco2_cm_s > low.papp_caco2_cm_s


def test_predict_papp_caco2_higher_hbd_decreases_papp():
    low_hbd = predict_papp_caco2(logP=2.0, mw=300.0, hbd=0, hba=4, psa=80.0)
    high_hbd = predict_papp_caco2(logP=2.0, mw=300.0, hbd=5, hba=4, psa=80.0)
    assert low_hbd.papp_caco2_cm_s > high_hbd.papp_caco2_cm_s


def test_predict_papp_caco2_higher_psa_decreases_papp():
    low_psa = predict_papp_caco2(logP=2.0, mw=300.0, hbd=2, hba=4, psa=20.0)
    high_psa = predict_papp_caco2(logP=2.0, mw=300.0, hbd=2, hba=4, psa=150.0)
    assert low_psa.papp_caco2_cm_s > high_psa.papp_caco2_cm_s


def test_predict_papp_caco2_higher_mw_decreases_papp():
    small = predict_papp_caco2(logP=2.0, mw=200.0, hbd=2, hba=4, psa=80.0)
    large = predict_papp_caco2(logP=2.0, mw=600.0, hbd=2, hba=4, psa=80.0)
    assert small.papp_caco2_cm_s > large.papp_caco2_cm_s


# ---------------------------------------------------------------------------
# predict_papp_caco2 — validation
# ---------------------------------------------------------------------------

def test_predict_papp_caco2_zero_mw_raises():
    with pytest.raises(ValueError, match="mw"):
        predict_papp_caco2(logP=2.0, mw=0.0, hbd=2, hba=4, psa=80.0)


def test_predict_papp_caco2_negative_mw_raises():
    with pytest.raises(ValueError, match="mw"):
        predict_papp_caco2(logP=2.0, mw=-100.0, hbd=2, hba=4, psa=80.0)


def test_predict_papp_caco2_negative_hbd_raises():
    with pytest.raises(ValueError, match="hbd"):
        predict_papp_caco2(logP=2.0, mw=300.0, hbd=-1, hba=4, psa=80.0)


def test_predict_papp_caco2_negative_hba_raises():
    with pytest.raises(ValueError, match="hba"):
        predict_papp_caco2(logP=2.0, mw=300.0, hbd=2, hba=-1, psa=80.0)


def test_predict_papp_caco2_negative_psa_raises():
    with pytest.raises(ValueError, match="psa"):
        predict_papp_caco2(logP=2.0, mw=300.0, hbd=2, hba=4, psa=-10.0)


# ---------------------------------------------------------------------------
# predict_papp_caco2 — category and BCS class fields
# ---------------------------------------------------------------------------

def test_permeability_category_low():
    # Very low logP, high PSA/HBD => low papp
    r = predict_papp_caco2(logP=-3.0, mw=500.0, hbd=8, hba=4, psa=200.0)
    assert r.permeability_category == "low"


def test_permeability_category_values():
    r = predict_papp_caco2(logP=2.0, mw=300.0, hbd=2, hba=4, psa=80.0)
    assert r.permeability_category in ("low", "medium", "high")


def test_bcs_permeability_class_values():
    r = predict_papp_caco2(logP=2.0, mw=300.0, hbd=2, hba=4, psa=80.0)
    assert r.bcs_permeability_class in ("BCS_high", "BCS_low")


def test_bcs_high_for_very_lipophilic():
    # Very high logP, zero HBD/PSA, small MW => BCS_high expected
    # logPapp = 0.6*8 - 0.01*150 - 0 - 0 + 0 - 5.5 = 4.8 - 1.5 - 5.5 = -2.2 => clamped -4 => 1e-4
    r = predict_papp_caco2(logP=8.0, mw=150.0, hbd=0, hba=0, psa=0.0)
    assert r.bcs_permeability_class == "BCS_high"


# ---------------------------------------------------------------------------
# predict_peff_intestinal
# ---------------------------------------------------------------------------

def test_peff_positive():
    peff = predict_peff_intestinal(1e-6)
    assert peff > 0


def test_peff_log_log_correlation():
    papp = 1e-6
    peff = predict_peff_intestinal(papp)
    expected_log = 0.98 * math.log10(papp) + 0.15
    assert abs(math.log10(peff) - expected_log) < 1e-9


def test_peff_increases_with_papp():
    peff_low = predict_peff_intestinal(1e-7)
    peff_high = predict_peff_intestinal(1e-5)
    assert peff_high > peff_low


def test_peff_zero_raises():
    with pytest.raises(ValueError):
        predict_peff_intestinal(0.0)


# ---------------------------------------------------------------------------
# mdck_to_caco2
# ---------------------------------------------------------------------------

def test_mdck_to_caco2_positive():
    result = mdck_to_caco2(1e-6)
    assert result > 0


def test_mdck_to_caco2_log_log():
    papp_mdck = 5e-6
    result = mdck_to_caco2(papp_mdck)
    expected_log = 0.85 * math.log10(papp_mdck) + 0.3
    assert abs(math.log10(result) - expected_log) < 1e-9


def test_mdck_to_caco2_zero_raises():
    with pytest.raises(ValueError):
        mdck_to_caco2(0.0)


def test_mdck_to_caco2_monotone():
    assert mdck_to_caco2(1e-5) > mdck_to_caco2(1e-7)


# ---------------------------------------------------------------------------
# permeability_classification
# ---------------------------------------------------------------------------

def test_classification_bcs_high():
    assert permeability_classification(2e-5) == "BCS_high"


def test_classification_bcs_low():
    assert permeability_classification(5e-7) == "BCS_low"


def test_classification_boundary():
    # Exactly at threshold (>1e-5 => BCS_high)
    assert permeability_classification(1e-5) == "BCS_low"
    assert permeability_classification(1.0001e-5) == "BCS_high"


# ---------------------------------------------------------------------------
# predict_cnss_permeability
# ---------------------------------------------------------------------------

def test_cnss_positive():
    result = predict_cnss_permeability(logP=2.0, mw=300.0, hbd=1, psa=50.0)
    assert result > 0


def test_cnss_higher_logp_increases():
    low = predict_cnss_permeability(logP=0.0, mw=300.0, hbd=1, psa=50.0)
    high = predict_cnss_permeability(logP=3.0, mw=300.0, hbd=1, psa=50.0)
    assert high > low


def test_cnss_higher_mw_decreases():
    small = predict_cnss_permeability(logP=2.0, mw=200.0, hbd=1, psa=50.0)
    large = predict_cnss_permeability(logP=2.0, mw=500.0, hbd=1, psa=50.0)
    assert small > large


def test_cnss_zero_mw_raises():
    with pytest.raises(ValueError, match="mw"):
        predict_cnss_permeability(logP=2.0, mw=0.0, hbd=1, psa=50.0)


def test_cnss_negative_hbd_raises():
    with pytest.raises(ValueError, match="hbd"):
        predict_cnss_permeability(logP=2.0, mw=300.0, hbd=-1, psa=50.0)


def test_cnss_negative_psa_raises():
    with pytest.raises(ValueError, match="psa"):
        predict_cnss_permeability(logP=2.0, mw=300.0, hbd=1, psa=-5.0)


def test_cnss_log_formula():
    logP, mw, hbd, psa = 2.0, 300.0, 1, 50.0
    expected_log = 0.4 * logP - 0.008 * mw - 0.2 * hbd - 0.003 * psa - 3.0
    result = predict_cnss_permeability(logP=logP, mw=mw, hbd=hbd, psa=psa)
    assert abs(math.log10(result) - expected_log) < 1e-9


# ---------------------------------------------------------------------------
# Legacy API — predict_permeability / screen_permeability (Phase 175)
# ---------------------------------------------------------------------------

def test_legacy_returns_result():
    r = predict_permeability("Drug", "CC", mw=200.0, logP=2.0)
    assert hasattr(r, "papp_cm_s")


def test_legacy_zero_mw_raises():
    with pytest.raises(ValueError, match="mw"):
        predict_permeability("Drug", "CC", mw=0.0, logP=2.0)


def test_legacy_negative_psa_raises():
    with pytest.raises(ValueError, match="psa"):
        predict_permeability("Drug", "CC", mw=200.0, logP=2.0, psa=-1.0)


def test_legacy_high_logp_higher_papp():
    low = predict_permeability("D", "CC", mw=200.0, logP=0.0)
    high = predict_permeability("D", "CC", mw=200.0, logP=4.0)
    assert high.papp_cm_s > low.papp_cm_s


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
    assert len(screen_permeability(compounds)) == 5
