"""Tests for Phase 255: drug_permeability_classifier_p255."""

import pytest

from omega_pbpk.prediction.drug_permeability_classifier_p255 import (
    PermeabilityClassificationResult,
    classify_permeability,
    screen_compound_series,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _high_logp_compound():
    """High logP, low PSA compound — should yield high permeability."""
    return classify_permeability("HighPermDrug", mw=300.0, logp=5.0, psa_A2=20.0, hbd_count=1)


def _high_psa_compound():
    """High PSA compound — should yield low permeability."""
    return classify_permeability("LowPermDrug", mw=400.0, logp=0.0, psa_A2=200.0, hbd_count=8)


# ---------------------------------------------------------------------------
# Return type tests
# ---------------------------------------------------------------------------

def test_return_type():
    result = classify_permeability("Aspirin", mw=180.16, logp=1.2, psa_A2=63.6, hbd_count=1)
    assert isinstance(result, PermeabilityClassificationResult)


def test_result_is_frozen():
    result = classify_permeability("TestDrug", mw=250.0, logp=2.0, psa_A2=50.0)
    with pytest.raises((AttributeError, TypeError)):
        result.drug_name = "Other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Positive values
# ---------------------------------------------------------------------------

def test_papp_caco2_positive():
    result = classify_permeability("Drug", mw=300.0, logp=2.5, psa_A2=60.0)
    assert result.papp_caco2_cm_s > 0


def test_papp_pampa_positive():
    result = classify_permeability("Drug", mw=300.0, logp=2.5, psa_A2=60.0)
    assert result.papp_pampa_cm_s > 0


def test_papp_pampa_greater_than_caco2():
    """PAMPA = 1.5 * Caco-2."""
    result = classify_permeability("Drug", mw=300.0, logp=2.5, psa_A2=60.0)
    assert abs(result.papp_pampa_cm_s - 1.5 * result.papp_caco2_cm_s) < 1e-20


# ---------------------------------------------------------------------------
# High logP → high permeability
# ---------------------------------------------------------------------------

def test_high_logp_gives_high_permeability_caco2():
    result = _high_logp_compound()
    assert result.permeability_class_caco2 == "high"


def test_high_logp_gives_high_permeability_pampa():
    result = _high_logp_compound()
    assert result.permeability_class_pampa == "high"


def test_high_logp_gives_class1():
    result = _high_logp_compound()
    assert result.absorption_class == "class1"


# ---------------------------------------------------------------------------
# High PSA → low permeability
# ---------------------------------------------------------------------------

def test_high_psa_gives_low_permeability_caco2():
    result = _high_psa_compound()
    assert result.permeability_class_caco2 == "low"


def test_high_psa_gives_class3():
    result = _high_psa_compound()
    assert result.absorption_class == "class3"


# ---------------------------------------------------------------------------
# absorption_class valid values
# ---------------------------------------------------------------------------

def test_absorption_class_valid_values():
    valid = {"class1", "class2", "class3"}
    for logp in [-2.0, 0.0, 2.0, 4.0]:
        result = classify_permeability("Drug", mw=300.0, logp=logp, psa_A2=80.0)
        assert result.absorption_class in valid


# ---------------------------------------------------------------------------
# bcs_permeability_class valid values
# ---------------------------------------------------------------------------

def test_bcs_permeability_class_valid_values():
    valid = {"high", "low"}
    for logp in [-3.0, 0.0, 3.0, 6.0]:
        result = classify_permeability("Drug", mw=350.0, logp=logp, psa_A2=90.0)
        assert result.bcs_permeability_class in valid


def test_high_logp_gives_bcs_high():
    result = _high_logp_compound()
    assert result.bcs_permeability_class == "high"


# ---------------------------------------------------------------------------
# passive_permeability_score in [0, 100]
# ---------------------------------------------------------------------------

def test_passive_permeability_score_range_low():
    result = _high_psa_compound()
    assert 0.0 <= result.passive_permeability_score <= 100.0


def test_passive_permeability_score_range_high():
    result = _high_logp_compound()
    assert 0.0 <= result.passive_permeability_score <= 100.0


def test_passive_permeability_score_range_middle():
    result = classify_permeability("Middle", mw=350.0, logp=2.0, psa_A2=80.0)
    assert 0.0 <= result.passive_permeability_score <= 100.0


# ---------------------------------------------------------------------------
# Efflux risk
# ---------------------------------------------------------------------------

def test_efflux_risk_true_for_low_logp_high_psa_small_mw():
    # logP < 2, psa > 60, mw < 500
    result = classify_permeability("EfFluxDrug", mw=300.0, logp=1.0, psa_A2=80.0)
    assert result.efflux_risk is True


def test_efflux_risk_false_for_high_logp():
    result = classify_permeability("NoEfFlux", mw=300.0, logp=4.0, psa_A2=80.0)
    assert result.efflux_risk is False


# ---------------------------------------------------------------------------
# screen_compound_series
# ---------------------------------------------------------------------------

def test_screen_returns_list():
    compounds = [
        {"name": "A", "mw": 300.0, "logp": 2.0, "psa_A2": 60.0},
        {"name": "B", "mw": 400.0, "logp": 4.0, "psa_A2": 30.0},
        {"name": "C", "mw": 500.0, "logp": -1.0, "psa_A2": 150.0},
    ]
    results = screen_compound_series(compounds)
    assert isinstance(results, list)
    assert len(results) == 3


def test_screen_sorted_descending():
    compounds = [
        {"name": "LowPerm", "mw": 500.0, "logp": -2.0, "psa_A2": 180.0},
        {"name": "HighPerm", "mw": 250.0, "logp": 5.0, "psa_A2": 15.0},
        {"name": "MidPerm", "mw": 350.0, "logp": 2.0, "psa_A2": 70.0},
    ]
    results = screen_compound_series(compounds)
    scores = [r.passive_permeability_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_screen_optional_hbd_count():
    compounds = [
        {"name": "A", "mw": 300.0, "logp": 2.0, "psa_A2": 60.0, "hbd_count": 3},
        {"name": "B", "mw": 300.0, "logp": 2.0, "psa_A2": 60.0},
    ]
    results = screen_compound_series(compounds)
    assert len(results) == 2


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

def test_validation_mw_zero():
    with pytest.raises(ValueError, match="mw"):
        classify_permeability("Bad", mw=0.0, logp=2.0, psa_A2=60.0)


def test_validation_mw_negative():
    with pytest.raises(ValueError, match="mw"):
        classify_permeability("Bad", mw=-100.0, logp=2.0, psa_A2=60.0)


def test_validation_logp_too_high():
    with pytest.raises(ValueError, match="logp"):
        classify_permeability("Bad", mw=300.0, logp=12.0, psa_A2=60.0)


def test_validation_logp_too_low():
    with pytest.raises(ValueError, match="logp"):
        classify_permeability("Bad", mw=300.0, logp=-6.0, psa_A2=60.0)


def test_validation_psa_negative():
    with pytest.raises(ValueError, match="psa_A2"):
        classify_permeability("Bad", mw=300.0, logp=2.0, psa_A2=-10.0)


def test_validation_hbd_negative():
    with pytest.raises(ValueError, match="hbd_count"):
        classify_permeability("Bad", mw=300.0, logp=2.0, psa_A2=60.0, hbd_count=-1)
