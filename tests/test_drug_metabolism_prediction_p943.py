"""Tests for Phase 943: drug_metabolism_prediction_p943."""

import pytest

from omega_pbpk.prediction.drug_metabolism_prediction_p943 import (
    DrugMetabolismPredictionResult,
    predict_metabolism,
    screen_metabolic_liability,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_ENZYMES = {"CYP3A4", "CYP2D6", "CYP2C9", "CYP2C19", "CYP1A2"}
VALID_LIABILITIES = {"low", "moderate", "high"}


def _base_drug(**kwargs):
    """Return a typical basic drug; override with kwargs."""
    defaults = dict(
        drug_name="TestBase",
        logp=2.5,
        pka=8.5,
        ionization_type="base",
        mw=350.0,
    )
    defaults.update(kwargs)
    return predict_metabolism(**defaults)


def _acid_drug(**kwargs):
    defaults = dict(
        drug_name="TestAcid",
        logp=2.5,
        pka=4.5,
        ionization_type="acid",
        mw=320.0,
    )
    defaults.update(kwargs)
    return predict_metabolism(**defaults)


# ---------------------------------------------------------------------------
# Return type tests
# ---------------------------------------------------------------------------

def test_return_type_is_frozen_dataclass():
    result = _base_drug()
    assert isinstance(result, DrugMetabolismPredictionResult)


def test_result_is_frozen():
    result = _base_drug()
    with pytest.raises((AttributeError, TypeError)):
        result.logp = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Fraction validity
# ---------------------------------------------------------------------------

def test_cyp_fractions_in_0_1():
    result = _base_drug()
    for f in [
        result.cyp3a4_fraction,
        result.cyp2d6_fraction,
        result.cyp2c9_fraction,
        result.cyp2c19_fraction,
        result.cyp1a2_fraction,
    ]:
        assert 0.0 <= f <= 1.0


def test_cyp_fractions_sum_to_1():
    result = _base_drug()
    total = (
        result.cyp3a4_fraction
        + result.cyp2d6_fraction
        + result.cyp2c9_fraction
        + result.cyp2c19_fraction
        + result.cyp1a2_fraction
    )
    assert abs(total - 1.0) < 1e-9


def test_cyp_fractions_sum_to_1_acid_drug():
    result = _acid_drug()
    total = (
        result.cyp3a4_fraction
        + result.cyp2d6_fraction
        + result.cyp2c9_fraction
        + result.cyp2c19_fraction
        + result.cyp1a2_fraction
    )
    assert abs(total - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Primary enzyme
# ---------------------------------------------------------------------------

def test_primary_enzyme_is_valid():
    result = _base_drug()
    assert result.primary_enzyme in VALID_ENZYMES


def test_primary_enzyme_has_highest_fraction():
    result = _base_drug()
    fractions = {
        "CYP3A4": result.cyp3a4_fraction,
        "CYP2D6": result.cyp2d6_fraction,
        "CYP2C9": result.cyp2c9_fraction,
        "CYP2C19": result.cyp2c19_fraction,
        "CYP1A2": result.cyp1a2_fraction,
    }
    assert fractions[result.primary_enzyme] == max(fractions.values())


# ---------------------------------------------------------------------------
# Phase I / II
# ---------------------------------------------------------------------------

def test_phase_i_likely_is_bool():
    result = _base_drug()
    assert isinstance(result.phase_i_likely, bool)


def test_phase_ii_likely_is_bool():
    result = _base_drug()
    assert isinstance(result.phase_ii_likely, bool)


def test_phase_i_likely_true_for_high_logp():
    result = _base_drug(logp=3.0)
    assert result.phase_i_likely is True


def test_phase_i_likely_false_for_low_logp():
    result = _base_drug(logp=0.5)
    assert result.phase_i_likely is False


def test_phase_ii_likely_for_large_mw():
    result = _base_drug(mw=650.0)
    assert result.phase_ii_likely is True


# ---------------------------------------------------------------------------
# Variability risk
# ---------------------------------------------------------------------------

def test_high_cyp_variability_risk_is_bool():
    result = _base_drug()
    assert isinstance(result.high_cyp_variability_risk, bool)


def test_high_variability_true_for_strong_base():
    # logp=1.5 (low: no CYP3A4/2C9 logP boost), pka=9.5 (outside CYP2C19 range),
    # mw=450 (above CYP2D6/1A2 small-MW thresholds, below 3A4 MW range top) →
    # CYP2D6 fraction ~0.316 > 0.3 → variability risk True
    result = predict_metabolism(
        drug_name="StrongBase",
        logp=1.5,
        pka=9.5,
        ionization_type="base",
        mw=450.0,
    )
    assert result.high_cyp_variability_risk is True


def test_high_variability_flag_when_cyp2d6_above_0p3():
    result = predict_metabolism(
        drug_name="StrongBase",
        logp=1.5,
        pka=9.5,
        ionization_type="base",
        mw=450.0,
    )
    assert result.cyp2d6_fraction > 0.3
    assert result.high_cyp_variability_risk is True


# ---------------------------------------------------------------------------
# Metabolic liability
# ---------------------------------------------------------------------------

def test_metabolic_liability_in_valid_set():
    result = _base_drug()
    assert result.metabolic_liability in VALID_LIABILITIES


def test_metabolic_liability_logic_consistent():
    result = predict_metabolism(
        drug_name="AnyDrug",
        logp=2.0,
        pka=9.0,
        ionization_type="base",
        mw=300.0,
    )
    fractions = {
        "CYP3A4": result.cyp3a4_fraction,
        "CYP2D6": result.cyp2d6_fraction,
        "CYP2C9": result.cyp2c9_fraction,
        "CYP2C19": result.cyp2c19_fraction,
        "CYP1A2": result.cyp1a2_fraction,
    }
    primary_fraction = fractions[result.primary_enzyme]
    if primary_fraction > 0.7:
        assert result.metabolic_liability == "high"
    elif primary_fraction < 0.3:
        assert result.metabolic_liability == "low"
    else:
        assert result.metabolic_liability == "moderate"


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------

def test_notes_is_non_empty():
    result = _base_drug()
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Drug name preserved
# ---------------------------------------------------------------------------

def test_drug_name_preserved():
    result = predict_metabolism(
        drug_name="Aspirin",
        logp=1.2,
        pka=3.5,
        ionization_type="acid",
        mw=180.0,
    )
    assert result.drug_name == "Aspirin"


# ---------------------------------------------------------------------------
# CYP2D6 higher for basic drugs
# ---------------------------------------------------------------------------

def test_base_drug_has_higher_cyp2d6_than_acid():
    base = predict_metabolism("B", logp=2.5, pka=8.5, ionization_type="base", mw=350.0)
    acid = predict_metabolism("A", logp=2.5, pka=4.5, ionization_type="acid", mw=350.0)
    assert base.cyp2d6_fraction > acid.cyp2d6_fraction


# ---------------------------------------------------------------------------
# CYP2C9 higher for acid drugs
# ---------------------------------------------------------------------------

def test_acid_drug_has_higher_cyp2c9_than_base():
    acid = predict_metabolism("A", logp=2.5, pka=4.5, ionization_type="acid", mw=320.0)
    base = predict_metabolism("B", logp=2.5, pka=8.5, ionization_type="base", mw=320.0)
    assert acid.cyp2c9_fraction > base.cyp2c9_fraction


# ---------------------------------------------------------------------------
# screen_metabolic_liability
# ---------------------------------------------------------------------------

def test_screen_returns_correct_length():
    compounds = [
        {"drug_name": "A", "logp": 2.0, "pka": 8.0, "ionization_type": "base", "mw": 300.0},
        {"drug_name": "B", "logp": 1.5, "pka": 4.5, "ionization_type": "acid", "mw": 280.0},
        {"drug_name": "C", "logp": 3.0, "pka": 7.0, "ionization_type": "neutral", "mw": 400.0},
    ]
    results = screen_metabolic_liability(compounds)
    assert len(results) == 3


def test_screen_results_are_correct_type():
    compounds = [
        {"drug_name": "D", "logp": 2.0, "pka": 8.0, "ionization_type": "base", "mw": 300.0},
    ]
    results = screen_metabolic_liability(compounds)
    for r in results:
        assert isinstance(r, DrugMetabolismPredictionResult)


def test_screen_sorted_by_primary_fraction_descending():
    compounds = [
        {"drug_name": "X", "logp": 0.5, "pka": 7.0, "ionization_type": "neutral", "mw": 500.0},
        {"drug_name": "Y", "logp": 2.0, "pka": 9.0, "ionization_type": "base", "mw": 300.0},
        {"drug_name": "Z", "logp": 4.0, "pka": 4.0, "ionization_type": "acid", "mw": 400.0},
    ]
    results = screen_metabolic_liability(compounds)
    fracs = []
    for r in results:
        fmap = {
            "CYP3A4": r.cyp3a4_fraction,
            "CYP2D6": r.cyp2d6_fraction,
            "CYP2C9": r.cyp2c9_fraction,
            "CYP2C19": r.cyp2c19_fraction,
            "CYP1A2": r.cyp1a2_fraction,
        }
        fracs.append(fmap[r.primary_enzyme])
    assert fracs == sorted(fracs, reverse=True)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_validation_mw_zero():
    with pytest.raises(ValueError, match="mw must be > 0"):
        predict_metabolism("X", logp=2.0, pka=7.0, ionization_type="base", mw=0.0)


def test_validation_mw_negative():
    with pytest.raises(ValueError, match="mw must be > 0"):
        predict_metabolism("X", logp=2.0, pka=7.0, ionization_type="base", mw=-10.0)


def test_validation_invalid_ionization_type():
    with pytest.raises(ValueError, match="ionization_type"):
        predict_metabolism("X", logp=2.0, pka=7.0, ionization_type="cation", mw=300.0)


# ---------------------------------------------------------------------------
# Low logP base drug: CYP2D6 high
# ---------------------------------------------------------------------------

def test_low_logp_base_drug_high_cyp2d6():
    result = predict_metabolism(
        drug_name="LowLogPBase",
        logp=1.5,
        pka=9.0,
        ionization_type="base",
        mw=350.0,
    )
    # CYP2D6 gets base+basic_pka boost and logP in [1,4] boost
    assert result.cyp2d6_fraction > 0.2


# ---------------------------------------------------------------------------
# Large MW -> phase II likely
# ---------------------------------------------------------------------------

def test_large_mw_phase_ii():
    result = predict_metabolism(
        drug_name="LargeMW",
        logp=2.5,
        pka=7.0,
        ionization_type="neutral",
        mw=650.0,
    )
    assert result.phase_ii_likely is True
