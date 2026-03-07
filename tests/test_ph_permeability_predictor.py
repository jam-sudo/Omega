"""Tests for Phase 755 — Drug Permeability Prediction by pH."""

import math

import pytest

from omega_pbpk.prediction.ph_permeability_predictor import (
    PHPermeabilityResult,
    predict_ph_permeability,
    screen_ph_permeability,
)

VALID_SEGMENTS = {"stomach", "duodenum", "jejunum", "ileum", "colon"}
VALID_CLASSIFICATIONS = {"high", "low"}


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_returns_result_type():
    result = predict_ph_permeability("DrugA")
    assert isinstance(result, PHPermeabilityResult)


# ---------------------------------------------------------------------------
# Neutral compound
# ---------------------------------------------------------------------------


def test_neutral_same_papp_all_segments():
    r = predict_ph_permeability("Neutral", compound_type="neutral", papp_max_nm_s=20.0, logp=1.0)
    papps = [
        r.papp_stomach_nm_s,
        r.papp_duodenum_nm_s,
        r.papp_jejunum_nm_s,
        r.papp_ileum_nm_s,
        r.papp_colon_nm_s,
    ]
    assert all(math.isclose(p, papps[0], rel_tol=1e-9) for p in papps)


def test_neutral_pka_none_stored_as_minus_one():
    r = predict_ph_permeability("NeutralNoPKa", pka=None, compound_type="neutral")
    assert r.pka == -1.0


def test_neutral_pka_none_same_papp_all_segments():
    r = predict_ph_permeability("NeutralNoPKa2", pka=None, papp_max_nm_s=15.0, logp=1.0)
    papps = [
        r.papp_stomach_nm_s,
        r.papp_duodenum_nm_s,
        r.papp_jejunum_nm_s,
        r.papp_ileum_nm_s,
        r.papp_colon_nm_s,
    ]
    assert all(math.isclose(p, papps[0], rel_tol=1e-9) for p in papps)


# ---------------------------------------------------------------------------
# Acidic compound (pKa=4)
# ---------------------------------------------------------------------------


def test_acid_high_papp_stomach():
    """Acid is mostly neutral at low pH → high Papp in stomach (pH 1.2)."""
    r = predict_ph_permeability(
        "AcidDrug", pka=4.0, compound_type="acid", papp_max_nm_s=30.0, logp=1.0
    )
    # stomach pH 1.2 << pKa=4 → mostly neutral → high Papp
    assert r.papp_stomach_nm_s > r.papp_ileum_nm_s


def test_acid_low_papp_ileum():
    """Acid is ionised at ileum pH 7.2 >> pKa=4 → lower Papp."""
    r = predict_ph_permeability(
        "AcidDrug2", pka=4.0, compound_type="acid", papp_max_nm_s=30.0, logp=1.0
    )
    assert r.papp_ileum_nm_s < 1.0  # nearly zero


def test_acid_optimal_segment_stomach():
    r = predict_ph_permeability(
        "AcidOpt", pka=4.0, compound_type="acid", papp_max_nm_s=30.0, logp=1.0
    )
    assert r.optimal_absorption_segment == "stomach"


# ---------------------------------------------------------------------------
# Basic compound (pKa=9)
# ---------------------------------------------------------------------------


def test_base_high_papp_ileum():
    """Base is mostly neutral at high pH → high Papp in ileum (pH 7.2)."""
    r = predict_ph_permeability(
        "BaseDrug", pka=9.0, compound_type="base", papp_max_nm_s=30.0, logp=1.0
    )
    # ileum pH 7.2 < pKa=9 → mostly neutral → high Papp
    assert r.papp_ileum_nm_s > r.papp_stomach_nm_s


def test_base_low_papp_stomach():
    """Base is protonated/ionised at stomach pH 1.2 << pKa=9 → low Papp."""
    r = predict_ph_permeability(
        "BaseDrug2", pka=9.0, compound_type="base", papp_max_nm_s=30.0, logp=1.0
    )
    assert r.papp_stomach_nm_s < 1.0


def test_base_optimal_segment_not_stomach():
    r = predict_ph_permeability(
        "BaseOpt", pka=9.0, compound_type="base", papp_max_nm_s=30.0, logp=1.0
    )
    assert r.optimal_absorption_segment != "stomach"


# ---------------------------------------------------------------------------
# Papp > 0 for all segments
# ---------------------------------------------------------------------------


def test_papp_positive_all_segments():
    r = predict_ph_permeability(
        "PositivePapp", pka=5.0, compound_type="acid", papp_max_nm_s=10.0, logp=2.0
    )
    assert r.papp_stomach_nm_s > 0
    assert r.papp_duodenum_nm_s > 0
    assert r.papp_jejunum_nm_s > 0
    assert r.papp_ileum_nm_s > 0
    assert r.papp_colon_nm_s > 0


# ---------------------------------------------------------------------------
# optimal_absorption_segment is valid
# ---------------------------------------------------------------------------


def test_optimal_segment_valid():
    r = predict_ph_permeability(
        "SegCheck", pka=7.0, compound_type="base", papp_max_nm_s=20.0, logp=2.0
    )
    assert r.optimal_absorption_segment in VALID_SEGMENTS


# ---------------------------------------------------------------------------
# permeability_classification valid
# ---------------------------------------------------------------------------


def test_permeability_classification_valid():
    r = predict_ph_permeability("ClassCheck")
    assert r.permeability_classification in VALID_CLASSIFICATIONS


def test_permeability_classification_high():
    # logp=5 → logP_factor=min(3, 1+(5-1)*0.3)=min(3,2.2)=2.2; papp_actual=20*2.2=44; jejunum=44 (neutral)
    r = predict_ph_permeability("HighPerm", papp_max_nm_s=20.0, logp=5.0, compound_type="neutral")
    assert r.permeability_classification == "high"


def test_permeability_classification_low():
    # small papp
    r = predict_ph_permeability("LowPerm", papp_max_nm_s=3.0, logp=1.0, compound_type="neutral")
    assert r.permeability_classification == "low"


# ---------------------------------------------------------------------------
# f_neutral_physiological_ph
# ---------------------------------------------------------------------------


def test_f_neutral_physiological_ph_between_0_and_1():
    r = predict_ph_permeability("FNeutral", pka=7.4, compound_type="acid")
    assert 0.0 <= r.f_neutral_physiological_ph <= 1.0


def test_f_neutral_neutral_compound_is_1():
    r = predict_ph_permeability("FNeutralNeutral", compound_type="neutral")
    assert math.isclose(r.f_neutral_physiological_ph, 1.0)


# ---------------------------------------------------------------------------
# screen_ph_permeability
# ---------------------------------------------------------------------------


def test_screen_returns_list():
    compounds = [
        {"compound_name": "A", "papp_max_nm_s": 10.0},
        {"compound_name": "B", "papp_max_nm_s": 5.0},
    ]
    results = screen_ph_permeability(compounds)
    assert isinstance(results, list)
    assert len(results) == 2


def test_screen_sorted_by_jejunum_desc():
    compounds = [
        {"compound_name": "Low", "papp_max_nm_s": 5.0, "logp": 1.0},
        {"compound_name": "High", "papp_max_nm_s": 25.0, "logp": 2.0},
        {"compound_name": "Mid", "papp_max_nm_s": 15.0, "logp": 1.5},
    ]
    results = screen_ph_permeability(compounds)
    jejunums = [r.papp_jejunum_nm_s for r in results]
    assert jejunums == sorted(jejunums, reverse=True)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validation_papp_max_zero_raises():
    with pytest.raises(ValueError):
        predict_ph_permeability("Bad", papp_max_nm_s=0.0)


def test_validation_papp_max_negative_raises():
    with pytest.raises(ValueError):
        predict_ph_permeability("Bad2", papp_max_nm_s=-5.0)


# ---------------------------------------------------------------------------
# Notes non-empty
# ---------------------------------------------------------------------------


def test_notes_nonempty():
    r = predict_ph_permeability("NotesCheck")
    assert isinstance(r.notes, str) and len(r.notes) > 0
