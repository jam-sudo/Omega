"""Tests for Phase 736 — Drug Stability in Biological Fluids."""

from __future__ import annotations

import math

from omega_pbpk.prediction.biological_fluid_stability import (
    DrugStabilityResult,
    predict_drug_stability,
    screen_stability,
)

_LN2 = math.log(2.0)

_VALID_PLASMA_STABILITIES = {"stable", "moderate", "labile"}
_VALID_OVERALL_STABILITIES = {"good", "moderate", "poor"}


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_returns_drug_stability_result():
    result = predict_drug_stability(compound_name="CompoundA")
    assert isinstance(result, DrugStabilityResult)


# ---------------------------------------------------------------------------
# Ester effect on plasma stability
# ---------------------------------------------------------------------------


def test_ester_increases_plasma_k_deg():
    base = predict_drug_stability(compound_name="Base")
    with_ester = predict_drug_stability(compound_name="Ester", has_ester=True)
    assert with_ester.k_deg_plasma_per_h > base.k_deg_plasma_per_h


def test_ester_reduces_plasma_half_life():
    base = predict_drug_stability(compound_name="Base")
    with_ester = predict_drug_stability(compound_name="Ester", has_ester=True)
    assert with_ester.t_half_plasma_h < base.t_half_plasma_h


# ---------------------------------------------------------------------------
# Acid-labile group effect on gastric stability
# ---------------------------------------------------------------------------


def test_acid_labile_increases_gastric_k_deg():
    base = predict_drug_stability(compound_name="Base")
    acid_labile = predict_drug_stability(compound_name="AcidLabile", has_acid_labile_group=True)
    assert acid_labile.k_deg_gastric_per_h > base.k_deg_gastric_per_h


def test_acid_labile_reduces_gastric_half_life():
    base = predict_drug_stability(compound_name="Base")
    acid_labile = predict_drug_stability(compound_name="AcidLabile", has_acid_labile_group=True)
    assert acid_labile.t_half_gastric_h < base.t_half_gastric_h


# ---------------------------------------------------------------------------
# Beta-lactam (acid-labile)
# ---------------------------------------------------------------------------


def test_lactam_gastric_labile():
    """Beta-lactam should give gastric t½ < 0.5 h (labile classification)."""
    result = predict_drug_stability(compound_name="Penicillin", has_lactam=True)
    assert result.t_half_gastric_h < 0.5


def test_lactam_gastric_stability_labile():
    result = predict_drug_stability(compound_name="Penicillin", has_lactam=True)
    assert result.gastric_stability == "labile"


# ---------------------------------------------------------------------------
# No reactive groups → stable
# ---------------------------------------------------------------------------


def test_no_reactive_groups_stable_plasma():
    result = predict_drug_stability(compound_name="Inert")
    assert result.plasma_stability == "stable"


def test_no_reactive_groups_stable_gastric():
    result = predict_drug_stability(compound_name="Inert")
    assert result.gastric_stability == "stable"


def test_no_reactive_groups_overall_good():
    result = predict_drug_stability(compound_name="Inert")
    assert result.overall_stability == "good"


# ---------------------------------------------------------------------------
# Mathematical consistency
# ---------------------------------------------------------------------------


def test_t_half_plasma_equals_ln2_over_k_deg():
    result = predict_drug_stability(compound_name="Math", has_ester=True)
    expected = _LN2 / result.k_deg_plasma_per_h
    assert abs(result.t_half_plasma_h - expected) < 1e-12


def test_t_half_gastric_equals_ln2_over_k_deg():
    result = predict_drug_stability(compound_name="Math", has_lactam=True)
    expected = _LN2 / result.k_deg_gastric_per_h
    assert abs(result.t_half_gastric_h - expected) < 1e-12


def test_fraction_intact_plasma_1h():
    result = predict_drug_stability(compound_name="Math", has_ester=True)
    expected = math.exp(-result.k_deg_plasma_per_h * 1.0)
    assert abs(result.fraction_intact_plasma_1h - expected) < 1e-12


def test_fraction_intact_gastric_30min():
    result = predict_drug_stability(compound_name="Math", has_lactam=True)
    expected = math.exp(-result.k_deg_gastric_per_h * 0.5)
    assert abs(result.fraction_intact_gastric_30min - expected) < 1e-12


# ---------------------------------------------------------------------------
# Classification strings
# ---------------------------------------------------------------------------


def test_plasma_stability_valid_string():
    result = predict_drug_stability(compound_name="Check")
    assert result.plasma_stability in _VALID_PLASMA_STABILITIES


def test_overall_stability_valid_string():
    result = predict_drug_stability(compound_name="Check")
    assert result.overall_stability in _VALID_OVERALL_STABILITIES


# ---------------------------------------------------------------------------
# screen_stability
# ---------------------------------------------------------------------------


def test_screen_stability_returns_list():
    compounds = [
        {"compound_name": "A"},
        {"compound_name": "B", "has_ester": True},
    ]
    results = screen_stability(compounds)
    assert isinstance(results, list)
    assert len(results) == 2


def test_screen_stability_sorted_good_first():
    compounds = [
        {"compound_name": "Labile", "has_ester": True, "has_lactam": True},
        {"compound_name": "Inert"},
        {"compound_name": "Moderate", "has_ester": True},
    ]
    results = screen_stability(compounds)
    order = {"good": 0, "moderate": 1, "poor": 2}
    for i in range(len(results) - 1):
        assert order[results[i].overall_stability] <= order[results[i + 1].overall_stability]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_compound_name_no_crash():
    """Empty compound_name must not raise an exception."""
    result = predict_drug_stability(compound_name="")
    assert isinstance(result, DrugStabilityResult)
    assert result.compound_name == ""


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


def test_notes_nonempty():
    result = predict_drug_stability(compound_name="DrugX")
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Combined labile groups → overall poor
# ---------------------------------------------------------------------------


def test_ester_and_lactam_overall_poor():
    """has_ester=True, has_lactam=True → gastric labile → overall poor."""
    result = predict_drug_stability(compound_name="VeryLabile", has_ester=True, has_lactam=True)
    assert result.overall_stability == "poor"
