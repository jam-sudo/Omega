"""Tests for Phase 1056 — Drug Plasma Stability (species-aware model)."""

import pytest

from omega_pbpk.prediction.plasma_stability_p1056 import (
    PlasmaStabilityResult,
    compare_species_stability,
    predict_plasma_stability,
)

_VALID_PATHWAYS = {"esterase", "peptidase", "oxidation", "hydrolysis", "stable"}
_VALID_CLASSES = {"unstable", "labile", "moderate", "stable"}


# ── Basic return type and frozen dataclass ─────────────────────────────────────

def test_returns_plasma_stability_result():
    result = predict_plasma_stability("DrugA", mw_Da=400.0, logp=2.0)
    assert isinstance(result, PlasmaStabilityResult)


def test_frozen_dataclass():
    result = predict_plasma_stability("DrugA", mw_Da=400.0, logp=2.0)
    with pytest.raises((AttributeError, TypeError)):
        result.t50_min = 999.0  # type: ignore[misc]


# ── Fraction remaining constraints ────────────────────────────────────────────

def test_fractions_in_zero_one():
    result = predict_plasma_stability("DrugA", mw_Da=400.0, logp=2.0, has_ester=True)
    assert 0.0 <= result.fraction_remaining_at_30min <= 1.0
    assert 0.0 <= result.fraction_remaining_at_60min <= 1.0
    assert 0.0 <= result.fraction_remaining_at_120min <= 1.0


def test_fractions_monotonically_decreasing():
    result = predict_plasma_stability("DrugA", mw_Da=400.0, logp=2.0, has_ester=True)
    assert result.fraction_remaining_at_30min >= result.fraction_remaining_at_60min
    assert result.fraction_remaining_at_60min >= result.fraction_remaining_at_120min


# ── t50 ────────────────────────────────────────────────────────────────────────

def test_t50_positive():
    result = predict_plasma_stability("DrugA", mw_Da=400.0, logp=2.0)
    assert result.t50_min > 0.0


def test_t50_stable_drug_large():
    """Drug with no reactive groups should have a very long t50."""
    result = predict_plasma_stability("StableDrug", mw_Da=300.0, logp=0.0)
    assert result.t50_min > 100.0


# ── Stability class and pathway ────────────────────────────────────────────────

def test_stability_class_in_valid_set():
    result = predict_plasma_stability("DrugA", mw_Da=400.0, logp=2.0)
    assert result.stability_class in _VALID_CLASSES


def test_primary_pathway_in_valid_set():
    result = predict_plasma_stability("DrugA", mw_Da=400.0, logp=2.0, has_ester=True)
    assert result.primary_degradation_pathway in _VALID_PATHWAYS


def test_stable_drug_pathway():
    result = predict_plasma_stability("StableDrug", mw_Da=300.0, logp=0.0)
    assert result.primary_degradation_pathway == "stable"
    assert result.stability_class == "stable"


# ── Ester effect ──────────────────────────────────────────────────────────────

def test_ester_lower_t50_than_no_ester():
    ester = predict_plasma_stability("DrugE", mw_Da=400.0, logp=2.0, has_ester=True)
    no_ester = predict_plasma_stability("DrugNE", mw_Da=400.0, logp=2.0, has_ester=False)
    assert ester.t50_min < no_ester.t50_min


def test_esterase_susceptibility_matches_has_ester():
    result_ester = predict_plasma_stability("DrugE", mw_Da=400.0, logp=2.0, has_ester=True)
    result_no_ester = predict_plasma_stability("DrugNE", mw_Da=400.0, logp=2.0, has_ester=False)
    assert result_ester.esterase_susceptibility is True
    assert result_no_ester.esterase_susceptibility is False


# ── Species effect ────────────────────────────────────────────────────────────

def test_rat_lower_t50_than_human_for_ester():
    """Rat has 2x esterase activity vs human."""
    human = predict_plasma_stability("DrugE", mw_Da=400.0, logp=2.0, has_ester=True, species="human")
    rat = predict_plasma_stability("DrugE", mw_Da=400.0, logp=2.0, has_ester=True, species="rat")
    assert rat.t50_min < human.t50_min


def test_mouse_faster_than_rat_for_ester():
    """Mouse has 2.5x vs rat's 2x esterase factor."""
    rat = predict_plasma_stability("DrugE", mw_Da=400.0, logp=2.0, has_ester=True, species="rat")
    mouse = predict_plasma_stability("DrugE", mw_Da=400.0, logp=2.0, has_ester=True, species="mouse")
    assert mouse.t50_min < rat.t50_min


# ── Notes ─────────────────────────────────────────────────────────────────────

def test_notes_nonempty():
    result = predict_plasma_stability("DrugA", mw_Da=400.0, logp=2.0)
    assert result.notes and len(result.notes) > 0


# ── compare_species_stability ─────────────────────────────────────────────────

def test_compare_species_returns_four():
    results = compare_species_stability("DrugA", mw_Da=400.0, logp=2.0, has_ester=True)
    assert len(results) == 4


def test_compare_species_sorted_by_t50_desc():
    results = compare_species_stability("DrugA", mw_Da=400.0, logp=2.0, has_ester=True)
    t50_values = [r.t50_min for r in results]
    assert t50_values == sorted(t50_values, reverse=True)


def test_compare_species_all_species_represented():
    results = compare_species_stability("DrugA", mw_Da=400.0, logp=2.0, has_ester=True)
    species_set = {r.species for r in results}
    assert species_set == {"human", "rat", "mouse", "dog"}


# ── Validation errors ─────────────────────────────────────────────────────────

def test_validation_mw_zero_raises():
    with pytest.raises(ValueError, match="mw_Da"):
        predict_plasma_stability("DrugA", mw_Da=0.0, logp=2.0)


def test_validation_mw_negative_raises():
    with pytest.raises(ValueError, match="mw_Da"):
        predict_plasma_stability("DrugA", mw_Da=-100.0, logp=2.0)


def test_validation_invalid_species_raises():
    with pytest.raises(ValueError, match="species"):
        predict_plasma_stability("DrugA", mw_Da=400.0, logp=2.0, species="rabbit")


def test_validation_temp_too_high_raises():
    with pytest.raises(ValueError, match="temperature_C"):
        predict_plasma_stability("DrugA", mw_Da=400.0, logp=2.0, temperature_C=60.0)


def test_validation_temp_too_low_raises():
    with pytest.raises(ValueError, match="temperature_C"):
        predict_plasma_stability("DrugA", mw_Da=400.0, logp=2.0, temperature_C=3.0)


def test_validation_conc_zero_raises():
    with pytest.raises(ValueError, match="initial_conc_uM"):
        predict_plasma_stability("DrugA", mw_Da=400.0, logp=2.0, initial_conc_uM=0.0)


# ── Additional edge cases ─────────────────────────────────────────────────────

def test_peptide_bond_pathway():
    result = predict_plasma_stability(
        "Peptide", mw_Da=800.0, logp=-1.0, has_peptide_bond=True
    )
    assert result.primary_degradation_pathway == "peptidase"


def test_temperature_effect():
    """Higher temperature → faster degradation → lower t50."""
    low_t = predict_plasma_stability(
        "DrugA", mw_Da=400.0, logp=2.0, has_ester=True, temperature_C=25.0
    )
    high_t = predict_plasma_stability(
        "DrugA", mw_Da=400.0, logp=2.0, has_ester=True, temperature_C=42.0
    )
    assert high_t.t50_min < low_t.t50_min
