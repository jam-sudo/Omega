"""Tests for Phase 1055 — Protein Corona Formation."""

import math
import pytest

from omega_pbpk.prediction.protein_corona_formation import (
    ProteinCoronaResult,
    compare_coatings_corona,
    predict_protein_corona,
)


# ── Basic return type and frozen dataclass ─────────────────────────────────────

def test_returns_protein_corona_result():
    result = predict_protein_corona("AuNP", 50.0)
    assert isinstance(result, ProteinCoronaResult)


def test_frozen_dataclass():
    result = predict_protein_corona("AuNP", 50.0)
    with pytest.raises((AttributeError, TypeError)):
        result.particle_size_nm = 100.0  # type: ignore[misc]


# ── Thickness and size constraints ─────────────────────────────────────────────

def test_hard_corona_positive():
    result = predict_protein_corona("AuNP", 100.0)
    assert result.hard_corona_thickness_nm > 0


def test_soft_corona_positive():
    result = predict_protein_corona("AuNP", 100.0)
    assert result.soft_corona_thickness_nm > 0


def test_soft_corona_is_fraction_of_hard():
    result = predict_protein_corona("AuNP", 100.0)
    assert math.isclose(result.soft_corona_thickness_nm, result.hard_corona_thickness_nm * 0.3, rel_tol=1e-6)


def test_total_corona_equals_sum():
    result = predict_protein_corona("AuNP", 100.0)
    expected = result.hard_corona_thickness_nm + result.soft_corona_thickness_nm
    assert math.isclose(result.total_corona_thickness_nm, expected, rel_tol=1e-9)


def test_effective_size_greater_than_particle():
    result = predict_protein_corona("AuNP", 50.0)
    assert result.effective_size_nm > result.particle_size_nm


def test_effective_size_equals_particle_plus_corona():
    result = predict_protein_corona("AuNP", 50.0)
    expected = result.particle_size_nm + result.total_corona_thickness_nm
    assert math.isclose(result.effective_size_nm, expected, rel_tol=1e-9)


# ── Range constraints ──────────────────────────────────────────────────────────

def test_protein_coverage_pct_in_range():
    result = predict_protein_corona("AuNP", 50.0)
    assert 5.0 <= result.protein_coverage_pct <= 99.0


def test_corona_stability_score_in_range():
    for coating in ("none", "PEG_2k", "PEG_5k", "citrate", "amine", "carboxyl"):
        result = predict_protein_corona("P", 50.0, surface_coating=coating)
        assert 10.0 <= result.corona_stability_score <= 100.0, f"Failed for coating={coating}"


def test_cellular_uptake_modifier_in_range():
    for coating in ("none", "PEG_2k", "PEG_5k", "citrate", "amine", "carboxyl"):
        result = predict_protein_corona("P", 50.0, surface_coating=coating)
        assert 0.1 <= result.cellular_uptake_modifier <= 3.0, f"Failed for coating={coating}"


def test_stealth_efficiency_in_range():
    for coating in ("none", "PEG_2k", "PEG_5k", "citrate", "amine", "carboxyl"):
        result = predict_protein_corona("P", 50.0, surface_coating=coating)
        assert 0.0 <= result.stealth_efficiency <= 1.0, f"Failed for coating={coating}"


# ── Coating effects ────────────────────────────────────────────────────────────

def test_peg5k_higher_stealth_than_none():
    peg = predict_protein_corona("P", 50.0, surface_coating="PEG_5k")
    bare = predict_protein_corona("P", 50.0, surface_coating="none")
    assert peg.stealth_efficiency > bare.stealth_efficiency


def test_peg5k_lower_uptake_than_amine():
    peg = predict_protein_corona("P", 50.0, surface_coating="PEG_5k")
    amine = predict_protein_corona("P", 50.0, surface_coating="amine")
    assert peg.cellular_uptake_modifier < amine.cellular_uptake_modifier


# ── Dominant proteins ─────────────────────────────────────────────────────────

def test_dominant_proteins_has_three_items():
    result = predict_protein_corona("AuNP", 50.0)
    assert len(result.dominant_proteins) == 3


def test_dominant_proteins_negative_zeta():
    result = predict_protein_corona("AuNP", 50.0, zeta_potential_mV=-30.0)
    assert result.dominant_proteins[0] == "IgG"


def test_dominant_proteins_zero_zeta():
    result = predict_protein_corona("AuNP", 50.0, zeta_potential_mV=-5.0)
    assert result.dominant_proteins[0] == "albumin"


def test_dominant_proteins_positive_zeta():
    result = predict_protein_corona("AuNP", 50.0, zeta_potential_mV=+15.0)
    assert result.dominant_proteins[0] == "vitronectin"


# ── Notes non-empty ────────────────────────────────────────────────────────────

def test_notes_nonempty():
    result = predict_protein_corona("AuNP", 50.0)
    assert result.notes and len(result.notes) > 0


# ── compare_coatings_corona ────────────────────────────────────────────────────

def test_compare_coatings_returns_six():
    results = compare_coatings_corona("AuNP", 50.0)
    assert len(results) == 6


def test_compare_coatings_sorted_by_stealth_desc():
    results = compare_coatings_corona("AuNP", 50.0)
    stealth_values = [r.stealth_efficiency for r in results]
    assert stealth_values == sorted(stealth_values, reverse=True)


def test_compare_coatings_all_types():
    results = compare_coatings_corona("AuNP", 50.0)
    coatings = {r.particle_name for r in results}
    # All results should have the same particle name
    assert len(coatings) == 1


# ── Validation errors ─────────────────────────────────────────────────────────

def test_validation_size_zero_raises():
    with pytest.raises(ValueError, match="particle_size_nm"):
        predict_protein_corona("P", 0.0)


def test_validation_size_negative_raises():
    with pytest.raises(ValueError):
        predict_protein_corona("P", -5.0)


def test_validation_invalid_coating_raises():
    with pytest.raises(ValueError, match="surface_coating"):
        predict_protein_corona("P", 50.0, surface_coating="unknown")


def test_validation_invalid_material_raises():
    with pytest.raises(ValueError, match="material"):
        predict_protein_corona("P", 50.0, material="titanium")


def test_validation_serum_conc_zero_raises():
    with pytest.raises(ValueError, match="serum_protein_conc"):
        predict_protein_corona("P", 50.0, serum_protein_conc_mg_mL=0.0)


def test_validation_incubation_zero_raises():
    with pytest.raises(ValueError, match="incubation_time_h"):
        predict_protein_corona("P", 50.0, incubation_time_h=0.0)


# ── Materials ─────────────────────────────────────────────────────────────────

def test_gold_higher_adsorption_than_lipid():
    """Gold has higher affinity (0.9) than lipid (0.5)."""
    gold = predict_protein_corona("P", 50.0, surface_coating="none", material="gold")
    lipid = predict_protein_corona("P", 50.0, surface_coating="none", material="lipid")
    assert gold.hard_corona_thickness_nm > lipid.hard_corona_thickness_nm
