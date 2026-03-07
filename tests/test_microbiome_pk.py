"""Tests for Phase 193 — microbiome_pk."""

import math

import pytest

from omega_pbpk.clinical.microbiome_pk import (
    MicrobiomeEffect,
    MicrobiomePKResult,
    assess_microbiome_impact,
    compare_microbiome_states,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assess(**kwargs):
    defaults = dict(
        drug_name="TestDrug",
        smiles="CC",
        f_oral_baseline=0.5,
        microbiome_state="normal",
    )
    defaults.update(kwargs)
    return assess_microbiome_impact(**defaults)


# ---------------------------------------------------------------------------
# Basic return-type checks
# ---------------------------------------------------------------------------


def test_returns_result_type():
    result = _assess()
    assert isinstance(result, MicrobiomePKResult)


def test_result_is_frozen():
    result = _assess()
    with pytest.raises((AttributeError, TypeError)):
        result.drug_name = "x"  # type: ignore[misc]


def test_effects_is_list():
    result = _assess()
    assert isinstance(result.effects, list)


def test_microbiome_effect_frozen():
    e = MicrobiomeEffect("azo_reduction", True, 1.5, "increase", "desc")
    with pytest.raises((AttributeError, TypeError)):
        e.pathway = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Germ-free: no microbiome activity
# ---------------------------------------------------------------------------


def test_germ_free_no_azo_effect():
    """With germ_free state and azo bond, activity=0 → magnitude=1.0 → no change."""
    result = assess_microbiome_impact(
        drug_name="AzoDrug",
        smiles="CC",
        f_oral_baseline=0.4,
        microbiome_state="germ_free",
        has_azo_bond=True,
    )
    # azo magnitude = 1 + 0.5*0 = 1.0 → f_adjusted == baseline
    assert math.isclose(result.f_oral_adjusted, 0.4, rel_tol=1e-6)
    assert math.isclose(result.auc_ratio, 1.0, rel_tol=1e-6)


def test_germ_free_no_glucuronide_effect():
    result = assess_microbiome_impact(
        drug_name="GlucDrug",
        smiles="CC",
        f_oral_baseline=0.6,
        microbiome_state="germ_free",
        has_glucuronide=True,
    )
    assert math.isclose(result.f_oral_adjusted, 0.6, rel_tol=1e-6)


def test_germ_free_ba_change_near_zero():
    result = assess_microbiome_impact(
        drug_name="Inert",
        smiles="CC",
        f_oral_baseline=0.5,
        microbiome_state="germ_free",
        has_azo_bond=True,
        has_glucuronide=True,
    )
    assert math.isclose(result.bioavailability_change_pct, 0.0, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# Normal microbiome: azo activation
# ---------------------------------------------------------------------------


def test_normal_azo_increases_ba():
    result = assess_microbiome_impact(
        drug_name="AzoDrug",
        smiles="CC",
        f_oral_baseline=0.4,
        microbiome_state="normal",
        has_azo_bond=True,
    )
    # magnitude = 1 + 0.5*1.0 = 1.5 → f_adjusted = min(1, 0.4*1.5) = 0.6
    assert result.f_oral_adjusted > result.f_oral_baseline
    assert math.isclose(result.f_oral_adjusted, 0.6, rel_tol=1e-6)


def test_normal_azo_auc_ratio_correct():
    result = assess_microbiome_impact(
        drug_name="AzoDrug",
        smiles="CC",
        f_oral_baseline=0.4,
        microbiome_state="normal",
        has_azo_bond=True,
    )
    expected_ratio = 0.6 / 0.4
    assert math.isclose(result.auc_ratio, expected_ratio, rel_tol=1e-6)


def test_normal_glucuronide_increases_ba():
    result = assess_microbiome_impact(
        drug_name="GlucDrug",
        smiles="CC",
        f_oral_baseline=0.5,
        microbiome_state="normal",
        has_glucuronide=True,
    )
    # magnitude = 1 + 0.3*1.0 = 1.3 → f_adjusted = 0.65
    assert math.isclose(result.f_oral_adjusted, 0.65, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# Antibiotic-treated: azo effect minimal
# ---------------------------------------------------------------------------


def test_antibiotic_treated_azo_minimal():
    result = assess_microbiome_impact(
        drug_name="AzoDrug",
        smiles="CC",
        f_oral_baseline=0.4,
        microbiome_state="antibiotic_treated",
        has_azo_bond=True,
    )
    # activity = 0.1 → mag = 1.05 → f_adjusted = 0.42
    assert result.f_oral_adjusted < 0.43
    assert result.f_oral_adjusted > result.f_oral_baseline


def test_antibiotic_less_effect_than_normal():
    normal = assess_microbiome_impact(
        drug_name="Drug",
        smiles="CC",
        f_oral_baseline=0.4,
        microbiome_state="normal",
        has_azo_bond=True,
    )
    abx = assess_microbiome_impact(
        drug_name="Drug",
        smiles="CC",
        f_oral_baseline=0.4,
        microbiome_state="antibiotic_treated",
        has_azo_bond=True,
    )
    assert abx.f_oral_adjusted < normal.f_oral_adjusted


# ---------------------------------------------------------------------------
# SMILES-based hydrazine detection
# ---------------------------------------------------------------------------


def test_hydrazine_smiles_detection():
    result = assess_microbiome_impact(
        drug_name="HydrazineDrug",
        smiles="CNN",
        f_oral_baseline=0.6,
        microbiome_state="normal",
    )
    pathways = [e.pathway for e in result.effects]
    assert "hydrazine_reduction" in pathways


def test_hydrazine_reduces_ba():
    result = assess_microbiome_impact(
        drug_name="HydrazineDrug",
        smiles="CNN",
        f_oral_baseline=0.6,
        microbiome_state="normal",
    )
    # hydrazine magnitude = 0.8 → f_adjusted = 0.48
    assert result.f_oral_adjusted < result.f_oral_baseline


# ---------------------------------------------------------------------------
# Lipophilic trapping (dysbiotic + logP > 4)
# ---------------------------------------------------------------------------


def test_lipophilic_trapping_dysbiotic():
    result = assess_microbiome_impact(
        drug_name="LipoPhil",
        smiles="CC",
        f_oral_baseline=0.7,
        microbiome_state="dysbiotic",
        logP=5.0,
    )
    pathways = [e.pathway for e in result.effects]
    assert "lipophilic_trapping" in pathways
    assert result.f_oral_adjusted < result.f_oral_baseline


def test_lipophilic_trapping_not_normal():
    result = assess_microbiome_impact(
        drug_name="LipoPhil",
        smiles="CC",
        f_oral_baseline=0.7,
        microbiome_state="normal",
        logP=5.0,
    )
    pathways = [e.pathway for e in result.effects]
    assert "lipophilic_trapping" not in pathways


# ---------------------------------------------------------------------------
# compare_microbiome_states
# ---------------------------------------------------------------------------


def test_compare_returns_four_results():
    results = compare_microbiome_states(
        drug_name="Drug",
        smiles="CC",
        f_oral_baseline=0.5,
        has_azo_bond=True,
    )
    assert len(results) == 4


def test_compare_states_ordered():
    results = compare_microbiome_states(
        drug_name="Drug",
        smiles="CC",
        f_oral_baseline=0.5,
        has_azo_bond=True,
    )
    states = [r.microbiome_state for r in results]
    assert states == ["normal", "dysbiotic", "antibiotic_treated", "germ_free"]


def test_compare_monotone_azo():
    """Normal > dysbiotic > antibiotic_treated > germ_free for azo drug."""
    results = compare_microbiome_states(
        drug_name="Drug",
        smiles="CC",
        f_oral_baseline=0.3,
        has_azo_bond=True,
    )
    f_vals = [r.f_oral_adjusted for r in results]
    assert f_vals[0] >= f_vals[1] >= f_vals[2] >= f_vals[3]


# ---------------------------------------------------------------------------
# Risk categories
# ---------------------------------------------------------------------------


def test_risk_low_no_effects():
    result = _assess()
    assert result.risk_category == "low"


def test_risk_moderate_azo_dysbiotic():
    # dysbiotic baseline 0.5, mag=1.3 → 0.65, change=30% → high
    result = assess_microbiome_impact(
        drug_name="D",
        smiles="CC",
        f_oral_baseline=0.5,
        microbiome_state="normal",
        has_azo_bond=True,
        has_glucuronide=True,
    )
    # 0.5 * 1.5 * 1.3 = 0.975 → clipped to 1.0 → change=100% → high
    assert result.risk_category in ("moderate", "high")


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_microbiome_state():
    with pytest.raises(ValueError, match="microbiome_state"):
        assess_microbiome_impact(
            drug_name="D",
            smiles="CC",
            f_oral_baseline=0.5,
            microbiome_state="space_gut",
        )


def test_invalid_f_oral_too_high():
    with pytest.raises(ValueError):
        assess_microbiome_impact(
            drug_name="D",
            smiles="CC",
            f_oral_baseline=1.5,
        )


def test_invalid_f_oral_zero():
    with pytest.raises(ValueError):
        assess_microbiome_impact(
            drug_name="D",
            smiles="CC",
            f_oral_baseline=0.0,
        )


def test_invalid_f_oral_negative():
    with pytest.raises(ValueError):
        assess_microbiome_impact(
            drug_name="D",
            smiles="CC",
            f_oral_baseline=-0.1,
        )
