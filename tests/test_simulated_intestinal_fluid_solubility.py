"""Tests for Phase 585 - Drug Solubility in Simulated Intestinal Fluids."""

import math

import pytest

from omega_pbpk.prediction.simulated_intestinal_fluid_solubility import (
    SIFSolubilityResult,
    compare_fluids,
    predict_sif_solubility,
)

# --- Basic return type and field tests ---


def test_return_type():
    r = predict_sif_solubility("DrugA", 2.0, 4.5, "acid", 0.5)
    assert isinstance(r, SIFSolubilityResult)


def test_all_fields_present():
    r = predict_sif_solubility("DrugA", 2.0, 4.5, "acid", 0.5)
    assert r.drug_name == "DrugA"
    assert r.logP == 2.0
    assert r.intrinsic_solubility_mg_mL == 0.5
    assert r.fluid_type == "FaSSIF"
    assert isinstance(r.solubility_sif_mg_mL, float)
    assert isinstance(r.enhancement_ratio, float)
    assert isinstance(r.micelle_solubilization_factor, float)
    assert isinstance(r.ph_correction_factor, float)
    assert isinstance(r.absorption_potential, float)
    assert isinstance(r.biorelevant_class, str)
    assert isinstance(r.notes, str)


def test_frozen_dataclass():
    r = predict_sif_solubility("DrugA", 2.0, 4.5, "acid", 0.5)
    with pytest.raises(AttributeError):
        r.drug_name = "changed"


# --- Solubility relationship tests ---


def test_solubility_sif_gte_intrinsic():
    """SIF solubility should be >= intrinsic solubility."""
    r = predict_sif_solubility("DrugA", 2.0, 4.5, "acid", 0.5)
    assert r.solubility_sif_mg_mL >= r.intrinsic_solubility_mg_mL


def test_enhancement_ratio_equals_ratio():
    r = predict_sif_solubility("DrugA", 2.0, 4.5, "acid", 0.5)
    expected = r.solubility_sif_mg_mL / r.intrinsic_solubility_mg_mL
    assert abs(r.enhancement_ratio - expected) < 1e-10


def test_enhancement_ratio_gte_1():
    r = predict_sif_solubility("DrugA", 2.0, 4.5, "acid", 0.5)
    assert r.enhancement_ratio >= 1.0


def test_micelle_factor_gte_1():
    r = predict_sif_solubility("DrugA", 2.0, 4.5, "acid", 0.5)
    assert r.micelle_solubilization_factor >= 1.0


# --- FeSSIF vs FaSSIF for lipophilic drug ---


def test_fessif_higher_than_fassif_lipophilic():
    """FeSSIF should give higher solubility than FaSSIF for lipophilic drug."""
    r_fassif = predict_sif_solubility("LipoD", 4.0, 4.5, "neutral", 0.01, "FaSSIF")
    r_fessif = predict_sif_solubility("LipoD", 4.0, 4.5, "neutral", 0.01, "FeSSIF")
    assert r_fessif.solubility_sif_mg_mL > r_fassif.solubility_sif_mg_mL


# --- pH correction tests ---


def test_ph_correction_neutral():
    """Neutral drugs should have ph_correction_factor = 1."""
    r = predict_sif_solubility("NeutD", 2.0, 7.0, "neutral", 1.0)
    assert r.ph_correction_factor == 1.0


def test_ph_correction_acid_fassif():
    """Acid in FaSSIF (pH 6.5): ionized_frac = 1/(1+10^(pKa-pH))."""
    pKa = 4.5
    ph = 6.5
    ionized = 1.0 / (1.0 + 10.0 ** (pKa - ph))
    expected = 1.0 + ionized * 2.0
    r = predict_sif_solubility("AcidD", 2.0, pKa, "acid", 1.0, "FaSSIF")
    assert abs(r.ph_correction_factor - expected) < 1e-10


def test_ph_correction_base_fassif():
    """Base in FaSSIF (pH 6.5)."""
    pKa = 8.0
    ph = 6.5
    ionized = 1.0 / (1.0 + 10.0 ** (ph - pKa))
    expected = 1.0 + ionized * 2.0
    r = predict_sif_solubility("BaseD", 2.0, pKa, "base", 1.0, "FaSSIF")
    assert abs(r.ph_correction_factor - expected) < 1e-10


# --- Biorelevant class tests ---


def test_biorelevant_class_solubility_limited():
    """Very low intrinsic solubility should yield solubility_limited."""
    r = predict_sif_solubility("LowSol", 1.0, 7.0, "neutral", 0.001, "FaSSGF")
    assert r.biorelevant_class == "solubility_limited"


def test_biorelevant_class_permeability_limited():
    """Low permeability should yield permeability_limited."""
    r = predict_sif_solubility("LowPerm", 2.0, 4.5, "acid", 1.0, "FaSSIF", peff_cm_per_s=5e-6)
    assert r.biorelevant_class == "permeability_limited"


def test_biorelevant_class_not_limited():
    """Good solubility and permeability should yield not_limited."""
    r = predict_sif_solubility("GoodD", 2.0, 4.5, "acid", 1.0, "FaSSIF", peff_cm_per_s=1e-4)
    assert r.biorelevant_class == "not_limited"


# --- Absorption potential ---


def test_absorption_potential_value():
    r = predict_sif_solubility("DrugA", 2.0, 4.5, "acid", 0.5, "FaSSIF", 1e-4)
    expected = math.log10(max(1e-10, r.solubility_sif_mg_mL * 1e-4))
    assert abs(r.absorption_potential - expected) < 1e-10


# --- Fluid type tests ---


def test_all_fluid_types_valid():
    for ft in ["FaSSIF", "FeSSIF", "FaSSGF"]:
        r = predict_sif_solubility("DrugA", 2.0, 4.5, "acid", 0.5, ft)
        assert r.fluid_type == ft


def test_default_fluid_type():
    r = predict_sif_solubility("DrugA", 2.0, 4.5, "acid", 0.5)
    assert r.fluid_type == "FaSSIF"


# --- Validation tests ---


def test_invalid_fluid_type():
    with pytest.raises(ValueError, match="Invalid fluid_type"):
        predict_sif_solubility("DrugA", 2.0, 4.5, "acid", 0.5, "INVALID")


def test_invalid_solubility_zero():
    with pytest.raises(ValueError, match="intrinsic_solubility"):
        predict_sif_solubility("DrugA", 2.0, 4.5, "acid", 0.0)


def test_invalid_solubility_negative():
    with pytest.raises(ValueError, match="intrinsic_solubility"):
        predict_sif_solubility("DrugA", 2.0, 4.5, "acid", -1.0)


def test_invalid_peff_zero():
    with pytest.raises(ValueError, match="peff"):
        predict_sif_solubility("DrugA", 2.0, 4.5, "acid", 0.5, peff_cm_per_s=0.0)


# --- compare_fluids tests ---


def test_compare_fluids_returns_list():
    results = compare_fluids("DrugA", 2.0, 4.5, "acid", 0.5)
    assert isinstance(results, list)
    assert len(results) == 3


def test_compare_fluids_sorted_descending():
    results = compare_fluids("DrugA", 3.0, 4.5, "acid", 0.5)
    solubilities = [r.solubility_sif_mg_mL for r in results]
    assert solubilities == sorted(solubilities, reverse=True)


def test_compare_fluids_all_types_present():
    results = compare_fluids("DrugA", 2.0, 4.5, "acid", 0.5)
    types = {r.fluid_type for r in results}
    assert types == {"FaSSIF", "FeSSIF", "FaSSGF"}
