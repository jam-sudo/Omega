"""Tests for protein_corona_predictor (Phase 796)."""

import pytest

from omega_pbpk.prediction.protein_corona_predictor import (
    ProteinCoronaResult,
    compare_surface_chemistries,
    predict_protein_corona,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _basic(surface="bare", diameter=100.0, zeta=0.0, peg_density=0.0, peg_mw=2.0):
    return predict_protein_corona(
        nanoparticle_name="TestNP",
        diameter_nm=diameter,
        surface_chemistry=surface,
        zeta_potential_mV=zeta,
        peg_density_chains_per_nm2=peg_density,
        peg_mw_kDa=peg_mw,
    )


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = _basic()
    assert isinstance(result, ProteinCoronaResult)


def test_frozen_dataclass():
    result = _basic()
    with pytest.raises(Exception):
        result.mps_uptake_factor = 0.0  # type: ignore[misc]


def test_nanoparticle_name_preserved():
    result = predict_protein_corona("MyNP", 50.0)
    assert result.nanoparticle_name == "MyNP"


def test_diameter_preserved():
    result = _basic(diameter=75.0)
    assert result.diameter_nm == 75.0


def test_surface_chemistry_preserved():
    for chem in ["PEG", "bare", "cationic", "anionic", "lipid"]:
        result = _basic(surface=chem)
        assert result.surface_chemistry == chem


# ---------------------------------------------------------------------------
# Effective diameter
# ---------------------------------------------------------------------------


def test_effective_diameter_larger_than_core():
    result = _basic(diameter=100.0)
    assert result.effective_diameter_nm > 100.0


def test_effective_diameter_formula():
    result = _basic(diameter=100.0)
    expected = 100.0 + 2.0 * (result.hard_corona_thickness_nm + result.soft_corona_thickness_nm)
    assert abs(result.effective_diameter_nm - expected) < 1e-9


# ---------------------------------------------------------------------------
# PEG reduces MPS uptake vs bare
# ---------------------------------------------------------------------------


def test_peg_lower_mps_than_bare():
    peg_result = _basic(surface="PEG")
    bare_result = _basic(surface="bare")
    assert peg_result.mps_uptake_factor < bare_result.mps_uptake_factor


def test_peg_density_further_reduces_mps():
    low_density = _basic(surface="PEG", peg_density=0.1, peg_mw=2.0)
    high_density = _basic(surface="PEG", peg_density=5.0, peg_mw=5.0)
    assert high_density.mps_uptake_factor < low_density.mps_uptake_factor


# ---------------------------------------------------------------------------
# Cationic vs anionic
# ---------------------------------------------------------------------------


def test_cationic_higher_mps_than_anionic():
    cationic = _basic(surface="cationic")
    anionic = _basic(surface="anionic")
    assert cationic.mps_uptake_factor > anionic.mps_uptake_factor


def test_cationic_complement_high():
    result = _basic(surface="cationic")
    assert result.complement_activation_risk == "high"


# ---------------------------------------------------------------------------
# Lipid corona properties
# ---------------------------------------------------------------------------


def test_lipid_lower_mps_than_bare():
    lipid = _basic(surface="lipid")
    bare = _basic(surface="bare")
    assert lipid.mps_uptake_factor < bare.mps_uptake_factor


# ---------------------------------------------------------------------------
# Corona thicknesses positive
# ---------------------------------------------------------------------------


def test_hard_corona_positive():
    result = _basic()
    assert result.hard_corona_thickness_nm > 0.0


def test_soft_corona_positive():
    result = _basic()
    assert result.soft_corona_thickness_nm > 0.0


# ---------------------------------------------------------------------------
# Protein adsorption percentages
# ---------------------------------------------------------------------------


def test_albumin_pct_in_range():
    for chem in ["PEG", "bare", "cationic", "anionic", "lipid"]:
        result = _basic(surface=chem)
        assert 0.0 <= result.albumin_adsorption_pct <= 100.0


def test_fibrinogen_pct_in_range():
    for chem in ["PEG", "bare", "cationic", "anionic", "lipid"]:
        result = _basic(surface=chem)
        assert 0.0 <= result.fibrinogen_adsorption_pct <= 100.0


# ---------------------------------------------------------------------------
# Zeta potential effect
# ---------------------------------------------------------------------------


def test_high_zeta_increases_corona():
    neutral = _basic(zeta=0.0)
    charged = _basic(zeta=40.0)
    assert charged.hard_corona_thickness_nm > neutral.hard_corona_thickness_nm


# ---------------------------------------------------------------------------
# compare_surface_chemistries
# ---------------------------------------------------------------------------


def test_compare_returns_list():
    results = compare_surface_chemistries("NP1", 100.0)
    assert isinstance(results, list)


def test_compare_returns_5_results():
    results = compare_surface_chemistries("NP1", 100.0)
    assert len(results) == 5


def test_compare_sorted_by_mps_ascending():
    results = compare_surface_chemistries("NP1", 100.0)
    factors = [r.mps_uptake_factor for r in results]
    assert factors == sorted(factors)


def test_compare_all_surface_types_present():
    results = compare_surface_chemistries("NP1", 100.0)
    chemistries = {r.surface_chemistry for r in results}
    assert chemistries == {"PEG", "bare", "cationic", "anionic", "lipid"}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_zero_diameter_raises():
    with pytest.raises(ValueError):
        predict_protein_corona("NP", 0.0)


def test_negative_diameter_raises():
    with pytest.raises(ValueError):
        predict_protein_corona("NP", -10.0)


def test_invalid_surface_chemistry_raises():
    with pytest.raises(ValueError):
        predict_protein_corona("NP", 100.0, surface_chemistry="unknown")


# ---------------------------------------------------------------------------
# Notes is string
# ---------------------------------------------------------------------------


def test_notes_is_string():
    result = _basic()
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Half-life factor
# ---------------------------------------------------------------------------


def test_half_life_factor_positive():
    for chem in ["PEG", "bare", "cationic", "anionic", "lipid"]:
        result = _basic(surface=chem)
        assert result.half_life_factor > 0.0
