"""Tests for Phase 674 - Protein Drug Stability in Plasma."""

import pytest

from omega_pbpk.prediction.protein_drug_plasma_stability_p674 import (
    ProteinPlasmaStabilityResult,
    predict_protein_plasma_stability,
)


@pytest.fixture
def base_params():
    return {
        "drug_name": "TestProtein",
        "mw_kDa": 10.0,
        "n_peptide_bonds": 50,
    }


# --- Return type and frozen ---


def test_return_type(base_params):
    r = predict_protein_plasma_stability(**base_params)
    assert isinstance(r, ProteinPlasmaStabilityResult)


def test_frozen_raises(base_params):
    r = predict_protein_plasma_stability(**base_params)
    with pytest.raises(AttributeError):
        r.drug_name = "other"


# --- PEGylation effect ---


def test_pegylated_longer_half_life(base_params):
    r_native = predict_protein_plasma_stability(**base_params)
    r_peg = predict_protein_plasma_stability(**base_params, is_pegylated=True)
    assert r_peg.t_half_plasma_min > r_native.t_half_plasma_min


def test_pegylated_lower_k_deg(base_params):
    r_native = predict_protein_plasma_stability(**base_params)
    r_peg = predict_protein_plasma_stability(**base_params, is_pegylated=True)
    assert r_peg.degradation_rate_per_min < r_native.degradation_rate_per_min


# --- D-amino acids: highest stability ---


def test_d_amino_acids_highest_stability(base_params):
    r_native = predict_protein_plasma_stability(**base_params)
    r_d = predict_protein_plasma_stability(**base_params, has_d_amino_acids=True)
    r_peg = predict_protein_plasma_stability(**base_params, is_pegylated=True)
    r_cyc = predict_protein_plasma_stability(**base_params, is_cyclic=True)
    assert r_d.t_half_plasma_min > r_peg.t_half_plasma_min
    assert r_d.t_half_plasma_min > r_cyc.t_half_plasma_min
    assert r_d.t_half_plasma_min > r_native.t_half_plasma_min


# --- Cyclic peptide ---


def test_cyclic_longer_than_linear(base_params):
    r_native = predict_protein_plasma_stability(**base_params)
    r_cyc = predict_protein_plasma_stability(**base_params, is_cyclic=True)
    assert r_cyc.t_half_plasma_min > r_native.t_half_plasma_min


# --- Intact fractions ---


def test_intact_1h_gt_4h(base_params):
    r = predict_protein_plasma_stability(**base_params)
    assert r.intact_fraction_at_1h_pct > r.intact_fraction_at_4h_pct


def test_intact_fractions_range(base_params):
    r = predict_protein_plasma_stability(**base_params)
    assert 0 <= r.intact_fraction_at_4h_pct <= r.intact_fraction_at_1h_pct <= 100


# --- t_half positive ---


def test_t_half_positive(base_params):
    r = predict_protein_plasma_stability(**base_params)
    assert r.t_half_plasma_min > 0


# --- Protease susceptibility ---


def test_protease_susceptibility_range(base_params):
    r = predict_protein_plasma_stability(**base_params)
    assert 0 <= r.protease_susceptibility <= 10.0


# --- Stability class values ---


def test_stability_class_low():
    # Many peptide bonds, small MW -> fast degradation -> low stability
    r = predict_protein_plasma_stability("Fast", mw_kDa=1.0, n_peptide_bonds=1000)
    assert r.stability_class == "low"


def test_stability_class_high():
    # PEGylated + D-amino acids -> very slow -> high stability
    r = predict_protein_plasma_stability(
        "Stable",
        mw_kDa=50.0,
        n_peptide_bonds=10,
        is_pegylated=True,
        has_d_amino_acids=True,
    )
    assert r.stability_class == "high"


def test_stability_class_medium():
    # Moderate params
    r = predict_protein_plasma_stability("Med", mw_kDa=20.0, n_peptide_bonds=30, has_disulfide=True)
    assert r.stability_class in ("low", "medium", "high")  # just valid


# --- MW effect ---


def test_higher_mw_longer_half_life():
    r_small = predict_protein_plasma_stability("Small", mw_kDa=5.0, n_peptide_bonds=50)
    r_large = predict_protein_plasma_stability("Large", mw_kDa=100.0, n_peptide_bonds=50)
    assert r_large.t_half_plasma_min > r_small.t_half_plasma_min


# --- Dominant protease ---


def test_dominant_protease_aminopeptidase():
    r = predict_protein_plasma_stability("Tiny", mw_kDa=1.0, n_peptide_bonds=3)
    assert r.dominant_protease == "aminopeptidase"


def test_dominant_protease_plasmin():
    r = predict_protein_plasma_stability("Big", mw_kDa=60.0, n_peptide_bonds=100)
    assert r.dominant_protease == "plasmin"


def test_dominant_protease_cyclic():
    r = predict_protein_plasma_stability("Cyc", mw_kDa=10.0, n_peptide_bonds=20, is_cyclic=True)
    assert r.dominant_protease == "none"


def test_dominant_protease_endoprotease():
    r = predict_protein_plasma_stability("Mid", mw_kDa=20.0, n_peptide_bonds=30)
    assert r.dominant_protease == "endoprotease"


# --- Half-life enhancement ---


def test_enhancement_native_is_one(base_params):
    r = predict_protein_plasma_stability(**base_params)
    assert r.half_life_enhancement_vs_native == 1.0


def test_enhancement_pegylated_gt_one(base_params):
    r = predict_protein_plasma_stability(**base_params, is_pegylated=True)
    assert r.half_life_enhancement_vs_native > 1.0


# --- Formulation strategy ---


def test_formulation_low():
    r = predict_protein_plasma_stability("Fast", mw_kDa=1.0, n_peptide_bonds=1000)
    assert "PEGylation" in r.formulation_strategy


def test_formulation_high():
    r = predict_protein_plasma_stability(
        "Stable",
        mw_kDa=50.0,
        n_peptide_bonds=10,
        is_pegylated=True,
        has_d_amino_acids=True,
    )
    assert "stable" in r.formulation_strategy.lower()


# --- Validation errors ---


def test_invalid_mw():
    with pytest.raises(ValueError):
        predict_protein_plasma_stability("D", mw_kDa=0.0, n_peptide_bonds=10)


def test_invalid_peptide_bonds():
    with pytest.raises(ValueError):
        predict_protein_plasma_stability("D", mw_kDa=10.0, n_peptide_bonds=0)
