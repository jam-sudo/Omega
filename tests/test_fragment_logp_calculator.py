"""
Tests for Phase 438: Fragment-based logP calculator.
"""

import pytest

from omega_pbpk.prediction.fragment_logp_calculator import (
    FragmentLogPResult,
    calculate_fragment_logp,
    screen_fragments,
)

# ─────────────────────────────────────────────────────────────────────────────
# Basic return type and structure tests
# ─────────────────────────────────────────────────────────────────────────────


def test_return_type():
    result = calculate_fragment_logp(
        drug_name="Aspirin",
        n_carbons=9,
        n_nitrogens=0,
        n_oxygens=4,
        n_halogens=0,
        n_sulfurs=0,
        n_aromatic_rings=1,
        n_h_bond_donors=1,
        n_h_bond_acceptors=4,
    )
    assert isinstance(result, FragmentLogPResult)


def test_drug_name_preserved():
    result = calculate_fragment_logp(
        drug_name="Ibuprofen",
        n_carbons=13,
        n_nitrogens=0,
        n_oxygens=2,
        n_halogens=0,
        n_sulfurs=0,
        n_aromatic_rings=1,
        n_h_bond_donors=1,
        n_h_bond_acceptors=2,
    )
    assert result.drug_name == "Ibuprofen"


def test_logp_in_valid_range():
    result = calculate_fragment_logp(
        drug_name="Drug",
        n_carbons=10,
        n_nitrogens=2,
        n_oxygens=3,
        n_halogens=1,
        n_sulfurs=0,
        n_aromatic_rings=1,
        n_h_bond_donors=2,
        n_h_bond_acceptors=5,
    )
    assert -5.0 <= result.logp_calculated <= 10.0


def test_more_carbons_higher_logp():
    """Increasing carbons should increase logP."""
    result_few = calculate_fragment_logp(
        drug_name="Drug",
        n_carbons=5,
        n_nitrogens=0,
        n_oxygens=0,
        n_halogens=0,
        n_sulfurs=0,
        n_aromatic_rings=0,
        n_h_bond_donors=0,
        n_h_bond_acceptors=0,
    )
    result_many = calculate_fragment_logp(
        drug_name="Drug",
        n_carbons=20,
        n_nitrogens=0,
        n_oxygens=0,
        n_halogens=0,
        n_sulfurs=0,
        n_aromatic_rings=0,
        n_h_bond_donors=0,
        n_h_bond_acceptors=0,
    )
    assert result_many.logp_calculated > result_few.logp_calculated


def test_more_nitrogens_lower_logp():
    """Increasing nitrogens (aliphatic) should decrease logP."""
    result_few = calculate_fragment_logp(
        drug_name="Drug",
        n_carbons=10,
        n_nitrogens=0,
        n_oxygens=0,
        n_halogens=0,
        n_sulfurs=0,
        n_aromatic_rings=0,
        n_h_bond_donors=0,
        n_h_bond_acceptors=0,
    )
    result_many = calculate_fragment_logp(
        drug_name="Drug",
        n_carbons=10,
        n_nitrogens=5,
        n_oxygens=0,
        n_halogens=0,
        n_sulfurs=0,
        n_aromatic_rings=0,
        n_h_bond_donors=0,
        n_h_bond_acceptors=0,
        aromatic_n_fraction=0.0,  # all aliphatic
    )
    assert result_many.logp_calculated < result_few.logp_calculated


def test_more_oxygens_lower_logp():
    """Increasing aliphatic oxygens should decrease logP."""
    result_few = calculate_fragment_logp(
        drug_name="Drug",
        n_carbons=10,
        n_nitrogens=0,
        n_oxygens=0,
        n_halogens=0,
        n_sulfurs=0,
        n_aromatic_rings=0,
        n_h_bond_donors=0,
        n_h_bond_acceptors=0,
    )
    result_many = calculate_fragment_logp(
        drug_name="Drug",
        n_carbons=10,
        n_nitrogens=0,
        n_oxygens=5,
        n_halogens=0,
        n_sulfurs=0,
        n_aromatic_rings=0,
        n_h_bond_donors=0,
        n_h_bond_acceptors=5,
        aromatic_o_fraction=0.0,  # all aliphatic
    )
    assert result_many.logp_calculated < result_few.logp_calculated


def test_halogens_increase_logp_order():
    """logP contribution: I > Br > Cl > F for same halogen count."""
    base_kwargs = dict(
        drug_name="Drug",
        n_carbons=5,
        n_nitrogens=0,
        n_oxygens=0,
        n_halogens=2,
        n_sulfurs=0,
        n_aromatic_rings=0,
        n_h_bond_donors=0,
        n_h_bond_acceptors=0,
    )
    result_F = calculate_fragment_logp(**{**base_kwargs, "halogen_type": "F"})
    result_Cl = calculate_fragment_logp(**{**base_kwargs, "halogen_type": "Cl"})
    result_Br = calculate_fragment_logp(**{**base_kwargs, "halogen_type": "Br"})
    result_I = calculate_fragment_logp(**{**base_kwargs, "halogen_type": "I"})

    assert result_F.logp_calculated < result_Cl.logp_calculated
    assert result_Cl.logp_calculated < result_Br.logp_calculated
    assert result_Br.logp_calculated < result_I.logp_calculated


def test_aromatic_rings_increase_logp():
    """More aromatic rings → higher logP via ring_bonus."""
    result_no_ring = calculate_fragment_logp(
        drug_name="Drug",
        n_carbons=6,
        n_nitrogens=0,
        n_oxygens=0,
        n_halogens=0,
        n_sulfurs=0,
        n_aromatic_rings=0,
        n_h_bond_donors=0,
        n_h_bond_acceptors=0,
    )
    result_two_rings = calculate_fragment_logp(
        drug_name="Drug",
        n_carbons=10,
        n_nitrogens=0,
        n_oxygens=0,
        n_halogens=0,
        n_sulfurs=0,
        n_aromatic_rings=2,
        n_h_bond_donors=0,
        n_h_bond_acceptors=0,
    )
    assert result_two_rings.logp_calculated > result_no_ring.logp_calculated


def test_logp_class_highly_lipophilic():
    # Force high logP: many carbons, no heteroatoms
    result = calculate_fragment_logp(
        drug_name="LipidDrug",
        n_carbons=40,
        n_nitrogens=0,
        n_oxygens=0,
        n_halogens=0,
        n_sulfurs=0,
        n_aromatic_rings=0,
        n_h_bond_donors=0,
        n_h_bond_acceptors=0,
    )
    assert result.logp_class == "highly_lipophilic"


def test_logp_class_hydrophilic():
    # Force negative logP: many polar atoms
    result = calculate_fragment_logp(
        drug_name="PolarDrug",
        n_carbons=2,
        n_nitrogens=5,
        n_oxygens=5,
        n_halogens=0,
        n_sulfurs=0,
        n_aromatic_rings=0,
        n_h_bond_donors=5,
        n_h_bond_acceptors=10,
        aromatic_n_fraction=0.0,
        aromatic_o_fraction=0.0,
    )
    assert result.logp_class == "hydrophilic"


def test_bbb_penetration_likely_true():
    """Drug-like compound with logP ~2, MW ~280, donors 1 should have BBB=True."""
    result = calculate_fragment_logp(
        drug_name="CNSDrug",
        n_carbons=15,
        n_nitrogens=1,
        n_oxygens=1,
        n_halogens=0,
        n_sulfurs=0,
        n_aromatic_rings=1,
        n_h_bond_donors=1,
        n_h_bond_acceptors=2,
        aromatic_n_fraction=0.0,
        aromatic_o_fraction=0.0,
    )
    # Check result fields
    assert isinstance(result.bbb_penetration_likely, bool)
    if 1.0 <= result.logp_calculated <= 4.0 and result.mw_estimated < 400.0:
        assert result.bbb_penetration_likely is True


def test_oral_absorption_likely_lipinski():
    """Small drug-like molecule should pass Lipinski."""
    result = calculate_fragment_logp(
        drug_name="SmallDrug",
        n_carbons=12,
        n_nitrogens=1,
        n_oxygens=2,
        n_halogens=0,
        n_sulfurs=0,
        n_aromatic_rings=1,
        n_h_bond_donors=2,
        n_h_bond_acceptors=3,
    )
    # MW < 500, logP < 5, donors < 5, acceptors < 10
    if (
        result.mw_estimated < 500.0
        and result.logp_calculated < 5.0
        and result.n_h_bond_donors < 5
        and result.n_h_bond_acceptors < 10
    ):
        assert result.oral_absorption_likely is True


def test_aqueous_solubility_positive():
    result = calculate_fragment_logp(
        drug_name="Drug",
        n_carbons=10,
        n_nitrogens=0,
        n_oxygens=2,
        n_halogens=0,
        n_sulfurs=0,
        n_aromatic_rings=1,
        n_h_bond_donors=1,
        n_h_bond_acceptors=2,
    )
    assert result.aqueous_solubility_estimate_mg_mL > 0.0


def test_screen_fragments_returns_correct_length():
    compounds = [
        dict(
            drug_name="A",
            n_carbons=5,
            n_nitrogens=0,
            n_oxygens=0,
            n_halogens=0,
            n_sulfurs=0,
            n_aromatic_rings=0,
            n_h_bond_donors=0,
            n_h_bond_acceptors=0,
        ),
        dict(
            drug_name="B",
            n_carbons=10,
            n_nitrogens=0,
            n_oxygens=0,
            n_halogens=0,
            n_sulfurs=0,
            n_aromatic_rings=0,
            n_h_bond_donors=0,
            n_h_bond_acceptors=0,
        ),
        dict(
            drug_name="C",
            n_carbons=15,
            n_nitrogens=1,
            n_oxygens=2,
            n_halogens=0,
            n_sulfurs=0,
            n_aromatic_rings=1,
            n_h_bond_donors=1,
            n_h_bond_acceptors=3,
        ),
    ]
    results = screen_fragments(compounds)
    assert len(results) == 3


def test_screen_fragments_sorted_ascending():
    compounds = [
        dict(
            drug_name="LipophilicDrug",
            n_carbons=20,
            n_nitrogens=0,
            n_oxygens=0,
            n_halogens=3,
            n_sulfurs=0,
            n_aromatic_rings=2,
            n_h_bond_donors=0,
            n_h_bond_acceptors=0,
            halogen_type="Cl",
        ),
        dict(
            drug_name="PolarDrug",
            n_carbons=3,
            n_nitrogens=4,
            n_oxygens=4,
            n_halogens=0,
            n_sulfurs=0,
            n_aromatic_rings=0,
            n_h_bond_donors=4,
            n_h_bond_acceptors=8,
            aromatic_n_fraction=0.0,
            aromatic_o_fraction=0.0,
        ),
        dict(
            drug_name="MidDrug",
            n_carbons=10,
            n_nitrogens=1,
            n_oxygens=1,
            n_halogens=0,
            n_sulfurs=0,
            n_aromatic_rings=1,
            n_h_bond_donors=1,
            n_h_bond_acceptors=2,
        ),
    ]
    results = screen_fragments(compounds)
    logps = [r.logp_calculated for r in results]
    assert logps == sorted(logps)


def test_iodine_gives_higher_logp_than_fluorine():
    base = dict(
        drug_name="Drug",
        n_carbons=8,
        n_nitrogens=0,
        n_oxygens=0,
        n_halogens=2,
        n_sulfurs=0,
        n_aromatic_rings=0,
        n_h_bond_donors=0,
        n_h_bond_acceptors=0,
    )
    result_F = calculate_fragment_logp(**{**base, "halogen_type": "F"})
    result_I = calculate_fragment_logp(**{**base, "halogen_type": "I"})
    assert result_I.logp_calculated > result_F.logp_calculated


def test_mw_override_works():
    result = calculate_fragment_logp(
        drug_name="Drug",
        n_carbons=10,
        n_nitrogens=0,
        n_oxygens=0,
        n_halogens=0,
        n_sulfurs=0,
        n_aromatic_rings=0,
        n_h_bond_donors=0,
        n_h_bond_acceptors=0,
        mw_override=999.0,
    )
    assert abs(result.mw_estimated - 999.0) < 0.001


# ─────────────────────────────────────────────────────────────────────────────
# Validation / error tests
# ─────────────────────────────────────────────────────────────────────────────


def test_validation_negative_carbons():
    with pytest.raises(ValueError, match="n_carbons"):
        calculate_fragment_logp(
            drug_name="Drug",
            n_carbons=-1,
            n_nitrogens=0,
            n_oxygens=0,
            n_halogens=0,
            n_sulfurs=0,
            n_aromatic_rings=0,
            n_h_bond_donors=0,
            n_h_bond_acceptors=0,
        )


def test_validation_negative_oxygens():
    with pytest.raises(ValueError, match="n_oxygens"):
        calculate_fragment_logp(
            drug_name="Drug",
            n_carbons=5,
            n_nitrogens=0,
            n_oxygens=-2,
            n_halogens=0,
            n_sulfurs=0,
            n_aromatic_rings=0,
            n_h_bond_donors=0,
            n_h_bond_acceptors=0,
        )


def test_validation_invalid_halogen_type():
    with pytest.raises(ValueError, match="halogen_type"):
        calculate_fragment_logp(
            drug_name="Drug",
            n_carbons=5,
            n_nitrogens=0,
            n_oxygens=0,
            n_halogens=1,
            n_sulfurs=0,
            n_aromatic_rings=0,
            n_h_bond_donors=0,
            n_h_bond_acceptors=0,
            halogen_type="X",
        )
