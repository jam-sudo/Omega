"""Tests for Phase 716 — molecular complexity scorer."""

import pytest

from omega_pbpk.prediction.molecular_complexity_scorer import (
    ComplexityResult,
    compare_complexity_optimization,
    score_molecular_complexity,
    screen_complexity,
)

# --- Basic return type ---


def test_returns_complexity_result():
    result = score_molecular_complexity(
        "Aspirin",
        mw=180,
        n_rings=1,
        n_stereocenters=0,
        n_rotatable_bonds=3,
        n_h_donors=1,
        n_h_acceptors=4,
    )
    assert isinstance(result, ComplexityResult)


# --- Simple molecule → low complexity ---


def test_simple_molecule_low_complexity():
    result = score_molecular_complexity(
        "Simple",
        mw=150,
        n_rings=1,
        n_stereocenters=0,
        n_rotatable_bonds=2,
        n_h_donors=1,
        n_h_acceptors=2,
    )
    assert result.complexity_score < 40


def test_simple_molecule_easy_sas():
    result = score_molecular_complexity(
        "Simple",
        mw=150,
        n_rings=1,
        n_stereocenters=0,
        n_rotatable_bonds=2,
        n_h_donors=1,
        n_h_acceptors=2,
    )
    assert result.sas_class == "easy"


# --- Complex molecule → high complexity ---


def test_complex_molecule_high_complexity():
    result = score_molecular_complexity(
        "Complex",
        mw=800,
        n_rings=6,
        n_stereocenters=4,
        n_rotatable_bonds=8,
        n_h_donors=4,
        n_h_acceptors=10,
        n_aromatic_rings=3,
        n_bridgeheads=2,
    )
    assert result.complexity_score > 60


def test_complex_molecule_hard_sas():
    result = score_molecular_complexity(
        "Complex",
        mw=800,
        n_rings=6,
        n_stereocenters=4,
        n_rotatable_bonds=8,
        n_h_donors=4,
        n_h_acceptors=10,
        n_aromatic_rings=3,
        n_bridgeheads=2,
    )
    assert result.sas_class in {"difficult", "very_difficult"}


# --- Score bounds ---


def test_complexity_score_in_range():
    result = score_molecular_complexity(
        "Extreme",
        mw=2000,
        n_rings=20,
        n_stereocenters=10,
        n_rotatable_bonds=30,
        n_h_donors=10,
        n_h_acceptors=20,
        n_aromatic_rings=10,
        n_bridgeheads=5,
    )
    assert 0 <= result.complexity_score <= 100


def test_sas_in_range():
    result = score_molecular_complexity(
        "DrugX",
        mw=400,
        n_rings=3,
        n_stereocenters=1,
        n_rotatable_bonds=5,
        n_h_donors=2,
        n_h_acceptors=5,
    )
    assert 1.0 <= result.sas <= 10.0


def test_sas_class_valid_values():
    for mw_val in [150, 400, 700, 1000]:
        result = score_molecular_complexity(
            "DrugX",
            mw=mw_val,
            n_rings=3,
            n_stereocenters=2,
            n_rotatable_bonds=5,
            n_h_donors=2,
            n_h_acceptors=5,
            n_aromatic_rings=1,
            n_bridgeheads=1,
        )
        assert result.sas_class in {"easy", "moderate", "difficult", "very_difficult"}


# --- Drug-likeness ---


def test_lipinski_pass_small_molecule():
    result = score_molecular_complexity(
        "Drug",
        mw=300,
        n_rings=2,
        n_stereocenters=0,
        n_rotatable_bonds=4,
        n_h_donors=2,
        n_h_acceptors=4,
    )
    assert result.lipinski_pass is True


def test_lipinski_fail_high_mw():
    result = score_molecular_complexity(
        "BigDrug",
        mw=600,
        n_rings=5,
        n_stereocenters=0,
        n_rotatable_bonds=6,
        n_h_donors=3,
        n_h_acceptors=8,
    )
    assert result.lipinski_pass is False


def test_veber_pass():
    result = score_molecular_complexity(
        "Drug",
        mw=300,
        n_rings=2,
        n_stereocenters=0,
        n_rotatable_bonds=5,
        n_h_donors=2,
        n_h_acceptors=4,
    )
    assert result.veber_pass is True


def test_veber_fail_too_many_rotatable():
    result = score_molecular_complexity(
        "Drug",
        mw=400,
        n_rings=2,
        n_stereocenters=0,
        n_rotatable_bonds=15,
        n_h_donors=2,
        n_h_acceptors=4,
    )
    assert result.veber_pass is False


def test_lead_like_true():
    result = score_molecular_complexity(
        "LeadCmpd",
        mw=200,
        n_rings=2,
        n_stereocenters=0,
        n_rotatable_bonds=3,
        n_h_donors=1,
        n_h_acceptors=3,
    )
    assert result.lead_like is True


def test_lead_like_false_high_mw():
    result = score_molecular_complexity(
        "Candidate",
        mw=420,
        n_rings=3,
        n_stereocenters=0,
        n_rotatable_bonds=5,
        n_h_donors=2,
        n_h_acceptors=5,
    )
    assert result.lead_like is False


def test_development_stage_lead():
    result = score_molecular_complexity(
        "LeadCmpd",
        mw=200,
        n_rings=2,
        n_stereocenters=0,
        n_rotatable_bonds=3,
        n_h_donors=1,
        n_h_acceptors=3,
    )
    assert result.development_stage_fit == "lead"


def test_development_stage_candidate():
    result = score_molecular_complexity(
        "Candidate",
        mw=400,
        n_rings=3,
        n_stereocenters=0,
        n_rotatable_bonds=6,
        n_h_donors=2,
        n_h_acceptors=6,
    )
    assert result.development_stage_fit == "candidate"


# --- Optimization priority ---


def test_optimization_priority_nonempty():
    result = score_molecular_complexity(
        "Drug",
        mw=400,
        n_rings=3,
        n_stereocenters=1,
        n_rotatable_bonds=5,
        n_h_donors=2,
        n_h_acceptors=5,
    )
    assert len(result.optimization_priority) > 0


def test_stereocenters_dominate_priority():
    # High stereocenters → stereocenters should be priority
    result = score_molecular_complexity(
        "ChiralDrug",
        mw=300,
        n_rings=1,
        n_stereocenters=4,
        n_rotatable_bonds=2,
        n_h_donors=1,
        n_h_acceptors=2,
    )
    assert result.optimization_priority == "stereocenters"


# --- screen_complexity ---


def test_screen_complexity_sorted_ascending():
    compounds = [
        {
            "compound_name": "Complex",
            "mw": 700,
            "n_rings": 5,
            "n_stereocenters": 3,
            "n_rotatable_bonds": 8,
            "n_h_donors": 3,
            "n_h_acceptors": 8,
        },
        {
            "compound_name": "Simple",
            "mw": 150,
            "n_rings": 1,
            "n_stereocenters": 0,
            "n_rotatable_bonds": 2,
            "n_h_donors": 1,
            "n_h_acceptors": 2,
        },
        {
            "compound_name": "Medium",
            "mw": 380,
            "n_rings": 3,
            "n_stereocenters": 1,
            "n_rotatable_bonds": 5,
            "n_h_donors": 2,
            "n_h_acceptors": 5,
        },
    ]
    results = screen_complexity(compounds)
    scores = [r.complexity_score for r in results]
    assert scores == sorted(scores)


def test_screen_complexity_returns_all():
    compounds = [
        {
            "compound_name": f"Cmpd{i}",
            "mw": 200 + i * 100,
            "n_rings": i,
            "n_stereocenters": 0,
            "n_rotatable_bonds": i,
            "n_h_donors": 1,
            "n_h_acceptors": 2,
        }
        for i in range(1, 5)
    ]
    results = screen_complexity(compounds)
    assert len(results) == 4


# --- compare_complexity_optimization ---


def test_compare_first_element_is_base():
    base = {
        "compound_name": "Base",
        "mw": 400,
        "n_rings": 3,
        "n_stereocenters": 2,
        "n_rotatable_bonds": 6,
        "n_h_donors": 2,
        "n_h_acceptors": 6,
    }
    variants = [
        {
            "compound_name": "VarA",
            "mw": 300,
            "n_rings": 2,
            "n_stereocenters": 1,
            "n_rotatable_bonds": 4,
            "n_h_donors": 1,
            "n_h_acceptors": 4,
        },
        {
            "compound_name": "VarB",
            "mw": 500,
            "n_rings": 4,
            "n_stereocenters": 3,
            "n_rotatable_bonds": 7,
            "n_h_donors": 3,
            "n_h_acceptors": 8,
        },
    ]
    results = compare_complexity_optimization(base, variants)
    assert results[0].compound_name == "Base"


def test_compare_variants_sorted():
    base = {
        "compound_name": "Base",
        "mw": 400,
        "n_rings": 3,
        "n_stereocenters": 2,
        "n_rotatable_bonds": 6,
        "n_h_donors": 2,
        "n_h_acceptors": 6,
    }
    variants = [
        {
            "compound_name": "Heavy",
            "mw": 700,
            "n_rings": 5,
            "n_stereocenters": 3,
            "n_rotatable_bonds": 8,
            "n_h_donors": 3,
            "n_h_acceptors": 8,
        },
        {
            "compound_name": "Light",
            "mw": 200,
            "n_rings": 1,
            "n_stereocenters": 0,
            "n_rotatable_bonds": 2,
            "n_h_donors": 1,
            "n_h_acceptors": 2,
        },
    ]
    results = compare_complexity_optimization(base, variants)
    variant_scores = [r.complexity_score for r in results[1:]]
    assert variant_scores == sorted(variant_scores)


# --- Validation ---


def test_mw_zero_raises():
    with pytest.raises(ValueError, match="mw"):
        score_molecular_complexity(
            "Bad",
            mw=0,
            n_rings=1,
            n_stereocenters=0,
            n_rotatable_bonds=2,
            n_h_donors=1,
            n_h_acceptors=2,
        )


def test_mw_negative_raises():
    with pytest.raises(ValueError, match="mw"):
        score_molecular_complexity(
            "Bad",
            mw=-100,
            n_rings=1,
            n_stereocenters=0,
            n_rotatable_bonds=2,
            n_h_donors=1,
            n_h_acceptors=2,
        )


def test_n_stereocenters_negative_raises():
    with pytest.raises(ValueError, match="n_stereocenters"):
        score_molecular_complexity(
            "Bad",
            mw=300,
            n_rings=1,
            n_stereocenters=-1,
            n_rotatable_bonds=2,
            n_h_donors=1,
            n_h_acceptors=2,
        )
