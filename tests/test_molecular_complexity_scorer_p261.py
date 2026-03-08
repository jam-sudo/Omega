"""Tests for Phase 261 — molecular complexity scorer (p261 variant)."""

import pytest

from omega_pbpk.prediction.molecular_complexity_scorer_p261 import (
    MolecularComplexityResult,
    compare_complexity,
    score_complexity,
)

# --- Return type ---


def test_returns_molecular_complexity_result():
    result = score_complexity(
        drug_name="Aspirin",
        mw=180.2,
        logp=1.2,
        psa_A2=63.6,
        hbd=1,
        hba=4,
    )
    assert isinstance(result, MolecularComplexityResult)


# --- SA score bounds ---


def test_sa_score_in_range_simple():
    result = score_complexity("Simple", mw=150.0, logp=2.0, psa_A2=30.0, hbd=1, hba=2)
    assert 1.0 <= result.synthetic_accessibility_score <= 10.0


def test_sa_score_in_range_complex():
    result = score_complexity(
        "Complex",
        mw=900.0,
        logp=1.0,
        psa_A2=200.0,
        hbd=6,
        hba=12,
        rotatable_bonds=15,
        rings=6,
        chiral_centers=5,
        stereocenters=5,
    )
    assert 1.0 <= result.synthetic_accessibility_score <= 10.0


def test_sa_score_clamped_at_10():
    # Extreme compound: high MW, many rings/chirals → SA should not exceed 10
    result = score_complexity(
        "Extreme",
        mw=2000.0,
        logp=0.0,
        psa_A2=500.0,
        hbd=20,
        hba=30,
        rotatable_bonds=20,
        rings=10,
        chiral_centers=10,
        stereocenters=10,
    )
    assert result.synthetic_accessibility_score <= 10.0


def test_sa_score_clamped_at_1():
    # High logP lowers SA; minimum should be 1
    result = score_complexity(
        "HighLogP",
        mw=100.0,
        logp=50.0,
        psa_A2=0.0,
        hbd=0,
        hba=0,
        rotatable_bonds=0,
        rings=0,
        chiral_centers=0,
        stereocenters=0,
    )
    assert result.synthetic_accessibility_score >= 1.0


# --- Fragment complexity score bounds ---


def test_fragment_complexity_in_range():
    result = score_complexity("Drug", mw=400.0, logp=2.5, psa_A2=80.0, hbd=2, hba=5)
    assert 0.0 <= result.fragment_complexity_score <= 100.0


def test_fragment_complexity_capped_at_100():
    result = score_complexity(
        "Extreme",
        mw=5000.0,
        logp=1.0,
        psa_A2=1000.0,
        hbd=10,
        hba=20,
        rotatable_bonds=50,
        rings=20,
        chiral_centers=20,
    )
    assert result.fragment_complexity_score == 100.0


# --- Higher MW → higher SA ---


def test_higher_mw_higher_sa():
    low = score_complexity("LowMW", mw=150.0, logp=2.0, psa_A2=50.0, hbd=1, hba=2)
    high = score_complexity("HighMW", mw=800.0, logp=2.0, psa_A2=50.0, hbd=1, hba=2)
    assert high.synthetic_accessibility_score >= low.synthetic_accessibility_score


# --- More chiral centers → higher SA ---


def test_more_chirals_higher_sa():
    no_chiral = score_complexity(
        "NoChiral", mw=400.0, logp=2.0, psa_A2=80.0, hbd=2, hba=5, chiral_centers=0, stereocenters=0
    )
    many_chiral = score_complexity(
        "ManyChiral",
        mw=400.0,
        logp=2.0,
        psa_A2=80.0,
        hbd=2,
        hba=5,
        chiral_centers=4,
        stereocenters=4,
    )
    assert many_chiral.synthetic_accessibility_score >= no_chiral.synthetic_accessibility_score


# --- development_risk valid values ---


def test_development_risk_valid_values():
    valid = {"low", "moderate", "high"}
    for mw_val in [150.0, 400.0, 900.0]:
        result = score_complexity("Drug", mw=mw_val, logp=2.0, psa_A2=70.0, hbd=2, hba=5)
        assert result.development_risk in valid


def test_development_risk_low_for_simple():
    # Low MW, high logP, no chirals → SA < 3 → risk = low
    result = score_complexity(
        "Simple",
        mw=150.0,
        logp=5.0,
        psa_A2=20.0,
        hbd=0,
        hba=1,
        rotatable_bonds=2,
        rings=1,
        chiral_centers=0,
        stereocenters=0,
    )
    assert result.development_risk == "low"


# --- molecular_complexity_class valid values ---


def test_molecular_complexity_class_valid_values():
    valid = {"simple", "moderate", "complex"}
    for mw_val in [100.0, 400.0, 900.0]:
        result = score_complexity("Drug", mw=mw_val, logp=2.0, psa_A2=70.0, hbd=2, hba=5)
        assert result.molecular_complexity_class in valid


# --- compare_complexity sorts ascending by SA ---


def test_compare_sorts_ascending_by_sa():
    compounds = [
        {
            "drug_name": "Complex",
            "mw": 900.0,
            "logp": 1.0,
            "psa_A2": 200.0,
            "hbd": 5,
            "hba": 10,
            "rings": 6,
            "chiral_centers": 4,
            "stereocenters": 4,
        },
        {"drug_name": "Simple", "mw": 150.0, "logp": 3.0, "psa_A2": 30.0, "hbd": 1, "hba": 2},
        {"drug_name": "Medium", "mw": 400.0, "logp": 2.0, "psa_A2": 80.0, "hbd": 2, "hba": 5},
    ]
    results = compare_complexity(compounds)
    scores = [r.synthetic_accessibility_score for r in results]
    assert scores == sorted(scores)


def test_compare_simplest_first():
    compounds = [
        {
            "drug_name": "Hard",
            "mw": 800.0,
            "logp": 0.5,
            "psa_A2": 150.0,
            "hbd": 4,
            "hba": 8,
            "rings": 5,
            "chiral_centers": 3,
            "stereocenters": 3,
        },
        {"drug_name": "Easy", "mw": 200.0, "logp": 4.0, "psa_A2": 40.0, "hbd": 1, "hba": 2},
    ]
    results = compare_complexity(compounds)
    assert results[0].drug_name == "Easy"


def test_compare_name_alias():
    """compare_complexity accepts 'name' key as alias for 'drug_name'."""
    compounds = [
        {"name": "DrugA", "mw": 300.0, "logp": 2.0, "psa_A2": 60.0, "hbd": 2, "hba": 4},
        {"name": "DrugB", "mw": 500.0, "logp": 1.0, "psa_A2": 100.0, "hbd": 3, "hba": 6},
    ]
    results = compare_complexity(compounds)
    assert len(results) == 2
    assert all(isinstance(r, MolecularComplexityResult) for r in results)


# --- estimated_synthesis_steps >= 1 ---


def test_estimated_synthesis_steps_positive():
    result = score_complexity("Drug", mw=300.0, logp=2.0, psa_A2=70.0, hbd=2, hba=4)
    assert result.estimated_synthesis_steps >= 1


def test_estimated_synthesis_steps_increases_with_complexity():
    simple = score_complexity("Simple", mw=150.0, logp=4.0, psa_A2=20.0, hbd=0, hba=1)
    hard = score_complexity(
        "Hard",
        mw=800.0,
        logp=0.0,
        psa_A2=200.0,
        hbd=5,
        hba=12,
        rings=6,
        chiral_centers=4,
        stereocenters=4,
    )
    assert hard.estimated_synthesis_steps >= simple.estimated_synthesis_steps


# --- Validation errors ---


def test_mw_zero_raises():
    with pytest.raises(ValueError, match="mw"):
        score_complexity("Bad", mw=0.0, logp=2.0, psa_A2=60.0, hbd=1, hba=2)


def test_mw_negative_raises():
    with pytest.raises(ValueError, match="mw"):
        score_complexity("Bad", mw=-100.0, logp=2.0, psa_A2=60.0, hbd=1, hba=2)


def test_psa_negative_raises():
    with pytest.raises(ValueError, match="psa_A2"):
        score_complexity("Bad", mw=300.0, logp=2.0, psa_A2=-10.0, hbd=1, hba=2)


def test_hbd_negative_raises():
    with pytest.raises(ValueError, match="hbd"):
        score_complexity("Bad", mw=300.0, logp=2.0, psa_A2=60.0, hbd=-1, hba=2)


def test_hba_negative_raises():
    with pytest.raises(ValueError, match="hba"):
        score_complexity("Bad", mw=300.0, logp=2.0, psa_A2=60.0, hbd=1, hba=-1)


def test_chiral_centers_negative_raises():
    with pytest.raises(ValueError, match="chiral_centers"):
        score_complexity("Bad", mw=300.0, logp=2.0, psa_A2=60.0, hbd=1, hba=2, chiral_centers=-1)
