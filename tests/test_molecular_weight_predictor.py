"""Tests for prediction/molecular_weight_predictor.py — Phase 630."""

import pytest

from omega_pbpk.prediction.molecular_weight_predictor import (
    MolecularWeightResult,
    classify_mw,
    estimate_atoms_from_mw,
    estimate_mw_from_formula,
    mw_from_fragments,
)

# ---------------------------------------------------------------------------
# Basic return type
# ---------------------------------------------------------------------------


def test_returns_result():
    result = estimate_mw_from_formula(n_carbon=9, n_hydrogen=8, n_oxygen=4, compound_name="aspirin")
    assert isinstance(result, MolecularWeightResult)


# ---------------------------------------------------------------------------
# MW accuracy for known compounds
# ---------------------------------------------------------------------------


def test_aspirin_mw():
    """Aspirin: C9H8O4 → MW ≈ 180.16."""
    result = estimate_mw_from_formula(n_carbon=9, n_hydrogen=8, n_oxygen=4, compound_name="aspirin")
    assert abs(result.estimated_mw - 180.16) / 180.16 < 0.01


def test_caffeine_mw():
    """Caffeine: C8H10N4O2 → MW ≈ 194.19."""
    result = estimate_mw_from_formula(
        n_carbon=8, n_hydrogen=10, n_nitrogen=4, n_oxygen=2, compound_name="caffeine"
    )
    assert abs(result.estimated_mw - 194.19) / 194.19 < 0.01


# ---------------------------------------------------------------------------
# classify_mw
# ---------------------------------------------------------------------------


def test_mw_class_small():
    result = estimate_mw_from_formula(n_carbon=20, n_hydrogen=20, n_oxygen=2)
    # MW ~316 → small
    assert result.mw_class == "small"


def test_mw_class_fragment():
    result = estimate_mw_from_formula(n_carbon=9, n_hydrogen=8, n_oxygen=4)
    # MW ~180 → fragment
    assert result.mw_class == "fragment"


def test_mw_class_macromolecule():
    result = estimate_mw_from_formula(n_carbon=60, n_hydrogen=80, n_oxygen=15, n_nitrogen=5)
    # MW >> 1000
    assert result.mw_class == "macromolecule"


def test_classify_mw_small():
    assert classify_mw(350.0) == "small"


def test_classify_mw_medium():
    assert classify_mw(500.0) == "medium"


# ---------------------------------------------------------------------------
# BCS size category
# ---------------------------------------------------------------------------


def test_bcs_small():
    result = estimate_mw_from_formula(n_carbon=9, n_hydrogen=8, n_oxygen=4)
    # MW ~180 → bcs_small
    assert result.bcs_size_category == "bcs_small"


def test_bcs_large():
    result = estimate_mw_from_formula(n_carbon=30, n_hydrogen=40, n_oxygen=10, n_nitrogen=5)
    # MW ~600 → bcs_large
    assert result.bcs_size_category == "bcs_large"


# ---------------------------------------------------------------------------
# Lipinski / Veber
# ---------------------------------------------------------------------------


def test_lipinski_ok_500():
    """A compound with MW exactly 500 should be Lipinski OK."""
    # MW = 500: C + H should give ~500
    # C33H36O4 ≈ 33*12.011 + 36*1.008 + 4*15.999 = 396.36 + 36.288 + 63.996 = 496.6
    result = estimate_mw_from_formula(n_carbon=33, n_hydrogen=36, n_oxygen=4)
    assert result.lipinski_mw_ok is True


def test_lipinski_fail_501():
    """A compound with MW > 500 should fail Lipinski."""
    # High MW compound
    result = estimate_mw_from_formula(n_carbon=40, n_hydrogen=50, n_oxygen=5)
    assert result.lipinski_mw_ok is False


# ---------------------------------------------------------------------------
# Formula guess
# ---------------------------------------------------------------------------


def test_formula_guess_nonempty():
    result = estimate_mw_from_formula(n_carbon=9, n_hydrogen=8, n_oxygen=4)
    assert len(result.formula_guess) > 0


def test_formula_has_C_for_organic():
    result = estimate_mw_from_formula(n_carbon=9, n_hydrogen=8, n_oxygen=4)
    assert "C" in result.formula_guess


# ---------------------------------------------------------------------------
# Heavy atom estimate
# ---------------------------------------------------------------------------


def test_heavy_atoms_estimate_positive():
    result = estimate_mw_from_formula(n_carbon=9, n_hydrogen=8, n_oxygen=4)
    assert result.n_heavy_atoms_estimate > 0


# ---------------------------------------------------------------------------
# estimate_atoms_from_mw
# ---------------------------------------------------------------------------


def test_estimate_atoms_from_mw_returns_dict():
    result = estimate_atoms_from_mw(300.0)
    assert isinstance(result, dict)


def test_estimate_atoms_keys():
    result = estimate_atoms_from_mw(300.0)
    assert "n_heavy" in result
    assert "n_carbon" in result
    assert "n_hydrogen" in result
    assert "n_heteroatoms" in result


def test_estimate_atoms_n_heavy_positive():
    result = estimate_atoms_from_mw(200.0)
    assert result["n_heavy"] > 0


# ---------------------------------------------------------------------------
# mw_from_fragments
# ---------------------------------------------------------------------------


def test_mw_from_fragments_sum():
    fragments = [
        {"smarts": "c1ccccc1", "count": 2, "fragment_mw": 77.0},
        {"smarts": "C=O", "count": 1, "fragment_mw": 28.0},
    ]
    total = mw_from_fragments(fragments)
    assert total == pytest.approx(2 * 77.0 + 28.0)


# ---------------------------------------------------------------------------
# MW range tuple
# ---------------------------------------------------------------------------


def test_mw_range_tuple():
    result = estimate_mw_from_formula(n_carbon=9, n_hydrogen=8, n_oxygen=4)
    low, high = result.mw_range_likely
    assert len(result.mw_range_likely) == 2
    assert low < result.estimated_mw < high or abs(low - result.estimated_mw) < 0.1


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_invalid_all_zero_raises():
    with pytest.raises(ValueError):
        estimate_mw_from_formula(n_carbon=0, n_hydrogen=5)


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


def test_notes_nonempty():
    result = estimate_mw_from_formula(n_carbon=9, n_hydrogen=8, n_oxygen=4)
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0
