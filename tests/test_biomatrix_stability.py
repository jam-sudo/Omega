"""Tests for Phase 474: Drug Stability in Biological Matrices."""

import math

import pytest

from omega_pbpk.prediction.biomatrix_stability import (
    BiomatrixStabilityResult,
    predict_biomatrix_stability,
    screen_matrices,
)

# ---------------------------------------------------------------------------
# Basic return type and structure
# ---------------------------------------------------------------------------


def test_return_type():
    result = predict_biomatrix_stability("DrugA", matrix="plasma")
    assert isinstance(result, BiomatrixStabilityResult)


def test_drug_name_preserved():
    result = predict_biomatrix_stability("Enalapril", matrix="plasma")
    assert result.drug_name == "Enalapril"


def test_matrix_preserved():
    result = predict_biomatrix_stability("DrugA", matrix="blood")
    assert result.matrix == "blood"


def test_logP_preserved():
    result = predict_biomatrix_stability("DrugA", matrix="plasma", logP=3.5)
    assert result.logP == 3.5


def test_ester_group_preserved():
    result = predict_biomatrix_stability("DrugA", matrix="plasma", ester_group=True)
    assert result.ester_group is True


# ---------------------------------------------------------------------------
# t_half calculation
# ---------------------------------------------------------------------------


def test_t_half_formula_plasma_default():
    """t_half = ln(2) / k_deg for default plasma."""
    # Default plasma k_base = 0.02/h, logP=2, no ester, no groups
    # No adjustments apply to plasma for logP (only CYP matrices)
    result = predict_biomatrix_stability("D", matrix="plasma", logP=2.0, ester_group=False)
    expected_k = 0.02
    expected_t_half = math.log(2) / expected_k
    assert abs(result.t_half_h - expected_t_half) < 1e-9


def test_t_half_formula_liver_microsomes():
    """t_half = ln(2) / k_deg for liver microsomes, default logP=2."""
    # k_deg = 0.5 * (1 + 0.1 * 2) = 0.5 * 1.2 = 0.6
    result = predict_biomatrix_stability("D", matrix="liver_microsomes", logP=2.0)
    expected_k = 0.5 * (1.0 + 0.1 * 2.0)
    expected_t_half = math.log(2) / expected_k
    assert abs(result.t_half_h - expected_t_half) < 1e-9


# ---------------------------------------------------------------------------
# pct_remaining calculations
# ---------------------------------------------------------------------------


def test_pct_remaining_30min_exact():
    """pct_remaining_30min = 100 * exp(-k_deg * 0.5)."""
    result = predict_biomatrix_stability("D", matrix="plasma", logP=2.0, ester_group=False)
    k_deg = result.k_deg_per_h
    expected = 100.0 * math.exp(-k_deg * 0.5)
    assert abs(result.pct_remaining_30min - expected) < 1e-9


def test_pct_remaining_60min_exact():
    result = predict_biomatrix_stability("D", matrix="blood", logP=2.0)
    k_deg = result.k_deg_per_h
    expected = 100.0 * math.exp(-k_deg * 1.0)
    assert abs(result.pct_remaining_60min - expected) < 1e-9


def test_pct_remaining_2h_exact():
    result = predict_biomatrix_stability("D", matrix="urine", logP=1.0)
    k_deg = result.k_deg_per_h
    expected = 100.0 * math.exp(-k_deg * 2.0)
    assert abs(result.pct_remaining_2h - expected) < 1e-9


def test_pct_remaining_decreases_over_time():
    result = predict_biomatrix_stability("D", matrix="liver_microsomes")
    assert result.pct_remaining_30min > result.pct_remaining_60min > result.pct_remaining_2h


# ---------------------------------------------------------------------------
# Ester group effect
# ---------------------------------------------------------------------------


def test_ester_group_shortens_t_half_in_plasma():
    """Ester group → shorter t_half in plasma (esterase lability)."""
    r_no_ester = predict_biomatrix_stability("D", matrix="plasma", ester_group=False)
    r_ester = predict_biomatrix_stability("D", matrix="plasma", ester_group=True)
    assert r_ester.t_half_h < r_no_ester.t_half_h


def test_ester_group_shortens_t_half_in_blood():
    r_no_ester = predict_biomatrix_stability("D", matrix="blood", ester_group=False)
    r_ester = predict_biomatrix_stability("D", matrix="blood", ester_group=True)
    assert r_ester.t_half_h < r_no_ester.t_half_h


def test_ester_group_multiplies_k_by_5_plasma():
    """Ester + plasma → k_deg = 5 * k_base."""
    r_no_ester = predict_biomatrix_stability("D", matrix="plasma", logP=0.0, ester_group=False)
    r_ester = predict_biomatrix_stability("D", matrix="plasma", logP=0.0, ester_group=True)
    assert abs(r_ester.k_deg_per_h - 5.0 * r_no_ester.k_deg_per_h) < 1e-9


def test_ester_group_no_effect_on_urine():
    """Ester group has no effect on urine (not an esterase-rich matrix)."""
    r_no_ester = predict_biomatrix_stability("D", matrix="urine", ester_group=False)
    r_ester = predict_biomatrix_stability("D", matrix="urine", ester_group=True)
    assert abs(r_ester.k_deg_per_h - r_no_ester.k_deg_per_h) < 1e-9


# ---------------------------------------------------------------------------
# Matrix comparison
# ---------------------------------------------------------------------------


def test_liver_microsomes_shorter_t_half_than_plasma():
    """Liver microsomes degrade drug faster than plasma."""
    r_plasma = predict_biomatrix_stability("D", matrix="plasma")
    r_liver = predict_biomatrix_stability("D", matrix="liver_microsomes")
    assert r_liver.t_half_h < r_plasma.t_half_h


def test_urine_most_stable():
    """Urine has the lowest k_base — most stable."""
    r_urine = predict_biomatrix_stability("D", matrix="urine", logP=0.0, ester_group=False)
    r_plasma = predict_biomatrix_stability("D", matrix="plasma", logP=0.0, ester_group=False)
    assert r_urine.t_half_h > r_plasma.t_half_h


# ---------------------------------------------------------------------------
# Functional group effects
# ---------------------------------------------------------------------------


def test_aldehyde_shortens_t_half_in_liver():
    """Aldehyde reactive group → shorter t_half in liver microsomes."""
    r_no_fg = predict_biomatrix_stability("D", matrix="liver_microsomes", functional_groups=())
    r_fg = predict_biomatrix_stability(
        "D", matrix="liver_microsomes", functional_groups=("aldehyde",)
    )
    assert r_fg.t_half_h < r_no_fg.t_half_h


def test_aldehyde_doubles_k_deg_in_liver():
    r_no_fg = predict_biomatrix_stability(
        "D", matrix="liver_microsomes", logP=0.0, functional_groups=()
    )
    r_fg = predict_biomatrix_stability(
        "D", matrix="liver_microsomes", logP=0.0, functional_groups=("aldehyde",)
    )
    assert abs(r_fg.k_deg_per_h - 2.0 * r_no_fg.k_deg_per_h) < 1e-9


def test_multiple_reactive_groups_multiply():
    """Two reactive groups → k_deg multiplied by 2*2=4."""
    r_no_fg = predict_biomatrix_stability(
        "D", matrix="liver_microsomes", logP=0.0, functional_groups=()
    )
    r_fg = predict_biomatrix_stability(
        "D", matrix="liver_microsomes", logP=0.0, functional_groups=("aldehyde", "nitro")
    )
    assert abs(r_fg.k_deg_per_h - 4.0 * r_no_fg.k_deg_per_h) < 1e-9


def test_functional_groups_no_effect_on_plasma():
    """Reactive groups do not affect plasma k_deg."""
    r_no_fg = predict_biomatrix_stability("D", matrix="plasma", functional_groups=())
    r_fg = predict_biomatrix_stability("D", matrix="plasma", functional_groups=("aldehyde",))
    assert abs(r_fg.k_deg_per_h - r_no_fg.k_deg_per_h) < 1e-9


# ---------------------------------------------------------------------------
# Stability classification
# ---------------------------------------------------------------------------


def test_stability_class_labile():
    """liver_microsomes with aldehyde + nitro + high logP → labile."""
    result = predict_biomatrix_stability(
        "D",
        matrix="liver_microsomes",
        logP=5.0,
        functional_groups=("aldehyde", "nitro", "Michael_acceptor"),
    )
    assert result.stability_class == "labile"


def test_stability_class_stable():
    """Urine with no modifiers → stable."""
    result = predict_biomatrix_stability("D", matrix="urine", logP=0.0, ester_group=False)
    assert result.stability_class == "stable"


def test_stability_class_values():
    """All stability classes are valid strings."""
    valid = {"labile", "unstable", "moderate", "stable"}
    for matrix in [
        "plasma",
        "blood",
        "liver_microsomes",
        "intestinal_microsomes",
        "brain_homogenate",
        "urine",
    ]:
        result = predict_biomatrix_stability("D", matrix=matrix)
        assert result.stability_class in valid


# ---------------------------------------------------------------------------
# screen_matrices
# ---------------------------------------------------------------------------


def test_screen_matrices_returns_six_items():
    results = screen_matrices("DrugA")
    assert len(results) == 6


def test_screen_matrices_sorted_by_t_half_ascending():
    results = screen_matrices("DrugA")
    t_halfs = [r.t_half_h for r in results]
    assert t_halfs == sorted(t_halfs)


def test_screen_matrices_all_matrices_present():
    results = screen_matrices("DrugA")
    matrices = {r.matrix for r in results}
    assert matrices == {
        "plasma",
        "blood",
        "liver_microsomes",
        "intestinal_microsomes",
        "brain_homogenate",
        "urine",
    }


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_matrix_raises():
    with pytest.raises(ValueError, match="matrix"):
        predict_biomatrix_stability("D", matrix="saliva")


def test_invalid_matrix_raises_message():
    with pytest.raises(ValueError):
        predict_biomatrix_stability("D", matrix="lung")
