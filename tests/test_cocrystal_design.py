"""Tests for Phase 245: cocrystal_design.py"""

import pytest

from omega_pbpk.prediction.cocrystal_design import (
    CocrystalResult,
    design_cocrystal,
    screen_coformers,
)

# --- Basic return type and structure ---


def test_design_cocrystal_returns_cocrystal_result():
    result = design_cocrystal("DrugA", 7.0, 1.0, "nicotinamide")
    assert isinstance(result, CocrystalResult)


def test_feasibility_valid_values_high():
    result = design_cocrystal("DrugA", 7.0, 1.0, "nicotinamide")
    assert result.cocrystal_feasibility in ("high", "low")


def test_feasibility_low_when_delta_pka_below_threshold():
    # nicotinamide threshold=3.0, coformer_pka=3.35; drug_pka=3.5 -> delta=0.15 < 3.0
    result = design_cocrystal("DrugA", 3.5, 1.0, "nicotinamide")
    assert result.cocrystal_feasibility == "low"


def test_feasibility_high_when_delta_pka_above_threshold():
    # nicotinamide threshold=3.0, coformer_pka=3.35; drug_pka=10.0 -> delta=6.65 > 3.0
    result = design_cocrystal("DrugB", 10.0, 1.0, "nicotinamide")
    assert result.cocrystal_feasibility == "high"


# --- delta_pka exact check ---


def test_delta_pka_exact_nicotinamide():
    drug_pka = 8.0
    result = design_cocrystal("DrugC", drug_pka, 1.0, "nicotinamide")
    expected = abs(drug_pka - 3.35)
    assert abs(result.delta_pka - expected) < 1e-10


def test_delta_pka_exact_glutaric_acid():
    drug_pka = 2.0
    result = design_cocrystal("DrugD", drug_pka, 1.0, "glutaric_acid")
    expected = abs(drug_pka - 4.34)
    assert abs(result.delta_pka - expected) < 1e-10


def test_delta_pka_exact_caffeine():
    drug_pka = 5.0
    result = design_cocrystal("DrugE", drug_pka, 1.0, "caffeine")
    expected = abs(drug_pka - 0.5)
    assert abs(result.delta_pka - expected) < 1e-10


# --- stability_score bounds ---


def test_stability_score_in_range():
    for coformer in ("nicotinamide", "saccharin", "caffeine", "glutaric_acid", "urea"):
        result = design_cocrystal("DrugF", 7.0, 1.0, coformer)
        assert 0.0 <= result.stability_score <= 100.0, f"Out of range for {coformer}"


# --- solubility and bioavailability ---


def test_predicted_solubility_greater_than_baseline():
    # All modifiers > 1, so solubility should always be > baseline
    baseline = 2.0
    for coformer in ("nicotinamide", "saccharin", "caffeine", "glutaric_acid", "urea"):
        result = design_cocrystal("DrugG", 7.0, baseline, coformer)
        assert result.predicted_solubility_mg_mL > baseline, f"Failed for {coformer}"


def test_bioavailability_improvement_positive():
    baseline = 1.0
    for coformer in ("nicotinamide", "saccharin", "caffeine", "glutaric_acid", "urea"):
        result = design_cocrystal("DrugH", 7.0, baseline, coformer)
        assert result.bioavailability_improvement_pct > 0.0, f"Failed for {coformer}"


# --- screen_coformers ---


def test_screen_returns_five_results():
    results = screen_coformers("DrugI", 7.0, 1.0)
    assert len(results) == 5


def test_screen_sorted_by_stability_descending():
    results = screen_coformers("DrugJ", 7.0, 1.0)
    scores = [r.stability_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_screen_all_coformers_represented():
    results = screen_coformers("DrugK", 5.0, 1.0)
    coformers = {r.coformer for r in results}
    assert coformers == {"nicotinamide", "saccharin", "caffeine", "glutaric_acid", "urea"}


# --- recommended logic ---


def test_recommended_true_requires_high_feasibility_and_stability():
    results = screen_coformers("DrugL", 7.0, 1.0)
    for r in results:
        if r.recommended:
            assert r.cocrystal_feasibility == "high"
            assert r.stability_score >= 50.0


def test_recommended_false_when_low_feasibility():
    # drug_pka very close to coformer_pka -> low feasibility
    result = design_cocrystal("DrugM", 3.35, 1.0, "nicotinamide")
    # delta_pka = 0, feasibility = "low"
    assert result.cocrystal_feasibility == "low"
    assert result.recommended is False


# --- melting point reduction ---


def test_melting_point_reduction_zero_when_low_feasibility():
    result = design_cocrystal("DrugN", 3.35, 1.0, "nicotinamide")
    assert result.melting_point_reduction_C == 0.0


def test_melting_point_reduction_positive_when_high_feasibility():
    result = design_cocrystal("DrugO", 10.0, 1.0, "nicotinamide")
    assert result.cocrystal_feasibility == "high"
    assert result.melting_point_reduction_C > 0.0


# --- Validation errors ---


def test_invalid_drug_pka_too_low():
    with pytest.raises(ValueError, match="drug_pka"):
        design_cocrystal("DrugP", -1.0, 1.0, "nicotinamide")


def test_invalid_drug_pka_too_high():
    with pytest.raises(ValueError, match="drug_pka"):
        design_cocrystal("DrugQ", 15.0, 1.0, "nicotinamide")


def test_invalid_baseline_solubility():
    with pytest.raises(ValueError, match="baseline_solubility"):
        design_cocrystal("DrugR", 7.0, 0.0, "nicotinamide")


def test_invalid_coformer():
    with pytest.raises(ValueError, match="coformer"):
        design_cocrystal("DrugS", 7.0, 1.0, "unknown_coformer")


def test_coformer_pka_stored_correctly():
    result = design_cocrystal("DrugT", 7.0, 1.0, "urea")
    assert abs(result.coformer_pka - 0.1) < 1e-10
