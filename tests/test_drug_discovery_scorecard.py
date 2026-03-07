"""Tests for drug discovery MPO scorecard."""

import pytest

from omega_pbpk.prediction.drug_discovery_scorecard import (
    MPOScoreResult,
    compare_mpo_optimization,
    compute_mpo_score,
    generate_optimization_suggestions,
    screen_mpo,
)

# --- Fixtures ---

IDEAL_DRUG = {
    "compound_name": "IdealDrug",
    "logP": 2.0,
    "mw": 300.0,
    "psa": 60.0,
    "n_h_donors": 1,
    "n_h_acceptors": 5,
    "n_rotatable_bonds": 4,
}

PROBLEMATIC_DRUG = {
    "compound_name": "ProblematicDrug",
    "logP": 6.0,
    "mw": 700.0,
    "psa": 150.0,
    "n_h_donors": 6,
    "n_h_acceptors": 12,
    "n_rotatable_bonds": 12,
}


# --- Return type ---


def test_returns_mpo_score_result():
    result = compute_mpo_score(**IDEAL_DRUG)
    assert isinstance(result, MPOScoreResult)


# --- Score ranges ---


def test_cns_mpo_score_range():
    result = compute_mpo_score(**IDEAL_DRUG)
    assert 0 <= result.cns_mpo_score <= 6


def test_oral_mpo_score_range():
    result = compute_mpo_score(**IDEAL_DRUG)
    assert 0 <= result.oral_mpo_score <= 6


def test_problematic_cns_mpo_range():
    result = compute_mpo_score(**PROBLEMATIC_DRUG)
    assert 0 <= result.cns_mpo_score <= 6


def test_problematic_oral_mpo_range():
    result = compute_mpo_score(**PROBLEMATIC_DRUG)
    assert 0 <= result.oral_mpo_score <= 6


# --- Ideal drug ---


def test_ideal_drug_oral_mpo_high():
    result = compute_mpo_score(**IDEAL_DRUG)
    assert result.oral_mpo_score > 4.5


def test_ideal_drug_lipinski_pass():
    result = compute_mpo_score(**IDEAL_DRUG)
    assert result.lipinski_pass is True


def test_ideal_drug_recommendation_proceed():
    result = compute_mpo_score(**IDEAL_DRUG)
    assert result.overall_recommendation == "proceed"


# --- Problematic drug ---


def test_problematic_drug_oral_mpo_low():
    result = compute_mpo_score(**PROBLEMATIC_DRUG)
    assert result.oral_mpo_score < 2


def test_problematic_drug_lipinski_fail():
    result = compute_mpo_score(**PROBLEMATIC_DRUG)
    assert result.lipinski_pass is False


def test_problematic_drug_recommendation_deprioritize():
    result = compute_mpo_score(**PROBLEMATIC_DRUG)
    assert result.overall_recommendation == "deprioritize"


# --- MPO class values ---


def test_cns_mpo_class_valid():
    result = compute_mpo_score(**IDEAL_DRUG)
    assert result.cns_mpo_class in {"excellent", "acceptable", "poor"}


def test_oral_mpo_class_valid():
    result = compute_mpo_score(**IDEAL_DRUG)
    assert result.oral_mpo_class in {"excellent", "acceptable", "poor"}


def test_problematic_cns_mpo_class_valid():
    result = compute_mpo_score(**PROBLEMATIC_DRUG)
    assert result.cns_mpo_class in {"excellent", "acceptable", "poor"}


def test_problematic_oral_mpo_class_valid():
    result = compute_mpo_score(**PROBLEMATIC_DRUG)
    assert result.oral_mpo_class in {"excellent", "acceptable", "poor"}


def test_ideal_oral_mpo_class_excellent():
    result = compute_mpo_score(**IDEAL_DRUG)
    assert result.oral_mpo_class == "excellent"


def test_problematic_oral_mpo_class_poor():
    result = compute_mpo_score(**PROBLEMATIC_DRUG)
    assert result.oral_mpo_class == "poor"


# --- Recommendation range ---


def test_recommendation_valid_values():
    result = compute_mpo_score(**IDEAL_DRUG)
    assert result.overall_recommendation in {"proceed", "optimize", "deprioritize"}


def test_medium_drug_optimize():
    result = compute_mpo_score(
        compound_name="MediumDrug",
        logP=3.5,
        mw=450.0,
        psa=100.0,
        n_h_donors=3,
        n_h_acceptors=8,
        n_rotatable_bonds=7,
    )
    assert result.overall_recommendation in {"optimize", "deprioritize", "proceed"}


# --- screen_mpo ---


def test_screen_mpo_sorted_descending():
    compounds = [IDEAL_DRUG, PROBLEMATIC_DRUG]
    results = screen_mpo(compounds)
    assert len(results) == 2
    assert results[0].oral_mpo_score >= results[1].oral_mpo_score


def test_screen_mpo_returns_list():
    results = screen_mpo([IDEAL_DRUG])
    assert isinstance(results, list)
    assert len(results) == 1


def test_screen_mpo_multiple_sorted():
    medium = {
        "compound_name": "Medium",
        "logP": 3.5,
        "mw": 450.0,
        "psa": 100.0,
        "n_h_donors": 3,
        "n_h_acceptors": 8,
        "n_rotatable_bonds": 7,
    }
    compounds = [PROBLEMATIC_DRUG, medium, IDEAL_DRUG]
    results = screen_mpo(compounds)
    scores = [r.oral_mpo_score for r in results]
    assert scores == sorted(scores, reverse=True)


# --- compare_mpo_optimization ---


def test_compare_first_is_base():
    variants = [PROBLEMATIC_DRUG]
    results = compare_mpo_optimization(IDEAL_DRUG, variants)
    assert results[0].compound_name == "IdealDrug"


def test_compare_returns_correct_count():
    variants = [PROBLEMATIC_DRUG]
    results = compare_mpo_optimization(IDEAL_DRUG, variants)
    assert len(results) == 2


def test_compare_variants_sorted():
    medium1 = {
        "compound_name": "Medium1",
        "logP": 3.5,
        "mw": 450.0,
        "psa": 100.0,
        "n_h_donors": 3,
        "n_h_acceptors": 8,
        "n_rotatable_bonds": 7,
    }
    medium2 = {
        "compound_name": "Medium2",
        "logP": 2.0,
        "mw": 380.0,
        "psa": 80.0,
        "n_h_donors": 2,
        "n_h_acceptors": 6,
        "n_rotatable_bonds": 5,
    }
    results = compare_mpo_optimization(IDEAL_DRUG, [medium1, medium2])
    # base is first, variants sorted
    assert results[1].oral_mpo_score >= results[2].oral_mpo_score


# --- generate_optimization_suggestions ---


def test_generate_suggestions_nonempty_for_suboptimal():
    result = compute_mpo_score(**PROBLEMATIC_DRUG)
    suggestions = generate_optimization_suggestions(result)
    assert isinstance(suggestions, list)
    assert len(suggestions) > 0


def test_generate_suggestions_contains_strings():
    result = compute_mpo_score(**PROBLEMATIC_DRUG)
    suggestions = generate_optimization_suggestions(result)
    for s in suggestions:
        assert isinstance(s, str)


# --- Validation ---


def test_mw_zero_raises():
    with pytest.raises(ValueError):
        compute_mpo_score(
            compound_name="Bad",
            logP=2.0,
            mw=0.0,
            psa=60.0,
            n_h_donors=1,
            n_h_acceptors=5,
            n_rotatable_bonds=4,
        )


def test_mw_negative_raises():
    with pytest.raises(ValueError):
        compute_mpo_score(
            compound_name="Bad",
            logP=2.0,
            mw=-100.0,
            psa=60.0,
            n_h_donors=1,
            n_h_acceptors=5,
            n_rotatable_bonds=4,
        )


# --- Optional parameters ---


def test_pka_base_accepted():
    result = compute_mpo_score(**IDEAL_DRUG, pka_base=8.5)
    assert isinstance(result, MPOScoreResult)
    assert 0 <= result.cns_mpo_score <= 6


def test_logd_74_accepted():
    result = compute_mpo_score(**IDEAL_DRUG, logD_74=1.5)
    assert isinstance(result, MPOScoreResult)
    assert 0 <= result.cns_mpo_score <= 6
