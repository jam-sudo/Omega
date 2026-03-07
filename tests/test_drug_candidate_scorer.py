"""Tests for prediction/drug_candidate_scorer.py — Phase 662."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.drug_candidate_scorer import (
    DrugCandidateScore,
    rank_drug_candidates,
    score_drug_candidate,
)

_ADMET_DIMS = {"absorption", "distribution", "metabolism", "excretion", "toxicity"}

# ---------------------------------------------------------------------------
# Ideal drug fixture: low MW, good logP, no alerts
# ---------------------------------------------------------------------------
IDEAL_KWARGS = dict(
    compound_name="IdealDrug",
    logP=2.0,
    mw=300.0,
    psa=50.0,
    n_hbd=1,
    n_hba=3,
    fup=0.3,
    clint_uL_per_min_per_mg=5.0,
    n_aromatic_rings=1,
    has_structural_alerts=False,
    herg_risk=False,
    n_chiral_centers=0,
    solubility_mg_mL=2.0,
)

PROBLEMATIC_KWARGS = dict(
    compound_name="BadDrug",
    logP=6.0,
    mw=600.0,
    psa=150.0,
    n_hbd=6,
    n_hba=10,
    fup=0.005,
    clint_uL_per_min_per_mg=300.0,
    n_aromatic_rings=4,
    has_structural_alerts=True,
    herg_risk=True,
    n_chiral_centers=5,
    solubility_mg_mL=0.01,
)


# ---------------------------------------------------------------------------
# Return type and field range tests
# ---------------------------------------------------------------------------


def test_returns_drug_candidate_score():
    result = score_drug_candidate(**IDEAL_KWARGS)
    assert isinstance(result, DrugCandidateScore)


def test_absorption_score_in_range():
    result = score_drug_candidate(**IDEAL_KWARGS)
    assert 0.0 <= result.absorption_score <= 20.0


def test_distribution_score_in_range():
    result = score_drug_candidate(**IDEAL_KWARGS)
    assert 0.0 <= result.distribution_score <= 20.0


def test_metabolism_score_in_range():
    result = score_drug_candidate(**IDEAL_KWARGS)
    assert 0.0 <= result.metabolism_score <= 20.0


def test_excretion_score_in_range():
    result = score_drug_candidate(**IDEAL_KWARGS)
    assert 0.0 <= result.excretion_score <= 20.0


def test_toxicity_score_in_range():
    result = score_drug_candidate(**IDEAL_KWARGS)
    assert 0.0 <= result.toxicity_score <= 20.0


def test_total_score_in_range():
    result = score_drug_candidate(**IDEAL_KWARGS)
    assert 0.0 <= result.total_score <= 100.0


def test_grade_valid():
    result = score_drug_candidate(**IDEAL_KWARGS)
    assert result.grade in {"A", "B", "C", "D"}


def test_optimization_priority_is_admet():
    result = score_drug_candidate(**IDEAL_KWARGS)
    assert result.optimization_priority in _ADMET_DIMS


def test_lead_likeness_true_when_total_gt_60():
    result = score_drug_candidate(**IDEAL_KWARGS)
    if result.total_score > 60.0:
        assert result.lead_likeness is True
    else:
        assert result.lead_likeness is False


def test_risk_flags_is_list():
    result = score_drug_candidate(**IDEAL_KWARGS)
    assert isinstance(result.risk_flags, list)


def test_notes_nonempty():
    result = score_drug_candidate(**IDEAL_KWARGS)
    assert isinstance(result.notes, str) and len(result.notes) > 0


# ---------------------------------------------------------------------------
# Drug quality tests
# ---------------------------------------------------------------------------


def test_ideal_drug_grade_a_or_b():
    result = score_drug_candidate(**IDEAL_KWARGS)
    assert result.grade in {"A", "B"}


def test_problematic_drug_grade_c_or_d():
    result = score_drug_candidate(**PROBLEMATIC_KWARGS)
    assert result.grade in {"C", "D"}


def test_structural_alerts_lower_metabolism():
    no_alert = score_drug_candidate(**{**IDEAL_KWARGS, "has_structural_alerts": False})
    with_alert = score_drug_candidate(**{**IDEAL_KWARGS, "has_structural_alerts": True})
    assert with_alert.metabolism_score < no_alert.metabolism_score


def test_herg_risk_lowers_toxicity():
    no_herg = score_drug_candidate(**{**IDEAL_KWARGS, "herg_risk": False})
    with_herg = score_drug_candidate(**{**IDEAL_KWARGS, "herg_risk": True})
    assert with_herg.toxicity_score < no_herg.toxicity_score


def test_higher_clint_lower_metabolism():
    low_cl = score_drug_candidate(**{**IDEAL_KWARGS, "clint_uL_per_min_per_mg": 5.0})
    high_cl = score_drug_candidate(**{**IDEAL_KWARGS, "clint_uL_per_min_per_mg": 300.0})
    assert high_cl.metabolism_score < low_cl.metabolism_score


def test_high_logp_lower_absorption():
    low_logp = score_drug_candidate(**{**IDEAL_KWARGS, "logP": 2.0})
    high_logp = score_drug_candidate(**{**IDEAL_KWARGS, "logP": 6.0})
    assert high_logp.absorption_score < low_logp.absorption_score


# ---------------------------------------------------------------------------
# Rank tests
# ---------------------------------------------------------------------------


def test_rank_returns_sorted_descending():
    compounds = [
        {**PROBLEMATIC_KWARGS, "compound_name": "Bad"},
        {**IDEAL_KWARGS, "compound_name": "Ideal"},
    ]
    results = rank_drug_candidates(compounds)
    assert len(results) == 2
    assert results[0].total_score >= results[1].total_score


def test_rank_empty_returns_empty():
    assert rank_drug_candidates([]) == []


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


def test_validation_mw_zero_raises():
    with pytest.raises(ValueError):
        score_drug_candidate(**{**IDEAL_KWARGS, "mw": 0.0})


def test_validation_fup_zero_raises():
    with pytest.raises(ValueError):
        score_drug_candidate(**{**IDEAL_KWARGS, "fup": 0.0})


def test_validation_fup_over_one_raises():
    with pytest.raises(ValueError):
        score_drug_candidate(**{**IDEAL_KWARGS, "fup": 1.2})


def test_validation_clint_negative_raises():
    with pytest.raises(ValueError):
        score_drug_candidate(**{**IDEAL_KWARGS, "clint_uL_per_min_per_mg": -1.0})


# ---------------------------------------------------------------------------
# Optimization priority test
# ---------------------------------------------------------------------------


def test_optimization_priority_is_lowest_dimension():
    result = score_drug_candidate(**PROBLEMATIC_KWARGS)
    scores = {
        "absorption": result.absorption_score,
        "distribution": result.distribution_score,
        "metabolism": result.metabolism_score,
        "excretion": result.excretion_score,
        "toxicity": result.toxicity_score,
    }
    min_dim = min(scores, key=lambda k: scores[k])
    assert result.optimization_priority == min_dim
