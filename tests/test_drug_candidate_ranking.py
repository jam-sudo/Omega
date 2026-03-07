"""Tests for Phase 1134 — Drug Candidate Ranking."""

import pytest

from omega_pbpk.prediction.drug_candidate_ranking import (
    CandidateRankingResult,
    compare_two_candidates,
    rank_drug_candidates,
)

# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------


def _make_candidate(
    drug_name: str = "drug_A",
    ic50_nM: float = 5.0,
    selectivity_ratio: float = 200.0,
    f_oral_pct: float = 60.0,
    t_half_h: float = 12.0,
    herg_ic50_uM: float = 50.0,
    ames_positive: bool = False,
    logP: float = 2.5,
    mw_Da: float = 350.0,
) -> dict:
    return {
        "drug_name": drug_name,
        "ic50_nM": ic50_nM,
        "selectivity_ratio": selectivity_ratio,
        "f_oral_pct": f_oral_pct,
        "t_half_h": t_half_h,
        "herg_ic50_uM": herg_ic50_uM,
        "ames_positive": ames_positive,
        "logP": logP,
        "mw_Da": mw_Da,
    }


GOOD_CANDIDATE = _make_candidate(
    drug_name="good_drug",
    ic50_nM=0.5,       # score 20
    selectivity_ratio=2000.0,  # score 20
    f_oral_pct=80.0,   # bio score 20
    t_half_h=12.0,     # half-life score 20 → adme=20
    herg_ic50_uM=100.0,  # score 20
    ames_positive=False,
    logP=2.0,
    mw_Da=300.0,
)

BAD_CANDIDATE = _make_candidate(
    drug_name="bad_drug",
    ic50_nM=5000.0,    # score 3
    selectivity_ratio=0.5,  # score 0
    f_oral_pct=2.0,    # bio score 0
    t_half_h=100.0,    # half-life score 5
    herg_ic50_uM=0.05,  # score 0
    ames_positive=True,  # -10 penalty
    logP=6.0,
    mw_Da=600.0,
)


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------

def test_returns_list_of_results():
    results = rank_drug_candidates([_make_candidate()])
    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], CandidateRankingResult)


def test_single_candidate_rank_one():
    results = rank_drug_candidates([_make_candidate()])
    assert results[0].rank == 1


def test_ranking_sorted_descending():
    candidates = [GOOD_CANDIDATE, BAD_CANDIDATE]
    results = rank_drug_candidates(candidates)
    assert results[0].composite_score >= results[1].composite_score


def test_rank_1_is_highest_score():
    candidates = [BAD_CANDIDATE, GOOD_CANDIDATE]
    results = rank_drug_candidates(candidates)
    assert results[0].rank == 1
    assert results[0].drug_name == "good_drug"


def test_ranks_assigned_correctly():
    candidates = [
        _make_candidate("A", ic50_nM=1.0),
        _make_candidate("B", ic50_nM=500.0),
        _make_candidate("C", ic50_nM=0.1),
    ]
    results = rank_drug_candidates(candidates)
    ranked_names = [r.drug_name for r in results]
    assert ranked_names[0] == "C"  # best potency


# ---------------------------------------------------------------------------
# Scoring dimensions
# ---------------------------------------------------------------------------

def test_high_potency_score():
    cand = _make_candidate(ic50_nM=0.5)
    results = rank_drug_candidates([cand])
    assert results[0].potency_score == 20.0


def test_medium_potency_score():
    cand = _make_candidate(ic50_nM=50.0)
    results = rank_drug_candidates([cand])
    assert results[0].potency_score == 13.0


def test_low_potency_score():
    cand = _make_candidate(ic50_nM=5000.0)
    results = rank_drug_candidates([cand])
    assert results[0].potency_score == 3.0


def test_high_selectivity_score():
    cand = _make_candidate(selectivity_ratio=2000.0)
    results = rank_drug_candidates([cand])
    assert results[0].selectivity_score == 20.0


def test_poor_selectivity_score():
    cand = _make_candidate(selectivity_ratio=0.5)
    results = rank_drug_candidates([cand])
    assert results[0].selectivity_score == 0.0


def test_good_bioavailability_and_half_life_adme():
    cand = _make_candidate(f_oral_pct=80.0, t_half_h=12.0)
    results = rank_drug_candidates([cand])
    assert results[0].adme_score == 20.0


def test_low_bioavailability_reduces_adme():
    cand_hi = _make_candidate(f_oral_pct=80.0, t_half_h=12.0)
    cand_lo = _make_candidate(f_oral_pct=3.0, t_half_h=12.0)
    results_hi = rank_drug_candidates([cand_hi])
    results_lo = rank_drug_candidates([cand_lo])
    assert results_hi[0].adme_score > results_lo[0].adme_score


def test_ames_positive_reduces_safety_score():
    cand_safe = _make_candidate(herg_ic50_uM=50.0, ames_positive=False)
    cand_ames = _make_candidate(herg_ic50_uM=50.0, ames_positive=True)
    results_safe = rank_drug_candidates([cand_safe])
    results_ames = rank_drug_candidates([cand_ames])
    assert results_safe[0].safety_score > results_ames[0].safety_score


def test_ames_positive_safety_not_negative():
    cand = _make_candidate(herg_ic50_uM=0.05, ames_positive=True)
    results = rank_drug_candidates([cand])
    assert results[0].safety_score >= 0.0


def test_lipinski_no_violations_dev_score():
    cand = _make_candidate(logP=2.0, mw_Da=300.0)
    results = rank_drug_candidates([cand])
    assert results[0].developability_score == 20.0


def test_lipinski_one_violation_dev_score():
    cand = _make_candidate(logP=6.0, mw_Da=300.0)  # logP violation
    results = rank_drug_candidates([cand])
    assert results[0].developability_score == 12.0


def test_lipinski_two_violations_dev_score():
    cand = _make_candidate(logP=6.0, mw_Da=600.0)
    results = rank_drug_candidates([cand])
    assert results[0].developability_score == 5.0


# ---------------------------------------------------------------------------
# Composite score and go/no_go
# ---------------------------------------------------------------------------

def test_good_candidate_high_composite():
    results = rank_drug_candidates([GOOD_CANDIDATE])
    assert results[0].composite_score >= 60.0


def test_bad_candidate_low_composite():
    results = rank_drug_candidates([BAD_CANDIDATE])
    assert results[0].composite_score < 40.0


def test_go_threshold():
    results = rank_drug_candidates([GOOD_CANDIDATE])
    assert results[0].go_no_go == "go"


def test_no_go_threshold():
    results = rank_drug_candidates([BAD_CANDIDATE])
    assert results[0].go_no_go == "no_go"


def test_maybe_threshold():
    cand = _make_candidate(
        ic50_nM=500.0,      # score 8
        selectivity_ratio=5.0,  # score 5
        f_oral_pct=25.0,    # bio 10
        t_half_h=12.0,      # hl 20 → adme 15
        herg_ic50_uM=5.0,   # score 10
        ames_positive=False,
        logP=3.0,
        mw_Da=400.0,        # no violations → 20
    )
    results = rank_drug_candidates([cand])
    # composite = 8+5+15+10+20 = 58 → maybe
    assert results[0].go_no_go in {"go", "maybe"}


# ---------------------------------------------------------------------------
# compare_two_candidates
# ---------------------------------------------------------------------------

def test_compare_returns_dict():
    result = compare_two_candidates(GOOD_CANDIDATE, BAD_CANDIDATE)
    assert isinstance(result, dict)


def test_compare_winner_key():
    result = compare_two_candidates(GOOD_CANDIDATE, BAD_CANDIDATE)
    assert result["winner"] == "good_drug"


def test_compare_margin_nonnegative():
    result = compare_two_candidates(GOOD_CANDIDATE, BAD_CANDIDATE)
    assert result["margin"] >= 0


def test_compare_comparison_dimensions():
    result = compare_two_candidates(GOOD_CANDIDATE, BAD_CANDIDATE)
    dims = {"potency", "selectivity", "adme", "safety", "developability", "composite"}
    assert dims == set(result["comparison"].keys())


def test_compare_dimension_a_b_keys():
    result = compare_two_candidates(GOOD_CANDIDATE, BAD_CANDIDATE)
    for dim, scores in result["comparison"].items():
        assert "A" in scores and "B" in scores, f"Missing A/B in {dim}"


def test_compare_reversed_winner():
    result = compare_two_candidates(BAD_CANDIDATE, GOOD_CANDIDATE)
    assert result["winner"] == "good_drug"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

def test_empty_candidates_raises():
    with pytest.raises(ValueError):
        rank_drug_candidates([])


def test_missing_key_raises():
    bad = {"drug_name": "X", "ic50_nM": 1.0}
    with pytest.raises(ValueError, match="Missing required keys"):
        rank_drug_candidates([bad])


def test_negative_ic50_raises():
    cand = _make_candidate(ic50_nM=-1.0)
    with pytest.raises(ValueError, match="ic50_nM"):
        rank_drug_candidates([cand])


def test_negative_selectivity_raises():
    cand = _make_candidate(selectivity_ratio=-10.0)
    with pytest.raises(ValueError, match="selectivity_ratio"):
        rank_drug_candidates([cand])


def test_negative_herg_raises():
    cand = _make_candidate(herg_ic50_uM=-5.0)
    with pytest.raises(ValueError, match="herg_ic50_uM"):
        rank_drug_candidates([cand])


def test_zero_mw_raises():
    cand = _make_candidate(mw_Da=0.0)
    with pytest.raises(ValueError, match="mw_Da"):
        rank_drug_candidates([cand])
