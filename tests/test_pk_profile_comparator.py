"""Tests for Phase 300 — PK Profile Comparator."""

import pytest

from omega_pbpk.clinical.pk_profile_comparator import (
    CandidateRankResult,
    PKComparisonResult,
    compare_pk_profiles,
    rank_candidates,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_candidate(
    drug_name="DrugA",
    dose_mg=100.0,
    cl_L_per_h=5.0,
    vd_L=50.0,
    f_oral=0.8,
    ka_per_h=1.5,
    tau_h=12.0,
    therapeutic_min_mg_L=1.0,
    therapeutic_max_mg_L=10.0,
):
    return dict(
        drug_name=drug_name,
        dose_mg=dose_mg,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
        f_oral=f_oral,
        ka_per_h=ka_per_h,
        tau_h=tau_h,
        therapeutic_min_mg_L=therapeutic_min_mg_L,
        therapeutic_max_mg_L=therapeutic_max_mg_L,
    )


def _two_candidates():
    return [
        _make_candidate(drug_name="DrugA"),
        _make_candidate(drug_name="DrugB", dose_mg=200.0, cl_L_per_h=8.0),
    ]


# ---------------------------------------------------------------------------
# compare_pk_profiles — basic structure
# ---------------------------------------------------------------------------


def test_compare_returns_pk_comparison_result():
    result = compare_pk_profiles(_two_candidates())
    assert isinstance(result, PKComparisonResult)


def test_compare_n_candidates_correct():
    result = compare_pk_profiles(_two_candidates())
    assert result.n_candidates == 2


def test_compare_three_candidates_n_correct():
    candidates = [
        _make_candidate(drug_name="A"),
        _make_candidate(drug_name="B"),
        _make_candidate(drug_name="C"),
    ]
    result = compare_pk_profiles(candidates)
    assert result.n_candidates == 3


def test_compare_candidate_names_length():
    result = compare_pk_profiles(_two_candidates())
    assert len(result.candidate_names) == 2


def test_compare_auc_values_length():
    result = compare_pk_profiles(_two_candidates())
    assert len(result.auc_values_mg_h_per_L) == 2


def test_compare_cmax_values_length():
    result = compare_pk_profiles(_two_candidates())
    assert len(result.cmax_values_mg_L) == 2


def test_compare_tmax_values_length():
    result = compare_pk_profiles(_two_candidates())
    assert len(result.tmax_values_h) == 2


def test_compare_t_half_values_length():
    result = compare_pk_profiles(_two_candidates())
    assert len(result.t_half_values_h) == 2


def test_compare_time_in_window_values_length():
    result = compare_pk_profiles(_two_candidates())
    assert len(result.time_in_window_pct_values) == 2


def test_compare_best_by_auc_is_valid_name():
    result = compare_pk_profiles(_two_candidates())
    assert result.best_candidate_by_auc in result.candidate_names


def test_compare_best_by_window_is_valid_name():
    result = compare_pk_profiles(_two_candidates())
    assert result.best_candidate_by_window in result.candidate_names


def test_compare_auc_values_positive():
    result = compare_pk_profiles(_two_candidates())
    for auc in result.auc_values_mg_h_per_L:
        assert auc > 0


def test_compare_t_half_values_positive():
    result = compare_pk_profiles(_two_candidates())
    for th in result.t_half_values_h:
        assert th > 0


def test_compare_time_in_window_bounded():
    result = compare_pk_profiles(_two_candidates())
    for pct in result.time_in_window_pct_values:
        assert 0 <= pct <= 100


# ---------------------------------------------------------------------------
# rank_candidates — basic structure
# ---------------------------------------------------------------------------


def test_rank_returns_list():
    result = rank_candidates(_two_candidates())
    assert isinstance(result, list)


def test_rank_returns_correct_length():
    result = rank_candidates(_two_candidates())
    assert len(result) == 2


def test_rank_returns_candidate_rank_result_instances():
    result = rank_candidates(_two_candidates())
    for r in result:
        assert isinstance(r, CandidateRankResult)


def test_rank_ranks_are_unique_and_sequential():
    candidates = [
        _make_candidate(drug_name="A"),
        _make_candidate(drug_name="B"),
        _make_candidate(drug_name="C"),
    ]
    result = rank_candidates(candidates)
    ranks = sorted(r.rank for r in result)
    assert ranks == [1, 2, 3]


def test_rank_composite_score_in_range():
    result = rank_candidates(_two_candidates())
    for r in result:
        assert 0 <= r.composite_score <= 100


def test_rank_sorted_descending_by_composite():
    candidates = [
        _make_candidate(drug_name="A"),
        _make_candidate(drug_name="B"),
        _make_candidate(drug_name="C"),
    ]
    result = rank_candidates(candidates)
    scores = [r.composite_score for r in result]
    assert scores == sorted(scores, reverse=True)


def test_rank_best_by_auc_with_auc_weight_only():
    """When auc weight=1, others=0, highest AUC candidate should rank first."""
    # DrugA: AUC = f*dose/cl/tau = 0.8*100/5/12 = 1.333
    # DrugB: AUC = 0.8*500/5/12 = 6.667
    candidates = [
        _make_candidate(drug_name="DrugA", dose_mg=100),
        _make_candidate(drug_name="DrugB", dose_mg=500),
    ]
    result = rank_candidates(
        candidates, weights={"auc": 1.0, "time_in_window": 0.0, "half_life": 0.0}
    )
    assert result[0].drug_name == "DrugB"
    assert result[0].rank == 1


def test_rank_first_has_rank_1():
    result = rank_candidates(_two_candidates())
    assert result[0].rank == 1


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_compare_fewer_than_two_candidates():
    with pytest.raises(ValueError, match="2 candidates"):
        compare_pk_profiles([_make_candidate()])


def test_compare_empty_list():
    with pytest.raises(ValueError):
        compare_pk_profiles([])


def test_compare_missing_key():
    bad = {"drug_name": "X", "dose_mg": 100}
    with pytest.raises(ValueError, match="missing required keys"):
        compare_pk_profiles([bad, _make_candidate()])


def test_compare_invalid_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        compare_pk_profiles(
            [
                _make_candidate(dose_mg=0),
                _make_candidate(drug_name="B"),
            ]
        )


def test_compare_invalid_f_oral_gt_one():
    with pytest.raises(ValueError, match="f_oral"):
        compare_pk_profiles(
            [
                _make_candidate(f_oral=1.5),
                _make_candidate(drug_name="B"),
            ]
        )


def test_compare_invalid_thera_max_lte_min():
    with pytest.raises(ValueError, match="therapeutic_max"):
        compare_pk_profiles(
            [
                _make_candidate(therapeutic_min_mg_L=5.0, therapeutic_max_mg_L=3.0),
                _make_candidate(drug_name="B"),
            ]
        )


def test_rank_fewer_than_two_candidates():
    with pytest.raises(ValueError):
        rank_candidates([_make_candidate()])
