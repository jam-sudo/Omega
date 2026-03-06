"""Tests for drug repurposing similarity scoring module."""

import pytest

from omega_pbpk.prediction.drug_repurposing import (
    DrugProfile,
    RepurposingScore,
    RepurposingTarget,
    create_target,
    get_default_drug_library,
    score_repurposing,
    screen_library_repurposing,
)


def _perfect_drug(target):
    """Drug that matches all target criteria perfectly."""
    t_min, t_max = target.required_t_half_range_h
    mw_min, mw_max = target.mw_range
    lp_min, lp_max = target.logP_range
    return DrugProfile(
        name="PerfectDrug",
        mw=(mw_min + mw_max) / 2,
        logP=(lp_min + lp_max) / 2,
        fup=0.5,
        t_half_h=(t_min + t_max) / 2,
        f_oral=max(target.required_f_oral, 0.5),
        cmax_mg_L=target.target_cmax_mg_L,
        herg_ic50_uM=1000.0,
        target_indication="other",
    )


def _bad_drug():
    """Drug with poor PK and safety."""
    return DrugProfile(
        name="BadDrug",
        mw=800,
        logP=7.0,
        fup=0.01,
        t_half_h=200,
        f_oral=0.05,
        cmax_mg_L=100.0,
        herg_ic50_uM=0.01,
        target_indication="other",
    )


def _target():
    return create_target("test_indication")


class TestScoreRepurposing:
    def test_returns_score(self):
        target = _target()
        drug = _perfect_drug(target)
        r = score_repurposing(drug, target)
        assert isinstance(r, RepurposingScore)

    def test_composite_in_range(self):
        target = _target()
        drug = _perfect_drug(target)
        r = score_repurposing(drug, target)
        assert 0 <= r.composite_score <= 100

    def test_pk_match_in_range(self):
        target = _target()
        drug = _perfect_drug(target)
        r = score_repurposing(drug, target)
        assert 0 <= r.pk_match_score <= 1

    def test_safety_in_range(self):
        target = _target()
        drug = _perfect_drug(target)
        r = score_repurposing(drug, target)
        assert 0 <= r.safety_score <= 1

    def test_similarity_in_range(self):
        target = _target()
        drug = _perfect_drug(target)
        r = score_repurposing(drug, target)
        assert 0 <= r.similarity_score <= 1

    def test_recommendation_valid(self):
        target = _target()
        drug = _perfect_drug(target)
        r = score_repurposing(drug, target)
        assert r.recommendation in {"strong", "moderate", "weak", "contraindicated"}

    def test_perfect_drug_full_pk(self):
        target = _target()
        drug = _perfect_drug(target)
        r = score_repurposing(drug, target)
        assert r.pk_match_score == pytest.approx(1.0)

    def test_bad_drug_zero_pk(self):
        target = _target()
        drug = _bad_drug()
        r = score_repurposing(drug, target)
        assert r.pk_match_score == pytest.approx(0.0)

    def test_low_herg_reduces_safety(self):
        target = _target()
        drug = _bad_drug()
        r = score_repurposing(drug, target)
        assert r.safety_score < 1.0

    def test_contraindicated_for_unsafe(self):
        target = _target()
        drug = _bad_drug()
        r = score_repurposing(drug, target)
        assert r.recommendation == "contraindicated"

    def test_pk_gaps_nonempty_when_mismatch(self):
        target = _target()
        drug = _bad_drug()
        r = score_repurposing(drug, target)
        assert len(r.pk_gaps) > 0

    def test_drug_name_preserved(self):
        target = _target()
        drug = _perfect_drug(target)
        r = score_repurposing(drug, target)
        assert r.drug_name == "PerfectDrug"

    def test_notes_nonempty(self):
        target = _target()
        drug = _perfect_drug(target)
        r = score_repurposing(drug, target)
        assert len(r.notes) > 0


class TestScreenLibrary:
    def test_returns_list(self):
        target = _target()
        results = screen_library_repurposing(target)
        assert isinstance(results, list)

    def test_default_library_10(self):
        lib = get_default_drug_library()
        assert len(lib) == 10

    def test_all_drugprofile(self):
        lib = get_default_drug_library()
        assert all(isinstance(d, DrugProfile) for d in lib)

    def test_sorted_descending(self):
        target = _target()
        results = screen_library_repurposing(target)
        scores = [r.composite_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_highest_first(self):
        target = _target()
        results = screen_library_repurposing(target)
        assert results[0].composite_score >= results[-1].composite_score

    def test_create_target_returns_target(self):
        t = create_target("diabetes")
        assert isinstance(t, RepurposingTarget)

    def test_empty_library_returns_empty(self):
        target = _target()
        results = screen_library_repurposing(target, library=[])
        assert results == []
