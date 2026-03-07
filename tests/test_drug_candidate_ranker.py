"""Tests for Phase 577: drug_candidate_ranker."""

import math

import pytest

from omega_pbpk.analysis.drug_candidate_ranker import (
    flag_alerts,
    pareto_front,
    rank_candidates,
    score_admet,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

IDEAL = dict(
    logP=2.0,
    mw=350.0,
    psa=80.0,
    n_hbd=2,
    n_hba=5,
    fu_plasma=0.3,
    cl_L_per_h=10.0,
    t_half_h=8.0,
    f_oral=0.9,
)

TERRIBLE = dict(
    logP=8.0,
    mw=900.0,
    psa=200.0,
    n_hbd=8,
    n_hba=15,
    fu_plasma=0.01,
    cl_L_per_h=80.0,
    t_half_h=200.0,
    f_oral=0.05,
)


# ---------------------------------------------------------------------------
# score_admet
# ---------------------------------------------------------------------------


class TestScoreAdmet:
    def test_ideal_drug_high_total_score(self):
        scores = score_admet(**IDEAL)
        assert scores["total_score"] > 70

    def test_ideal_drug_perfect_drug_likeness(self):
        scores = score_admet(**IDEAL)
        assert scores["drug_likeness"] == 10.0

    def test_ideal_drug_high_pk_score(self):
        scores = score_admet(**IDEAL)
        # f_oral=0.9, t_half=8 → exp(0)=1 → raw=0.9 → score=9.0
        assert scores["pk_score"] == pytest.approx(9.0, abs=0.01)

    def test_ideal_drug_perfect_cl_score(self):
        scores = score_admet(**IDEAL)
        assert scores["cl_score"] == 10.0

    def test_ideal_drug_fu_score(self):
        scores = score_admet(**IDEAL)
        # fu=0.3 → 0.3*5=1.5 → capped at 1 → score=10
        assert scores["fu_score"] == 10.0

    def test_terrible_drug_low_total_score(self):
        scores = score_admet(**TERRIBLE)
        assert scores["total_score"] < 30

    def test_terrible_drug_low_drug_likeness(self):
        # 4 Ro5 violations (logP, mw, n_hbd, n_hba) → 10 - 2*4 = 2 (min 0 not triggered)
        scores = score_admet(**TERRIBLE)
        assert scores["drug_likeness"] == pytest.approx(2.0, abs=1e-9)

    def test_total_score_in_range(self):
        for params in [IDEAL, TERRIBLE]:
            s = score_admet(**params)
            assert 0.0 <= s["total_score"] <= 100.0

    def test_two_violations_drug_likeness(self):
        params = dict(IDEAL, logP=6.0, mw=550.0)
        scores = score_admet(**params)
        assert scores["drug_likeness"] == 6.0

    def test_pk_score_short_half_life(self):
        # t_half=0.5 → |0.5-8|/8=0.9375 → exp(-0.9375)≈0.392 → *f_oral*10
        params = dict(IDEAL, t_half_h=0.5)
        scores = score_admet(**params)
        expected = IDEAL["f_oral"] * math.exp(-abs(0.5 - 8.0) / 8.0) * 10.0
        assert scores["pk_score"] == pytest.approx(expected, abs=0.01)

    def test_fu_score_capped_at_10(self):
        # fu=1.0 → 1*5=5 → capped → score=10
        scores = score_admet(**dict(IDEAL, fu_plasma=1.0))
        assert scores["fu_score"] == 10.0

    def test_fu_score_low_fu(self):
        # fu=0.05 → 0.05*5=0.25 → score=2.5
        scores = score_admet(**dict(IDEAL, fu_plasma=0.05))
        assert scores["fu_score"] == pytest.approx(2.5, abs=0.01)

    def test_all_keys_present(self):
        scores = score_admet(**IDEAL)
        for key in ["drug_likeness", "pk_score", "cl_score", "fu_score", "total_score"]:
            assert key in scores


# ---------------------------------------------------------------------------
# rank_candidates
# ---------------------------------------------------------------------------


class TestRankCandidates:
    def _make_candidate(self, name, **overrides):
        c = dict(IDEAL, name=name)
        c.update(overrides)
        return c

    def test_best_candidate_ranked_first(self):
        c1 = self._make_candidate("best", f_oral=0.9, t_half_h=8.0)
        c2 = self._make_candidate(
            "worst",
            f_oral=0.05,
            t_half_h=200.0,
            logP=8.0,
            mw=900.0,
            n_hbd=8,
            n_hba=15,
            fu_plasma=0.01,
            cl_L_per_h=80.0,
        )
        ranked = rank_candidates([c2, c1])
        assert ranked[0]["name"] == "best"
        assert ranked[0]["rank"] == 1

    def test_returns_same_length(self):
        candidates = [self._make_candidate(f"c{i}") for i in range(5)]
        ranked = rank_candidates(candidates)
        assert len(ranked) == 5

    def test_ranks_are_sequential(self):
        candidates = [self._make_candidate(f"c{i}", fu_plasma=0.1 * (i + 1)) for i in range(3)]
        ranked = rank_candidates(candidates)
        assert [r["rank"] for r in ranked] == [1, 2, 3]

    def test_empty_list(self):
        assert rank_candidates([]) == []

    def test_single_candidate_rank_one(self):
        ranked = rank_candidates([self._make_candidate("only")])
        assert ranked[0]["rank"] == 1

    def test_scores_added_to_dict(self):
        ranked = rank_candidates([self._make_candidate("x")])
        for key in ["drug_likeness", "pk_score", "cl_score", "fu_score", "total_score"]:
            assert key in ranked[0]


# ---------------------------------------------------------------------------
# flag_alerts
# ---------------------------------------------------------------------------


class TestFlagAlerts:
    def test_ideal_drug_no_alerts(self):
        alerts = flag_alerts(
            logP=2.0, mw=350.0, psa=80.0, cl_L_per_h=10.0, fu_plasma=0.3, t_half_h=8.0
        )
        assert alerts == []

    def test_high_logp_alert(self):
        alerts = flag_alerts(
            logP=6.0, mw=350.0, psa=80.0, cl_L_per_h=10.0, fu_plasma=0.3, t_half_h=8.0
        )
        assert any("lipophilicity" in a for a in alerts)

    def test_high_mw_alert(self):
        alerts = flag_alerts(
            logP=2.0, mw=600.0, psa=80.0, cl_L_per_h=10.0, fu_plasma=0.3, t_half_h=8.0
        )
        assert any("molecular weight" in a for a in alerts)

    def test_high_psa_alert(self):
        alerts = flag_alerts(
            logP=2.0, mw=350.0, psa=160.0, cl_L_per_h=10.0, fu_plasma=0.3, t_half_h=8.0
        )
        assert any("permeability" in a for a in alerts)

    def test_very_high_cl_alert(self):
        alerts = flag_alerts(
            logP=2.0, mw=350.0, psa=80.0, cl_L_per_h=40.0, fu_plasma=0.3, t_half_h=8.0
        )
        assert any("clearance" in a for a in alerts)

    def test_low_fu_alert(self):
        alerts = flag_alerts(
            logP=2.0, mw=350.0, psa=80.0, cl_L_per_h=10.0, fu_plasma=0.02, t_half_h=8.0
        )
        assert any("protein binding" in a for a in alerts)

    def test_very_long_half_life_alert(self):
        alerts = flag_alerts(
            logP=2.0, mw=350.0, psa=80.0, cl_L_per_h=10.0, fu_plasma=0.3, t_half_h=100.0
        )
        assert any("long half-life" in a for a in alerts)

    def test_very_short_half_life_alert(self):
        alerts = flag_alerts(
            logP=2.0, mw=350.0, psa=80.0, cl_L_per_h=10.0, fu_plasma=0.3, t_half_h=0.5
        )
        assert any("short half-life" in a for a in alerts)

    def test_multiple_alerts(self):
        alerts = flag_alerts(
            logP=6.0, mw=600.0, psa=80.0, cl_L_per_h=10.0, fu_plasma=0.3, t_half_h=8.0
        )
        assert len(alerts) == 2


# ---------------------------------------------------------------------------
# pareto_front
# ---------------------------------------------------------------------------


class TestParetoFront:
    def _scored(self, name, a, b):
        return {"name": name, "obj_a": a, "obj_b": b}

    def test_single_candidate_on_front(self):
        c = self._scored("only", 5.0, 5.0)
        front = pareto_front([c], ["obj_a", "obj_b"])
        assert len(front) == 1

    def test_two_non_dominated_both_on_front(self):
        # c1 better on obj_a, c2 better on obj_b → neither dominates
        c1 = self._scored("c1", 8.0, 3.0)
        c2 = self._scored("c2", 3.0, 8.0)
        front = pareto_front([c1, c2], ["obj_a", "obj_b"])
        assert len(front) == 2

    def test_dominated_candidate_excluded(self):
        # c3 is dominated by c1 (c1 >= c3 on both, > on obj_a)
        c1 = self._scored("c1", 8.0, 5.0)
        c2 = self._scored("c2", 3.0, 8.0)
        c3 = self._scored("c3", 4.0, 4.0)  # dominated by c1
        front = pareto_front([c1, c2, c3], ["obj_a", "obj_b"])
        names = {c["name"] for c in front}
        assert "c3" not in names
        assert "c1" in names
        assert "c2" in names

    def test_empty_candidates(self):
        front = pareto_front([], ["obj_a"])
        assert front == []
