"""Tests for ADR risk scoring (Phase 70)."""
from __future__ import annotations

import pytest

from omega_pbpk.risk.adr_risk import ADRRiskResult, score_adr_risk, batch_adr_screen


def _default(**kw):
    defaults = dict(
        drug_name="TestDrug",
        logP=2.0,
        MW=300.0,
        fup=0.5,
        cmax_mg_L=5.0,
        dose_mg=100.0,
    )
    defaults.update(kw)
    return score_adr_risk(**defaults)


class TestScoreADRRisk:
    def test_returns_result_type(self):
        assert isinstance(_default(), ADRRiskResult)

    def test_composite_score_bounded_0_100(self):
        r = _default()
        assert 0 <= r.composite_risk_score <= 100

    def test_category_scores_has_5_categories(self):
        r = _default()
        assert len(r.category_scores) == 5

    def test_all_category_scores_bounded(self):
        r = _default()
        for v in r.category_scores.values():
            assert 0 <= v <= 10

    def test_risk_tier_valid(self):
        r = _default()
        assert r.risk_tier in ("low", "moderate", "high", "very_high")

    def test_drug_name_preserved(self):
        r = _default(drug_name="Warfarin")
        assert r.drug_name == "Warfarin"

    def test_reactive_metabolite_increases_hepato(self):
        r_no = _default(reactive_metabolite=False)
        r_yes = _default(reactive_metabolite=True)
        assert r_yes.category_scores["hepatotoxicity"] > r_no.category_scores["hepatotoxicity"]

    def test_high_qtc_increases_cardio(self):
        r_low = _default(delta_qtc_ms=0.0)
        r_high = _default(delta_qtc_ms=50.0)
        assert r_high.category_scores["cardiotoxicity"] > r_low.category_scores["cardiotoxicity"]

    def test_high_kp_uu_increases_neuro(self):
        r_low = _default(kp_uu_brain=0.0)
        r_high = _default(kp_uu_brain=0.8)
        assert r_high.category_scores["neurotoxicity"] > r_low.category_scores["neurotoxicity"]

    def test_low_sol_increases_gi(self):
        r_high = _default(sol_mg_mL=10.0)
        r_low = _default(sol_mg_mL=0.001)
        assert r_low.category_scores["gi_toxicity"] > r_high.category_scores["gi_toxicity"]

    def test_safety_margin_positive(self):
        assert _default().safety_margin > 0

    def test_bsa_normalized_dose(self):
        r = _default(dose_mg=100.0)
        assert abs(r.bsa_mg_per_m2 - 100.0 / 1.73) < 0.1

    def test_risk_flags_is_list(self):
        assert isinstance(_default().risk_flags, list)

    def test_safe_drug_low_score(self):
        r = score_adr_risk("Safe", logP=0.0, MW=150, fup=1.0, cmax_mg_L=1.0,
                           dose_mg=10.0, fraction_renal=0.0, kp_uu_brain=0.0,
                           delta_qtc_ms=0.0, sol_mg_mL=10.0)
        assert r.risk_tier == "low"


class TestBatchADRScreen:
    def _compounds(self):
        return [
            {"drug_name": "DrugA", "logP": 1.0, "MW": 200, "fup": 0.9, "cmax_mg_L": 1.0, "dose_mg": 100},
            {"drug_name": "DrugB", "logP": 4.0, "MW": 400, "fup": 0.1, "cmax_mg_L": 5.0, "dose_mg": 100,
             "reactive_metabolite": True},
        ]

    def test_returns_list(self):
        assert isinstance(batch_adr_screen(self._compounds()), list)

    def test_sorted_by_risk_descending(self):
        r = batch_adr_screen(self._compounds())
        assert r[0].composite_risk_score >= r[-1].composite_risk_score

    def test_all_results_valid(self):
        for result in batch_adr_screen(self._compounds()):
            assert isinstance(result, ADRRiskResult)

    def test_empty_list_returns_empty(self):
        assert batch_adr_screen([]) == []


class TestRiskTierConsistency:
    def test_low_tier_threshold(self):
        r = score_adr_risk("Safe", logP=0.0, MW=150, fup=1.0, cmax_mg_L=0.5,
                           dose_mg=5.0, fraction_renal=0.0)
        assert r.risk_tier == "low"

    def test_risky_drug_higher_tier(self):
        r = score_adr_risk("Risky", logP=5.0, MW=500, fup=0.01, cmax_mg_L=10.0,
                           dose_mg=500.0, reactive_metabolite=True,
                           delta_qtc_ms=60.0, kp_uu_brain=0.9, sol_mg_mL=0.001)
        assert r.risk_tier != "low"
