"""Tests for PK-PK interaction predictor (Phase 101)."""

from __future__ import annotations

from omega_pbpk.prediction.pk_interaction_predictor import (
    PKInteractionResult,
    _aucr_cyp_inhibition,
    _aucr_induction,
    predict_pk_interaction,
    screen_interaction_pairs,
)


class TestCYPInhibition:
    def test_inhibition_increases_auc(self):
        aucr = _aucr_cyp_inhibition(ki_uM=1.0, inhibitor_conc_uM=10.0, fm=0.8)
        assert aucr > 1.0

    def test_low_inhibitor_minimal_effect(self):
        aucr = _aucr_cyp_inhibition(ki_uM=100.0, inhibitor_conc_uM=0.01)
        assert aucr < 1.5

    def test_aucr_clamped_max(self):
        aucr = _aucr_cyp_inhibition(ki_uM=0.001, inhibitor_conc_uM=1000.0)
        assert aucr <= 20.0

    def test_aucr_geq_1(self):
        aucr = _aucr_cyp_inhibition(ki_uM=5.0, inhibitor_conc_uM=0.0)
        assert aucr >= 1.0

    def test_higher_inhibitor_higher_aucr(self):
        aucr_low = _aucr_cyp_inhibition(ki_uM=1.0, inhibitor_conc_uM=0.1)
        aucr_high = _aucr_cyp_inhibition(ki_uM=1.0, inhibitor_conc_uM=10.0)
        assert aucr_high > aucr_low


class TestInduction:
    def test_induction_decreases_auc(self):
        aucr = _aucr_induction(emax=10.0, ec50_uM=1.0, inducer_conc_uM=5.0)
        assert aucr < 1.0

    def test_no_inducer_auc_unchanged(self):
        aucr = _aucr_induction(emax=5.0, ec50_uM=1.0, inducer_conc_uM=0.0)
        assert abs(aucr - 1.0) < 1e-6

    def test_higher_emax_lower_aucr(self):
        aucr_low = _aucr_induction(emax=2.0, ec50_uM=1.0, inducer_conc_uM=1.0)
        aucr_high = _aucr_induction(emax=10.0, ec50_uM=1.0, inducer_conc_uM=1.0)
        assert aucr_high < aucr_low


class TestPredictInteraction:
    def test_returns_result_type(self):
        r = predict_pk_interaction("DrugA", "DrugB", ki_uM=1.0, inhibitor_conc_uM=5.0)
        assert isinstance(r, PKInteractionResult)

    def test_severe_risk_for_high_inhibition(self):
        r = predict_pk_interaction("DrugA", "DrugB", ki_uM=0.01, inhibitor_conc_uM=100.0, fm=0.9)
        assert r.risk_level in ("moderate", "severe")

    def test_inhibition_mechanism_label(self):
        r = predict_pk_interaction("A", "B", ki_uM=1.0, inhibitor_conc_uM=1.0)
        assert "inhibit" in r.mechanism

    def test_induction_mechanism_label(self):
        r = predict_pk_interaction("A", "B", emax=5.0, ec50_uM=1.0, inducer_conc_uM=1.0)
        assert "induct" in r.mechanism

    def test_confidence_between_0_and_1(self):
        r = predict_pk_interaction("A", "B", ki_uM=1.0)
        assert 0.0 < r.confidence <= 1.0

    def test_cmax_ratio_positive(self):
        r = predict_pk_interaction("A", "B", ki_uM=1.0, inhibitor_conc_uM=10.0)
        assert r.cmax_ratio > 0.0

    def test_low_fm_lower_confidence(self):
        r_high = predict_pk_interaction("A", "B", ki_uM=1.0, fm=0.9)
        r_low = predict_pk_interaction("A", "B", ki_uM=1.0, fm=0.3)
        assert r_low.confidence <= r_high.confidence

    def test_fda_recommendation_is_string(self):
        r = predict_pk_interaction("A", "B", ki_uM=1.0)
        assert isinstance(r.fda_recommendation, str)
        assert len(r.fda_recommendation) > 0


class TestScreenPairs:
    def test_returns_list(self):
        pairs = [
            {"perpetrator": "A", "victim": "B", "ki_uM": 1.0},
            {"perpetrator": "C", "victim": "D", "emax": 5.0, "ec50_uM": 1.0},
        ]
        results = screen_interaction_pairs(pairs)
        assert isinstance(results, list)

    def test_length_matches_input(self):
        pairs = [{"perpetrator": f"A{i}", "victim": "B", "ki_uM": 1.0} for i in range(5)]
        results = screen_interaction_pairs(pairs)
        assert len(results) == 5

    def test_empty_pairs_returns_empty(self):
        assert screen_interaction_pairs([]) == []
