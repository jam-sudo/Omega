"""Tests for hERG/QT prolongation risk prediction (Phase 64)."""
from __future__ import annotations

import pytest

from omega_pbpk.risk.herg_qt import HERGRiskResult, predict_herg_risk, qtc_prolongation_risk_table


class TestHERGRisk:
    def test_returns_result_type(self):
        r = predict_herg_risk("TestDrug", logP=2.0, pka_basic=7.0, MW=300)
        assert isinstance(r, HERGRiskResult)

    def test_basic_amine_increases_risk(self):
        r_no = predict_herg_risk("A", logP=2.0, pka_basic=0.0, MW=300)
        r_basic = predict_herg_risk("A", logP=2.0, pka_basic=8.5, MW=300)
        assert r_basic.risk_score > r_no.risk_score

    def test_high_logP_increases_score(self):
        r_low = predict_herg_risk("A", logP=0.0, pka_basic=0.0, MW=300)
        r_high = predict_herg_risk("A", logP=4.0, pka_basic=0.0, MW=300)
        assert r_high.risk_score > r_low.risk_score

    @pytest.mark.parametrize("logP,pka_basic,MW", [
        (0, 0, 100), (2, 5, 300), (5, 10, 600), (-1, 0, 150), (3, 8, 450),
    ])
    def test_risk_score_bounded_0_10(self, logP, pka_basic, MW):
        r = predict_herg_risk("X", logP=logP, pka_basic=pka_basic, MW=MW)
        assert 0 <= r.risk_score <= 10

    @pytest.mark.parametrize("logP,pka_basic,MW,cmax_free", [
        (0, 0, 100, 0.01), (5, 10, 600, 10.0), (2, 5, 300, 1.0),
    ])
    def test_herg_inhibition_pct_bounded(self, logP, pka_basic, MW, cmax_free):
        r = predict_herg_risk("X", logP=logP, pka_basic=pka_basic, MW=MW,
                              cmax_free_mg_L=cmax_free)
        assert 0 <= r.herg_inhibition_pct <= 95

    def test_delta_qtc_nonnegative(self):
        r = predict_herg_risk("X", logP=2.0, pka_basic=0.0, MW=300)
        assert r.delta_qtc_ms >= 0

    def test_low_risk_drug(self):
        r = predict_herg_risk("Safe", logP=0.0, pka_basic=0.0, MW=200)
        assert r.risk_category in ("low", "moderate")

    def test_high_risk_drug(self):
        r = predict_herg_risk("Risky", logP=4.0, pka_basic=9.0, MW=400)
        assert r.risk_category in ("high", "very_high")

    def test_risk_factors_populated_for_risky_drug(self):
        r = predict_herg_risk("Risky", logP=4.0, pka_basic=8.5, MW=400)
        assert len(r.risk_factors) > 0

    def test_drug_name_preserved(self):
        r = predict_herg_risk("Amiodarone", logP=3.0, pka_basic=6.5, MW=645)
        assert r.drug_name == "Amiodarone"

    def test_cmax_free_increases_inhibition(self):
        r_low = predict_herg_risk("A", logP=3.0, pka_basic=5.0, MW=350, cmax_free_mg_L=0.1)
        r_high = predict_herg_risk("A", logP=3.0, pka_basic=5.0, MW=350, cmax_free_mg_L=10.0)
        assert r_high.herg_inhibition_pct > r_low.herg_inhibition_pct

    def test_classification_consistency(self):
        r = predict_herg_risk("Safe", logP=0.0, pka_basic=0.0, MW=200)
        if r.delta_qtc_ms < 5:
            assert r.risk_category == "low"


class TestQTcRiskTable:
    def test_returns_list(self):
        r = qtc_prolongation_risk_table("A", logP=2.0, pka_basic=5.0, MW=300)
        assert isinstance(r, list)

    def test_has_4_entries(self):
        r = qtc_prolongation_risk_table("A", logP=2.0, pka_basic=5.0, MW=300)
        assert len(r) == 4

    def test_entries_have_required_keys(self):
        r = qtc_prolongation_risk_table("A", logP=2.0, pka_basic=5.0, MW=300)
        required = {"cmax_free_mg_L", "herg_inhibition_pct", "delta_qtc_ms", "risk_category"}
        for entry in r:
            assert required.issubset(entry.keys())

    def test_higher_cmax_higher_inhibition(self):
        r = qtc_prolongation_risk_table("A", logP=3.0, pka_basic=7.0, MW=350)
        inhibitions = [e["herg_inhibition_pct"] for e in r]
        cmaxes = [e["cmax_free_mg_L"] for e in r]
        paired = sorted(zip(cmaxes, inhibitions))
        for i in range(len(paired) - 1):
            assert paired[i][1] <= paired[i + 1][1]
