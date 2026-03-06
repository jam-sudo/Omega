"""Tests for plasma protein binding prediction (Phase 105)."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.plasma_protein_binding import (
    PPBResult,
    predict_ppb,
    ppb_sensitivity,
)


class TestPredictPPB:
    def test_returns_result_type(self):
        r = predict_ppb("Drug", logP=2.0, MW=300.0)
        assert isinstance(r, PPBResult)

    def test_fup_between_0_and_1(self):
        r = predict_ppb("Drug", logP=2.0, MW=300.0)
        assert 0.0 < r.fup < 1.0

    def test_ppb_pct_consistent_with_fup(self):
        r = predict_ppb("Drug", logP=2.0, MW=300.0)
        assert abs(r.ppb_pct - 100.0 * (1.0 - r.fup)) < 0.01

    def test_high_logP_lower_fup(self):
        r_low = predict_ppb("Drug", logP=0.0, MW=300.0)
        r_high = predict_ppb("Drug", logP=5.0, MW=300.0)
        assert r_high.fup < r_low.fup

    def test_classification_high_binding(self):
        r = predict_ppb("Drug", logP=5.0, MW=400.0, PSA=30.0)
        assert r.classification in ("moderate", "high")

    def test_classification_low_binding(self):
        r = predict_ppb("Drug", logP=-2.0, MW=200.0, PSA=120.0)
        assert r.classification in ("low", "moderate")

    def test_basic_drug_binds_aag(self):
        r = predict_ppb("BasicDrug", logP=2.0, MW=300.0, pka_basic=9.0)
        assert r.primary_protein == "AAG"

    def test_acidic_drug_binds_hsa(self):
        r = predict_ppb("AcidicDrug", logP=2.0, MW=300.0, pka_acid=4.5)
        assert r.primary_protein == "HSA"

    def test_log_ka_positive(self):
        r = predict_ppb("Drug", logP=2.0, MW=300.0)
        assert r.log_ka > 0.0

    def test_confidence_between_0_and_1(self):
        r = predict_ppb("Drug", logP=2.0, MW=300.0)
        assert 0.0 < r.confidence <= 1.0

    def test_drug_like_higher_confidence(self):
        r_drug = predict_ppb("Drug", logP=2.0, MW=350.0, PSA=60.0)
        r_outlier = predict_ppb("Drug", logP=8.0, MW=900.0, PSA=200.0)
        assert r_drug.confidence >= r_outlier.confidence


class TestPPBSensitivity:
    def test_returns_dict_with_keys(self):
        result = ppb_sensitivity("Drug", logP=2.0, MW=300.0)
        assert "base" in result
        assert "high_logP" in result
        assert "low_logP" in result
        assert "dfup_dlogP" in result

    def test_dfup_dlogP_negative(self):
        result = ppb_sensitivity("Drug", logP=2.0, MW=300.0)
        # Higher logP → lower fup → negative derivative
        assert result["dfup_dlogP"] < 0.0
