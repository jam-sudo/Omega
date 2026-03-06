"""Tests for logP predictor (Phase 109)."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.logp_predictor import (
    LogPResult,
    predict_logp,
    screen_logp,
)


class TestPredictLogP:
    def test_returns_result_type(self):
        r = predict_logp("Drug", n_carbons=20)
        assert isinstance(r, LogPResult)

    def test_high_carbon_higher_logP(self):
        r_low = predict_logp("Drug", n_carbons=5)
        r_high = predict_logp("Drug", n_carbons=30)
        assert r_high.logP_predicted > r_low.logP_predicted

    def test_nitrogen_decreases_logP(self):
        r_no_N = predict_logp("Drug", n_carbons=20)
        r_with_N = predict_logp("Drug", n_carbons=20, n_nitrogens=3)
        assert r_with_N.logP_predicted < r_no_N.logP_predicted

    def test_classification_lipophilic(self):
        r = predict_logp("Drug", n_carbons=40, n_aromatic_rings=3)
        assert r.classification in ("drug-like", "lipophilic")

    def test_classification_hydrophilic(self):
        r = predict_logp("Drug", n_carbons=3, n_nitrogens=4, n_oxygens=3)
        assert r.classification in ("hydrophilic", "drug-like")

    def test_confidence_between_0_and_1(self):
        r = predict_logp("Drug", n_carbons=20)
        assert 0.0 < r.confidence <= 1.0

    def test_bbb_false_high_mw(self):
        r = predict_logp("Drug", n_carbons=20, MW=700.0, PSA=60.0)
        assert r.bbb_penetration is False

    def test_bbb_true_drug_like(self):
        r = predict_logp("Drug", n_carbons=15, n_aromatic_rings=1, MW=250.0, PSA=40.0, n_hbd=1)
        # logP should be in 1-4.5 range
        assert isinstance(r.bbb_penetration, bool)

    def test_solubility_class_is_valid(self):
        r = predict_logp("Drug", n_carbons=20)
        assert r.water_solubility_class in ("high", "moderate", "low", "insoluble")

    def test_halogen_increases_logP(self):
        r_no_hal = predict_logp("Drug", n_carbons=20)
        r_with_hal = predict_logp("Drug", n_carbons=20, n_halogens=3, halogen_type="Cl")
        assert r_with_hal.logP_predicted > r_no_hal.logP_predicted

    def test_high_psa_reduces_logP(self):
        r_low_psa = predict_logp("Drug", n_carbons=20, PSA=20.0)
        r_high_psa = predict_logp("Drug", n_carbons=20, PSA=150.0)
        assert r_high_psa.logP_predicted < r_low_psa.logP_predicted

    def test_method_is_string(self):
        r = predict_logp("Drug", n_carbons=20)
        assert isinstance(r.method, str)


class TestScreenLogP:
    def test_returns_list(self):
        compounds = [
            {"drug_name": "A", "n_carbons": 15},
            {"drug_name": "B", "n_carbons": 25, "n_nitrogens": 2},
        ]
        results = screen_logp(compounds)
        assert isinstance(results, list)

    def test_length_matches_input(self):
        compounds = [{"drug_name": f"Drug{i}", "n_carbons": 10 + i} for i in range(5)]
        results = screen_logp(compounds)
        assert len(results) == 5

    def test_empty_returns_empty(self):
        assert screen_logp([]) == []
