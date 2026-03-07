"""Tests for BBB permeability prediction (Phase 201)."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.blood_brain_barrier import (
    BBBPredictionResult,
    predict_bbb_permeability,
    screen_cns_candidates,
)


class TestPredictBBBPermeability:
    def test_returns_result_type(self):
        smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
        r = predict_bbb_permeability("Caffeine", smiles, mw=194.2, logP=-0.07)
        assert isinstance(r, BBBPredictionResult)

    def test_high_psa_poor_bbb(self):
        """High PSA (>200) should produce low bbb_score."""
        # score = 100 - (220-60)*0.5 + min(1,3)*3 = 100 - 80 + 3 = 23
        r = predict_bbb_permeability("HighPSA", "CC(=O)O", mw=300.0, logP=1.0, psa=220.0)
        assert r.bbb_score < 40.0

    def test_high_psa_penetration_class_poor_or_none(self):
        r = predict_bbb_permeability("HighPSA", "CC(=O)O", mw=300.0, logP=1.0, psa=220.0)
        assert r.penetration_class in ("poor", "none")

    def test_low_psa_good_bbb(self):
        """Low PSA, moderate logP, low MW → excellent or good score."""
        r = predict_bbb_permeability("CNSDrug", "c1ccccc1", mw=250.0, logP=2.0, psa=20.0, n_hbd=0)
        assert r.bbb_score >= 60.0

    def test_low_psa_excellent_class(self):
        r = predict_bbb_permeability("CNSDrug", "c1ccccc1", mw=250.0, logP=2.0, psa=20.0, n_hbd=0)
        assert r.penetration_class in ("excellent", "good")

    def test_log_ps_range(self):
        """log_ps must be within clamped range [-6, 0]."""
        r = predict_bbb_permeability("Drug", "C", mw=200.0, logP=2.0, psa=50.0)
        assert -6.0 <= r.log_ps <= 0.0

    def test_ps_consistent_with_log_ps(self):
        r = predict_bbb_permeability("Drug", "C", mw=200.0, logP=2.0, psa=50.0)
        assert abs(r.ps_cm_per_s - 10.0**r.log_ps) < 1e-10

    def test_cns_mpo_range(self):
        """CNS MPO score must be between 0 and 6."""
        r = predict_bbb_permeability("Drug", "CC", mw=300.0, logP=2.0, psa=60.0, pka=7.0)
        assert 0.0 <= r.cns_mpo_score <= 6.0

    def test_cns_mpo_ideal_compound(self):
        """Compound meeting all MPO criteria should score 6."""
        # logP=2 (<=3 +1, >=1 +1), psa=80 (<=90 +1), mw=350 (<=360 +1),
        # n_hbd=1 (<=1 +1), pka=7 (<=8 +1) → 6/6
        r = predict_bbb_permeability("Ideal", "CN", mw=350.0, logP=2.0, psa=80.0, n_hbd=1, pka=7.0)
        assert r.cns_mpo_score == 6.0

    def test_lipinski_cns_true(self):
        r = predict_bbb_permeability("Small", "C", mw=200.0, logP=2.0, psa=40.0, n_hbd=1)
        assert r.lipinski_cns is True

    def test_lipinski_cns_false_large_mw(self):
        r = predict_bbb_permeability("Large", "C", mw=450.0, logP=2.0, psa=40.0, n_hbd=1)
        assert r.lipinski_cns is False

    def test_lipinski_cns_false_high_psa(self):
        r = predict_bbb_permeability("PSAFail", "C", mw=300.0, logP=2.0, psa=70.0, n_hbd=1)
        assert r.lipinski_cns is False

    def test_pgp_risk_large_basic_amine(self):
        """Large MW + N in SMILES + logP > 2 → pgp_substrate_risk True."""
        r = predict_bbb_permeability(
            "PgpSub", "CCN(CC)CCNC(=O)c1cc(Cl)c(N)cc1OC", mw=450.0, logP=3.5, psa=60.0
        )
        assert r.pgp_substrate_risk is True

    def test_pgp_risk_small_no_nitrogen(self):
        """Small MW, no N → pgp_substrate_risk False."""
        r = predict_bbb_permeability("NoPgp", "c1ccccc1", mw=250.0, logP=2.0, psa=20.0)
        assert r.pgp_substrate_risk is False

    def test_invalid_mw_raises(self):
        with pytest.raises(ValueError, match="mw must be positive"):
            predict_bbb_permeability("Bad", "C", mw=0.0, logP=2.0)

    def test_negative_mw_raises(self):
        with pytest.raises(ValueError, match="mw must be positive"):
            predict_bbb_permeability("Bad", "C", mw=-100.0, logP=2.0)

    def test_empty_smiles_raises(self):
        with pytest.raises(ValueError, match="smiles"):
            predict_bbb_permeability("Bad", "", mw=300.0, logP=2.0)

    def test_bbb_score_bounded(self):
        """bbb_score must always be 0–100."""
        for logP in [-3.0, 0.0, 2.0, 5.0, 8.0]:
            for psa in [10.0, 60.0, 150.0]:
                r = predict_bbb_permeability("X", "C", mw=300.0, logP=logP, psa=psa)
                assert 0.0 <= r.bbb_score <= 100.0, f"Out of range for logP={logP}, psa={psa}"


class TestScreenCNSCandidates:
    def _make_compounds(self):
        return [
            {"name": "PoorCNS", "smiles": "CC(=O)O", "mw": 400.0, "logP": 0.5, "psa": 150.0},
            {"name": "GoodCNS", "smiles": "c1ccccc1", "mw": 250.0, "logP": 2.0, "psa": 20.0},
            {"name": "MediumCNS", "smiles": "CC", "mw": 300.0, "logP": 1.5, "psa": 60.0},
        ]

    def test_returns_list_of_results(self):
        results = screen_cns_candidates(self._make_compounds())
        assert isinstance(results, list)
        assert all(isinstance(r, BBBPredictionResult) for r in results)

    def test_sorted_by_bbb_score_descending(self):
        results = screen_cns_candidates(self._make_compounds())
        scores = [r.bbb_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_best_candidate_first(self):
        results = screen_cns_candidates(self._make_compounds())
        assert results[0].compound_name == "GoodCNS"

    def test_empty_list(self):
        results = screen_cns_candidates([])
        assert results == []

    def test_single_compound(self):
        compounds = [{"name": "Only", "smiles": "C", "mw": 200.0, "logP": 1.5}]
        results = screen_cns_candidates(compounds)
        assert len(results) == 1
