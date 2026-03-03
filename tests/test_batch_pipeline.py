"""Tests for Phase 24 batch prediction pipeline.

Covers run_batch_prediction() unit behaviour and POST /predict/batch endpoint.
Run with:
    PYTHONPATH=src python3 -m pytest tests/test_batch_pipeline.py -v --tb=short
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from omega_pbpk.api.app import app
from omega_pbpk.prediction.batch_pipeline import (
    BatchPredictionResult,
    run_batch_prediction,
)

client = TestClient(app)

CAFFEINE = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
ASPIRIN = "CC(=O)Oc1ccccc1C(O)=O"
IBUPROFEN = "CC(C)Cc1ccc(cc1)C(C)C(O)=O"

THREE_SMILES = [CAFFEINE, ASPIRIN, IBUPROFEN]


# ---------------------------------------------------------------------------
# Shared fixtures — run once per module to avoid repeated slow pipeline calls
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def single_result() -> BatchPredictionResult:
    return run_batch_prediction([CAFFEINE])


@pytest.fixture(scope="module")
def triple_result() -> BatchPredictionResult:
    return run_batch_prediction(THREE_SMILES)


@pytest.fixture(scope="module")
def risk_ranked_result() -> BatchPredictionResult:
    return run_batch_prediction(THREE_SMILES, ranking="risk")


@pytest.fixture(scope="module")
def auc_ranked_result() -> BatchPredictionResult:
    return run_batch_prediction(THREE_SMILES, ranking="auc")


# ---------------------------------------------------------------------------
# TestBatchPipelineUnit
# ---------------------------------------------------------------------------


class TestBatchPipelineUnit:
    """Unit tests against run_batch_prediction()."""

    def test_single_compound(self, single_result):
        assert single_result.n_total == 1
        assert single_result.n_succeeded == 1

    def test_multiple_ranked(self, triple_result):
        assert len(triple_result.compounds) == 3
        ranks = [c.rank for c in triple_result.compounds]
        assert sorted(ranks) == [1, 2, 3]
        assert len(set(ranks)) == 3  # unique ranks

    def test_score_range(self, triple_result):
        for compound in triple_result.compounds:
            assert 0.0 <= compound.composite_score <= 100.0

    def test_ranking_by_risk(self, risk_ranked_result):
        """ranking='risk': rank-1 compound has the lowest risk_count."""
        rank1 = next(c for c in risk_ranked_result.compounds if c.rank == 1)
        for other in risk_ranked_result.compounds:
            if other.rank != 1:
                assert rank1.risk_count <= other.risk_count

    def test_ranking_by_auc(self, auc_ranked_result):
        """ranking='auc': rank-1 compound has the highest AUC."""
        rank1 = next(c for c in auc_ranked_result.compounds if c.rank == 1)
        for other in auc_ranked_result.compounds:
            if other.rank != 1:
                assert rank1.auc0t_mg_h_L >= other.auc0t_mg_h_L

    def test_failed_counted(self, triple_result):
        """n_succeeded + n_failed must always equal n_total."""
        assert triple_result.n_succeeded + triple_result.n_failed == triple_result.n_total


# ---------------------------------------------------------------------------
# TestBatchPipelineEdge
# ---------------------------------------------------------------------------


class TestBatchPipelineEdge:
    """Edge-case tests for run_batch_prediction()."""

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="empty"):
            run_batch_prediction([])

    def test_max_compounds_limit(self):
        too_many = [CAFFEINE] * 51
        with pytest.raises(ValueError, match="Maximum"):
            run_batch_prediction(too_many)

    def test_duplicate_smiles(self):
        """Same SMILES twice should be handled gracefully (no crash)."""
        result = run_batch_prediction([CAFFEINE, CAFFEINE])
        assert result.n_total == 2
        assert result.n_succeeded + result.n_failed == result.n_total


# ---------------------------------------------------------------------------
# TestBatchPipelineAPI
# ---------------------------------------------------------------------------


class TestBatchPipelineAPI:
    """API tests for POST /predict/batch."""

    def test_endpoint_200(self):
        resp = client.post(
            "/predict/batch",
            json={"smiles_list": [CAFFEINE, ASPIRIN]},
        )
        assert resp.status_code == 200

    def test_response_structure(self):
        resp = client.post(
            "/predict/batch",
            json={"smiles_list": [CAFFEINE, ASPIRIN]},
        )
        body = resp.json()
        expected_keys = [
            "compounds",
            "n_total",
            "n_succeeded",
            "n_failed",
            "ranking_criterion",
            "best_compound",
            "warnings",
        ]
        for key in expected_keys:
            assert key in body, f"Missing response key: {key}"
        assert len(body["compounds"]) == 2
        compound_keys = [
            "rank",
            "smiles",
            "drug_name",
            "cmax_mg_L",
            "tmax_h",
            "auc0t_mg_h_L",
            "t_half_h",
            "cmax_p50",
            "auc_p50",
            "risk_count",
            "overall_risk_level",
            "confidence",
            "adme_confidence",
            "transporter_ddi_flags",
            "composite_score",
            "warnings",
        ]
        for key in compound_keys:
            assert key in body["compounds"][0], f"Missing compound key: {key}"

    def test_invalid_ranking(self):
        resp = client.post(
            "/predict/batch",
            json={"smiles_list": [CAFFEINE], "ranking": "invalid"},
        )
        assert resp.status_code == 422

    def test_too_many_rejected(self):
        resp = client.post(
            "/predict/batch",
            json={"smiles_list": [CAFFEINE] * 51},
        )
        assert resp.status_code == 422
