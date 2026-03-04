"""Tests for Phase 23: End-to-end full prediction pipeline.

Covers run_full_prediction() unit behaviour and POST /predict/full endpoint.
Run with:
    PYTHONPATH=src python3 -m pytest tests/test_full_pipeline.py -v --tb=short
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from omega_pbpk.api.app import app
from omega_pbpk.prediction.full_pipeline import FullPredictionResult, run_full_prediction

client = TestClient(app)

CAFFEINE_SMILES = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"


# ---------------------------------------------------------------------------
# Shared fixture — run once per class to avoid repeated slow pipeline calls
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def caffeine_result() -> FullPredictionResult:
    return run_full_prediction(CAFFEINE_SMILES, dose_mg=200.0)


# ---------------------------------------------------------------------------
# TestFullPipelineUnit
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestFullPipelineUnit:
    """~10 unit tests against run_full_prediction() with caffeine SMILES."""

    def test_returns_full_prediction_result(self, caffeine_result):
        assert isinstance(caffeine_result, FullPredictionResult)

    def test_adme_populated(self, caffeine_result):
        adme = caffeine_result.adme
        for key in ("mw", "logP", "fup", "rbp", "clint_3a4"):
            assert key in adme, f"Missing ADME key: {key}"

    def test_transporter_lists(self, caffeine_result):
        assert isinstance(caffeine_result.transporter_substrates, tuple)
        assert isinstance(caffeine_result.transporter_inhibitors, tuple)
        assert all(isinstance(s, str) for s in caffeine_result.transporter_substrates)
        assert all(isinstance(s, str) for s in caffeine_result.transporter_inhibitors)

    def test_pk_positive(self, caffeine_result):
        assert caffeine_result.cmax_mg_L > 0
        assert caffeine_result.auc0t_mg_h_L > 0
        assert caffeine_result.tmax_h >= 0

    def test_uq_ordered(self, caffeine_result):
        assert caffeine_result.cmax_p5 <= caffeine_result.cmax_p50 <= caffeine_result.cmax_p95
        assert caffeine_result.auc_p5 <= caffeine_result.auc_p50 <= caffeine_result.auc_p95

    def test_risk_flags_dict(self, caffeine_result):
        assert isinstance(caffeine_result.risk_flags, dict)
        assert all(isinstance(v, bool) for v in caffeine_result.risk_flags.values())

    def test_risk_count_matches(self, caffeine_result):
        expected = sum(1 for v in caffeine_result.risk_flags.values() if v)
        assert caffeine_result.risk_count == expected

    def test_risk_level_valid(self, caffeine_result):
        assert caffeine_result.overall_risk_level in ("low", "moderate", "high")

    def test_confidence_valid(self, caffeine_result):
        assert caffeine_result.confidence in ("low", "medium", "high")

    def test_time_cp_match(self, caffeine_result):
        assert len(caffeine_result.time_h) == len(caffeine_result.cp_mg_L)
        assert len(caffeine_result.time_h) > 1


# ---------------------------------------------------------------------------
# TestFullPipelineEdge
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestFullPipelineEdge:
    """Edge-case tests."""

    def test_iv_route(self):
        result = run_full_prediction(CAFFEINE_SMILES, dose_mg=100.0, route="iv")
        # IV bolus: peak is at or very near t = 0
        assert result.tmax_h < 1.0
        assert result.cmax_mg_L > 0
        assert result.auc0t_mg_h_L > 0
        assert result.overall_risk_level in ("low", "moderate", "high")

    def test_invalid_smiles(self):
        # Should not raise — graceful degradation regardless of rdkit availability.
        # Without rdkit the simplified heuristic predictor accepts any string, so
        # we only assert the result type and valid field values (not a specific confidence).
        result = run_full_prediction("INVALID_XYZ_###")
        assert isinstance(result, FullPredictionResult)
        assert result.overall_risk_level in ("low", "moderate", "high")
        assert result.confidence in ("low", "medium", "high")


# ---------------------------------------------------------------------------
# TestFullPipelineAPI
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestFullPipelineAPI:
    """API tests for POST /predict/full."""

    def test_endpoint_200(self):
        resp = client.post(
            "/predict/full",
            json={"smiles": CAFFEINE_SMILES, "dose_mg": 200.0},
        )
        assert resp.status_code == 200

    def test_response_structure(self):
        resp = client.post(
            "/predict/full",
            json={"smiles": CAFFEINE_SMILES, "dose_mg": 200.0},
        )
        body = resp.json()
        expected_keys = [
            "drug_name",
            "smiles",
            "adme",
            "adme_confidence",
            "transporter_substrates",
            "transporter_inhibitors",
            "transporter_ddi_flags",
            "transporter_confidence",
            "cmax_mg_L",
            "tmax_h",
            "auc0t_mg_h_L",
            "t_half_h",
            "time_h",
            "cp_mg_L",
            "cmax_p5",
            "cmax_p50",
            "cmax_p95",
            "auc_p5",
            "auc_p50",
            "auc_p95",
            "risk_flags",
            "risk_count",
            "overall_risk_level",
            "confidence",
            "warnings",
        ]
        for key in expected_keys:
            assert key in body, f"Missing response key: {key}"

    def test_long_smiles_rejected(self):
        resp = client.post(
            "/predict/full",
            json={"smiles": "C" * 501},
        )
        assert resp.status_code == 422
