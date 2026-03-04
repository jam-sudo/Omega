"""Tests for ML-based tissue Kp predictor (Phase 42)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from omega_pbpk.ml_models.kp_predictor import (
    TISSUES,
    KpEnsemble,
    KpInput,
    KpOutput,
    _encode_features,
    generate_kp_training_data,
)

# ---------------------------------------------------------------------------
# TestKpPredictorInput
# ---------------------------------------------------------------------------


class TestKpPredictorInput:
    @pytest.mark.unit
    def test_kp_input_defaults(self):
        inp = KpInput(logP=2.0, pKa=7.0, fup=0.1, mw=300.0)
        assert inp.hbd == 0
        assert inp.hba == 0
        assert inp.compound_type == "neutral"

    @pytest.mark.unit
    def test_kp_input_custom(self):
        inp = KpInput(logP=1.5, pKa=5.0, fup=0.3, mw=250.0, compound_type="acid")
        assert inp.compound_type == "acid"
        assert inp.logP == 1.5
        assert inp.fup == 0.3


# ---------------------------------------------------------------------------
# TestFeatureEncoding
# ---------------------------------------------------------------------------


class TestFeatureEncoding:
    @pytest.mark.unit
    def test_encode_features_shape(self):
        inp = KpInput(logP=2.0, pKa=7.0, fup=0.1, mw=300.0)
        features = _encode_features(inp)
        assert features.shape == (8,)
        assert features.dtype == np.float32

    @pytest.mark.unit
    def test_encode_features_one_hot(self):
        base_kw = {"logP": 2.0, "pKa": 7.0, "fup": 0.1, "mw": 300.0}
        neutral = _encode_features(KpInput(**base_kw, compound_type="neutral"))
        acid = _encode_features(KpInput(**base_kw, compound_type="acid"))
        base = _encode_features(KpInput(**base_kw, compound_type="base"))
        ampholyte = _encode_features(KpInput(**base_kw, compound_type="ampholyte"))

        # neutral: is_neutral=1, is_acid=0, is_base=0
        assert neutral[4] == pytest.approx(1.0)
        assert neutral[5] == pytest.approx(0.0)
        assert neutral[6] == pytest.approx(0.0)

        # acid: is_neutral=0, is_acid=1, is_base=0
        assert acid[4] == pytest.approx(0.0)
        assert acid[5] == pytest.approx(1.0)
        assert acid[6] == pytest.approx(0.0)

        # base: is_neutral=0, is_acid=0, is_base=1
        assert base[4] == pytest.approx(0.0)
        assert base[5] == pytest.approx(0.0)
        assert base[6] == pytest.approx(1.0)

        # ampholyte: is_neutral=0, is_acid=1, is_base=1
        assert ampholyte[4] == pytest.approx(0.0)
        assert ampholyte[5] == pytest.approx(1.0)
        assert ampholyte[6] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# TestTrainingDataGeneration
# ---------------------------------------------------------------------------


class TestTrainingDataGeneration:
    @pytest.mark.unit
    def test_generate_kp_data_shapes(self):
        X, y = generate_kp_training_data(n_samples=500, seed=0)
        assert X.shape == (500, 8)
        assert y.shape == (500, 13)
        assert X.dtype == np.float32
        assert y.dtype == np.float32

    @pytest.mark.unit
    def test_generate_kp_data_positive(self):
        X, y = generate_kp_training_data(n_samples=500, seed=0)
        # y contains log10(Kp); exp transform should give all positive Kp
        kp_vals = 10.0**y
        assert np.all(kp_vals > 0)


# ---------------------------------------------------------------------------
# TestKpMLP
# ---------------------------------------------------------------------------


class TestKpMLP:
    @pytest.mark.unit
    def test_kp_mlp_forward_shape(self):
        from omega_pbpk.ml_models.pk_surrogate import NumpyMLP

        mlp = NumpyMLP(layer_sizes=(8, 32, 32, 13), seed=0)
        X = np.random.default_rng(0).random((10, 8)).astype(np.float32)
        out = mlp.forward(X)
        assert out.shape == (10, 13)

    @pytest.mark.unit
    def test_kp_mlp_deterministic(self):
        from omega_pbpk.ml_models.pk_surrogate import NumpyMLP

        mlp1 = NumpyMLP(layer_sizes=(8, 32, 32, 13), seed=42)
        mlp2 = NumpyMLP(layer_sizes=(8, 32, 32, 13), seed=42)
        X = np.ones((5, 8), dtype=np.float32)
        out1 = mlp1.forward(X)
        out2 = mlp2.forward(X)
        np.testing.assert_array_equal(out1, out2)


# ---------------------------------------------------------------------------
# TestKpEnsemble
# ---------------------------------------------------------------------------


class TestKpEnsemble:
    @pytest.fixture
    def trained_ensemble(self):
        ensemble = KpEnsemble(seed=42)
        X, y = generate_kp_training_data(n_samples=200, seed=42)
        ensemble.train(X, y, epochs=20, lr=0.005, batch_size=32)
        return ensemble

    @pytest.mark.integration
    def test_ensemble_train_predict(self, trained_ensemble):
        inp = KpInput(logP=2.0, pKa=7.4, fup=0.1, mw=300.0, compound_type="neutral")
        out = trained_ensemble.predict(inp)
        assert isinstance(out, KpOutput)
        assert set(out.tissue_kp.keys()) == set(TISSUES)
        assert len(out.tissue_kp) == 13

    @pytest.mark.integration
    def test_ensemble_confidence(self, trained_ensemble):
        # Typical drug-like compound
        inp = KpInput(logP=2.0, pKa=7.4, fup=0.1, mw=300.0, compound_type="neutral")
        out = trained_ensemble.predict(inp)
        assert out.confidence in ("high", "medium", "low")

    @pytest.mark.integration
    def test_ensemble_save_load(self, trained_ensemble):
        inp = KpInput(logP=2.0, pKa=7.4, fup=0.1, mw=300.0, compound_type="neutral")
        out_before = trained_ensemble.predict(inp)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "kp_ensemble.npz"
            trained_ensemble.save(save_path)

            loaded = KpEnsemble(seed=42)
            loaded.load(save_path)

            out_after = loaded.predict(inp)

        for tissue in TISSUES:
            assert abs(out_before.tissue_kp[tissue] - out_after.tissue_kp[tissue]) < 1e-3


# ---------------------------------------------------------------------------
# TestKpValidation
# ---------------------------------------------------------------------------


class TestKpValidation:
    @pytest.mark.integration
    def test_divergence_fallback(self, caplog):
        import logging

        ensemble = KpEnsemble(seed=42)
        X, y = generate_kp_training_data(n_samples=200, seed=42)
        ensemble.train(X, y, epochs=20, lr=0.005, batch_size=32)

        inp = KpInput(logP=2.0, pKa=7.4, fup=0.1, mw=300.0, compound_type="neutral")

        # Inject extreme ML output that diverges > 2x from RR for all tissues
        extreme_ml_kp = {tissue: 9999.0 for tissue in TISSUES}

        with caplog.at_level(logging.DEBUG):
            corrected, warnings = ensemble._validate_vs_rr(inp, extreme_ml_kp)

        # Some tissues should have been replaced with RR values
        assert len(warnings) > 0
        # At least one tissue should have been corrected away from 9999
        assert any(v < 9999.0 for v in corrected.values())


# ---------------------------------------------------------------------------
# TestHeuristicsIntegration
# ---------------------------------------------------------------------------


class TestHeuristicsIntegration:
    @pytest.mark.integration
    def test_ml_kp_method_registered(self):
        from omega_pbpk.core.heuristics import get_partition_method

        fn = get_partition_method("ml")
        assert callable(fn)


# ---------------------------------------------------------------------------
# TestAPIEndpoint
# ---------------------------------------------------------------------------


class TestAPIEndpoint:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        from omega_pbpk.api.app import app

        return TestClient(app)

    @pytest.mark.unit
    def test_kp_endpoint_ml(self, client):
        payload = {
            "logP": 2.0,
            "pKa": 7.4,
            "fup": 0.1,
            "mw": 300.0,
            "compound_type": "neutral",
            "method": "ml",
        }
        resp = client.post("/predict/kp", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "tissue_kp" in data
        assert len(data["tissue_kp"]) == 13

    @pytest.mark.unit
    def test_kp_endpoint_rr(self, client):
        payload = {
            "logP": 2.0,
            "pKa": 7.4,
            "fup": 0.1,
            "mw": 300.0,
            "compound_type": "neutral",
            "method": "rodgers_rowland",
        }
        resp = client.post("/predict/kp", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "tissue_kp" in data
        assert data["method_used"] == "rodgers_rowland"
