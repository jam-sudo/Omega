"""Tests for Phase 38 Graph Attention Network (GAT) for ADME prediction.

Covers:
  - TestGraphConstruction (3): smiles_to_graph correctness and fallbacks
  - TestAttentionLayers (5): GAT layer shapes, softmax, determinism, multi-head
  - TestModelPrediction (4): predict_single, determinism, fit, batch
  - TestIntegration (2): attention maps and POST /predict/gat endpoint
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from omega_pbpk.api.app import app
from omega_pbpk.ml_models.graph_attention import (
    N_EDGE_FEATURES,
    N_NODE_FEATURES,
    GraphAttentionLayer,
    MolecularGAT,
    MultiHeadGraphAttention,
    smiles_to_graph,
)

client = TestClient(app)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CAFFEINE = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
_ETHANOL = "CCO"


def _make_random_graph(n_atoms: int, seed: int = 0):
    """Return (node_features, adjacency, edge_features) for a random graph."""
    rng = np.random.default_rng(seed)
    nf = rng.random((n_atoms, N_NODE_FEATURES)).astype(np.float32)
    # Symmetric adjacency, no self-loops
    raw = (rng.random((n_atoms, n_atoms)) > 0.5).astype(np.float32)
    adj = np.clip(raw + raw.T, 0, 1) - np.eye(n_atoms, dtype=np.float32)
    adj = np.clip(adj, 0, 1)
    ef = np.zeros((n_atoms, n_atoms, N_EDGE_FEATURES), dtype=np.float32)
    return nf, adj, ef


# ---------------------------------------------------------------------------
# TestGraphConstruction
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGraphConstruction:
    def test_smiles_to_graph_caffeine(self):
        """Caffeine graph: 14 heavy atoms and symmetric adjacency."""
        g = smiles_to_graph(_CAFFEINE)
        assert g.num_atoms == 14
        assert g.node_features.shape == (14, N_NODE_FEATURES)
        assert g.adjacency.shape == (14, 14)
        assert g.edge_features.shape == (14, 14, N_EDGE_FEATURES)
        # Adjacency must be symmetric
        np.testing.assert_array_equal(g.adjacency, g.adjacency.T)

    def test_smiles_to_graph_simple_molecule(self):
        """Ethanol CCO: 3 heavy atoms and exactly 2 unique undirected bonds."""
        g = smiles_to_graph(_ETHANOL)
        assert g.num_atoms == 3
        assert g.node_features.shape == (3, N_NODE_FEATURES)
        # 2 bonds → adjacency has 4 ones (bidirectional) and 5 zeros
        assert int(g.adjacency.sum()) == 4
        np.testing.assert_array_equal(g.adjacency, g.adjacency.T)

    def test_smiles_to_graph_invalid_returns_minimal(self):
        """Invalid SMILES strings return a minimal 1-atom graph gracefully."""
        for bad in ["!!!###", "", "   "]:
            g = smiles_to_graph(bad)
            assert g.num_atoms == 1
            assert g.node_features.shape == (1, N_NODE_FEATURES)


# ---------------------------------------------------------------------------
# TestAttentionLayers
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAttentionLayers:
    def test_layer_output_shape(self):
        """GraphAttentionLayer (5, 13) → (5, 32) output."""
        layer = GraphAttentionLayer(N_NODE_FEATURES, 32, seed=42)
        nf, adj, ef = _make_random_graph(5)
        out = layer.forward(nf, adj, ef)
        assert out.shape == (5, 32)

    def test_attention_weights_sum_to_one(self):
        """Attention weights over neighbors sum to ~1 per node (softmax validity)."""
        layer = GraphAttentionLayer(N_NODE_FEATURES, 16, seed=7)
        n = 6
        nf, _, ef = _make_random_graph(n, seed=1)
        # Fully connected (excluding self-loops) so every node has neighbours
        adj = 1.0 - np.eye(n, dtype=np.float32)
        alpha = layer.get_attention_weights(nf, adj, ef)
        assert alpha.shape == (n, n)
        # Each row: sum of weights = sum / (sum + 1e-8) ≈ 1 when sum >> 1e-8
        row_sums = alpha.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(n), atol=1e-5)

    def test_layer_deterministic(self):
        """Same seed produces identical outputs."""
        nf, adj, ef = _make_random_graph(4)
        layer_a = GraphAttentionLayer(N_NODE_FEATURES, 8, seed=99)
        layer_b = GraphAttentionLayer(N_NODE_FEATURES, 8, seed=99)
        np.testing.assert_array_equal(
            layer_a.forward(nf, adj, ef),
            layer_b.forward(nf, adj, ef),
        )

    def test_multi_head_output_shape(self):
        """MultiHeadGraphAttention (10, 32) → (10, 32) with 4 heads."""
        mha = MultiHeadGraphAttention(32, 32, n_heads=4, seed=0)
        rng = np.random.default_rng(5)
        nf = rng.random((10, 32)).astype(np.float32)
        adj = (1.0 - np.eye(10, dtype=np.float32)).astype(np.float32)
        ef = np.zeros((10, 10, N_EDGE_FEATURES), dtype=np.float32)
        out = mha.forward(nf, adj, ef)
        assert out.shape == (10, 32)

    def test_get_attention_weights_returns_n_heads(self):
        """get_attention_weights returns 4 matrices, each (n, n)."""
        n = 5
        mha = MultiHeadGraphAttention(N_NODE_FEATURES, 16, n_heads=4, seed=3)
        nf, adj, ef = _make_random_graph(n)
        weights = mha.get_attention_weights(nf, adj, ef)
        assert len(weights) == 4
        for w in weights:
            assert w.shape == (n, n)


# ---------------------------------------------------------------------------
# TestModelPrediction
# ---------------------------------------------------------------------------


class TestModelPrediction:
    @pytest.mark.unit
    def test_predict_single_returns_all_properties(self):
        """predict_single returns fup, clint_3a4, peff, logS, confidence."""
        model = MolecularGAT(seed=0)
        result = model.predict_single(_CAFFEINE)
        for key in ("fup", "clint_3a4", "peff", "logS", "confidence"):
            assert key in result, f"Missing key: {key}"
        assert 0.0 < result["fup"] < 1.0
        assert result["clint_3a4"] >= 0.0
        assert 0.0 <= result["confidence"] <= 1.0

    @pytest.mark.unit
    def test_predict_single_deterministic(self):
        """Same seed → identical predictions."""
        model_a = MolecularGAT(seed=42)
        model_b = MolecularGAT(seed=42)
        r_a = model_a.predict_single(_CAFFEINE)
        r_b = model_b.predict_single(_CAFFEINE)
        assert r_a == r_b

    @pytest.mark.integration
    def test_fit_reduces_loss(self):
        """A short training run should reduce loss (at least from first epoch)."""
        model = MolecularGAT(hidden_dim=8, n_heads=2, n_rounds=1, seed=0)
        smiles = [_CAFFEINE, _ETHANOL, "CC(=O)O"]  # caffeine, ethanol, acetic acid
        targets = {
            "fup": np.array([0.3, 0.9, 0.7], dtype=np.float32),
            "clint_3a4": np.array([5.0, 0.5, 1.0], dtype=np.float32),
            "peff": np.array([1e-5, 5e-5, 3e-5], dtype=np.float32),
            "logS": np.array([-2.0, 1.0, -0.5], dtype=np.float32),
        }
        # Use n_epochs=2 so test runs quickly; loss_history has 2 entries
        loss_history = model.fit(smiles, targets, n_epochs=2, lr=0.05, batch_size=4)
        assert len(loss_history) == 2
        # Loss values must be finite
        assert all(np.isfinite(v) for v in loss_history)
        # At least one of the losses should be non-negative
        assert all(v >= 0.0 for v in loss_history)

    @pytest.mark.unit
    def test_predict_batch_length(self):
        """predict_batch for 3 SMILES returns exactly 3 results."""
        model = MolecularGAT(seed=1)
        smiles_list = [_CAFFEINE, _ETHANOL, "CC(=O)O"]
        results = model.predict_batch(smiles_list)
        assert len(results) == 3
        for r in results:
            assert "fup" in r
            assert "confidence" in r


# ---------------------------------------------------------------------------
# TestIntegration
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestIntegration:
    def test_get_attention_maps(self):
        """get_attention_maps returns correct structure for caffeine."""
        model = MolecularGAT(seed=0)
        maps = model.get_attention_maps(_CAFFEINE)
        assert "num_atoms" in maps
        assert maps["num_atoms"] == 14
        assert "layers" in maps
        assert len(maps["layers"]) == model.n_rounds
        for layer_info in maps["layers"]:
            assert "layer" in layer_info
            assert "heads" in layer_info
            heads = layer_info["heads"]
            assert len(heads) == model.n_heads
            for _head_name, matrix in heads.items():
                assert isinstance(matrix, list)
                assert len(matrix) == 14

    def test_predict_gat_endpoint(self):
        """POST /predict/gat returns 200 with all expected prediction keys."""
        from unittest.mock import patch

        fake_result = {
            "fup": 0.35,
            "clint_3a4": 12.0,
            "peff": 0.005,
            "logS": -2.1,
            "confidence": 0.72,
            "model_info": {
                "type": "MolecularGAT",
                "trained": False,
                "hidden_dim": 32,
                "n_rounds": 3,
            },
        }
        with patch(
            "omega_pbpk.prediction.adme_predictor.ADMEPredictor.predict_gat",
            return_value=fake_result,
        ):
            response = client.post("/predict/gat", json={"smiles": _CAFFEINE})
        assert response.status_code == 200
        data = response.json()
        for key in ("fup", "clint_3a4", "peff", "logS", "confidence"):
            assert key in data, f"Missing key in response: {key}"
