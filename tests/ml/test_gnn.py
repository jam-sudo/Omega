"""Tests for GNN architecture: graph builder, encoder, param head, losses, full pipeline."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")

from omega_pbpk.ml.features.graphs import (  # noqa: E402
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
    HAS_RDKIT,
    _FallbackBatch,
    _FallbackData,
    smiles_to_graph,
    smiles_to_graph_batch,
)
from omega_pbpk.ml.models.foundation.gnn_encoder import MolecularEncoder  # noqa: E402
from omega_pbpk.ml.models.foundation.param_head import PKParameterHead  # noqa: E402
from omega_pbpk.ml.training.losses import PKLoss  # noqa: E402

# Skip all RDKit-dependent tests if not installed
requires_rdkit = pytest.mark.skipif(not HAS_RDKIT, reason="RDKit not installed")


# ---------------------------------------------------------------------------
# Task C.1 Tests: smiles_to_graph
# ---------------------------------------------------------------------------


class TestSmilesToGraph:
    """Tests for SMILES-to-graph conversion."""

    @requires_rdkit
    def test_ethanol_atom_count(self):
        """Ethanol (CCO) has 3 heavy atoms."""
        data = smiles_to_graph("CCO")
        assert data is not None
        assert data.x.shape[0] == 3  # C, C, O
        assert data.x.shape[1] == ATOM_FEATURE_DIM

    @requires_rdkit
    def test_ethanol_bond_count(self):
        """Ethanol has 2 bonds -> 4 directed edges."""
        data = smiles_to_graph("CCO")
        assert data is not None
        assert data.edge_index.shape == (2, 4)
        assert data.edge_attr.shape == (4, BOND_FEATURE_DIM)

    @requires_rdkit
    def test_aspirin_structure(self):
        """Aspirin (CC(=O)Oc1ccccc1C(=O)O) - 13 heavy atoms."""
        data = smiles_to_graph("CC(=O)Oc1ccccc1C(=O)O")
        assert data is not None
        assert data.x.shape[0] == 13

    @requires_rdkit
    def test_invalid_smiles_returns_none(self):
        """Invalid SMILES should return None."""
        result = smiles_to_graph("NOT_A_MOLECULE")
        assert result is None

    @requires_rdkit
    def test_single_atom(self):
        """Single atom (no bonds)."""
        data = smiles_to_graph("[Na]")
        assert data is not None
        assert data.x.shape[0] == 1
        assert data.edge_index.shape == (2, 0)

    def test_no_rdkit_raises(self):
        """Should raise ImportError with clear message when RDKit missing."""
        if HAS_RDKIT:
            pytest.skip("RDKit is installed")
        with pytest.raises(ImportError, match="RDKit"):
            smiles_to_graph("CCO")

    @requires_rdkit
    def test_feature_dimensions(self):
        """Check that ATOM_FEATURE_DIM and BOND_FEATURE_DIM are correct."""
        assert ATOM_FEATURE_DIM == 32
        assert BOND_FEATURE_DIM == 10

    @requires_rdkit
    def test_batch_processing(self):
        """Test batch conversion of multiple SMILES."""
        batch = smiles_to_graph_batch(["CCO", "CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1"])
        assert batch is not None
        assert batch.x.shape[1] == ATOM_FEATURE_DIM
        # 3 + 13 + 6 = 22 atoms total
        assert batch.x.shape[0] == 22
        assert batch.batch is not None
        assert batch.batch.max().item() == 2  # 3 graphs (0, 1, 2)

    @requires_rdkit
    def test_batch_with_invalid_skips(self):
        """Invalid SMILES in batch should be skipped."""
        batch = smiles_to_graph_batch(["CCO", "INVALID", "c1ccccc1"])
        assert batch is not None
        # Only 2 valid molecules
        assert batch.batch.max().item() == 1

    @requires_rdkit
    def test_all_invalid_returns_none(self):
        """All invalid SMILES should return None."""
        result = smiles_to_graph_batch(["INVALID1", "INVALID2"])
        assert result is None


# ---------------------------------------------------------------------------
# Task C.2 Tests: MolecularEncoder
# ---------------------------------------------------------------------------


class TestMolecularEncoder:
    """Tests for the GNN encoder."""

    def _make_dummy_graph(self, num_atoms: int = 5, num_edges: int = 8):
        """Create a dummy graph for testing without RDKit."""
        x = torch.randn(num_atoms, ATOM_FEATURE_DIM)
        edge_index = torch.randint(0, num_atoms, (2, num_edges))
        edge_attr = torch.randn(num_edges, BOND_FEATURE_DIM)
        return _FallbackData(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def _make_dummy_batch(self, batch_size: int = 4, atoms_per: int = 5, edges_per: int = 8):
        """Create a dummy batched graph."""
        xs, eis, eas, bis = [], [], [], []
        offset = 0
        for i in range(batch_size):
            xs.append(torch.randn(atoms_per, ATOM_FEATURE_DIM))
            eis.append(torch.randint(0, atoms_per, (2, edges_per)) + offset)
            eas.append(torch.randn(edges_per, BOND_FEATURE_DIM))
            bis.append(torch.full((atoms_per,), i, dtype=torch.long))
            offset += atoms_per
        return _FallbackBatch(
            x=torch.cat(xs),
            edge_index=torch.cat(eis, dim=1),
            edge_attr=torch.cat(eas),
            batch=torch.cat(bis),
        )

    def test_output_shape_single(self):
        """Single graph -> (1, 256) embedding."""
        encoder = MolecularEncoder()
        data = self._make_dummy_graph()
        out = encoder(data)
        assert out.shape == (1, 256)

    def test_output_shape_batch(self):
        """Batch of 4 graphs -> (4, 256) embedding."""
        encoder = MolecularEncoder()
        batch = self._make_dummy_batch(batch_size=4)
        out = encoder(batch)
        assert out.shape == (4, 256)

    def test_custom_dimensions(self):
        """Custom hidden dims and embedding dim."""
        encoder = MolecularEncoder(
            hidden_dims=(64, 128, 128),
            embedding_dim=128,
        )
        data = self._make_dummy_graph()
        out = encoder(data)
        assert out.shape == (1, 128)

    def test_no_edges(self):
        """Single atom graph (no edges) should not crash."""
        x = torch.randn(1, ATOM_FEATURE_DIM)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, BOND_FEATURE_DIM))
        data = _FallbackData(x=x, edge_index=edge_index, edge_attr=edge_attr)
        encoder = MolecularEncoder()
        out = encoder(data)
        assert out.shape == (1, 256)

    def test_gradients_flow(self):
        """Ensure gradients flow through the encoder."""
        encoder = MolecularEncoder()
        data = self._make_dummy_graph()
        out = encoder(data)
        loss = out.sum()
        loss.backward()
        for p in encoder.parameters():
            if p.requires_grad:
                assert p.grad is not None

    @requires_rdkit
    def test_with_real_molecule(self, sample_smiles):
        """Test encoder with a real molecular graph."""
        data = smiles_to_graph(sample_smiles["aspirin"])
        assert data is not None
        encoder = MolecularEncoder()
        out = encoder(data)
        assert out.shape == (1, 256)


# ---------------------------------------------------------------------------
# Task C.3 Tests: PKParameterHead
# ---------------------------------------------------------------------------


class TestPKParameterHead:
    """Tests for the PK parameter prediction head."""

    def test_output_keys(self):
        """All expected parameter names are in output."""
        head = PKParameterHead()
        embedding = torch.randn(2, 256)
        result = head(embedding)
        for name in PKParameterHead.ALL_PARAMS:
            assert name in result
            assert result[name].shape == (2,)

    def test_positive_params_positive(self):
        """Positive params (softplus) should always be > 0."""
        head = PKParameterHead()
        # Test with many random inputs
        embedding = torch.randn(1000, 256)
        result = head(embedding)
        for name in PKParameterHead.POSITIVE_PARAMS:
            assert (result[name] > 0).all(), f"{name} has non-positive values"

    def test_fup_range(self):
        """fup should be in [0, 1]."""
        head = PKParameterHead()
        embedding = torch.randn(1000, 256)
        result = head(embedding)
        assert (result["fup"] >= 0).all()
        assert (result["fup"] <= 1).all()

    def test_rbp_range(self):
        """rbp should be in [0.5, 3.0]."""
        head = PKParameterHead()
        embedding = torch.randn(1000, 256)
        result = head(embedding)
        assert (result["rbp"] >= 0.5).all()
        assert (result["rbp"] <= 3.0).all()

    def test_bioavailability_range(self):
        """bioavailability should be in [0, 1]."""
        head = PKParameterHead()
        embedding = torch.randn(1000, 256)
        result = head(embedding)
        assert (result["bioavailability"] >= 0).all()
        assert (result["bioavailability"] <= 1).all()

    def test_gradients_flow(self):
        """Ensure gradients flow through the head."""
        head = PKParameterHead()
        embedding = torch.randn(4, 256, requires_grad=True)
        result = head(embedding)
        loss = sum(v.sum() for v in result.values())
        loss.backward()
        assert embedding.grad is not None

    def test_to_drug_kwargs(self):
        """Test conversion to Drug-compatible kwargs."""
        head = PKParameterHead()
        embedding = torch.randn(2, 256)
        result = head(embedding)
        kwargs = head.to_drug_kwargs(result, idx=0)
        assert "mw" in kwargs
        assert "logP" in kwargs
        assert "fup" in kwargs
        assert "rbp" in kwargs
        assert "clint" in kwargs
        assert "CYP3A4" in kwargs["clint"]
        assert isinstance(kwargs["mw"], float)


# ---------------------------------------------------------------------------
# Task C.4 Tests: PKLoss
# ---------------------------------------------------------------------------


class TestPKLoss:
    """Tests for physics-informed loss components."""

    def test_zero_loss_perfect_predictions(self):
        """Perfect predictions should give zero PK MSE."""
        loss_fn = PKLoss()
        pk = {"cmax": torch.tensor([10.0, 20.0]), "auc": torch.tensor([100.0, 200.0])}
        result = loss_fn.pk_metric_mse(pk, pk)
        assert result.item() == pytest.approx(0.0, abs=1e-6)

    def test_positive_loss_imperfect_predictions(self):
        """Imperfect predictions should give positive PK MSE."""
        loss_fn = PKLoss()
        pred = {"cmax": torch.tensor([10.0]), "auc": torch.tensor([100.0])}
        true = {"cmax": torch.tensor([15.0]), "auc": torch.tensor([150.0])}
        result = loss_fn.pk_metric_mse(pred, true)
        assert result.item() > 0

    def test_mass_conservation_perfect(self):
        """Perfect mass balance should give zero loss."""
        loss_fn = PKLoss()
        tissue = torch.tensor([[5.0, 3.0, 2.0]])
        eliminated = torch.tensor([90.0])
        dose = torch.tensor([100.0])
        result = loss_fn.mass_conservation(tissue, eliminated, dose)
        assert result.item() == pytest.approx(0.0, abs=1e-6)

    def test_mass_conservation_violation(self):
        """Mass balance violation should give positive loss."""
        loss_fn = PKLoss()
        tissue = torch.tensor([[5.0, 3.0, 2.0]])
        eliminated = torch.tensor([50.0])  # Missing 40
        dose = torch.tensor([100.0])
        result = loss_fn.mass_conservation(tissue, eliminated, dose)
        assert result.item() > 0
        assert result.item() == pytest.approx(0.4, abs=1e-6)

    def test_non_negativity_all_positive(self):
        """All positive concentrations should give zero penalty."""
        loss_fn = PKLoss()
        conc = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        result = loss_fn.non_negativity(conc)
        assert result.item() == pytest.approx(0.0, abs=1e-6)

    def test_non_negativity_has_negative(self):
        """Negative concentrations should give positive penalty."""
        loss_fn = PKLoss()
        conc = torch.tensor([[1.0, -2.0, 3.0, -4.0]])
        result = loss_fn.non_negativity(conc)
        assert result.item() > 0

    def test_monotonic_terminal_decreasing(self):
        """Monotonically decreasing after peak should give zero penalty."""
        loss_fn = PKLoss()
        curve = torch.tensor([[1.0, 5.0, 4.0, 3.0, 2.0, 1.0]])
        result = loss_fn.monotonic_terminal(curve)
        assert result.item() == pytest.approx(0.0, abs=1e-6)

    def test_monotonic_terminal_violation(self):
        """Non-monotonic terminal phase should give positive penalty."""
        loss_fn = PKLoss()
        # Peak at index 1, then violation at index 3 (3 > 2)
        curve = torch.tensor([[1.0, 5.0, 2.0, 3.0, 1.0]])
        result = loss_fn.monotonic_terminal(curve)
        assert result.item() > 0

    def test_param_plausibility_in_range(self):
        """In-range parameters should give zero penalty."""
        loss_fn = PKLoss()
        params = {
            "clint_3a4": torch.tensor([1.0]),
            "fup": torch.tensor([0.5]),
            "rbp": torch.tensor([1.0]),
            "vd": torch.tensor([50.0]),
        }
        result = loss_fn.parameter_plausibility(params)
        assert result.item() == pytest.approx(0.0, abs=1e-6)

    def test_param_plausibility_out_of_range(self):
        """Out-of-range parameters should give positive penalty."""
        loss_fn = PKLoss()
        params = {
            "clint_3a4": torch.tensor([5000.0]),  # too high
            "fup": torch.tensor([-0.5]),  # negative
            "vd": torch.tensor([50000.0]),  # too high
        }
        result = loss_fn.parameter_plausibility(params)
        assert result.item() > 0

    def test_forward_returns_all_components(self):
        """Forward should return dict with total_loss and all components."""
        loss_fn = PKLoss()
        pred_pk = {"cmax": torch.tensor([10.0]), "auc": torch.tensor([100.0])}
        true_pk = {"cmax": torch.tensor([12.0]), "auc": torch.tensor([110.0])}
        params = {"clint_3a4": torch.tensor([1.0]), "fup": torch.tensor([0.5])}
        result = loss_fn(pred_pk, true_pk, params)
        assert "total_loss" in result
        assert "pk_mse" in result
        assert "mass_conservation" in result
        assert "non_negativity" in result
        assert "monotonic_terminal" in result
        assert "param_plausibility" in result

    def test_forward_with_curve(self):
        """Forward with curve data should compute all physics losses."""
        loss_fn = PKLoss()
        pred_pk = {"cmax": torch.tensor([10.0])}
        true_pk = {"cmax": torch.tensor([10.0])}
        params = {"fup": torch.tensor([0.5])}
        curve = torch.tensor([[1.0, 5.0, 4.0, 3.0, 2.0]])
        tissue = torch.tensor([[3.0, 2.0]])
        elim = torch.tensor([95.0])
        dose = torch.tensor([100.0])
        result = loss_fn(
            pred_pk,
            true_pk,
            params,
            predicted_curve=curve,
            tissue_amounts=tissue,
            eliminated=elim,
            dose=dose,
        )
        assert result["total_loss"].item() >= 0

    def test_custom_weights(self):
        """Custom weights should affect total loss."""
        loss_heavy = PKLoss(w_pk_mse=10.0)
        loss_light = PKLoss(w_pk_mse=0.1)
        pred_pk = {"cmax": torch.tensor([10.0])}
        true_pk = {"cmax": torch.tensor([20.0])}
        params = {"fup": torch.tensor([0.5])}
        r_heavy = loss_heavy(pred_pk, true_pk, params)
        r_light = loss_light(pred_pk, true_pk, params)
        assert r_heavy["total_loss"].item() > r_light["total_loss"].item()


# ---------------------------------------------------------------------------
# Task C.5: Full forward pass test
# ---------------------------------------------------------------------------


class TestFullPipeline:
    """End-to-end test: SMILES -> graph -> encoder -> param_head -> params."""

    def test_full_forward_dummy(self):
        """Full pipeline with dummy graph data."""
        # Create dummy graph
        x = torch.randn(5, ATOM_FEATURE_DIM)
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]])
        edge_attr = torch.randn(8, BOND_FEATURE_DIM)
        data = _FallbackData(x=x, edge_index=edge_index, edge_attr=edge_attr)

        encoder = MolecularEncoder()
        param_head = PKParameterHead()

        # Forward pass
        embedding = encoder(data)
        assert embedding.shape == (1, 256)

        params = param_head(embedding)
        assert "fup" in params
        assert "rbp" in params
        assert "logP" in params
        assert "clint_3a4" in params

        # Check constraints
        assert (params["fup"] >= 0).all() and (params["fup"] <= 1).all()
        assert (params["rbp"] >= 0.5).all() and (params["rbp"] <= 3.0).all()
        assert (params["mw"] > 0).all()

    @requires_rdkit
    def test_full_forward_real_molecule(self, sample_smiles):
        """Full pipeline with real SMILES (aspirin)."""
        data = smiles_to_graph(sample_smiles["aspirin"])
        assert data is not None

        encoder = MolecularEncoder()
        param_head = PKParameterHead()

        embedding = encoder(data)
        params = param_head(embedding)

        # All expected params present
        for name in PKParameterHead.ALL_PARAMS:
            assert name in params

        # Drug-compatible kwargs
        kwargs = param_head.to_drug_kwargs(params, idx=0)
        assert 0 < kwargs["fup"] <= 1
        assert 0.5 <= kwargs["rbp"] <= 3.0

    @requires_rdkit
    def test_full_forward_batch(self, sample_smiles):
        """Full pipeline with batch of molecules."""
        smiles_list = [
            sample_smiles["ethanol"],
            sample_smiles["aspirin"],
            sample_smiles["caffeine"],
        ]
        batch = smiles_to_graph_batch(smiles_list)
        assert batch is not None

        encoder = MolecularEncoder()
        param_head = PKParameterHead()

        embedding = encoder(batch)
        assert embedding.shape == (3, 256)

        params = param_head(embedding)
        for name in PKParameterHead.ALL_PARAMS:
            assert params[name].shape == (3,)

    def test_gradients_end_to_end(self):
        """Gradients should flow from loss back to encoder."""
        x = torch.randn(5, ATOM_FEATURE_DIM)
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]])
        edge_attr = torch.randn(8, BOND_FEATURE_DIM)
        data = _FallbackData(x=x, edge_index=edge_index, edge_attr=edge_attr)

        encoder = MolecularEncoder()
        param_head = PKParameterHead()
        loss_fn = PKLoss()

        embedding = encoder(data)
        params = param_head(embedding)

        # Create fake targets
        pred_pk = {"cmax": params["mw"], "auc": params["vd"]}
        true_pk = {"cmax": torch.tensor([300.0]), "auc": torch.tensor([50.0])}

        losses = loss_fn(pred_pk, true_pk, params)
        losses["total_loss"].backward()

        # Check gradients exist in encoder
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.parameters())
        assert has_grad, "No gradients in encoder"
