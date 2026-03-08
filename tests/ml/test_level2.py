"""Tests for Level 2 end-to-end training pipeline.

Tests cover:
- SMILESToPKModel forward pass
- Gradient flow through entire pipeline
- PKTrainer 1-epoch training
- MultiFidelityCurriculum stage transitions
- MLPKPredictor end-to-end (with mocked ODE)
- PK metric extraction correctness

Note: Tests use synthetic graph tensors to avoid RDKit dependency.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from omega_pbpk.ml.data.synthetic import N_TIMEPOINTS
from omega_pbpk.ml.features.graphs import ATOM_FEATURE_DIM, BOND_FEATURE_DIM, _FallbackData
from omega_pbpk.ml.models.foundation.end_to_end import SMILESToPKModel
from omega_pbpk.ml.models.foundation.param_head import PKParameterHead
from omega_pbpk.ml.training.losses import PKLoss
from omega_pbpk.ml.training.trainer import PKTrainer, TrainingConfig, TrainingHistory


# ---------------------------------------------------------------------------
# Helpers: synthetic graph data (avoids RDKit dependency)
# ---------------------------------------------------------------------------


def _make_fake_graph(n_atoms: int = 3, n_bonds: int = 2) -> _FallbackData:
    """Create a synthetic graph mimicking a small molecule (e.g., ethanol).

    Parameters
    ----------
    n_atoms : int
        Number of atoms (nodes).
    n_bonds : int
        Number of bonds (edges, doubled for undirected).

    Returns
    -------
    _FallbackData with x, edge_index, edge_attr.
    """
    x = torch.randn(n_atoms, ATOM_FEATURE_DIM)
    # Create undirected edges (each bond doubled)
    src = list(range(n_bonds)) + list(range(1, n_bonds + 1))
    dst = list(range(1, n_bonds + 1)) + list(range(n_bonds))
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.randn(len(src), BOND_FEATURE_DIM)
    return _FallbackData(x=x, edge_index=edge_index, edge_attr=edge_attr)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def model() -> SMILESToPKModel:
    """Create a SMILESToPKModel for testing."""
    return SMILESToPKModel(embedding_dim=256, surrogate_n_output=N_TIMEPOINTS)


@pytest.fixture
def fake_graph() -> _FallbackData:
    """Pre-computed synthetic graph (3 atoms, 2 bonds like ethanol)."""
    return _make_fake_graph(n_atoms=3, n_bonds=2)


@pytest.fixture
def tiny_dataset() -> TensorDataset:
    """Create a tiny synthetic dataset for training tests."""
    n = 32
    params = torch.randn(n, 6).abs()  # 6D positive params
    curves = torch.randn(n, N_TIMEPOINTS).abs()  # positive curves
    metrics = torch.randn(n, 4).abs()  # positive metrics
    return TensorDataset(params, curves, metrics)


@pytest.fixture
def tiny_train_val(tiny_dataset):
    """Split tiny dataset into train/val."""
    n = len(tiny_dataset)
    n_val = max(1, n // 4)
    n_train = n - n_val
    params, curves, metrics = tiny_dataset.tensors
    train = TensorDataset(params[:n_train], curves[:n_train], metrics[:n_train])
    val = TensorDataset(params[n_train:], curves[n_train:], metrics[n_train:])
    return train, val


# ---------------------------------------------------------------------------
# Task L2.1: SMILESToPKModel
# ---------------------------------------------------------------------------


class TestSMILESToPKModel:
    """Tests for end-to-end model wiring."""

    def test_forward_with_graph(self, model: SMILESToPKModel, fake_graph):
        """Forward pass with a pre-computed graph produces expected output dict."""
        model.eval()
        with torch.no_grad():
            result = model(fake_graph)

        assert "params" in result
        assert "curve" in result
        assert "pk_metrics" in result
        assert "embedding" in result

    def test_output_shapes(self, model: SMILESToPKModel, fake_graph):
        """Output tensors have correct shapes (batch=1 for single molecule)."""
        model.eval()
        with torch.no_grad():
            result = model(fake_graph)

        # Curve should be (1, 241)
        assert result["curve"].dim() == 2
        assert result["curve"].shape[0] == 1
        assert result["curve"].shape[1] == N_TIMEPOINTS

        # Embedding should be (1, 256)
        assert result["embedding"].shape == (1, 256)

        # PK metrics should each be (1,)
        for key in ("cmax", "auc", "tmax", "t_half"):
            assert key in result["pk_metrics"]
            assert result["pk_metrics"][key].shape == (1,)

    def test_params_dict_keys(self, model: SMILESToPKModel, fake_graph):
        """Predicted params dict has all expected parameter names."""
        model.eval()
        with torch.no_grad():
            result = model(fake_graph)

        expected_params = set(PKParameterHead.ALL_PARAMS)
        actual_params = set(result["params"].keys())
        assert expected_params == actual_params

    def test_curve_non_negative(self, model: SMILESToPKModel, fake_graph):
        """Predicted curve values are non-negative."""
        model.eval()
        with torch.no_grad():
            result = model(fake_graph)

        assert (result["curve"] >= 0).all()

    def test_param_constraints(self, model: SMILESToPKModel, fake_graph):
        """Predicted params respect their activation constraints."""
        model.eval()
        with torch.no_grad():
            result = model(fake_graph)

        params = result["params"]
        # fup should be in [0, 1]
        assert 0.0 <= params["fup"].item() <= 1.0
        # rbp should be in [0.5, 3.0]
        assert 0.5 <= params["rbp"].item() <= 3.0
        # bioavailability should be in [0, 1]
        assert 0.0 <= params["bioavailability"].item() <= 1.0
        # Positive params should be > 0
        assert params["mw"].item() > 0
        assert params["peff"].item() > 0
        assert params["clint_3a4"].item() > 0

    def test_save_and_load(self, model: SMILESToPKModel, fake_graph):
        """Model can be saved and loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_model.pt"
            model.save(path)
            assert path.exists()

            loaded = SMILESToPKModel.load(path)
            loaded.eval()
            with torch.no_grad():
                result = loaded(fake_graph)
            assert "curve" in result

    def test_parameter_count(self, model: SMILESToPKModel):
        """Parameter count returns reasonable values."""
        counts = model.parameter_count()
        assert counts["total"] > 0
        assert counts["encoder"] > 0
        assert counts["param_head"] > 0
        assert counts["surrogate"] > 0
        assert counts["total"] == counts["encoder"] + counts["param_head"] + counts["surrogate"]

    def test_surrogate_param_extraction(self, model: SMILESToPKModel, fake_graph):
        """Surrogate param extraction produces a (batch, 6) tensor."""
        model.eval()
        with torch.no_grad():
            embedding = model.encoder(fake_graph)
            pk_params = model.param_head(embedding)
            surrogate_input = model._extract_surrogate_params(pk_params)

        assert surrogate_input.shape == (1, 6)
        # All values should be finite
        assert torch.isfinite(surrogate_input).all()


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


class TestGradientFlow:
    """Test that gradients flow through the entire pipeline."""

    def test_loss_backward(self, model: SMILESToPKModel, fake_graph):
        """loss.backward() completes without error on the full pipeline."""
        model.train()
        result = model(fake_graph)

        # Fake target metrics
        target_pk = {
            "cmax": torch.tensor([1.0]),
            "auc": torch.tensor([10.0]),
            "tmax": torch.tensor([2.0]),
            "t_half": torch.tensor([5.0]),
        }

        loss_fn = PKLoss()
        loss_dict = loss_fn(
            predicted_pk=result["pk_metrics"],
            true_pk=target_pk,
            predicted_params=result["params"],
            predicted_curve=result["curve"],
        )

        loss = loss_dict["total_loss"]
        loss.backward()

        # Check that gradients exist on encoder parameters
        encoder_grads = [
            p.grad for p in model.encoder.parameters() if p.grad is not None
        ]
        assert len(encoder_grads) > 0, "No gradients on encoder parameters"

        # Check param_head gradients
        head_grads = [
            p.grad for p in model.param_head.parameters() if p.grad is not None
        ]
        assert len(head_grads) > 0, "No gradients on param_head parameters"

        # Check surrogate gradients
        surrogate_grads = [
            p.grad for p in model.surrogate.parameters() if p.grad is not None
        ]
        assert len(surrogate_grads) > 0, "No gradients on surrogate parameters"

    def test_no_nan_gradients(self, model: SMILESToPKModel, fake_graph):
        """Gradients should not contain NaN values."""
        model.train()
        result = model(fake_graph)

        target_pk = {
            "cmax": torch.tensor([0.5]),
            "auc": torch.tensor([5.0]),
            "tmax": torch.tensor([1.0]),
            "t_half": torch.tensor([3.0]),
        }

        loss_fn = PKLoss()
        loss = loss_fn(
            predicted_pk=result["pk_metrics"],
            true_pk=target_pk,
            predicted_params=result["params"],
            predicted_curve=result["curve"],
        )["total_loss"]

        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), (
                    f"NaN gradient in {name}"
                )


# ---------------------------------------------------------------------------
# Task L2.2: PKTrainer
# ---------------------------------------------------------------------------


class TestPKTrainer:
    """Tests for the training loop."""

    def test_train_1_epoch(self, model: SMILESToPKModel, tiny_train_val):
        """Trainer runs 1 epoch without error."""
        train_data, val_data = tiny_train_val
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                lr=1e-3,
                batch_size=8,
                n_epochs=1,
                patience=5,
                checkpoint_dir=tmpdir,
                device="cpu",
            )
            trainer = PKTrainer(model, config)
            history = trainer.train(train_data, val_data)

        assert isinstance(history, TrainingHistory)
        assert len(history.train_losses) == 1
        assert len(history.val_losses) == 1
        assert history.train_losses[0] > 0

    def test_train_3_epochs_loss_decreases_or_stable(
        self, model: SMILESToPKModel, tiny_train_val
    ):
        """Training loss should not diverge over 3 epochs."""
        train_data, val_data = tiny_train_val
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                lr=1e-4,
                batch_size=8,
                n_epochs=3,
                patience=10,
                checkpoint_dir=tmpdir,
                device="cpu",
            )
            trainer = PKTrainer(model, config)
            history = trainer.train(train_data, val_data)

        assert len(history.train_losses) == 3
        # Loss should not explode (allow 10x tolerance from initial)
        assert history.train_losses[-1] < history.train_losses[0] * 10

    def test_evaluate_returns_metrics(
        self, model: SMILESToPKModel, tiny_train_val
    ):
        """Evaluate returns expected metric keys."""
        _, val_data = tiny_train_val
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(checkpoint_dir=tmpdir, device="cpu")
            trainer = PKTrainer(model, config)
            metrics = trainer.evaluate(val_data)

        assert "aafe_cmax" in metrics
        assert "aafe_auc" in metrics
        assert "pct_within_2fold_cmax" in metrics
        assert "n_samples" in metrics
        assert metrics["n_samples"] > 0

    def test_early_stopping(self, model: SMILESToPKModel, tiny_train_val):
        """Early stopping triggers before max epochs."""
        train_data, val_data = tiny_train_val
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                lr=1e-3,
                n_epochs=1000,  # very high
                patience=2,  # very short patience
                checkpoint_dir=tmpdir,
                device="cpu",
                batch_size=8,
            )
            trainer = PKTrainer(model, config)
            history = trainer.train(train_data, val_data)

        # Should stop well before 1000 epochs
        assert len(history.train_losses) < 1000


# ---------------------------------------------------------------------------
# Task L2.3: MultiFidelityCurriculum
# ---------------------------------------------------------------------------


class TestMultiFidelityCurriculum:
    """Tests for curriculum training."""

    def test_stage_transitions(self, model: SMILESToPKModel, tiny_train_val):
        """Curriculum transitions through stages correctly."""
        from omega_pbpk.ml.training.curriculum import (
            CurriculumResult,
            MultiFidelityCurriculum,
            StageConfig,
        )

        train_data, val_data = tiny_train_val

        with tempfile.TemporaryDirectory() as tmpdir:
            # Use very fast configs
            stages = {
                "1cpt_pretrain": StageConfig(
                    name="1cpt_pretrain", lr=1e-3, n_epochs=1, batch_size=8, patience=5
                ),
                "pbpk_finetune": StageConfig(
                    name="pbpk_finetune", lr=1e-4, n_epochs=1, batch_size=8, patience=5
                ),
                "clinical_finetune": StageConfig(
                    name="clinical_finetune", lr=1e-5, n_epochs=1, batch_size=8, patience=5
                ),
            }

            curriculum = MultiFidelityCurriculum(
                model, checkpoint_dir=tmpdir, device="cpu", stages=stages
            )

            result = curriculum.run_curriculum(
                stage1_train=train_data,
                stage1_val=val_data,
                stage2_train=train_data,
                stage2_val=val_data,
                # Skip stage 3 (no clinical data)
            )

        assert isinstance(result, CurriculumResult)
        assert "1cpt_pretrain" in result.stages_completed
        assert "pbpk_finetune" in result.stages_completed
        assert "clinical_finetune" not in result.stages_completed
        assert len(result.stage_histories) == 2

    def test_skip_all_stages(self, model: SMILESToPKModel):
        """Curriculum handles no data gracefully."""
        from omega_pbpk.ml.training.curriculum import MultiFidelityCurriculum

        with tempfile.TemporaryDirectory() as tmpdir:
            curriculum = MultiFidelityCurriculum(
                model, checkpoint_dir=tmpdir, device="cpu"
            )
            result = curriculum.run_curriculum()

        assert len(result.stages_completed) == 0

    def test_single_stage(self, model: SMILESToPKModel, tiny_train_val):
        """Running a single stage works."""
        from omega_pbpk.ml.training.curriculum import MultiFidelityCurriculum, StageConfig

        train_data, val_data = tiny_train_val
        with tempfile.TemporaryDirectory() as tmpdir:
            curriculum = MultiFidelityCurriculum(
                model, checkpoint_dir=tmpdir, device="cpu"
            )
            history = curriculum.run_stage(
                "1cpt_pretrain",
                train_data,
                val_data,
                StageConfig(name="1cpt_pretrain", lr=1e-3, n_epochs=1, batch_size=8),
            )

        assert isinstance(history, TrainingHistory)
        assert len(history.train_losses) >= 1


# ---------------------------------------------------------------------------
# Task L2.4: MLPKPredictor
# ---------------------------------------------------------------------------


class TestMLPKPredictor:
    """Tests for inference pipeline."""

    def test_predict_with_mocked_ode(self, model: SMILESToPKModel, fake_graph):
        """End-to-end prediction with mocked ODE."""
        from omega_pbpk.ml.models.foundation.inference import MLPKPredictor

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save model
            model_path = Path(tmpdir) / "test.pt"
            model.save(model_path)

            predictor = MLPKPredictor(model_path=str(model_path), device="cpu")

            # Mock the real ODE and smiles_to_graph
            mock_sim_result = MagicMock()
            mock_sim_result.pk_summary.return_value = {
                "Cmax_mg_L": 1.5,
                "Tmax_h": 2.0,
                "AUC_mg_h_L": 15.0,
                "half_life_h": 6.0,
                "CL_L_h": 0.67,
                "Vss_L": 50.0,
                "dose_mg": 100.0,
                "route": "oral",
            }

            with patch.object(
                predictor, "_run_real_ode", return_value=mock_sim_result
            ), patch(
                "omega_pbpk.ml.models.foundation.end_to_end.smiles_to_graph",
                return_value=fake_graph,
            ):
                result = predictor.predict("CCO")

            assert "drug" in result
            assert "params" in result
            assert "pk_profile" in result
            assert "surrogate_curve" in result
            assert result["pk_profile"]["Cmax_mg_L"] == 1.5

    def test_predict_fallback_on_ode_failure(self, model: SMILESToPKModel, fake_graph):
        """Predictor falls back to surrogate when ODE fails."""
        from omega_pbpk.ml.models.foundation.inference import MLPKPredictor

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test.pt"
            model.save(model_path)

            predictor = MLPKPredictor(model_path=str(model_path), device="cpu")

            with patch.object(
                predictor, "_run_real_ode", side_effect=RuntimeError("ODE failed")
            ), patch(
                "omega_pbpk.ml.models.foundation.end_to_end.smiles_to_graph",
                return_value=fake_graph,
            ):
                result = predictor.predict("CCO")

            assert len(result["warnings"]) > 0
            assert "ODE" in result["warnings"][0]
            assert result["pk_profile"] is not None

    def test_no_model_raises(self):
        """Predictor raises RuntimeError when no model is loaded."""
        from omega_pbpk.ml.models.foundation.inference import MLPKPredictor

        predictor = MLPKPredictor(
            model_path="nonexistent_model.pt", device="cpu"
        )
        with pytest.raises(RuntimeError, match="No model loaded"):
            predictor.predict("CCO")

    def test_params_to_drug(self, model: SMILESToPKModel):
        """_params_to_drug creates a valid Drug object."""
        from omega_pbpk.ml.models.foundation.inference import MLPKPredictor

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test.pt"
            model.save(model_path)

            predictor = MLPKPredictor(model_path=str(model_path), device="cpu")

        params = {
            "mw": 300.0,
            "logP": 2.0,
            "fup": 0.5,
            "rbp": 1.0,
            "peff": 1.0,
            "clint_3a4": 5.0,
            "clint_2d6": 1.0,
        }
        drug = predictor._params_to_drug(params, "CCO", 100.0, "oral")

        assert drug.mw == 300.0
        assert drug.logP == 2.0
        assert drug.fup == 0.5
        assert drug.dose_mg == 100.0


# ---------------------------------------------------------------------------
# PK Metric Extraction
# ---------------------------------------------------------------------------


class TestPKMetricExtraction:
    """Tests for PK metric extraction from known curves."""

    def test_known_curve_cmax(self, model: SMILESToPKModel):
        """Cmax extraction from a known curve is correct."""
        # Create a simple peaked curve
        t = torch.linspace(0, 24, N_TIMEPOINTS)
        # C(t) = 10 * t * exp(-0.5 * t) -> Cmax ~= 7.36 at t = 2.0
        curve = 10.0 * t * torch.exp(-0.5 * t)
        curve = curve.unsqueeze(0)  # (1, 241)

        metrics = model._extract_pk_metrics(curve)

        assert abs(metrics["cmax"].item() - curve.max().item()) < 0.01

    def test_known_curve_auc(self):
        """AUC via trapezoidal rule matches expected value."""
        model = SMILESToPKModel()
        dt = model.dt_h

        # Constant curve C(t) = 5.0 for 24h -> AUC = 5 * 24 = 120
        curve = torch.full((1, N_TIMEPOINTS), 5.0)
        metrics = model._extract_pk_metrics(curve)

        expected_auc = 5.0 * (N_TIMEPOINTS - 1) * dt  # 5 * 240 * 0.1 = 120
        assert abs(metrics["auc"].item() - expected_auc) < 1.0

    def test_tmax_correct(self):
        """Tmax is at the correct time index."""
        model = SMILESToPKModel()

        # Peak at index 50 -> Tmax = 50 * 0.1 = 5.0 h
        curve = torch.zeros(1, N_TIMEPOINTS)
        curve[0, 50] = 10.0  # peak at index 50

        metrics = model._extract_pk_metrics(curve)
        assert abs(metrics["tmax"].item() - 5.0) < 0.01

    def test_thalf_reasonable(self):
        """Half-life estimate is in a reasonable range for exponential decay."""
        model = SMILESToPKModel()

        # Mono-exponential: C(t) = 10 * exp(-0.1 * t)
        # t_half = ln(2) / 0.1 = 6.93 h
        t = torch.linspace(0, 24, N_TIMEPOINTS)
        curve = (10.0 * torch.exp(-0.1 * t)).unsqueeze(0)

        metrics = model._extract_pk_metrics(curve)
        t_half = metrics["t_half"].item()

        # Should be approximately 6.93, allow generous tolerance
        assert 3.0 < t_half < 15.0, f"t_half = {t_half}, expected ~6.93"

    def test_metrics_differentiable(self):
        """PK metrics should be differentiable (support backward)."""
        model = SMILESToPKModel()
        t = torch.linspace(0, 24, N_TIMEPOINTS, requires_grad=False)
        curve = (10.0 * t * torch.exp(-0.5 * t)).unsqueeze(0)
        curve.requires_grad_(True)

        metrics = model._extract_pk_metrics(curve)

        # Sum all metrics and backward
        total = (
            metrics["cmax"]
            + metrics["auc"]
            + metrics["tmax"]
            + metrics["t_half"]
        )
        total.backward()

        assert curve.grad is not None
        assert not torch.isnan(curve.grad).any()


# ---------------------------------------------------------------------------
# Task L2.5: Benchmark
# ---------------------------------------------------------------------------


class TestLevel2Benchmark:
    """Tests for the benchmarking module."""

    def test_benchmark_with_mock_predictor(self):
        """Benchmark runs with a mocked predictor."""
        from omega_pbpk.ml.evaluation.level2_benchmark import (
            BenchmarkResult,
            run_level2_benchmark,
        )

        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = {
            "params": {"mw": 180.0, "logP": 1.2, "fup": 0.5},
            "pk_profile": {
                "Cmax_mg_L": 1.5,
                "AUC_mg_h_L": 10.0,
                "Tmax_h": 2.0,
                "half_life_h": 5.0,
            },
            "surrogate_curve": np.random.rand(N_TIMEPOINTS),
            "simulation_result": None,
            "warnings": [],
        }

        ref_drugs = [
            {
                "smiles": "CC(=O)Oc1ccccc1C(=O)O",
                "name": "Aspirin",
                "known_params": {"mw": 180.16, "logP": 1.2, "fup": 0.5},
                "known_pk": {"Cmax_mg_L": 1.5, "AUC_mg_h_L": 10.0},
            }
        ]

        result = run_level2_benchmark(mock_predictor, reference_drugs=ref_drugs)

        assert isinstance(result, BenchmarkResult)
        assert result.n_drugs == 1
        assert len(result.param_aafe) > 0

    def test_aafe_computation(self):
        """AAFE computation is correct for known values."""
        from omega_pbpk.ml.evaluation.level2_benchmark import _compute_aafe

        # Perfect prediction
        pred = np.array([1.0, 2.0, 3.0])
        obs = np.array([1.0, 2.0, 3.0])
        assert abs(_compute_aafe(pred, obs) - 1.0) < 0.01

        # 2-fold off
        pred = np.array([2.0, 4.0, 6.0])
        obs = np.array([1.0, 2.0, 3.0])
        assert abs(_compute_aafe(pred, obs) - 2.0) < 0.01

    def test_pct_within_nfold(self):
        """Percentage within n-fold is correct."""
        from omega_pbpk.ml.evaluation.level2_benchmark import _pct_within_nfold

        pred = np.array([1.0, 2.0, 4.0, 8.0])
        obs = np.array([1.0, 1.0, 1.0, 1.0])

        # Within 2-fold: 1.0/1.0=1, 2.0/1.0=2 -> 2 of 4 = 50%
        pct = _pct_within_nfold(pred, obs, 2.0)
        assert abs(pct - 50.0) < 0.01
