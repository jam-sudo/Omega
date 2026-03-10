"""Multi-fidelity curriculum training for SMILESToPKModel.

Implements a 3-stage training curriculum:
  Stage 1: Pre-train on 1-compartment analytical data (100K samples, fast)
  Stage 2: Fine-tune on 35-state ODE data (50K samples, high fidelity)
  Stage 3: Fine-tune on clinical data (if available, real PK data)

Each stage reduces the learning rate by 10x and uses stage-specific
hyperparameters for optimal convergence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import TensorDataset

from omega_pbpk.ml.training.trainer import PKTrainer, TrainingConfig, TrainingHistory

logger = logging.getLogger(__name__)


@dataclass
class StageConfig:
    """Configuration for a single curriculum stage.

    Attributes
    ----------
    name : str
        Human-readable stage name.
    lr : float
        Learning rate for this stage.
    n_epochs : int
        Maximum epochs for this stage.
    batch_size : int
        Batch size.
    patience : int
        Early stopping patience.
    weight_decay : float
        L2 regularization weight.
    loss_weights : dict
        PKLoss component weights (can differ per stage).
    """

    name: str = "stage"
    lr: float = 1e-3
    n_epochs: int = 50
    batch_size: int = 64
    patience: int = 10
    weight_decay: float = 1e-5
    loss_weights: dict[str, float] = field(
        default_factory=lambda: {
            "w_pk_mse": 1.0,
            "w_mass": 0.0,  # disabled: tissue_amounts not in training data
            "w_nonneg": 1.0,
            "w_monotone": 0.05,
            "w_plausible": 0.1,
        }
    )


@dataclass
class CurriculumResult:
    """Result from running the full multi-fidelity curriculum.

    Attributes
    ----------
    stage_histories : dict mapping stage name to TrainingHistory
    final_metrics : dict with evaluation metrics after all stages
    stages_completed : list of stage names that were executed
    """

    stage_histories: dict[str, TrainingHistory] = field(default_factory=dict)
    final_metrics: dict[str, float] = field(default_factory=dict)
    stages_completed: list[str] = field(default_factory=list)


class MultiFidelityCurriculum:
    """Multi-fidelity curriculum trainer for progressive PK model training.

    Orchestrates 3 training stages with decreasing learning rates and
    increasing data fidelity:

    1. Pre-train on 1-compartment analytical data (low fidelity, high volume)
    2. Fine-tune on 35-state ODE data (high fidelity, medium volume)
    3. Fine-tune on clinical data (highest fidelity, low volume)

    Parameters
    ----------
    model : nn.Module
        The SMILESToPKModel to train.
    checkpoint_dir : str
        Base directory for checkpoints.
    device : str
        Device string ('auto', 'cuda', 'cpu').
    base_lr : float
        Base learning rate for stage 1. Stage 2 uses base_lr/10,
        stage 3 uses base_lr/100.
    """

    DEFAULT_STAGES = {
        "1cpt_pretrain": StageConfig(
            name="1cpt_pretrain",
            lr=1e-3,
            n_epochs=50,
            batch_size=128,
            patience=10,
            weight_decay=1e-5,
            loss_weights={
                "w_pk_mse": 1.0,
                "w_mass": 0.0,  # disabled: tissue_amounts not in training data
                "w_nonneg": 1.0,
                "w_monotone": 0.02,
                "w_plausible": 0.05,
            },
        ),
        "pbpk_finetune": StageConfig(
            name="pbpk_finetune",
            lr=1e-4,
            n_epochs=100,
            batch_size=64,
            patience=15,
            weight_decay=1e-4,
            loss_weights={
                "w_pk_mse": 1.0,
                "w_mass": 0.0,  # disabled: tissue_amounts not in training data
                "w_nonneg": 1.0,
                "w_monotone": 0.05,
                "w_plausible": 0.1,
            },
        ),
        "clinical_finetune": StageConfig(
            name="clinical_finetune",
            lr=1e-5,
            n_epochs=200,
            batch_size=32,
            patience=20,
            weight_decay=1e-3,
            loss_weights={
                "w_pk_mse": 2.0,
                "w_mass": 0.0,  # disabled: tissue_amounts not in training data
                "w_nonneg": 1.0,
                "w_monotone": 0.1,
                "w_plausible": 0.2,
            },
        ),
    }

    def __init__(
        self,
        model: torch.nn.Module,
        checkpoint_dir: str = "models/level2",
        device: str = "auto",
        base_lr: float = 1e-3,
        stages: dict[str, StageConfig] | None = None,
    ) -> None:
        self.model = model
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device
        self.base_lr = base_lr

        if stages is not None:
            self.stages = stages
        else:
            # Scale LRs relative to base_lr
            self.stages = {}
            for key, default in self.DEFAULT_STAGES.items():
                stage = StageConfig(
                    name=default.name,
                    lr=default.lr * (base_lr / 1e-3),
                    n_epochs=default.n_epochs,
                    batch_size=default.batch_size,
                    patience=default.patience,
                    weight_decay=default.weight_decay,
                    loss_weights=dict(default.loss_weights),
                )
                self.stages[key] = stage

    def run_stage(
        self,
        stage_name: str,
        train_data: TensorDataset,
        val_data: TensorDataset,
        stage_config: StageConfig | None = None,
    ) -> TrainingHistory:
        """Run a single curriculum stage.

        Parameters
        ----------
        stage_name : str
            Stage identifier.
        train_data : TensorDataset
            Training data (params, curves, metrics).
        val_data : TensorDataset
            Validation data.
        stage_config : StageConfig or None
            Override stage config.

        Returns
        -------
        TrainingHistory for this stage.
        """
        config = stage_config or self.stages.get(stage_name)
        if config is None:
            raise ValueError(f"Unknown stage '{stage_name}'. Available: {list(self.stages.keys())}")

        logger.info(
            "=== Starting stage '%s': lr=%.2e, epochs=%d, batch=%d ===",
            stage_name,
            config.lr,
            config.n_epochs,
            config.batch_size,
        )

        stage_checkpoint_dir = self.checkpoint_dir / stage_name
        training_config = TrainingConfig(
            lr=config.lr,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            weight_decay=config.weight_decay,
            grad_clip=1.0,
            patience=config.patience,
            checkpoint_dir=str(stage_checkpoint_dir),
            loss_weights=config.loss_weights,
            device=self.device,
            lr_scheduler="cosine",
        )

        trainer = PKTrainer(self.model, training_config)
        history = trainer.train(train_data, val_data)

        logger.info(
            "Stage '%s' complete: best_val=%.6f at epoch %d (%.1fs)",
            stage_name,
            history.best_val_loss,
            history.best_epoch + 1,
            history.elapsed_seconds,
        )

        return history

    def run_curriculum(
        self,
        stage1_train: TensorDataset | None = None,
        stage1_val: TensorDataset | None = None,
        stage2_train: TensorDataset | None = None,
        stage2_val: TensorDataset | None = None,
        stage3_train: TensorDataset | None = None,
        stage3_val: TensorDataset | None = None,
    ) -> CurriculumResult:
        """Run the full multi-fidelity curriculum.

        Stages are skipped if their data is not provided.

        Parameters
        ----------
        stage1_train, stage1_val : TensorDataset or None
            1-compartment analytical data for pre-training.
        stage2_train, stage2_val : TensorDataset or None
            35-state ODE data for fine-tuning.
        stage3_train, stage3_val : TensorDataset or None
            Clinical data for final fine-tuning.

        Returns
        -------
        CurriculumResult with histories and final metrics.
        """
        result = CurriculumResult()

        # Stage 1: 1-compartment pre-training
        if stage1_train is not None and stage1_val is not None:
            history = self.run_stage("1cpt_pretrain", stage1_train, stage1_val)
            result.stage_histories["1cpt_pretrain"] = history
            result.stages_completed.append("1cpt_pretrain")
        else:
            logger.info("Skipping stage 1 (1-cpt pre-training): no data provided")

        # Stage 2: PBPK fine-tuning
        if stage2_train is not None and stage2_val is not None:
            history = self.run_stage("pbpk_finetune", stage2_train, stage2_val)
            result.stage_histories["pbpk_finetune"] = history
            result.stages_completed.append("pbpk_finetune")
        else:
            logger.info("Skipping stage 2 (PBPK fine-tuning): no data provided")

        # Stage 3: Clinical fine-tuning
        if stage3_train is not None and stage3_val is not None:
            history = self.run_stage("clinical_finetune", stage3_train, stage3_val)
            result.stage_histories["clinical_finetune"] = history
            result.stages_completed.append("clinical_finetune")
        else:
            logger.info("Skipping stage 3 (clinical fine-tuning): no data provided")

        # Save final model
        final_path = self.checkpoint_dir / "final.pt"
        if hasattr(self.model, "save"):
            self.model.save(final_path)
        else:
            torch.save(self.model.state_dict(), str(final_path))

        logger.info(
            "Curriculum complete: %d stages run. Final model saved to %s",
            len(result.stages_completed),
            final_path,
        )

        return result


def _make_tensor_dataset(
    params: torch.Tensor,
    curves: torch.Tensor,
    metrics: torch.Tensor,
) -> TensorDataset:
    """Create a TensorDataset from numpy-like arrays.

    Parameters
    ----------
    params : array-like (N, n_params)
    curves : array-like (N, n_timepoints)
    metrics : array-like (N, n_metrics)

    Returns
    -------
    TensorDataset
    """
    if not isinstance(params, torch.Tensor):
        params = torch.tensor(params, dtype=torch.float32)
    if not isinstance(curves, torch.Tensor):
        curves = torch.tensor(curves, dtype=torch.float32)
    if not isinstance(metrics, torch.Tensor):
        metrics = torch.tensor(metrics, dtype=torch.float32)
    return TensorDataset(params, curves, metrics)


def run_curriculum(
    model: torch.nn.Module,
    data_dir: str = "data/ml/",
    checkpoint_dir: str = "models/level2",
    device: str = "auto",
    base_lr: float = 1e-3,
    val_fraction: float = 0.1,
) -> CurriculumResult:
    """Convenience function to run the full curriculum from HDF5 data files.

    Loads data from:
    - data_dir/1cpt_curves.h5 (stage 1)
    - data_dir/pbpk_curves.h5 (stage 2)
    - data_dir/clinical_curves.h5 (stage 3, if exists)

    Parameters
    ----------
    model : nn.Module
        The SMILESToPKModel to train.
    data_dir : str
        Directory containing HDF5 data files.
    checkpoint_dir : str
        Directory for checkpoints.
    device : str
        Device string.
    base_lr : float
        Base learning rate.
    val_fraction : float
        Fraction of data for validation.

    Returns
    -------
    CurriculumResult
    """
    import numpy as np

    from omega_pbpk.ml.data.synthetic import CurveDataset

    data_path = Path(data_dir)
    curriculum = MultiFidelityCurriculum(
        model, checkpoint_dir=checkpoint_dir, device=device, base_lr=base_lr
    )

    def _load_and_split(filename: str) -> tuple[TensorDataset, TensorDataset] | None:
        filepath = data_path / filename
        if not filepath.exists():
            logger.info("Data file not found: %s", filepath)
            return None

        ds = CurveDataset.load_hdf5(filepath)
        n = ds.n_samples
        n_val = max(1, int(n * val_fraction))
        n_train = n - n_val

        rng = np.random.default_rng(42)
        idx = rng.permutation(n)
        train_idx, val_idx = idx[:n_train], idx[n_train:]

        train_ds = _make_tensor_dataset(
            ds.params[train_idx], ds.curves[train_idx], ds.metrics[train_idx]
        )
        val_ds = _make_tensor_dataset(ds.params[val_idx], ds.curves[val_idx], ds.metrics[val_idx])
        logger.info("Loaded %s: %d train, %d val samples", filename, n_train, n_val)
        return train_ds, val_ds

    # Load datasets
    s1 = _load_and_split("1cpt_curves.h5")
    s2 = _load_and_split("pbpk_curves.h5")
    s3 = _load_and_split("clinical_curves.h5")

    return curriculum.run_curriculum(
        stage1_train=s1[0] if s1 else None,
        stage1_val=s1[1] if s1 else None,
        stage2_train=s2[0] if s2 else None,
        stage2_val=s2[1] if s2 else None,
        stage3_train=s3[0] if s3 else None,
        stage3_val=s3[1] if s3 else None,
    )


__all__ = [
    "StageConfig",
    "CurriculumResult",
    "MultiFidelityCurriculum",
    "run_curriculum",
]
