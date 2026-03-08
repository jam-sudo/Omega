"""Training loop orchestrator for ML models with logging and checkpointing.

Provides PKTrainer for training SMILESToPKModel using physics-informed
PKLoss with Adam optimizer, LR scheduling, gradient clipping, early
stopping, and model checkpointing.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for PKTrainer.

    Attributes
    ----------
    lr : float
        Initial learning rate.
    batch_size : int
        Mini-batch size.
    n_epochs : int
        Maximum number of training epochs.
    weight_decay : float
        L2 regularization weight.
    grad_clip : float
        Maximum gradient norm for clipping.
    patience : int
        Early stopping patience (epochs without improvement).
    checkpoint_dir : str
        Directory for saving model checkpoints.
    loss_weights : dict
        Weights for PKLoss components.
    device : str
        Device ('cuda', 'cpu', or 'auto').
    lr_scheduler : str
        LR scheduler type: 'cosine' or 'plateau'.
    log_interval : int
        Log every N epochs.
    """

    lr: float = 1e-3
    batch_size: int = 64
    n_epochs: int = 100
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    patience: int = 15
    checkpoint_dir: str = "models/level2"
    loss_weights: dict[str, float] = field(
        default_factory=lambda: {
            "w_pk_mse": 1.0,
            "w_mass": 0.1,
            "w_nonneg": 1.0,
            "w_monotone": 0.05,
            "w_plausible": 0.1,
        }
    )
    device: str = "auto"
    lr_scheduler: str = "cosine"
    log_interval: int = 5

    def get_device(self) -> torch.device:
        """Resolve device string to torch.device."""
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


@dataclass
class TrainingHistory:
    """Training run history.

    Attributes
    ----------
    train_losses : list of total loss per epoch
    val_losses : list of validation loss per epoch
    component_losses : list of dicts with per-component losses
    learning_rates : list of LR per epoch
    best_epoch : epoch with best validation loss
    best_val_loss : best validation loss value
    elapsed_seconds : total training time
    """

    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    component_losses: list[dict[str, float]] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    elapsed_seconds: float = 0.0


class PKTrainer:
    """Training loop for SMILESToPKModel with physics-informed loss.

    Parameters
    ----------
    model : nn.Module
        The SMILESToPKModel to train.
    config : TrainingConfig
        Training configuration.
    loss_fn : nn.Module or None
        Loss function (default: PKLoss with config weights).
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        loss_fn: Optional[nn.Module] = None,
    ) -> None:
        self.config = config
        self.device = config.get_device()
        self.model = model.to(self.device)

        if loss_fn is None:
            from omega_pbpk.ml.training.losses import PKLoss

            self.loss_fn = PKLoss(**config.loss_weights).to(self.device)
        else:
            self.loss_fn = loss_fn.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        if config.lr_scheduler == "cosine":
            self.scheduler: Any = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.n_epochs, eta_min=config.lr * 0.01
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", patience=config.patience // 2, factor=0.5
            )

        self._checkpoint_dir = Path(config.checkpoint_dir)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        train_data: TensorDataset | DataLoader,
        val_data: TensorDataset | DataLoader,
        n_epochs: Optional[int] = None,
    ) -> TrainingHistory:
        """Run training loop.

        Parameters
        ----------
        train_data : TensorDataset or DataLoader
            Training data. If TensorDataset, expects tensors:
            (params [N, 6], curves [N, 241], metrics [N, 4]).
        val_data : TensorDataset or DataLoader
            Validation data (same format).
        n_epochs : int or None
            Override config n_epochs.

        Returns
        -------
        TrainingHistory with per-epoch losses and metadata.
        """
        n_epochs = n_epochs or self.config.n_epochs
        history = TrainingHistory()

        # Wrap datasets in DataLoaders if needed
        train_loader = self._ensure_dataloader(train_data, shuffle=True)
        val_loader = self._ensure_dataloader(val_data, shuffle=False)

        best_val_loss = float("inf")
        epochs_without_improvement = 0
        start_time = time.monotonic()

        for epoch in range(n_epochs):
            # Training step
            train_loss, comp_losses = self._train_epoch(train_loader)
            history.train_losses.append(train_loss)
            history.component_losses.append(comp_losses)

            # Validation step
            val_loss = self._validate(val_loader)
            history.val_losses.append(val_loss)

            # Learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            history.learning_rates.append(current_lr)

            # LR scheduling
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            # Checkpoint best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                history.best_epoch = epoch
                history.best_val_loss = val_loss
                epochs_without_improvement = 0
                self._save_checkpoint("best.pt")
            else:
                epochs_without_improvement += 1

            # Logging
            if (epoch + 1) % self.config.log_interval == 0 or epoch == 0:
                logger.info(
                    "Epoch %d/%d: train=%.6f val=%.6f lr=%.2e best=%.6f@%d",
                    epoch + 1,
                    n_epochs,
                    train_loss,
                    val_loss,
                    current_lr,
                    best_val_loss,
                    history.best_epoch + 1,
                )

            # Early stopping
            if epochs_without_improvement >= self.config.patience:
                logger.info(
                    "Early stopping at epoch %d (patience=%d)",
                    epoch + 1,
                    self.config.patience,
                )
                break

        history.elapsed_seconds = time.monotonic() - start_time

        # Load best model
        best_path = self._checkpoint_dir / "best.pt"
        if best_path.exists():
            self._load_checkpoint("best.pt")
            logger.info(
                "Loaded best model from epoch %d (val_loss=%.6f)",
                history.best_epoch + 1,
                history.best_val_loss,
            )

        return history

    def _train_epoch(
        self, loader: DataLoader
    ) -> tuple[float, dict[str, float]]:
        """Run one training epoch.

        Returns
        -------
        (average_total_loss, dict_of_component_losses)
        """
        self.model.train()
        total_loss_sum = 0.0
        comp_sums: dict[str, float] = {}
        n_batches = 0

        for batch in loader:
            params_batch, curves_batch, metrics_batch = self._unpack_batch(batch)

            self.optimizer.zero_grad()

            # Forward: params -> surrogate -> curve
            # The training data provides ground truth params and curves
            # We feed params through the surrogate and compare
            loss_dict = self._compute_loss(
                params_batch, curves_batch, metrics_batch
            )

            loss = loss_dict["total_loss"]
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip
            )

            self.optimizer.step()

            total_loss_sum += loss.item()
            for key, val in loss_dict.items():
                comp_sums[key] = comp_sums.get(key, 0.0) + val.item()
            n_batches += 1

        n = max(n_batches, 1)
        avg_comps = {k: v / n for k, v in comp_sums.items()}
        return total_loss_sum / n, avg_comps

    def _validate(self, loader: DataLoader) -> float:
        """Run validation and return average loss."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in loader:
                params_batch, curves_batch, metrics_batch = self._unpack_batch(batch)
                loss_dict = self._compute_loss(
                    params_batch, curves_batch, metrics_batch
                )
                total_loss += loss_dict["total_loss"].item()
                n_batches += 1

        return total_loss / max(n_batches, 1)

    def _compute_loss(
        self,
        params_batch: torch.Tensor,
        curves_batch: torch.Tensor,
        metrics_batch: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute physics-informed loss.

        Feeds ground-truth params through the surrogate to get predicted
        curves, then compares with ground-truth curves and metrics.

        Parameters
        ----------
        params_batch : (batch, 6) ground truth ODE parameters
        curves_batch : (batch, 241) ground truth C(t) curves
        metrics_batch : (batch, 4) ground truth [Cmax, AUC, Tmax, t_half]
        """
        # Run params through surrogate
        curve_log_pred = self.model.surrogate(params_batch)
        curve_pred = torch.expm1(torch.clamp(curve_log_pred, max=20.0))
        curve_pred = torch.clamp(curve_pred, min=0.0)

        # Extract PK metrics from predicted curve
        pred_metrics = self.model._extract_pk_metrics(curve_pred)

        # Ground truth metrics dict
        true_metrics = {
            "cmax": metrics_batch[:, 0],
            "auc": metrics_batch[:, 1],
            "tmax": metrics_batch[:, 2],
            "t_half": metrics_batch[:, 3],
        }

        # Construct a minimal predicted_params dict for plausibility loss
        # Map the 6D param vector back to named params
        predicted_params = {
            "logP": params_batch[:, 0],
            "fup": params_batch[:, 1],
            "clint_3a4": params_batch[:, 2],  # approximate; using clint_L_h
            "mw": params_batch[:, 3],
            "rbp": params_batch[:, 4],
            "peff": params_batch[:, 5],
        }

        # Compute loss
        return self.loss_fn(
            predicted_pk=pred_metrics,
            true_pk=true_metrics,
            predicted_params=predicted_params,
            predicted_curve=curve_pred,
        )

    def _unpack_batch(
        self, batch: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Unpack batch and move to device."""
        params = batch[0].to(self.device)
        curves = batch[1].to(self.device)
        metrics = batch[2].to(self.device)
        return params, curves, metrics

    def _ensure_dataloader(
        self, data: TensorDataset | DataLoader, shuffle: bool
    ) -> DataLoader:
        """Wrap TensorDataset in DataLoader if needed."""
        if isinstance(data, DataLoader):
            return data
        return DataLoader(
            data,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            drop_last=False,
        )

    def _save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        path = self._checkpoint_dir / filename
        if hasattr(self.model, "save"):
            self.model.save(path)
        else:
            torch.save(self.model.state_dict(), str(path))

    def _load_checkpoint(self, filename: str) -> None:
        """Load model checkpoint."""
        path = self._checkpoint_dir / filename
        if hasattr(self.model, "load"):
            loaded = type(self.model).load(path, device=str(self.device))
            self.model.load_state_dict(loaded.state_dict())
        else:
            state = torch.load(str(path), map_location=self.device, weights_only=False)
            if isinstance(state, dict) and "model_state_dict" in state:
                self.model.load_state_dict(state["model_state_dict"])
            else:
                self.model.load_state_dict(state)

    def evaluate(self, test_data: TensorDataset | DataLoader) -> dict[str, float]:
        """Evaluate model on test data.

        Computes AAFE, fold errors, and related metrics.

        Parameters
        ----------
        test_data : TensorDataset or DataLoader
            Test data (same format as train/val).

        Returns
        -------
        dict with evaluation metrics: aafe_cmax, aafe_auc, pct_within_2fold, etc.
        """
        self.model.eval()
        loader = self._ensure_dataloader(test_data, shuffle=False)

        all_pred_cmax = []
        all_true_cmax = []
        all_pred_auc = []
        all_true_auc = []

        with torch.no_grad():
            for batch in loader:
                params_batch, curves_batch, metrics_batch = self._unpack_batch(batch)

                # Surrogate prediction
                curve_log_pred = self.model.surrogate(params_batch)
                curve_pred = torch.expm1(torch.clamp(curve_log_pred, max=20.0))
                curve_pred = torch.clamp(curve_pred, min=0.0)

                pred_cmax = curve_pred.max(dim=-1).values
                true_cmax = metrics_batch[:, 0]
                true_auc = metrics_batch[:, 1]

                # Trapezoidal AUC for predicted
                pred_auc = torch.trapz(curve_pred, dx=self.model.dt_h)

                all_pred_cmax.append(pred_cmax.cpu())
                all_true_cmax.append(true_cmax.cpu())
                all_pred_auc.append(pred_auc.cpu())
                all_true_auc.append(true_auc.cpu())

        pred_cmax = torch.cat(all_pred_cmax)
        true_cmax = torch.cat(all_true_cmax)
        pred_auc = torch.cat(all_pred_auc)
        true_auc = torch.cat(all_true_auc)

        eps = 1e-12

        # AAFE: geometric mean of max(pred/true, true/pred)
        def _aafe(pred: torch.Tensor, true: torch.Tensor) -> float:
            pred_safe = pred.clamp(min=eps)
            true_safe = true.clamp(min=eps)
            fold = torch.max(pred_safe / true_safe, true_safe / pred_safe)
            return float(torch.exp(torch.log(fold).mean()).item())

        aafe_cmax = _aafe(pred_cmax, true_cmax)
        aafe_auc = _aafe(pred_auc, true_auc)

        # % within 2-fold
        def _pct_within_nfold(pred: torch.Tensor, true: torch.Tensor, n: float) -> float:
            pred_safe = pred.clamp(min=eps)
            true_safe = true.clamp(min=eps)
            fold = torch.max(pred_safe / true_safe, true_safe / pred_safe)
            return float((fold <= n).float().mean().item()) * 100.0

        pct_2fold_cmax = _pct_within_nfold(pred_cmax, true_cmax, 2.0)
        pct_2fold_auc = _pct_within_nfold(pred_auc, true_auc, 2.0)

        return {
            "aafe_cmax": round(aafe_cmax, 4),
            "aafe_auc": round(aafe_auc, 4),
            "pct_within_2fold_cmax": round(pct_2fold_cmax, 2),
            "pct_within_2fold_auc": round(pct_2fold_auc, 2),
            "n_samples": len(pred_cmax),
        }


__all__ = ["TrainingConfig", "TrainingHistory", "PKTrainer"]
