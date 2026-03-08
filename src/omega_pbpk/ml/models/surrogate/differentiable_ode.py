"""Differentiable surrogate of the PBPK ODE for backpropagation during training.

Approximates the 35-state ODE engine with a neural network:
  params (6D) -> Linear layers -> C(t) curve (241 timepoints)

This enables gradient-based optimization of parameters that feed into the ODE,
without needing to differentiate through the stiff ODE solver itself.

Training uses MSE loss in both linear and log space for balanced accuracy
across the full concentration range.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Try to import PyTorch; fall back gracefully
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning(
        "PyTorch not available. DifferentiableODESurrogate requires PyTorch. "
        "Install with: pip install torch"
    )


def _check_torch() -> None:
    """Raise ImportError if PyTorch is not available."""
    if not HAS_TORCH:
        msg = "PyTorch is required for DifferentiableODESurrogate. Install with: pip install torch"
        raise ImportError(msg)


class DifferentiableODESurrogate(nn.Module if HAS_TORCH else object):
    """Neural network surrogate for the 35-state PBPK ODE.

    Architecture:
        params (n_input) -> Linear(256) -> ReLU -> Linear(256) -> ReLU
        -> Linear(256) -> ReLU -> Linear(n_output)

    The network predicts log-transformed concentrations to handle the
    wide dynamic range of PK profiles. Output is exponentiated at inference.

    Args:
        n_input: Number of input parameters (default: 6).
        n_output: Number of output timepoints (default: 241).
        hidden_dim: Hidden layer dimension (default: 256).
    """

    def __init__(
        self,
        n_input: int = 6,
        n_output: int = 241,
        hidden_dim: int = 256,
    ) -> None:
        _check_torch()
        super().__init__()

        self.n_input = n_input
        self.n_output = n_output
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(n_input, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_output),
        )

        # Normalization statistics (set during training)
        self.register_buffer("param_mean", torch.zeros(n_input, dtype=torch.float32))
        self.register_buffer("param_std", torch.ones(n_input, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: normalized params -> predicted C(t) curve.

        Args:
            x: Input parameters, shape (batch, n_input).

        Returns:
            Predicted concentration curves, shape (batch, n_output).
            Values are in log1p space during training; use predict() for
            actual concentrations.
        """
        x_norm = (x - self.param_mean) / (self.param_std + 1e-8)
        return self.net(x_norm)

    def predict(self, params: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict C(t) curves from numpy parameter arrays.

        Args:
            params: Parameters, shape (n_samples, n_input) or (n_input,).

        Returns:
            Predicted concentration curves, shape (n_samples, n_output).
        """
        _check_torch()
        was_1d = params.ndim == 1
        if was_1d:
            params = params.reshape(1, -1)

        self.eval()
        with torch.no_grad():
            x = torch.tensor(params, dtype=torch.float32)
            y_log = self.forward(x)
            # Convert from log1p space back to concentration
            y = torch.expm1(torch.clamp(y_log, max=20.0))
            y = torch.clamp(y, min=0.0)

        result = y.numpy()
        if was_1d:
            result = result.squeeze(0)
        return result


def train_surrogate(
    params: NDArray[np.float64],
    curves: NDArray[np.float64],
    n_epochs: int = 200,
    batch_size: int = 256,
    lr: float = 1e-3,
    val_fraction: float = 0.1,
    log_space_weight: float = 0.5,
    save_dir: str | Path | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train the differentiable ODE surrogate on generated data.

    Uses a combined loss: linear MSE + log-space MSE (weighted) to ensure
    accuracy across the full concentration range.

    Args:
        params: Input parameters, shape (n_samples, n_params).
        curves: Target C(t) curves, shape (n_samples, n_timepoints).
        n_epochs: Training epochs.
        batch_size: Mini-batch size.
        lr: Learning rate.
        val_fraction: Fraction of data for validation.
        log_space_weight: Weight for log-space MSE loss (0-1).
        save_dir: Directory to save trained model. None = don't save.
        verbose: Print training progress.

    Returns:
        Dictionary with:
            - model: Trained DifferentiableODESurrogate
            - train_losses: List of training losses per epoch
            - val_losses: List of validation losses per epoch
            - aafe: Average absolute fold error on validation set
    """
    _check_torch()

    n_samples = params.shape[0]
    n_input = params.shape[1]
    n_output = curves.shape[1]

    # Train/val split
    n_val = max(1, int(n_samples * val_fraction))
    n_train = n_samples - n_val

    indices = np.random.default_rng(42).permutation(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    X_train = torch.tensor(params[train_idx], dtype=torch.float32)
    X_val = torch.tensor(params[val_idx], dtype=torch.float32)

    # Transform curves to log1p space for training
    y_train_raw = torch.tensor(curves[train_idx], dtype=torch.float32)
    y_val_raw = torch.tensor(curves[val_idx], dtype=torch.float32)
    y_train = torch.log1p(torch.clamp(y_train_raw, min=0.0))
    y_val = torch.log1p(torch.clamp(y_val_raw, min=0.0))

    # Create model
    model = DifferentiableODESurrogate(n_input=n_input, n_output=n_output)

    # Compute and set normalization stats from training data
    model.param_mean = X_train.mean(dim=0)
    model.param_std = X_train.std(dim=0)

    # DataLoader
    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=10, factor=0.5
    )
    mse_fn = nn.MSELoss()

    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(x_batch)

            # Combined loss: MSE in log1p space + optional linear space term
            loss_log = mse_fn(y_pred, y_batch)

            if log_space_weight < 1.0:
                # Also compute loss in linear space
                y_pred_lin = torch.expm1(torch.clamp(y_pred, max=20.0))
                y_true_lin = torch.expm1(y_batch)
                loss_lin = mse_fn(y_pred_lin, y_true_lin)
                loss = log_space_weight * loss_log + (1.0 - log_space_weight) * loss_lin
            else:
                loss = loss_log

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val)
            val_loss = mse_fn(y_val_pred, y_val).item()
            val_losses.append(val_loss)

        scheduler.step(val_loss)

        if verbose and (epoch + 1) % 20 == 0:
            logger.info(
                "Epoch %d/%d: train_loss=%.6f, val_loss=%.6f",
                epoch + 1,
                n_epochs,
                avg_train_loss,
                val_loss,
            )

    # Compute AAFE on validation set
    model.eval()
    with torch.no_grad():
        y_val_pred_log = model(X_val)
        y_val_pred_lin = torch.expm1(torch.clamp(y_val_pred_log, max=20.0))
        y_val_pred_lin = torch.clamp(y_val_pred_lin, min=0.0)

    pred_np = y_val_pred_lin.numpy()
    true_np = curves[val_idx]

    # AAFE: geometric mean of max(pred/true, true/pred) for Cmax
    eps = 1e-12
    pred_cmax = np.maximum(pred_np.max(axis=1), eps)
    true_cmax = np.maximum(true_np.max(axis=1), eps)
    fold_errors = np.maximum(pred_cmax / true_cmax, true_cmax / pred_cmax)
    aafe = float(np.exp(np.mean(np.log(fold_errors))))

    if verbose:
        logger.info("Final validation AAFE (Cmax): %.3f", aafe)

    # Save model
    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        model_file = save_path / "surrogate_model.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "n_input": n_input,
                "n_output": n_output,
                "hidden_dim": model.hidden_dim,
                "param_mean": model.param_mean,
                "param_std": model.param_std,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "aafe": aafe,
            },
            str(model_file),
        )
        logger.info("Saved surrogate model to %s", model_file)

    return {
        "model": model,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "aafe": aafe,
    }


def load_surrogate(
    path: str | Path,
) -> DifferentiableODESurrogate:
    """Load a trained surrogate model from disk.

    Args:
        path: Path to saved model checkpoint (.pt file).

    Returns:
        Loaded DifferentiableODESurrogate in eval mode.
    """
    _check_torch()

    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    model = DifferentiableODESurrogate(
        n_input=checkpoint["n_input"],
        n_output=checkpoint["n_output"],
        hidden_dim=checkpoint["hidden_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


__all__ = [
    "DifferentiableODESurrogate",
    "train_surrogate",
    "load_surrogate",
    "HAS_TORCH",
]
