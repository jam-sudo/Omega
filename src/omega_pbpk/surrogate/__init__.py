"""Neural surrogate model for fast PBPK PK prediction.

Pure numpy/scipy MLP that approximates the 34-state ODE solver.
Maps drug compound parameters → PK summary (Cmax, AUC, Tmax, t½)
in ~0.1ms vs ~500ms for the mechanistic model.

Usage:
    from omega_pbpk.surrogate import PKSurrogate
    from omega_pbpk.surrogate.data_generator import generate_training_data

    data = generate_training_data(n_samples=500)
    model = PKSurrogate(n_input=data.n_params, n_output=data.n_outputs)
    model.train(data.X, data.y)
    pk_fast = model.predict(new_params)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def _relu(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.maximum(x, 0.0)


def _relu_deriv(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return (x > 0).astype(np.float64)


@dataclass
class PKSurrogate:
    """Multi-layer perceptron surrogate for PBPK PK prediction.

    Architecture: Input → [hidden_dim] × n_layers (ReLU) → Output
    Training: Mini-batch SGD with Adam optimizer.
    Normalization: Input/output are standardized internally.

    Attributes:
        n_input: Number of input features (drug parameters).
        n_output: Number of output targets (PK metrics).
        hidden_dim: Hidden layer width.
        n_layers: Number of hidden layers.
    """

    n_input: int = 6
    n_output: int = 4
    hidden_dim: int = 64
    n_layers: int = 3

    # Internal state (initialized by _init_weights or load)
    weights: list[NDArray[np.float64]] = field(default_factory=list)
    biases: list[NDArray[np.float64]] = field(default_factory=list)

    # Normalization statistics
    x_mean: NDArray[np.float64] = field(default_factory=lambda: np.zeros(1))
    x_std: NDArray[np.float64] = field(default_factory=lambda: np.ones(1))
    y_mean: NDArray[np.float64] = field(default_factory=lambda: np.zeros(1))
    y_std: NDArray[np.float64] = field(default_factory=lambda: np.ones(1))

    # Metadata
    is_trained: bool = False
    training_loss: float = float("inf")
    param_names: list[str] = field(default_factory=list)
    output_names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.weights:
            self._init_weights()

    def _init_weights(self) -> None:
        """He initialization for ReLU networks."""
        rng = np.random.default_rng(0)
        self.weights = []
        self.biases = []

        dims = [self.n_input] + [self.hidden_dim] * self.n_layers + [self.n_output]
        for i in range(len(dims) - 1):
            fan_in = dims[i]
            std = np.sqrt(2.0 / fan_in)
            w = rng.normal(0, std, (dims[i], dims[i + 1])).astype(np.float64)
            b = np.zeros(dims[i + 1], dtype=np.float64)
            self.weights.append(w)
            self.biases.append(b)

    def _forward(
        self, X: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], list[NDArray[np.float64]]]:
        """Forward pass returning output and all pre-activation layers."""
        activations = [X]
        h = X
        for i in range(len(self.weights) - 1):
            z = h @ self.weights[i] + self.biases[i]
            h = _relu(z)
            activations.append(z)
        # Output layer (no activation)
        out = h @ self.weights[-1] + self.biases[-1]
        activations.append(out)
        return out, activations

    def predict_raw(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict from normalized input, return normalized output."""
        out, _ = self._forward(X)
        return out

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict PK metrics from raw drug parameter values.

        Args:
            X: Array of shape (n_samples, n_input) or (n_input,).

        Returns:
            Array of shape (n_samples, n_output) with PK predictions.
        """
        single = X.ndim == 1
        if single:
            X = X.reshape(1, -1)

        # Normalize input
        X_norm = (X - self.x_mean) / np.maximum(self.x_std, 1e-8)

        # Forward pass
        y_norm = self.predict_raw(X_norm)

        # Denormalize output
        y = y_norm * self.y_std + self.y_mean

        # Clamp non-negative
        y = np.maximum(y, 0.0)

        return y[0] if single else y

    def predict_dict(self, params: dict[str, float]) -> dict[str, float]:
        """Predict PK from a parameter dict, return named dict."""
        x = np.array([params.get(name, 0.0) for name in self.param_names])
        y = self.predict(x)
        return {name: float(y[i]) for i, name in enumerate(self.output_names)}

    def train(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        epochs: int = 300,
        lr: float = 0.001,
        batch_size: int = 32,
        val_frac: float = 0.1,
        seed: int = 42,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Train the surrogate on (X, y) data using Adam optimizer.

        Args:
            X: Input features, shape (n_samples, n_input).
            y: Output targets, shape (n_samples, n_output).
            epochs: Training epochs.
            lr: Learning rate.
            batch_size: Mini-batch size.
            val_frac: Fraction held out for validation.
            seed: Random seed.
            verbose: Whether to log progress.

        Returns:
            Dict with training history (train_loss, val_loss per epoch).
        """
        rng = np.random.default_rng(seed)

        # Log-transform positive outputs for better scaling
        y_log = np.copy(y)
        y_log[:, :3] = np.log1p(np.maximum(y[:, :3], 0))  # Cmax, AUC, Tmax
        y_log[:, 3] = np.log1p(np.maximum(y[:, 3], 0))  # half-life

        # Compute normalization
        self.x_mean = X.mean(axis=0)
        self.x_std = X.std(axis=0)
        self.y_mean = y_log.mean(axis=0)
        self.y_std = y_log.std(axis=0)

        X_norm = (X - self.x_mean) / np.maximum(self.x_std, 1e-8)
        y_norm = (y_log - self.y_mean) / np.maximum(self.y_std, 1e-8)

        # Train/val split
        n = X_norm.shape[0]
        n_val = max(int(n * val_frac), 1)
        idx = rng.permutation(n)
        val_idx, train_idx = idx[:n_val], idx[n_val:]
        X_train, y_train = X_norm[train_idx], y_norm[train_idx]
        X_val, y_val = X_norm[val_idx], y_norm[val_idx]

        # Re-initialize weights
        self._init_weights()

        # Adam state
        m_w = [np.zeros_like(w) for w in self.weights]
        v_w = [np.zeros_like(w) for w in self.weights]
        m_b = [np.zeros_like(b) for b in self.biases]
        v_b = [np.zeros_like(b) for b in self.biases]
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        n_train = X_train.shape[0]
        t_step = 0

        for epoch in range(epochs):
            perm = rng.permutation(n_train)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_train, batch_size):
                batch_idx = perm[start : start + batch_size]
                xb = X_train[batch_idx]
                yb = y_train[batch_idx]
                bs = xb.shape[0]

                # Forward
                pred, acts = self._forward(xb)
                loss = float(np.mean((pred - yb) ** 2))
                epoch_loss += loss
                n_batches += 1

                # Backward (manual backprop)
                d_out = 2.0 * (pred - yb) / bs  # MSE gradient

                grads_w: list[NDArray[np.float64]] = [np.empty(0)] * len(self.weights)
                grads_b: list[NDArray[np.float64]] = [np.empty(0)] * len(self.biases)

                # Output layer
                L = len(self.weights) - 1
                h_prev = acts[L]
                h_prev_act = _relu(h_prev) if L > 0 else xb
                grads_w[L] = h_prev_act.T @ d_out
                grads_b[L] = d_out.sum(axis=0)
                d_h = d_out @ self.weights[L].T

                # Hidden layers
                for i in range(L - 1, -1, -1):
                    d_h = d_h * _relu_deriv(acts[i + 1])
                    h_in = _relu(acts[i]) if i > 0 else xb
                    grads_w[i] = h_in.T @ d_h
                    grads_b[i] = d_h.sum(axis=0)
                    if i > 0:
                        d_h = d_h @ self.weights[i].T

                # Adam update
                t_step += 1
                for i in range(len(self.weights)):
                    m_w[i] = beta1 * m_w[i] + (1 - beta1) * grads_w[i]
                    v_w[i] = beta2 * v_w[i] + (1 - beta2) * grads_w[i] ** 2
                    m_b[i] = beta1 * m_b[i] + (1 - beta1) * grads_b[i]
                    v_b[i] = beta2 * v_b[i] + (1 - beta2) * grads_b[i] ** 2

                    mw_hat = m_w[i] / (1 - beta1**t_step)
                    vw_hat = v_w[i] / (1 - beta2**t_step)
                    mb_hat = m_b[i] / (1 - beta1**t_step)
                    vb_hat = v_b[i] / (1 - beta2**t_step)

                    self.weights[i] -= lr * mw_hat / (np.sqrt(vw_hat) + eps)
                    self.biases[i] -= lr * mb_hat / (np.sqrt(vb_hat) + eps)

            train_loss = epoch_loss / max(n_batches, 1)
            val_pred = self.predict_raw(X_val)
            val_loss = float(np.mean((val_pred - y_val) ** 2))

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if verbose and (epoch + 1) % 50 == 0:
                logger.info(
                    "Epoch %d/%d: train_loss=%.6f, val_loss=%.6f",
                    epoch + 1,
                    epochs,
                    train_loss,
                    val_loss,
                )

        self.is_trained = True
        self.training_loss = history["val_loss"][-1] if history["val_loss"] else float("inf")

        # Store log-transform info
        self._log_transform = True  # flag for inverse transform

        return history

    def save(self, path: str | Path) -> None:
        """Save model weights and metadata to directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save weights
        for i, (w, b) in enumerate(zip(self.weights, self.biases, strict=True)):
            np.save(path / f"w{i}.npy", w)
            np.save(path / f"b{i}.npy", b)

        # Save normalization
        np.save(path / "x_mean.npy", self.x_mean)
        np.save(path / "x_std.npy", self.x_std)
        np.save(path / "y_mean.npy", self.y_mean)
        np.save(path / "y_std.npy", self.y_std)

        # Save metadata
        meta = {
            "n_input": self.n_input,
            "n_output": self.n_output,
            "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers,
            "is_trained": self.is_trained,
            "training_loss": self.training_loss,
            "param_names": self.param_names,
            "output_names": self.output_names,
            "n_weight_arrays": len(self.weights),
        }
        (path / "meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> PKSurrogate:
        """Load a saved surrogate model."""
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())

        model = cls(
            n_input=meta["n_input"],
            n_output=meta["n_output"],
            hidden_dim=meta["hidden_dim"],
            n_layers=meta["n_layers"],
        )

        model.weights = [np.load(path / f"w{i}.npy") for i in range(meta["n_weight_arrays"])]
        model.biases = [np.load(path / f"b{i}.npy") for i in range(meta["n_weight_arrays"])]
        model.x_mean = np.load(path / "x_mean.npy")
        model.x_std = np.load(path / "x_std.npy")
        model.y_mean = np.load(path / "y_mean.npy")
        model.y_std = np.load(path / "y_std.npy")
        model.is_trained = meta["is_trained"]
        model.training_loss = meta["training_loss"]
        model.param_names = meta["param_names"]
        model.output_names = meta["output_names"]

        return model


__all__ = ["PKSurrogate"]
