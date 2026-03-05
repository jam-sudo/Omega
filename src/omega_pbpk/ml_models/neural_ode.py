"""Phase 46 — Neural ODE Surrogate for PBPK C(t) trajectory prediction.

Architecture:
  - Dynamics function: 2-layer MLP (8→32→32→1, tanh, tanh, linear)
    Inputs: [C_norm, t_norm, logP, log_fup, log_clint, log_mw, log_rbp, log_peff]
    Output: dC_norm/dt_norm
  - Training: Euler BPTT with Adam optimizer on PBPK-generated C(t) trajectories
  - Inference: RK4 integration for higher accuracy
  - Data: 100 LHS-sampled drug param sets × 25 timepoints each

Constraints (no PyTorch, no RDKit):
  - Pure NumPy
  - Manual backprop through unrolled Euler steps
  - Lazy singleton: trained on first call, reused after

Input normalisation (from training data statistics):
  C_norm = C / C_scale  (C_scale = 95th-pct Cmax across training set)
  t_norm = t / T_end_h
  drug features: z-score standardised using training set mean/std

PK metrics extracted from predicted C(t):
  Cmax, AUC (trapz), Tmax, t_half (log-linear terminal slope)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from omega_pbpk._compat import np_trapz

__all__ = [
    "NeuralODEInput",
    "NeuralODEOutput",
    "NeuralODEFunc",
    "NeuralODE",
    "build_node_dataset",
    "predict_neural_ode",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_N_INPUT: int = 8       # [C_norm, t_norm, logP, log_fup, log_clint, log_mw, log_rbp, log_peff]
_N_HIDDEN: int = 32
_T_END_H: float = 24.0
_REF_DOSE_MG: float = 100.0
_REF_VD_L: float = 42.0        # approx Vd for 70 kg subject
_DT_TRAIN: float = 0.5          # Euler step during training (h)
_DT_INFER: float = 0.1          # RK4 step during inference (h)
_N_OBS: int = 25                # observation timepoints per trajectory
_DRUG_FEATURES = ["logP", "log_fup", "log_clint", "log_mw", "log_rbp", "log_peff"]

# Singleton
_NODE_SINGLETON: NeuralODE | None = None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NeuralODEInput:
    """Input specification for the Neural ODE surrogate.

    Attributes:
        logP: Log octanol-water partition coefficient.
        fup: Plasma unbound fraction (0–1).
        clint_L_per_h: Hepatic intrinsic clearance (L/h).
        mw: Molecular weight (g/mol).
        rbp: Red blood cell-to-plasma ratio.
        peff: Effective intestinal permeability (cm/s × 10⁻⁴).
        dose_mg: Administered dose (mg).
        route: 'oral' or 'iv'.
        t_end_h: Simulation end time (h). Default 24.
    """

    logP: float = 2.0
    fup: float = 0.5
    clint_L_per_h: float = 10.0
    mw: float = 300.0
    rbp: float = 1.0
    peff: float = 1.0
    dose_mg: float = 100.0
    route: str = "oral"
    t_end_h: float = 24.0


@dataclass
class NeuralODEOutput:
    """Output of the Neural ODE surrogate.

    Attributes:
        t_h: Time points (h), shape (T,).
        C_mg_L: Predicted plasma concentration (mg/L), shape (T,).
        cmax_mg_L: Peak plasma concentration (mg/L).
        auc_mg_h_L: AUC₀₋ₜ (mg·h/L).
        tmax_h: Time to peak (h).
        t_half_h: Terminal elimination half-life (h).
        train_r2: R² of Neural ODE on training set (set after training).
        n_train: Number of training trajectories used.
        solve_time_ms: Wall-clock time for inference (ms).
    """

    t_h: list[float] = field(default_factory=list)
    C_mg_L: list[float] = field(default_factory=list)
    cmax_mg_L: float = 0.0
    auc_mg_h_L: float = 0.0
    tmax_h: float = 0.0
    t_half_h: float = 0.0
    train_r2: float = 0.0
    n_train: int = 0
    solve_time_ms: float = 0.0


# ---------------------------------------------------------------------------
# NeuralODEFunc — 2-layer MLP dynamics network
# ---------------------------------------------------------------------------


class NeuralODEFunc:
    """MLP parameterising dC_norm/dt_norm = f(C_norm, t_norm, drug_feat).

    Architecture: 8 → 32 → 32 → 1 (tanh, tanh, linear).
    All operations are pure NumPy.
    """

    def __init__(self, n_hidden: int = _N_HIDDEN, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        scale1 = np.sqrt(2.0 / _N_INPUT)
        scale2 = np.sqrt(2.0 / n_hidden)
        self.W1: NDArray[np.float64] = rng.standard_normal((n_hidden, _N_INPUT)) * scale1
        self.b1: NDArray[np.float64] = np.zeros(n_hidden)
        self.W2: NDArray[np.float64] = rng.standard_normal((n_hidden, n_hidden)) * scale2
        self.b2: NDArray[np.float64] = np.zeros(n_hidden)
        self.w3: NDArray[np.float64] = rng.standard_normal(n_hidden) * scale2
        self.b3: float = 0.0
        self._n_hidden = n_hidden

    # ------------------------------------------------------------------
    # Forward pass — returns output + intermediates for backprop
    # ------------------------------------------------------------------

    def forward(
        self, x: NDArray[np.float64]
    ) -> tuple[float, NDArray, NDArray, NDArray, NDArray]:
        """Forward pass.

        Args:
            x: Input vector, shape (8,).

        Returns:
            (out, z1, a1, z2, a2) where out = scalar dC_norm/dt_norm,
            z1/a1 are pre/post-activation for layer 1,
            z2/a2 are pre/post-activation for layer 2.
        """
        z1 = self.W1 @ x + self.b1          # (H,)
        a1 = np.tanh(z1)                     # (H,)
        z2 = self.W2 @ a1 + self.b2          # (H,)
        a2 = np.tanh(z2)                     # (H,)
        out = float(self.w3 @ a2 + self.b3)  # scalar
        return out, z1, a1, z2, a2

    # ------------------------------------------------------------------
    # Parameter serialisation
    # ------------------------------------------------------------------

    def get_flat_params(self) -> NDArray[np.float64]:
        return np.concatenate(
            [self.W1.ravel(), self.b1, self.W2.ravel(), self.b2, self.w3, [self.b3]]
        )

    def set_flat_params(self, p: NDArray[np.float64]) -> None:
        H, D = self._n_hidden, _N_INPUT
        idx = 0
        self.W1 = p[idx : idx + H * D].reshape(H, D); idx += H * D
        self.b1 = p[idx : idx + H].copy(); idx += H
        self.W2 = p[idx : idx + H * H].reshape(H, H); idx += H * H
        self.b2 = p[idx : idx + H].copy(); idx += H
        self.w3 = p[idx : idx + H].copy(); idx += H
        self.b3 = float(p[idx])

    @property
    def n_params(self) -> int:
        H, D = self._n_hidden, _N_INPUT
        return H * D + H + H * H + H + H + 1


# ---------------------------------------------------------------------------
# NeuralODE — integrator + training
# ---------------------------------------------------------------------------


class NeuralODE:
    """Neural ODE surrogate with Euler-BPTT training and RK4 inference.

    Usage::

        node = NeuralODE()
        t_eval, C_traj, p_norm = build_node_dataset(n_samples=100, seed=42)
        node.fit(p_norm, C_traj, t_eval, epochs=50)
        out = node.simulate(p_vec, dose_mg=100.0, route='oral', t_end_h=24.0)
    """

    def __init__(self, n_hidden: int = _N_HIDDEN, seed: int = 42) -> None:
        self._func = NeuralODEFunc(n_hidden=n_hidden, seed=seed)
        self._trained: bool = False
        self._C_scale: float = 5.0          # mg/L, updated from training data
        self._feat_mean: NDArray | None = None   # shape (6,)
        self._feat_std: NDArray | None = None    # shape (6,)
        self._train_r2: float = 0.0
        self._n_train: int = 0
        # Adam state — one slot per parameter array
        self._adam_m: NDArray | None = None
        self._adam_v: NDArray | None = None
        self._adam_t: int = 0

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------

    def _norm_drug(self, p_raw: NDArray[np.float64]) -> NDArray[np.float64]:
        """Z-score normalise 6 drug features using stored training statistics."""
        if self._feat_mean is None or self._feat_std is None:
            return p_raw
        return (p_raw - self._feat_mean) / (self._feat_std + 1e-8)

    @staticmethod
    def _raw_drug_features(
        logP: float, fup: float, clint: float, mw: float, rbp: float, peff: float
    ) -> NDArray[np.float64]:
        """Convert physical drug params to the 6 raw log-space features."""
        return np.array([
            logP,
            np.log(max(fup, 1e-6)),
            np.log(max(clint, 1e-3)),
            np.log(max(mw, 1.0) / 300.0),
            np.log(max(rbp, 1e-3)),
            np.log(max(peff, 1e-4)),
        ], dtype=np.float64)

    def _make_input(
        self, C_norm: float, t_norm: float, p_norm: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Assemble the 8-feature MLP input vector."""
        return np.concatenate([[C_norm, t_norm], p_norm])

    # ------------------------------------------------------------------
    # Euler forward + backward
    # ------------------------------------------------------------------

    def _euler_forward(
        self,
        p_norm: NDArray[np.float64],
        C0_norm: float,
        n_steps: int,
        dt_norm: float,
    ) -> tuple[NDArray, list]:
        """Euler integration; caches intermediates for BPTT.

        Returns:
            C_traj: shape (n_steps+1,) normalised concentrations.
            cache: list of (x, z1, a1, z2, a2) per step.
        """
        C = np.zeros(n_steps + 1)
        C[0] = C0_norm
        cache: list[tuple] = []

        for n in range(n_steps):
            t_norm = n * dt_norm
            x = self._make_input(C[n], t_norm, p_norm)
            out, z1, a1, z2, a2 = self._func.forward(x)
            C[n + 1] = C[n] + dt_norm * out
            cache.append((x, z1, a1, z2, a2))

        return C, cache

    def _euler_backward(
        self,
        C: NDArray[np.float64],
        cache: list[tuple],
        C_true_norm: NDArray[np.float64],
        t_obs_idx: NDArray[np.intp],
        dt_norm: float,
    ) -> dict[str, NDArray]:
        """Backprop through unrolled Euler steps.

        Args:
            C: Forward concentrations, shape (n_steps+1,).
            cache: Per-step (x, z1, a1, z2, a2).
            C_true_norm: Target concentrations at observation indices.
            t_obs_idx: Indices into C where observations occur.
            dt_norm: Normalised Euler step size.

        Returns:
            Dict of gradients: dW1, db1, dW2, db2, dw3, db3.
        """
        n_steps = len(C) - 1
        H = self._func._n_hidden
        func = self._func

        dW1 = np.zeros_like(func.W1)
        db1 = np.zeros_like(func.b1)
        dW2 = np.zeros_like(func.W2)
        db2 = np.zeros_like(func.b2)
        dw3 = np.zeros_like(func.w3)
        db3 = 0.0

        n_obs = len(C_true_norm)
        g_C = np.zeros(n_steps + 1)
        for k, idx in enumerate(t_obs_idx):
            g_C[idx] += 2.0 * (C[idx] - C_true_norm[k]) / n_obs

        for n in range(n_steps - 1, -1, -1):
            x, z1, a1, z2, a2 = cache[n]

            # ---- gradient of loss wrt dC_norm/dt_norm at step n ----
            g_out = dt_norm * g_C[n + 1]   # scalar

            # ---- backprop through output layer (w3, b3) ----
            dw3 += g_out * a2
            db3 += g_out
            g_a2 = func.w3 * g_out         # (H,)

            # ---- backprop through tanh (layer 2) ----
            g_z2 = g_a2 * (1.0 - a2 ** 2)  # (H,)

            # ---- backprop through W2, b2 ----
            dW2 += np.outer(g_z2, a1)
            db2 += g_z2
            g_a1 = func.W2.T @ g_z2        # (H,)

            # ---- backprop through tanh (layer 1) ----
            g_z1 = g_a1 * (1.0 - a1 ** 2)  # (H,)

            # ---- backprop through W1, b1 ----
            dW1 += np.outer(g_z1, x)
            db1 += g_z1

            # ---- propagate C gradient: dL/dC[n] from Euler step ----
            # C[n+1] = C[n] + dt * out, and out = f(x) where x[0] = C[n]
            # dC[n+1]/dC[n] = 1 + dt * d(out)/d(C[n])
            # d(out)/d(x[0]) = w3 @ diag(1-a2^2) @ W2 @ diag(1-a1^2) @ W1[:, 0]
            g_x0_via_net = func.W1[:, 0]          # (H,) — gradient in W1 direction
            d_out_d_x0 = float(
                func.w3 @ ((1.0 - a2 ** 2) * (func.W2 @ ((1.0 - a1 ** 2) * g_x0_via_net)))
            )
            # x[0] = C_norm, so d(out)/d(C[n]) = d(out)/d(x[0]) (no extra scaling
            # since we're already working in normalised space)
            g_C[n] += g_C[n + 1] * (1.0 + dt_norm * d_out_d_x0)

        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2, "dw3": dw3, "db3": db3}

    # ------------------------------------------------------------------
    # Adam update
    # ------------------------------------------------------------------

    def _adam_step(
        self, grads: dict[str, NDArray], lr: float = 1e-3,
        beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8,
    ) -> None:
        """Apply one Adam update to NeuralODEFunc parameters."""
        func = self._func
        flat_grad = np.concatenate([
            grads["dW1"].ravel(), grads["db1"],
            grads["dW2"].ravel(), grads["db2"],
            grads["dw3"], [grads["db3"]],
        ])
        flat_grad = np.clip(flat_grad, -5.0, 5.0)   # gradient clipping

        if self._adam_m is None:
            self._adam_m = np.zeros_like(flat_grad)
            self._adam_v = np.zeros_like(flat_grad)

        self._adam_t += 1
        t = self._adam_t
        self._adam_m = beta1 * self._adam_m + (1 - beta1) * flat_grad
        self._adam_v = beta2 * self._adam_v + (1 - beta2) * flat_grad ** 2
        m_hat = self._adam_m / (1 - beta1 ** t)
        v_hat = self._adam_v / (1 - beta2 ** t)
        update = lr * m_hat / (np.sqrt(v_hat) + eps)

        func.set_flat_params(func.get_flat_params() - update)

    # ------------------------------------------------------------------
    # RK4 inference
    # ------------------------------------------------------------------

    def _rk4_trajectory(
        self,
        p_norm: NDArray[np.float64],
        C0_norm: float,
        t_eval: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """RK4 integration returning C_norm at t_eval points.

        Args:
            p_norm: Normalised drug features, shape (6,).
            C0_norm: Initial normalised concentration.
            t_eval: Evaluation times (normalised), shape (T,).

        Returns:
            C_norm at t_eval, shape (T,).
        """
        T_max = float(t_eval[-1])
        dt = _DT_INFER / _T_END_H  # normalised step
        n_steps = max(1, int(T_max / dt))
        dt = T_max / n_steps

        C = C0_norm
        t = 0.0
        results = []
        eval_idx = 0

        def f(c: float, t_: float) -> float:
            x = self._make_input(c, t_, p_norm)
            out, *_ = self._func.forward(x)
            return out

        for n in range(n_steps):
            # Record if any t_eval falls in this step
            while eval_idx < len(t_eval) and t_eval[eval_idx] <= t + 1e-9:
                results.append(C)
                eval_idx += 1

            k1 = f(C, t)
            k2 = f(C + 0.5 * dt * k1, t + 0.5 * dt)
            k3 = f(C + 0.5 * dt * k2, t + 0.5 * dt)
            k4 = f(C + dt * k3, t + dt)
            C = C + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            C = max(C, 0.0)   # non-negativity
            t += dt

        while eval_idx < len(t_eval):
            results.append(C)
            eval_idx += 1

        return np.array(results, dtype=np.float64)

    # ------------------------------------------------------------------
    # Public API: fit
    # ------------------------------------------------------------------

    def fit(
        self,
        p_raw_array: NDArray[np.float64],
        C_traj: NDArray[np.float64],
        t_eval: NDArray[np.float64],
        epochs: int = 60,
        lr: float = 1e-3,
        route: str = "oral",
    ) -> "NeuralODE":
        """Train the Neural ODE on PBPK-generated C(t) trajectories.

        Args:
            p_raw_array: Drug params (raw log-space features), shape (N, 6).
            C_traj: Plasma C(t) trajectories in mg/L, shape (N, T).
            t_eval: Evaluation time points in hours, shape (T,).
            epochs: Training epochs.
            lr: Adam learning rate.
            route: 'oral' or 'iv' (determines C0).

        Returns:
            self (fitted).
        """
        N, T = C_traj.shape
        logger.info("NeuralODE.fit: %d trajectories × %d timepoints, %d epochs", N, T, epochs)

        # Compute normalisation statistics
        self._feat_mean = p_raw_array.mean(axis=0)
        self._feat_std = p_raw_array.std(axis=0)
        self._C_scale = float(np.percentile(C_traj, 95)) + 1e-6
        self._n_train = N

        p_norm_array = (p_raw_array - self._feat_mean) / (self._feat_std + 1e-8)

        # Normalise trajectories and time
        C_norm_traj = C_traj / self._C_scale            # (N, T)
        t_norm_eval = t_eval / _T_END_H                 # (T,)

        dt_norm = _DT_TRAIN / _T_END_H
        n_steps = int(_T_END_H / _DT_TRAIN)

        # Map t_eval → Euler step indices
        t_obs_idx = np.round(t_norm_eval / dt_norm).astype(np.intp)
        t_obs_idx = np.clip(t_obs_idx, 0, n_steps)

        for epoch in range(epochs):
            epoch_loss = 0.0
            perm = np.random.default_rng(epoch).permutation(N)

            acc_grads: dict[str, Any] = {
                k: np.zeros_like(v)
                for k, v in [
                    ("dW1", self._func.W1), ("db1", self._func.b1),
                    ("dW2", self._func.W2), ("db2", self._func.b2),
                    ("dw3", self._func.w3), ("db3", np.zeros(1)),
                ]
            }
            acc_grads["db3"] = 0.0

            for i in perm:
                p_norm = p_norm_array[i]
                C_true_norm = C_norm_traj[i]
                C0_norm = 0.0 if route == "oral" else C_true_norm[0]

                C_pred, cache = self._euler_forward(p_norm, C0_norm, n_steps, dt_norm)
                C_pred_at_obs = C_pred[t_obs_idx]
                loss_i = float(np.mean((C_pred_at_obs - C_true_norm) ** 2))
                epoch_loss += loss_i

                grads = self._euler_backward(C_pred, cache, C_true_norm, t_obs_idx, dt_norm)
                for k in acc_grads:
                    if k == "db3":
                        acc_grads[k] += grads[k] / N
                    else:
                        acc_grads[k] += grads[k] / N

            self._adam_step(acc_grads, lr=lr)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info("Epoch %3d/%d — loss=%.6f", epoch + 1, epochs, epoch_loss / N)

        # Compute training R²
        ss_res, ss_tot = 0.0, 0.0
        C_mean = C_norm_traj.mean()
        for i in range(N):
            p_norm = p_norm_array[i]
            C_true_norm = C_norm_traj[i]
            C0_norm = 0.0 if route == "oral" else C_true_norm[0]
            C_pred, _ = self._euler_forward(p_norm, C0_norm, n_steps, dt_norm)
            C_pred_at_obs = C_pred[t_obs_idx]
            ss_res += float(np.sum((C_pred_at_obs - C_true_norm) ** 2))
            ss_tot += float(np.sum((C_true_norm - C_mean) ** 2))
        self._train_r2 = 1.0 - ss_res / (ss_tot + 1e-12)
        logger.info("NeuralODE training R² = %.4f", self._train_r2)

        self._trained = True
        return self

    # ------------------------------------------------------------------
    # Public API: simulate
    # ------------------------------------------------------------------

    def simulate(
        self,
        logP: float,
        fup: float,
        clint_L_per_h: float,
        mw: float,
        rbp: float,
        peff: float,
        dose_mg: float = _REF_DOSE_MG,
        route: str = "oral",
        t_end_h: float = _T_END_H,
        n_points: int = 49,
    ) -> NeuralODEOutput:
        """Predict plasma C(t) profile for a new drug.

        Args:
            logP: Log octanol-water partition coefficient.
            fup: Unbound plasma fraction.
            clint_L_per_h: Hepatic CLint (L/h).
            mw: Molecular weight (g/mol).
            rbp: Red blood cell-to-plasma ratio.
            peff: Intestinal effective permeability (×10⁻⁴ cm/s).
            dose_mg: Dose in mg.
            route: 'oral' or 'iv'.
            t_end_h: Simulation horizon (h).
            n_points: Number of output time points.

        Returns:
            NeuralODEOutput with trajectory and PK metrics.
        """
        t0 = time.perf_counter()

        p_raw = self._raw_drug_features(logP, fup, clint_L_per_h, mw, rbp, peff)
        p_norm = self._norm_drug(p_raw)

        # Scale C0 by dose ratio (linear superposition)
        dose_scale = dose_mg / _REF_DOSE_MG
        if route == "iv":
            C0_norm = (dose_mg / _REF_VD_L) / self._C_scale
        else:
            C0_norm = 0.0

        t_eval_h = np.linspace(0.0, t_end_h, n_points)
        t_eval_norm = t_eval_h / _T_END_H

        C_norm = self._rk4_trajectory(p_norm, C0_norm, t_eval_norm)

        # Scale back to physical units
        if route == "oral":
            # Apply dose-linear scaling post-prediction
            C_mg_L = C_norm * self._C_scale * dose_scale
        else:
            C_mg_L = C_norm * self._C_scale  # already scaled via C0

        C_mg_L = np.maximum(C_mg_L, 0.0)

        # PK metrics from trajectory
        cmax = float(np.max(C_mg_L))
        tmax_idx = int(np.argmax(C_mg_L))
        tmax = float(t_eval_h[tmax_idx])
        auc = float(np_trapz(C_mg_L, t_eval_h))

        # Terminal t_half from last 30% of timepoints
        n_pts = len(t_eval_h)
        tail_start = int(0.70 * n_pts)
        tail_C = C_mg_L[tail_start:]
        tail_t = t_eval_h[tail_start:]
        if len(tail_C) >= 3 and np.all(tail_C > 1e-9):
            slope, _ = np.polyfit(tail_t, np.log(tail_C + 1e-12), 1)
            t_half = float(-np.log(2) / slope) if slope < -1e-9 else 99.0
        else:
            t_half = 99.0
        t_half = min(t_half, 99.0)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return NeuralODEOutput(
            t_h=t_eval_h.tolist(),
            C_mg_L=C_mg_L.tolist(),
            cmax_mg_L=cmax,
            auc_mg_h_L=auc,
            tmax_h=tmax,
            t_half_h=t_half,
            train_r2=self._train_r2,
            n_train=self._n_train,
            solve_time_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save weights and normalisation stats to a .npz file."""
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")
        params = self._func.get_flat_params()
        np.savez(
            path,
            params=params,
            feat_mean=self._feat_mean if self._feat_mean is not None else np.zeros(6),
            feat_std=self._feat_std if self._feat_std is not None else np.ones(6),
            C_scale=np.array([self._C_scale]),
            train_r2=np.array([self._train_r2]),
            n_train=np.array([self._n_train]),
        )
        logger.info("NeuralODE saved to %s", path)

    def load(self, path: str | Path) -> None:
        """Load weights and normalisation stats from a .npz file."""
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")
        data = np.load(path)
        self._func.set_flat_params(data["params"])
        self._feat_mean = data["feat_mean"]
        self._feat_std = data["feat_std"]
        self._C_scale = float(data["C_scale"][0])
        self._train_r2 = float(data["train_r2"][0])
        self._n_train = int(data["n_train"][0])
        self._trained = True
        logger.info("NeuralODE loaded from %s (R²=%.4f)", path, self._train_r2)


# ---------------------------------------------------------------------------
# Dataset builder — generates PBPK C(t) training trajectories
# ---------------------------------------------------------------------------


def build_node_dataset(
    n_samples: int = 100,
    dose_mg: float = _REF_DOSE_MG,
    route: str = "oral",
    t_end_h: float = _T_END_H,
    n_obs: int = _N_OBS,
    seed: int = 42,
    n_workers: int = 1,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Generate PBPK C(t) trajectories for Neural ODE training.

    Uses LHS sampling over the same 6-dimensional drug parameter space
    as the existing surrogate data generator.

    Args:
        n_samples: Number of PBPK simulations to run.
        dose_mg: Fixed dose (mg) for all samples.
        route: 'oral' or 'iv'.
        t_end_h: Simulation horizon (h).
        n_obs: Number of observation timepoints per trajectory.
        seed: Random seed.
        n_workers: Worker processes (1 = serial, avoids fork issues in tests).

    Returns:
        (p_raw_array, C_traj, t_eval_h):
            p_raw_array: Shape (N, 6) log-space drug features.
            C_traj: Shape (N, n_obs) plasma concentrations (mg/L).
            t_eval_h: Shape (n_obs,) evaluation times (h).
    """
    from omega_pbpk.core.body import WholeBodyPBPK
    from omega_pbpk.drugs.drug import Drug

    rng = np.random.default_rng(seed)

    # LHS sampling (same ranges as data_generator.py)
    param_ranges = {
        "logP": (-2.0, 6.0),
        "fup": (0.01, 1.0),
        "clint": (1.0, 200.0),
        "mw": (100.0, 800.0),
        "rbp": (0.5, 3.0),
        "peff": (0.1, 20.0),
    }
    n_p = len(param_ranges)
    lhs = np.zeros((n_samples, n_p))
    for j in range(n_p):
        perm = rng.permutation(n_samples)
        lhs[:, j] = (perm + rng.uniform(size=n_samples)) / n_samples

    # Scale to physical ranges (log-uniform for positive params)
    names = list(param_ranges.keys())
    X_phys = np.empty((n_samples, n_p))
    for j, name in enumerate(names):
        lo, hi = param_ranges[name]
        if name == "logP":
            X_phys[:, j] = lo + lhs[:, j] * (hi - lo)
        else:
            X_phys[:, j] = np.exp(np.log(lo) + lhs[:, j] * (np.log(hi) - np.log(lo)))

    t_eval_h = np.linspace(0.0, t_end_h, n_obs)

    # Run PBPK simulations
    p_raw_list: list[NDArray] = []
    C_traj_list: list[NDArray] = []

    for i in range(n_samples):
        logP, fup, clint, mw, rbp, peff = [float(X_phys[i, j]) for j in range(n_p)]
        try:
            drug = Drug(
                name=f"node_train_{i}",
                logP=logP,
                fup=max(fup, 0.001),
                rbp=rbp,
                peff=peff,
                mw=mw,
                clint_hepatic_L_per_h=clint,
            )
            model = WholeBodyPBPK(drug=drug)
            if route == "iv":
                model.setup_iv(dose_mg=dose_mg)
            else:
                model.setup_oral(dose_mg=dose_mg)
            result = model.simulate(t_end_h=t_end_h, dt_h=0.1)
            cp_full = result.plasma_concentration()

            # Subsample at n_obs evenly spaced points
            full_t = np.linspace(0.0, t_end_h, len(cp_full))
            C_obs = np.interp(t_eval_h, full_t, cp_full)

            if not np.all(np.isfinite(C_obs)) or np.any(C_obs < 0):
                continue

            # Build raw log-space feature vector
            p_raw = np.array([
                logP,
                np.log(max(fup, 1e-6)),
                np.log(max(clint, 1e-3)),
                np.log(max(mw, 1.0) / 300.0),
                np.log(max(rbp, 1e-3)),
                np.log(max(peff, 1e-4)),
            ], dtype=np.float64)
            p_raw_list.append(p_raw)
            C_traj_list.append(C_obs)

        except Exception as exc:
            logger.debug("Node dataset sample %d failed: %s", i, exc)
            continue

        if (i + 1) % 25 == 0:
            logger.info("NODE dataset: %d/%d simulations (%d valid)", i + 1, n_samples, len(p_raw_list))

    if len(p_raw_list) < 10:
        raise RuntimeError(
            f"Only {len(p_raw_list)} valid PBPK trajectories generated from {n_samples} attempts. "
            "Check drug parameter ranges and PBPK model configuration."
        )

    p_raw_array = np.array(p_raw_list, dtype=np.float64)
    C_traj_array = np.array(C_traj_list, dtype=np.float64)
    logger.info("NODE dataset: %d valid trajectories", len(p_raw_list))
    return p_raw_array, C_traj_array, t_eval_h


# ---------------------------------------------------------------------------
# Singleton convenience function
# ---------------------------------------------------------------------------


def _build_singleton(n_samples: int = 100, epochs: int = 60, seed: int = 42) -> NeuralODE:
    """Lazily train and return the module-level NeuralODE singleton."""
    logger.info("NeuralODE: building training dataset (n=%d)…", n_samples)
    p_raw, C_traj, t_eval = build_node_dataset(n_samples=n_samples, seed=seed)
    node = NeuralODE(seed=seed)
    node.fit(p_raw, C_traj, t_eval, epochs=epochs)
    return node


def predict_neural_ode(inp: NeuralODEInput) -> NeuralODEOutput:
    """Predict C(t) profile via the module-level Neural ODE singleton.

    The singleton is lazily trained on the first call using 100 PBPK
    simulations (≈15 s).  Subsequent calls are near-instant (~1 ms).

    Args:
        inp: NeuralODEInput with drug params + dosing.

    Returns:
        NeuralODEOutput with trajectory and PK summary metrics.
    """
    global _NODE_SINGLETON
    if _NODE_SINGLETON is None or not _NODE_SINGLETON._trained:
        _NODE_SINGLETON = _build_singleton()
    return _NODE_SINGLETON.simulate(
        logP=inp.logP,
        fup=inp.fup,
        clint_L_per_h=inp.clint_L_per_h,
        mw=inp.mw,
        rbp=inp.rbp,
        peff=inp.peff,
        dose_mg=inp.dose_mg,
        route=inp.route,
        t_end_h=inp.t_end_h,
    )
