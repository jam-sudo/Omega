"""Neural PK Surrogate — NumPy MLP ensemble for fast 1-compartment PK prediction.

Architecture:
  - Pure NumPy (no PyTorch/TensorFlow)
  - Ensemble of 5 MLP models (6→32→32→4, ReLU, He init)
  - Mini-batch SGD with numerical gradients (finite differences)
  - Bootstrap resampling per ensemble member
  - log1p output transform + z-score normalisation

Analytical 1-compartment PK:
  - ORAL: ke=CL/Vd, AUC=F·dose/CL, tmax=ln(ka/ke)/(ka-ke), Cmax formula
  - IV:   AUC=dose/CL, Cmax=dose/Vd, tmax=0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "PKSurrogateInput",
    "PKSurrogateOutput",
    "NumpyMLP",
    "PKSurrogateEnsemble",
    "generate_analytical_pk_data",
    "analytical_pk_1cpt",
    "predict_pk_surrogate",
]

logger = logging.getLogger(__name__)

_LN2: float = 0.693147180559945
_N_INPUT: int = 6
_N_OUTPUT: int = 4
_N_ENSEMBLE: int = 5

# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PKSurrogateInput:
    """Input specification for 1-compartment PK surrogate.

    Attributes:
        dose_mg: Administered dose (mg).
        cl_L_per_h: Clearance (L/h).
        vd_L: Volume of distribution (L).
        ka_per_h: Absorption rate constant (1/h, oral only).
        f_bioavail: Oral bioavailability fraction (0–1).
        route: Administration route ('oral' or 'iv').
    """

    dose_mg: float
    cl_L_per_h: float
    vd_L: float
    ka_per_h: float
    f_bioavail: float
    route: str = "oral"


@dataclass(frozen=True)
class PKSurrogateOutput:
    """Output of the PK surrogate or analytical computation.

    Attributes:
        auc_mg_h_per_L: AUC₀₋∞ (mg·h/L).
        cmax_mg_per_L: Peak concentration (mg/L).
        tmax_h: Time to peak (h).
        t_half_h: Elimination half-life (h).
        confidence: 'high', 'medium', 'low', or 'analytical'.
        auc_std / cmax_std / tmax_std / t_half_std: Ensemble std devs.
    """

    auc_mg_h_per_L: float
    cmax_mg_per_L: float
    tmax_h: float
    t_half_h: float
    confidence: str
    auc_std: float = 0.0
    cmax_std: float = 0.0
    tmax_std: float = 0.0
    t_half_std: float = 0.0


# ---------------------------------------------------------------------------
# Analytical 1-compartment PK helpers
# ---------------------------------------------------------------------------


def _anal_oral(
    dose: float, cl: float, vd: float, ka: float, f: float
) -> tuple[float, float, float, float]:
    """Analytical oral 1-cpt PK. Returns (auc, cmax, tmax, t_half)."""
    ke = cl / vd
    t_half = _LN2 / ke
    auc = f * dose / cl
    ke_eff = ke + 1e-6 if abs(ka - ke) < 1e-6 else ke
    denom = ka - ke_eff
    tmax = max(0.0, float(np.log(ka / ke_eff) / denom))
    cmax = (f * dose / vd) * (ka / denom) * ((ke_eff / ka) ** (ke_eff / denom))
    return float(auc), max(0.0, float(cmax)), tmax, float(t_half)


def _anal_iv(dose: float, cl: float, vd: float) -> tuple[float, float, float, float]:
    """Analytical IV bolus 1-cpt PK. Returns (auc, cmax, tmax, t_half)."""
    ke = cl / vd
    return float(dose / cl), float(dose / vd), 0.0, float(_LN2 / ke)


def analytical_pk_1cpt(inp: PKSurrogateInput) -> PKSurrogateOutput:
    """Exact 1-compartment PK computation (no ML, confidence='analytical').

    Args:
        inp: PKSurrogateInput.

    Returns:
        PKSurrogateOutput with confidence='analytical'.
    """
    if inp.route.lower() == "iv":
        auc, cmax, tmax, t_half = _anal_iv(inp.dose_mg, inp.cl_L_per_h, inp.vd_L)
    else:
        auc, cmax, tmax, t_half = _anal_oral(
            inp.dose_mg, inp.cl_L_per_h, inp.vd_L, inp.ka_per_h, inp.f_bioavail
        )
    return PKSurrogateOutput(
        auc_mg_h_per_L=round(auc, 6),
        cmax_mg_per_L=round(cmax, 6),
        tmax_h=round(tmax, 6),
        t_half_h=round(t_half, 6),
        confidence="analytical",
    )


# ---------------------------------------------------------------------------
# Training data generator
# ---------------------------------------------------------------------------


def generate_analytical_pk_data(
    n_samples: int = 2000,
    seed: int = 42,
) -> tuple[NDArray, NDArray]:
    """Generate LHS-sampled training data with analytical 1-cpt PK targets.

    Feature columns (X):
        0: log(dose_mg)
        1: log(cl_L_per_h)
        2: log(vd_L)
        3: log(ka_per_h)
        4: f_bioavail
        5: route (0=oral, 1=IV)

    Target columns (y — raw, NOT log-transformed):
        0: AUC (mg·h/L)
        1: Cmax (mg/L)
        2: tmax (h)
        3: t_half (h)

    Args:
        n_samples: Number of samples.
        seed: RNG seed.

    Returns:
        (X, y) as float32 arrays of shape (n_samples, 6) and (n_samples, 4).
    """
    # Latin Hypercube Sampling (fast numpy implementation for speed)
    rng = np.random.default_rng(seed)
    base = np.arange(n_samples, dtype=np.float64).reshape(-1, 1)
    cuts = (base + rng.random((n_samples, 6))) / n_samples  # (n, 6)
    lhs: NDArray = np.empty((n_samples, 6), dtype=np.float64)
    for col in range(6):
        lhs[:, col] = rng.permutation(cuts[:, col])

    # Map uniform [0,1] to parameter ranges (log-uniform or linear)
    log_dose = lhs[:, 0] * (np.log(1000.0) - np.log(1.0)) + np.log(1.0)
    log_cl = lhs[:, 1] * (np.log(100.0) - np.log(0.1)) + np.log(0.1)
    log_vd = lhs[:, 2] * (np.log(500.0) - np.log(5.0)) + np.log(5.0)
    log_ka = lhs[:, 3] * (np.log(5.0) - np.log(0.1)) + np.log(0.1)
    f_vals = lhs[:, 4] * (1.0 - 0.05) + 0.05
    route_bin = (lhs[:, 5] >= 0.5).astype(np.float32)

    dose = np.exp(log_dose)
    cl = np.exp(log_cl)
    vd = np.exp(log_vd)
    ka = np.exp(log_ka)

    # Vectorised analytical PK —————————————————————————————————————————
    ke = cl / vd
    t_halfs = _LN2 / ke

    # Oral (all samples, then override IV below)
    ke_eff = np.where(np.abs(ka - ke) < 1e-6, ke + 1e-6, ke)
    denom = ka - ke_eff
    tmaxs = np.maximum(0.0, np.log(ka / ke_eff) / denom)
    cmax_oral = (f_vals * dose / vd) * (ka / denom) * ((ke_eff / ka) ** (ke_eff / denom))
    cmaxs = np.maximum(0.0, cmax_oral)
    aucs = f_vals * dose / cl

    # Override IV samples
    iv_mask = route_bin > 0.5
    aucs = np.where(iv_mask, dose / cl, aucs)
    cmaxs = np.where(iv_mask, dose / vd, cmaxs)
    tmaxs = np.where(iv_mask, 0.0, tmaxs)

    X = np.column_stack([log_dose, log_cl, log_vd, log_ka, f_vals, route_bin]).astype(np.float32)
    y = np.column_stack([aucs, cmaxs, tmaxs, t_halfs]).astype(np.float32)

    return X, y


# ---------------------------------------------------------------------------
# NumpyMLP
# ---------------------------------------------------------------------------


class NumpyMLP:
    """Pure NumPy fully-connected network with ReLU hidden layers.

    Default architecture: 6 → 32 → 32 → 4.
    Uses He (Kaiming) initialisation for weights.
    Training uses analytical backpropagation for speed; get_params /
    set_params support the same finite-difference SGD interface as
    graph_attention.py for parameter introspection and fine-tuning.
    """

    def __init__(
        self,
        layer_sizes: tuple[int, ...] = (_N_INPUT, 32, 32, _N_OUTPUT),
        seed: int = 42,
    ) -> None:
        self.layer_sizes = layer_sizes
        rng = np.random.default_rng(seed)
        self._weights: list[NDArray] = []
        self._biases: list[NDArray] = []
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            he_std = float(np.sqrt(2.0 / fan_in))
            W = (rng.standard_normal((fan_in, fan_out)) * he_std).astype(np.float32)
            b = np.zeros(fan_out, dtype=np.float32)
            self._weights.append(W)
            self._biases.append(b)

    def forward(self, X: NDArray) -> NDArray:
        """Forward pass. X: (N, in_dim) → (N, out_dim).

        Args:
            X: Input feature matrix, shape (N, input_size).

        Returns:
            Output array, shape (N, output_size).
        """
        h: NDArray = np.asarray(X, dtype=np.float32)
        n_hidden = len(self._weights) - 1
        for i, (W, b) in enumerate(zip(self._weights, self._biases, strict=True)):
            h = h @ W + b
            if i < n_hidden:  # ReLU on hidden layers only
                np.maximum(h, 0.0, out=h)
        return h

    def forward_and_cache(self, X: NDArray) -> tuple[NDArray, list[NDArray], list[NDArray]]:
        """Forward pass saving pre-activations and activations for backprop.

        Args:
            X: Input, shape (N, in_dim).

        Returns:
            (output, pre_acts, acts) where pre_acts[i] and acts[i] are the
            pre- and post-activation values at layer i.
        """
        h: NDArray = np.asarray(X, dtype=np.float32)
        pre_acts: list[NDArray] = []
        acts: list[NDArray] = [h]
        n_hidden = len(self._weights) - 1
        for i, (W, b) in enumerate(zip(self._weights, self._biases, strict=True)):
            z = h @ W + b
            pre_acts.append(z)
            if i < n_hidden:
                h = np.maximum(z, 0.0)
            else:
                h = z  # linear output
            acts.append(h)
        return h, pre_acts, acts

    def backward(
        self,
        pre_acts: list[NDArray],
        acts: list[NDArray],
        loss_grad_out: NDArray,
        lr: float,
    ) -> None:
        """Backprop + SGD step. Updates weights in-place.

        Args:
            pre_acts: Pre-activation arrays from forward_and_cache.
            acts: Activation arrays from forward_and_cache.
            loss_grad_out: dL/d(output), shape (N, out_dim).
            lr: Learning rate.
        """
        delta = loss_grad_out.astype(np.float32)
        n = float(acts[0].shape[0])
        n_layers = len(self._weights)
        for i in reversed(range(n_layers)):
            self._weights[i] -= lr * (acts[i].T @ delta / n)
            self._biases[i] -= lr * delta.mean(axis=0)
            if i > 0:  # propagate through ReLU
                delta = (delta @ self._weights[i].T) * (pre_acts[i - 1] > 0)

    def get_params(self) -> list[NDArray]:
        """Return direct references to parameter arrays [W0, b0, W1, b1, ...].

        Callers may ravel() each array and modify in-place (finite-difference
        SGD pattern from graph_attention.py).

        Returns:
            Alternating weight / bias array references (not copies).
        """
        params: list[NDArray] = []
        for W, b in zip(self._weights, self._biases, strict=True):
            params.append(W)
            params.append(b)
        return params

    def set_params(self, params: list[NDArray]) -> None:
        """Copy parameters in-place from a flat list [W0, b0, W1, b1, ...].

        Args:
            params: List of arrays matching the shape of get_params().
        """
        idx = 0
        for i in range(len(self._weights)):
            self._weights[i][:] = params[idx]
            self._biases[i][:] = params[idx + 1]
            idx += 2


# ---------------------------------------------------------------------------
# PKSurrogateEnsemble
# ---------------------------------------------------------------------------


class PKSurrogateEnsemble:
    """Ensemble of NumpyMLP models for 1-compartment PK prediction.

    Training:
        - Each model is trained on a bootstrap resample of the data.
        - Mini-batch SGD with finite-difference numerical gradients.
        - Targets are log1p-transformed then z-score normalised.

    Prediction:
        - Ensemble mean ± std across models.
        - Confidence from mean coefficient of variation:
            CV < 0.15  → 'high'
            CV < 0.40  → 'medium'
            else       → 'low'
    """

    def __init__(self, n_models: int = _N_ENSEMBLE, seed: int = 42) -> None:
        self._n_models = n_models
        self._seed = seed
        self._models: list[NumpyMLP] = [
            NumpyMLP(layer_sizes=(_N_INPUT, 32, 32, _N_OUTPUT), seed=seed + i)
            for i in range(n_models)
        ]
        self._x_mean: NDArray = np.zeros(_N_INPUT, dtype=np.float32)
        self._x_std: NDArray = np.ones(_N_INPUT, dtype=np.float32)
        self._y_mean: NDArray = np.zeros(_N_OUTPUT, dtype=np.float32)
        self._y_std: NDArray = np.ones(_N_OUTPUT, dtype=np.float32)
        self._trained = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mse(pred: NDArray, target: NDArray) -> float:
        diff = pred - target
        return float(np.mean(diff * diff))

    def _to_features(self, inp: PKSurrogateInput) -> NDArray:
        """Convert PKSurrogateInput to normalised feature vector (1, 6)."""
        route_bin = 1.0 if inp.route.lower() == "iv" else 0.0
        raw = np.array(
            [
                np.log(max(inp.dose_mg, 1e-9)),
                np.log(max(inp.cl_L_per_h, 1e-9)),
                np.log(max(inp.vd_L, 1e-9)),
                np.log(max(inp.ka_per_h, 1e-9)),
                inp.f_bioavail,
                route_bin,
            ],
            dtype=np.float32,
        )
        return ((raw - self._x_mean) / self._x_std).reshape(1, _N_INPUT)

    def _decode(self, pred_norm: NDArray) -> NDArray:
        """Denormalise then inverse-log1p. pred_norm shape: (N, 4)."""
        pred_log = pred_norm * self._y_std + self._y_mean
        return np.expm1(np.clip(pred_log, -10.0, 20.0))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X: NDArray,
        y: NDArray,
        epochs: int = 200,
        lr: float = 0.005,
        batch_size: int = 64,
    ) -> None:
        """Train ensemble with bootstrap resampling + vectorised SGD.

        Each model is trained on a bootstrap resample of the data using
        mini-batch SGD with analytical backpropagation (vectorised over
        the batch).  Targets are log1p-transformed then z-score normalised
        before training so all outputs share comparable magnitude.

        For parameter introspection and finite-difference fine-tuning,
        use get_params() / set_params() on individual NumpyMLP instances
        (same interface as graph_attention.py numerical-gradient SGD).

        Args:
            X: Feature matrix (N, 6) — log-domain PK parameters.
            y: Target matrix (N, 4) — raw AUC, Cmax, tmax, t_half values.
            epochs: Training epochs per ensemble member.
            lr: SGD learning rate.
            batch_size: Mini-batch size.
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        n = len(X)

        # Normalise inputs
        self._x_mean = X.mean(axis=0).astype(np.float32)
        x_std = X.std(axis=0)
        self._x_std = np.where(x_std < 1e-8, 1.0, x_std).astype(np.float32)
        X_norm = ((X - self._x_mean) / self._x_std).astype(np.float32)

        # log1p-transform then normalise targets
        y_t = np.log1p(np.abs(y)).astype(np.float32)
        self._y_mean = y_t.mean(axis=0).astype(np.float32)
        y_std = y_t.std(axis=0)
        self._y_std = np.where(y_std < 1e-8, 1.0, y_std).astype(np.float32)
        y_norm = ((y_t - self._y_mean) / self._y_std).astype(np.float32)

        rng = np.random.default_rng(self._seed)

        for model_idx, model in enumerate(self._models):
            # Bootstrap resample (with replacement)
            boot = rng.integers(0, n, size=n)
            Xb = X_norm[boot]
            yb = y_norm[boot]

            model_rng = np.random.default_rng(self._seed + model_idx * 997)

            for epoch in range(epochs):
                perm = model_rng.permutation(n)
                epoch_loss = 0.0
                n_batches = 0

                for start in range(0, n, batch_size):
                    bidx = perm[start : start + batch_size]
                    Xmb = Xb[bidx]
                    ymb = yb[bidx]

                    # Vectorised analytical backpropagation
                    out, pre_acts, acts = model.forward_and_cache(Xmb)
                    # MSE gradient: dL/d(out) = 2*(out - target)/N
                    grad_out = 2.0 * (out - ymb) / float(ymb.shape[0])
                    # Track pre-update loss for logging (reuse 'out')
                    epoch_loss += float(np.mean((out - ymb) ** 2))
                    n_batches += 1
                    model.backward(pre_acts, acts, grad_out, lr)

                if epoch % 50 == 0:
                    avg = epoch_loss / max(n_batches, 1)
                    logger.debug(
                        "PKSurrogate model %d/%d  epoch %d/%d  loss=%.4f",
                        model_idx + 1,
                        self._n_models,
                        epoch + 1,
                        epochs,
                        avg,
                    )

        self._trained = True
        logger.info("PKSurrogateEnsemble trained (%d models).", self._n_models)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, inp: PKSurrogateInput) -> PKSurrogateOutput:
        """Predict PK metrics for a single input.

        If not trained, falls back to analytical computation.

        Args:
            inp: PKSurrogateInput.

        Returns:
            PKSurrogateOutput with ensemble mean/std and confidence.
        """
        if not self._trained:
            logger.warning("Ensemble not trained; returning analytical result.")
            return analytical_pk_1cpt(inp)

        x = self._to_features(inp)
        preds: list[NDArray] = []
        for model in self._models:
            raw = model.forward(x)  # (1, 4)
            decoded = self._decode(raw)  # (1, 4)
            preds.append(decoded[0])

        arr = np.stack(preds, axis=0)  # (n_models, 4)
        means = arr.mean(axis=0)
        stds = arr.std(axis=0)

        # Coefficient of variation → confidence
        cv = (stds / (np.abs(means) + 1e-8)).mean()
        if cv < 0.15:
            confidence = "high"
        elif cv < 0.40:
            confidence = "medium"
        else:
            confidence = "low"

        return PKSurrogateOutput(
            auc_mg_h_per_L=float(max(0.0, means[0])),
            cmax_mg_per_L=float(max(0.0, means[1])),
            tmax_h=float(max(0.0, means[2])),
            t_half_h=float(max(0.0, means[3])),
            confidence=confidence,
            auc_std=float(stds[0]),
            cmax_std=float(stds[1]),
            tmax_std=float(stds[2]),
            t_half_std=float(stds[3]),
        )

    def predict_batch(self, inputs: list[PKSurrogateInput]) -> list[PKSurrogateOutput]:
        """Predict PK metrics for a list of inputs.

        Args:
            inputs: List of PKSurrogateInput.

        Returns:
            List of PKSurrogateOutput (same length and order).
        """
        return [self.predict(inp) for inp in inputs]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save ensemble weights and normalisation stats to a .npz file.

        Args:
            path: Destination file path (will add .npz if absent).
        """
        path = Path(path)
        sd: dict[str, NDArray] = {
            "x_mean": self._x_mean,
            "x_std": self._x_std,
            "y_mean": self._y_mean,
            "y_std": self._y_std,
        }
        for mi, model in enumerate(self._models):
            for pi, p_arr in enumerate(model.get_params()):
                sd[f"m{mi}_p{pi}"] = p_arr.copy()
        np.savez(path, **sd)
        logger.info("PKSurrogateEnsemble saved to %s", path)

    def load(self, path: str | Path) -> None:
        """Load ensemble weights and normalisation stats from a .npz file.

        Args:
            path: Source .npz file.
        """
        path = Path(path)
        data = np.load(path)
        self._x_mean = data["x_mean"].astype(np.float32)
        self._x_std = data["x_std"].astype(np.float32)
        self._y_mean = data["y_mean"].astype(np.float32)
        self._y_std = data["y_std"].astype(np.float32)
        for mi, model in enumerate(self._models):
            params = model.get_params()
            for pi in range(len(params)):
                params[pi][:] = data[f"m{mi}_p{pi}"]
        self._trained = True
        logger.info("PKSurrogateEnsemble loaded from %s", path)


# ---------------------------------------------------------------------------
# Singleton convenience function
# ---------------------------------------------------------------------------

_SINGLETON: PKSurrogateEnsemble | None = None


def predict_pk_surrogate(inp: PKSurrogateInput) -> PKSurrogateOutput:
    """Predict 1-compartment PK via singleton ensemble (auto-trains on first call).

    Falls back to analytical computation when surrogate confidence is 'low'.

    Args:
        inp: PKSurrogateInput specification.

    Returns:
        PKSurrogateOutput. confidence='analytical' on fallback.
    """
    global _SINGLETON  # noqa: PLW0603
    if _SINGLETON is None:
        logger.info(
            "PKSurrogateEnsemble not initialised — generating training data "
            "and training now (this may take a moment)."
        )
        _SINGLETON = PKSurrogateEnsemble()
        X, y = generate_analytical_pk_data(n_samples=2000, seed=42)
        _SINGLETON.train(X, y, epochs=200, lr=0.005, batch_size=64)

    result = _SINGLETON.predict(inp)
    if result.confidence == "low":
        logger.debug("Surrogate confidence low; falling back to analytical PK.")
        return analytical_pk_1cpt(inp)
    return result
