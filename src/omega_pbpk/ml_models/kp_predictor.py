"""ML-based tissue partition coefficient (Kp) predictor.

Architecture:
  - Pure NumPy (no PyTorch/TensorFlow)
  - KpEnsemble: 3 NumpyMLP models, 8→32→32→13
  - Trained on LHS-sampled simplified Rodgers-Rowland data (3 000 points)
  - Validated against mechanistic RR formula on each prediction
  - Singleton auto-training via predict_kp_ml()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

from omega_pbpk.ml_models.pk_surrogate import NumpyMLP

__all__ = [
    "KpInput",
    "KpOutput",
    "KpEnsemble",
    "TISSUES",
    "generate_kp_training_data",
    "predict_kp_ml",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TISSUES: Final[list[str]] = [
    "adipose",
    "bone",
    "brain",
    "gut_wall",
    "heart",
    "kidney",
    "liver",
    "lung",
    "muscle",
    "skin",
    "spleen",
    "stomach",
    "thymus",
]

_N_TISSUES: int = len(TISSUES)  # 13
_N_FEATURES: int = 8
_N_ENSEMBLE: int = 3

# (fnl, fw, ka_factor) per tissue — simplified from literature
_TISSUE_CONSTANTS: dict[str, tuple[float, float, float]] = {
    "adipose": (0.853, 0.15, 0.01),
    "bone": (0.017, 0.44, 0.10),
    "brain": (0.051, 0.77, 0.05),
    "gut_wall": (0.038, 0.71, 0.10),
    "heart": (0.014, 0.76, 0.16),
    "kidney": (0.013, 0.78, 0.16),
    "liver": (0.014, 0.71, 0.08),
    "lung": (0.020, 0.76, 0.25),
    "muscle": (0.010, 0.75, 0.06),
    "skin": (0.060, 0.71, 0.05),
    "spleen": (0.013, 0.74, 0.16),
    "stomach": (0.016, 0.75, 0.10),
    "thymus": (0.013, 0.74, 0.10),
}

# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KpInput:
    """Input specification for ML Kp predictor.

    Attributes:
        logP: Octanol-water log partition coefficient.
        pKa: Primary pKa value.
        fup: Fraction unbound in plasma (0-1).
        mw: Molecular weight (g/mol).
        compound_type: Ionization class:
            'neutral' | 'acid' | 'base' | 'ampholyte'.
        hbd: Hydrogen bond donor count (reserved for future features).
        hba: Hydrogen bond acceptor count (reserved for future features).
    """

    logP: float
    pKa: float
    fup: float
    mw: float
    compound_type: str = "neutral"
    hbd: int = 0
    hba: int = 0


@dataclass(frozen=True)
class KpOutput:
    """Output of the ML Kp predictor.

    Attributes:
        tissue_kp: Tissue:plasma Kp for each of the 13 standard tissues.
        confidence: 'high', 'medium', or 'low' (ensemble CV).
        warnings: Quality/validation warnings encountered.
    """

    tissue_kp: dict[str, float]
    confidence: str
    warnings: tuple[str, ...]


# ---------------------------------------------------------------------------
# Feature encoding
# ---------------------------------------------------------------------------


def _encode_features(inp: KpInput) -> NDArray:
    """Encode KpInput to an 8-element feature vector (float32).

    Feature layout:
        0: logP
        1: pKa
        2: fup
        3: log10(MW)
        4: one-hot neutral   (ampholyte → 0)
        5: one-hot acid      (ampholyte → 1)
        6: one-hot base      (ampholyte → 1)
        7: logP − pKa        (lipophilicity–ionisation cross term)

    Args:
        inp: KpInput specification.

    Returns:
        Float32 ndarray of shape (8,).
    """
    ct = inp.compound_type.lower()
    is_neutral = 1.0 if ct == "neutral" else 0.0
    is_acid = 1.0 if ct in ("acid", "ampholyte") else 0.0
    is_base = 1.0 if ct in ("base", "ampholyte") else 0.0
    log_mw = float(np.log10(max(inp.mw, 1.0)))
    return np.array(
        [
            inp.logP,
            inp.pKa,
            inp.fup,
            log_mw,
            is_neutral,
            is_acid,
            is_base,
            inp.logP - inp.pKa,
        ],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Simplified Rodgers-Rowland formula (training data + RR validation)
# ---------------------------------------------------------------------------


def _rr_simplified(
    logP: float,
    fup: float,
    fnl: float,
    fw: float,
    ka_factor: float,
    compound_type: str,
) -> float:
    """Simplified Rodgers-Rowland Kp for a single tissue.

    Formula:
        base = fnl * P + fw / max(fup, 0.001)
        base  → scaled by ka_factor for base/acid compounds.

    Args:
        logP: Octanol-water log partition coefficient.
        fup: Fraction unbound in plasma.
        fnl: Fractional neutral lipid volume in tissue.
        fw: Fractional water volume in tissue.
        ka_factor: Tissue-specific ionisation binding factor.
        compound_type: One of 'neutral', 'acid', 'base', 'ampholyte'.

    Returns:
        Estimated Kp (always ≥ 0.01).
    """
    P = 10.0**logP
    base = fnl * P + fw / max(fup, 0.001)
    ct = compound_type.lower()
    if ct == "base":
        kp = base * (1.0 + ka_factor)
    elif ct == "acid":
        kp = base * (1.0 - 0.5 * ka_factor)
    else:  # neutral, ampholyte
        kp = base
    return max(kp, 0.01)


# ---------------------------------------------------------------------------
# Training data generator
# ---------------------------------------------------------------------------


def generate_kp_training_data(
    n_samples: int = 3000,
    seed: int = 42,
) -> tuple[NDArray, NDArray]:
    """Generate LHS-sampled training data with simplified RR Kp targets.

    Feature columns X (n, 8): see _encode_features docstring.
    Target columns y (n, 13): log10-transformed Kp for each tissue
        in TISSUES order.

    LHS parameter ranges:
        logP ∈ [−2, 5], pKa ∈ [3, 10], fup ∈ [0.01, 1.0],
        MW ∈ [100, 800], compound_type sampled randomly.

    5 % multiplicative noise is added to each Kp target.

    Args:
        n_samples: Number of training samples.
        seed: RNG seed.

    Returns:
        (X, y) as float32 arrays of shape (n, 8) and (n, 13).
    """
    rng = np.random.default_rng(seed)

    # --- Latin Hypercube Sampling (4 continuous dims) ---
    try:
        from scipy.stats.qmc import LatinHypercube  # type: ignore[import]

        sampler = LatinHypercube(d=4, seed=int(seed))
        lhs_raw: NDArray = sampler.random(n=n_samples)
    except Exception:  # scipy absent or version mismatch
        lhs_raw = np.empty((n_samples, 4), dtype=np.float64)
        base = np.arange(n_samples, dtype=np.float64).reshape(-1, 1)
        cuts = (base + rng.random((n_samples, 4))) / n_samples
        for col in range(4):
            lhs_raw[:, col] = rng.permutation(cuts[:, col])

    # Map [0, 1] → parameter ranges
    logP_arr = lhs_raw[:, 0] * 7.0 - 2.0  # [-2, 5]
    pKa_arr = lhs_raw[:, 1] * 7.0 + 3.0  # [3, 10]
    fup_arr = lhs_raw[:, 2] * 0.99 + 0.01  # [0.01, 1.0]
    mw_arr = lhs_raw[:, 3] * 700.0 + 100.0  # [100, 800]

    ctype_choices = ["neutral", "acid", "base", "ampholyte"]
    ctypes = rng.choice(ctype_choices, size=n_samples)

    X_rows: list[NDArray] = []
    y_rows: list[list[float]] = []

    for i in range(n_samples):
        lp = float(logP_arr[i])
        pk = float(pKa_arr[i])
        fu = float(fup_arr[i])
        mw = float(mw_arr[i])
        ct = str(ctypes[i])

        inp = KpInput(logP=lp, pKa=pk, fup=fu, mw=mw, compound_type=ct)
        X_rows.append(_encode_features(inp))

        kp_row: list[float] = []
        for tissue in TISSUES:
            fnl, fw, ka_factor = _TISSUE_CONSTANTS[tissue]
            kp = _rr_simplified(lp, fu, fnl, fw, ka_factor, ct)
            noise = 1.0 + rng.standard_normal() * 0.05
            kp = max(kp * noise, 0.01)
            kp_row.append(float(np.log10(kp)))
        y_rows.append(kp_row)

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.float32)
    return X, y


# ---------------------------------------------------------------------------
# KpEnsemble
# ---------------------------------------------------------------------------


class KpEnsemble:
    """Ensemble of 3 NumpyMLP models for tissue Kp prediction.

    Architecture: 8 → 32 → 32 → 13 (He init, ReLU hidden, linear output).

    Training:
        - Bootstrap resampling per model.
        - Mini-batch SGD with analytical backpropagation.
        - Targets are log10-Kp, z-score normalised before training.

    Confidence from mean coefficient of variation across tissues:
        CV < 0.15 → 'high', CV < 0.40 → 'medium', else → 'low'
    """

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed
        self._models: list[NumpyMLP] = [
            NumpyMLP(layer_sizes=(8, 32, 32, 13), seed=seed + i) for i in range(_N_ENSEMBLE)
        ]
        self._x_mean = np.zeros(_N_FEATURES, dtype=np.float32)
        self._x_std = np.ones(_N_FEATURES, dtype=np.float32)
        self._y_mean = np.zeros(_N_TISSUES, dtype=np.float32)
        self._y_std = np.ones(_N_TISSUES, dtype=np.float32)
        self._trained = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _decode(self, pred_norm: NDArray) -> NDArray:
        """Denormalise z-scored log10-Kp. pred_norm shape: (N, 13)."""
        return pred_norm * self._y_std + self._y_mean

    def _validate_vs_rr(
        self,
        inp: KpInput,
        ml_kp: dict[str, float],
    ) -> tuple[dict[str, float], list[str]]:
        """Replace tissues where ML diverges > 2× from simplified RR.

        Args:
            inp: Original KpInput.
            ml_kp: ML-predicted Kp dict (tissue → Kp).

        Returns:
            (corrected_kp, warning_messages) tuple.
        """
        warnings: list[str] = []
        corrected = dict(ml_kp)
        ct = inp.compound_type
        for tissue in TISSUES:
            fnl, fw, ka_factor = _TISSUE_CONSTANTS[tissue]
            rr_val = _rr_simplified(inp.logP, inp.fup, fnl, fw, ka_factor, ct)
            ml_val = corrected[tissue]
            lo = min(ml_val, rr_val)
            hi = max(ml_val, rr_val)
            ratio = hi / max(lo, 1e-6)
            if ratio > 2.0:
                corrected[tissue] = round(float(rr_val), 4)
                warnings.append(
                    f"{tissue}: ML Kp {ml_val:.3f} diverges from RR {rr_val:.3f}; replaced."
                )
        return corrected, warnings

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X: NDArray,
        y: NDArray,
        epochs: int = 150,
        lr: float = 0.005,
        batch_size: int = 64,
    ) -> None:
        """Train ensemble with bootstrap resampling + backpropagation.

        Args:
            X: Feature matrix (n, 8) — log10/linear physicochemical params.
            y: Log10-Kp target matrix (n, 13).
            epochs: Training epochs per ensemble member.
            lr: SGD learning rate.
            batch_size: Mini-batch size.
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        n = len(X)

        self._x_mean = X.mean(axis=0).astype(np.float32)
        x_std = X.std(axis=0)
        self._x_std = np.where(x_std < 1e-8, 1.0, x_std).astype(np.float32)
        X_norm = ((X - self._x_mean) / self._x_std).astype(np.float32)

        self._y_mean = y.mean(axis=0).astype(np.float32)
        y_std = y.std(axis=0)
        self._y_std = np.where(y_std < 1e-8, 1.0, y_std).astype(np.float32)
        y_norm = ((y - self._y_mean) / self._y_std).astype(np.float32)

        rng = np.random.default_rng(self._seed)

        for mi, model in enumerate(self._models):
            boot = rng.integers(0, n, size=n)
            Xb, yb = X_norm[boot], y_norm[boot]
            model_rng = np.random.default_rng(self._seed + mi * 997)

            for epoch in range(epochs):
                perm = model_rng.permutation(n)
                epoch_loss = 0.0
                n_batches = 0
                for start in range(0, n, batch_size):
                    bidx = perm[start : start + batch_size]
                    Xmb, ymb = Xb[bidx], yb[bidx]
                    out, pre_acts, acts = model.forward_and_cache(Xmb)
                    grad = 2.0 * (out - ymb) / float(ymb.shape[0])
                    epoch_loss += float(np.mean((out - ymb) ** 2))
                    n_batches += 1
                    model.backward(pre_acts, acts, grad, lr)

                if epoch % 50 == 0:
                    avg = epoch_loss / max(n_batches, 1)
                    logger.debug(
                        "KpEnsemble model %d/%d epoch %d/%d loss=%.4f",
                        mi + 1,
                        _N_ENSEMBLE,
                        epoch + 1,
                        epochs,
                        avg,
                    )

        self._trained = True
        logger.info("KpEnsemble trained (%d models, %d samples).", _N_ENSEMBLE, n)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, inp: KpInput) -> KpOutput:
        """Predict tissue Kp for a single compound.

        Args:
            inp: KpInput specification.

        Returns:
            KpOutput with tissue_kp (13 tissues), confidence, warnings.

        Raises:
            RuntimeError: If ensemble has not been trained.
        """
        if not self._trained:
            raise RuntimeError("KpEnsemble has not been trained.")

        x_raw = _encode_features(inp).reshape(1, _N_FEATURES)
        x_norm = ((x_raw - self._x_mean) / self._x_std).astype(np.float32)

        preds: list[NDArray] = []
        for model in self._models:
            raw = model.forward(x_norm)  # (1, 13)
            log10_kp = self._decode(raw)[0]  # (13,)
            kp_vals = np.clip(10.0**log10_kp, 0.1, 100.0)
            preds.append(kp_vals)

        arr = np.stack(preds, axis=0)  # (3, 13)
        means = arr.mean(axis=0)
        stds = arr.std(axis=0)

        cv = float((stds / (np.abs(means) + 1e-8)).mean())
        if cv < 0.15:
            confidence = "high"
        elif cv < 0.40:
            confidence = "medium"
        else:
            confidence = "low"

        ml_kp = {tissue: float(round(means[i], 4)) for i, tissue in enumerate(TISSUES)}
        corrected, warn_list = self._validate_vs_rr(inp, ml_kp)

        # Final clamp after RR validation
        clamped = {t: float(np.clip(v, 0.1, 100.0)) for t, v in corrected.items()}

        return KpOutput(
            tissue_kp=clamped,
            confidence=confidence,
            warnings=tuple(warn_list),
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save ensemble weights and normalisation stats to .npz.

        Args:
            path: Destination file path.
        """
        path = Path(path)
        sd: dict[str, NDArray] = {
            "x_mean": self._x_mean,
            "x_std": self._x_std,
            "y_mean": self._y_mean,
            "y_std": self._y_std,
        }
        for mi, model in enumerate(self._models):
            for pi, arr in enumerate(model.get_params()):
                sd[f"m{mi}_p{pi}"] = arr.copy()
        np.savez(path, **sd)
        logger.info("KpEnsemble saved to %s", path)

    def load(self, path: str | Path) -> None:
        """Load ensemble weights and normalisation stats from .npz.

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
        logger.info("KpEnsemble loaded from %s", path)


# ---------------------------------------------------------------------------
# Singleton convenience function
# ---------------------------------------------------------------------------

_KP_SINGLETON: KpEnsemble | None = None


def predict_kp_ml(inp: KpInput) -> KpOutput:
    """Predict tissue Kp via singleton ensemble (auto-trains on first call).

    On first invocation, generates 3 000 LHS training samples and trains
    the KpEnsemble (≈ a few seconds on CPU).  Subsequent calls are fast.

    Args:
        inp: KpInput specification.

    Returns:
        KpOutput with 13-tissue Kp dict, confidence, and warnings.
    """
    global _KP_SINGLETON  # noqa: PLW0603
    if _KP_SINGLETON is None:
        logger.info(
            "KpEnsemble not initialised — generating training data "
            "and training now (this may take a moment)."
        )
        _KP_SINGLETON = KpEnsemble()
        X, y = generate_kp_training_data(n_samples=3000, seed=42)
        _KP_SINGLETON.train(X, y, epochs=150, lr=0.005, batch_size=64)
    return _KP_SINGLETON.predict(inp)
