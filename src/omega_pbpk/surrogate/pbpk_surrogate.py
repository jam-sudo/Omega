"""PBPK Property Surrogate — persistent drug-property → PK prediction model.

Bridges the gap between the PBPK training pipeline (surrogate/train.py) and
the end-to-end prediction pipeline (prediction/full_pipeline.py).

Architecture:
    PKSurrogate (6→64×3→4) trained on real 35-state PBPK ODE simulations.
    Inputs:  [logP, log(fup), log(clint_L_h), log(mw), log(peff), rbp]
    Outputs: [Cmax (mg/L), AUC (mg·h/L), Tmax (h), t_half (h)]

Model lifecycle:
    1. First call: generates n_samples PBPK simulations, trains, saves to cache.
    2. Subsequent calls: loads from cache instantly (< 1ms).
    3. Cache invalidated manually or via clear_cache().

Training AAFE is reported after training as a quality indicator.
Target: AAFE < 3.0 for the training set (model approximation, not validation).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Default cache location (relative to package root)
_DEFAULT_CACHE_DIR = Path(__file__).parent.parent.parent.parent / "models" / "pbpk_surrogate"

# Feature names in the order expected by the model
FEATURE_NAMES = ["logP", "fup", "clint_L_h", "mw", "rbp", "peff"]
OUTPUT_NAMES = ["cmax_mg_L", "auc_mg_h_L", "tmax_h", "t_half_h"]


@dataclass
class PBPKSurrogateOutput:
    """Prediction output from the PBPK property surrogate."""

    cmax_mg_L: float
    """Predicted peak plasma concentration (mg/L)."""

    auc_mg_h_L: float
    """Predicted AUC₀₋₂₄ (mg·h/L)."""

    tmax_h: float
    """Predicted time to peak (h)."""

    t_half_h: float
    """Predicted terminal half-life (h)."""

    source: str = "pbpk_surrogate"
    """'pbpk_surrogate' when using this model, 'fallback' when using defaults."""


@dataclass
class TrainingReport:
    """Summary of surrogate training quality."""

    n_samples: int
    n_successful: int
    aafe_cmax: float
    aafe_auc: float
    aafe_tmax: float
    train_loss: float
    cache_dir: str

    @property
    def mean_aafe(self) -> float:
        return float(np.mean([self.aafe_cmax, self.aafe_auc]))

    def __str__(self) -> str:
        return (
            f"PBPKSurrogate training: {self.n_successful}/{self.n_samples} samples, "
            f"AAFE Cmax={self.aafe_cmax:.2f}, AUC={self.aafe_auc:.2f}, "
            f"Tmax={self.aafe_tmax:.2f}, mean={self.mean_aafe:.2f}"
        )


def _compute_aafe(y_true: np.ndarray, y_pred: np.ndarray) -> list[float]:
    """Compute per-output AAFE (Average Absolute Fold Error)."""
    aafes = []
    for i in range(y_true.shape[1]):
        mask = y_true[:, i] > 1e-6
        if mask.sum() < 2:
            aafes.append(float("nan"))
            continue
        log10_fe = np.abs(np.log10(np.maximum(y_pred[mask, i], 1e-9) / y_true[mask, i]))
        aafes.append(float(10.0 ** np.mean(log10_fe)))
    return aafes


def _featurize(
    logP: float,
    fup: float,
    clint_L_h: float,
    mw: float,
    peff: float,
    rbp: float,
) -> np.ndarray:
    """Convert drug properties to feature vector."""
    return np.array([logP, fup, clint_L_h, mw, rbp, peff], dtype=np.float64)


def train_and_cache(
    cache_dir: Path | str | None = None,
    n_samples: int = 200,
    epochs: int = 300,
    seed: int = 42,
    noise_cv: float = 0.3,
    force_retrain: bool = False,
) -> tuple[Any, TrainingReport]:
    """Generate PBPK training data, train surrogate, save to cache.

    Args:
        cache_dir: Directory to save trained model. Defaults to ``models/pbpk_surrogate/``.
        n_samples: Number of PBPK simulations to run for training data.
        epochs: Adam optimizer epochs.
        seed: Random seed.
        noise_cv: Log-normal noise coefficient of variation for augmentation.
        force_retrain: Retrain even if a cached model exists.

    Returns:
        (trained_model, TrainingReport)
    """
    from omega_pbpk.surrogate import PKSurrogate
    from omega_pbpk.surrogate.train import build_training_dataset

    cache_path = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR

    logger.info("Generating PBPK training data: %d samples...", n_samples)
    X, y = build_training_dataset(n_samples=n_samples, seed=seed, noise_cv=noise_cv)
    n_successful = len(X)

    if n_successful < 10:
        raise RuntimeError(
            f"Too few successful PBPK simulations ({n_successful}/{n_samples}). "
            "Check the drug parameter ranges in surrogate/train.py."
        )

    # 80/20 train/val split for AAFE evaluation
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_successful)
    n_train = int(0.8 * n_successful)
    X_train, y_train = X[perm[:n_train]], y[perm[:n_train]]
    X_val, y_val = X[perm[n_train:]], y[perm[n_train:]]

    model = PKSurrogate(n_input=6, n_output=4)
    model.param_names = FEATURE_NAMES
    model.output_names = OUTPUT_NAMES

    logger.info(
        "Training PBPKSurrogate: %d train, %d val, %d epochs...", n_train, len(X_val), epochs
    )
    history = model.train(X_train, y_train, epochs=epochs, lr=0.001, batch_size=32, seed=seed)
    train_loss = history["val_loss"][-1] if history["val_loss"] else float("inf")

    # AAFE on validation set
    y_val_pred = model.predict(X_val)
    aafes = _compute_aafe(y_val, y_val_pred)

    report = TrainingReport(
        n_samples=n_samples,
        n_successful=n_successful,
        aafe_cmax=aafes[0] if not np.isnan(aafes[0]) else 99.0,
        aafe_auc=aafes[1] if not np.isnan(aafes[1]) else 99.0,
        aafe_tmax=aafes[2] if not np.isnan(aafes[2]) else 99.0,
        train_loss=train_loss,
        cache_dir=str(cache_path),
    )
    logger.info("%s", report)

    model.save(cache_path)
    logger.info("PBPKSurrogate saved to %s", cache_path)

    return model, report


def load_or_train(
    cache_dir: Path | str | None = None,
    n_samples: int = 200,
    epochs: int = 300,
    seed: int = 42,
) -> Any:
    """Load a cached PBPKSurrogate, or train one if no cache exists.

    Args:
        cache_dir: Cache directory. Defaults to ``models/pbpk_surrogate/``.
        n_samples: Samples to generate if training from scratch.
        epochs: Training epochs if training from scratch.
        seed: Random seed.

    Returns:
        Trained PKSurrogate model.
    """
    from omega_pbpk.surrogate import PKSurrogate

    cache_path = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
    meta_file = cache_path / "meta.json"

    if meta_file.exists():
        try:
            model = PKSurrogate.load(cache_path)
            logger.info("PBPKSurrogate loaded from cache: %s", cache_path)
            return model
        except Exception as exc:
            logger.warning("Failed to load cached surrogate (%s); retraining...", exc)

    model, _ = train_and_cache(
        cache_dir=cache_path,
        n_samples=n_samples,
        epochs=epochs,
        seed=seed,
    )
    return model


def predict_from_adme_dict(
    adme: dict[str, Any],
    dose_mg: float = 100.0,
    route: str = "oral",
    model: Any = None,
    cache_dir: Path | str | None = None,
) -> PBPKSurrogateOutput:
    """Predict PK from an ADME property dictionary.

    This is the primary interface for the full prediction pipeline.

    Args:
        adme: Dict with keys: mw, logP, fup, rbp, peff, clint_3a4 (at minimum).
        dose_mg: Dose in mg (used to derive CLint if needed).
        route: 'oral' or 'iv'.
        model: Pre-loaded PKSurrogate (avoids reload overhead). If None, uses
               cached model from cache_dir.
        cache_dir: Cache directory for model weights.

    Returns:
        PBPKSurrogateOutput with Cmax, AUC, Tmax, t_half predictions.
    """
    try:
        if model is None:
            model = load_or_train(cache_dir=cache_dir, n_samples=200, epochs=200)

        # Extract features with defaults
        logP = float(adme.get("logP", 2.0))
        fup = float(np.clip(adme.get("fup", 0.5), 0.001, 1.0))
        # CLint: prefer hepatic_L_h, fall back to clint_3a4 with IVIVE scaling
        # IVIVE: µL/min/pmol × MPPGL(40) × mg/g(45) × liver(1800g) / 1e6 / 60 ≈ ×0.054
        _ivive_factor = 40.0 * 45.0 * 1800.0 / 1e6 / 60.0  # ~0.054
        clint = float(adme.get("clint_hepatic_L_h", adme.get("clint_3a4", 5.0) * _ivive_factor))
        clint = max(clint, 0.01)
        mw = float(adme.get("mw", 300.0))
        peff = float(adme.get("peff", 1.0))
        rbp = float(adme.get("rbp", 1.0))

        x = _featurize(logP, fup, clint, mw, peff, rbp)
        y = model.predict(x)

        return PBPKSurrogateOutput(
            cmax_mg_L=max(float(y[0]), 0.0),
            auc_mg_h_L=max(float(y[1]), 0.0),
            tmax_h=max(float(y[2]), 0.0),
            t_half_h=max(float(y[3]), 0.0),
            source="pbpk_surrogate",
        )

    except Exception as exc:
        logger.warning("PBPKSurrogate prediction failed: %s; returning defaults", exc)
        # Simple 1-cpt analytical fallback
        fup_safe = float(np.clip(adme.get("fup", 0.5), 0.001, 1.0))
        clint_fallback = float(adme.get("clint_3a4", 5.0) * 40.0 * 45.0 * 1800.0 / 1e6 / 60.0)
        cl_approx = fup_safe * max(clint_fallback, 0.01)
        vd_approx = float(adme.get("mw", 300.0)) / 15.0 + 5.0  # rough allometric
        cmax = (dose_mg * fup_safe) / vd_approx if route == "oral" else dose_mg / vd_approx
        auc = dose_mg / max(cl_approx, 0.01)
        return PBPKSurrogateOutput(
            cmax_mg_L=float(np.clip(cmax, 0.001, 1000.0)),
            auc_mg_h_L=float(np.clip(auc, 0.001, 10000.0)),
            tmax_h=1.0 if route == "oral" else 0.0,
            t_half_h=float(np.log(2) * vd_approx / max(cl_approx, 0.01)),
            source="fallback",
        )


def clear_cache(cache_dir: Path | str | None = None) -> None:
    """Delete cached model files to force retraining on next call."""
    import shutil

    cache_path = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
    if cache_path.exists():
        shutil.rmtree(cache_path)
        logger.info("PBPKSurrogate cache cleared: %s", cache_path)
    else:
        logger.info("No cache to clear at %s", cache_path)


__all__ = [
    "PBPKSurrogateOutput",
    "TrainingReport",
    "train_and_cache",
    "load_or_train",
    "predict_from_adme_dict",
    "clear_cache",
    "FEATURE_NAMES",
    "OUTPUT_NAMES",
]
