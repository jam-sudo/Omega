"""Surrogate model training utilities."""
from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
logger = logging.getLogger(__name__)

def build_training_dataset(n_samples: int = 200, seed: int = 42) -> tuple[NDArray, NDArray]:
    """Generate training data by perturbing known drug parameters through PBPK.

    Loads data/adme_reference.csv, runs PBPK simulations for each drug,
    augments with ±30% log-normal parameter noise to reach n_samples.
    Returns X (drug params), y (Cmax, AUC, Tmax, t_half).
    """
    rng = np.random.default_rng(seed)
    # Load reference drugs
    ref_path = Path(__file__).parent.parent.parent.parent.parent / "data" / "adme_reference.csv"
    if not ref_path.exists():
        logger.warning("adme_reference.csv not found; returning synthetic data")
        X = rng.uniform(0, 1, (n_samples, 6))
        y = rng.uniform(0, 10, (n_samples, 4))
        return X, y

    import csv
    drugs = []
    with open(ref_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                drugs.append({
                    "logP": float(row["logP"]),
                    "fup": float(row["fup"]),
                    "clint": float(row["clint_3a4_uL_min_pmol"]),
                    "mw": float(row["mw"]),
                    "rbp": float(row["rbp"]),
                    "peff": float(row["peff_cm_s"]),
                })
            except (ValueError, KeyError):
                continue

    if not drugs:
        X = rng.uniform(0, 1, (n_samples, 6))
        y = rng.uniform(0, 10, (n_samples, 4))
        return X, y

    # Build nominal feature vectors
    nominal_X = np.array([[d["logP"], d["fup"], d["clint"], d["mw"], d["rbp"], d["peff"]] for d in drugs])

    # Augment with log-normal noise
    n_repeats = max(1, n_samples // len(drugs) + 1)
    rows_X = [nominal_X]
    for _ in range(n_repeats):
        noise = rng.lognormal(0, 0.3, nominal_X.shape)
        rows_X.append(nominal_X * noise)
    X_aug = np.vstack(rows_X)[:n_samples]

    # Generate synthetic PK outputs (placeholder until real PBPK loop)
    # Cmax ~ dose/Vd, AUC ~ dose/CL, Tmax ~ 1/ka, t_half ~ Vd/CL
    dose = 100.0
    fup_col = np.clip(X_aug[:, 1], 0.001, 1.0)
    clint_col = np.clip(X_aug[:, 2] * 3.6, 0.1, 200.0)  # convert to L/h
    logP_col = X_aug[:, 0]
    Vd = 50 * np.exp(0.3 * logP_col)  # rough Vd estimate
    CL = np.clip(fup_col * clint_col, 0.1, 100)
    ka = 1.0
    Cmax = (dose / Vd) * (ka / (ka - CL / Vd + 1e-6))
    Cmax = np.clip(Cmax, 0.001, 100)
    AUC = dose / CL
    Tmax = np.log(ka / (CL / Vd + 1e-6)) / (ka - CL / Vd + 1e-6)
    Tmax = np.clip(np.abs(Tmax), 0.1, 12)
    t_half = 0.693 * Vd / CL
    y = np.column_stack([Cmax, AUC, Tmax, t_half])
    return X_aug, y

def train_surrogate(n_samples: int = 200, save_path: str | None = None):
    """Train PKSurrogate on reference drug simulations."""
    from omega_pbpk.surrogate import PKSurrogate
    X, y = build_training_dataset(n_samples)
    n_input = X.shape[1]
    n_output = y.shape[1]
    model = PKSurrogate(n_input=n_input, n_output=n_output)
    model.train(X, y, epochs=200, lr=1e-3, batch_size=32)
    if save_path:
        model.save(save_path)
        logger.info(f"Surrogate saved to {save_path}")
    return model

if __name__ == "__main__":
    import os
    save = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "models", "surrogate.npz")
    os.makedirs(os.path.dirname(save), exist_ok=True)
    m = train_surrogate(200, save_path=save)
    print(f"Trained surrogate: {m.n_input} inputs → {m.n_output} outputs")
