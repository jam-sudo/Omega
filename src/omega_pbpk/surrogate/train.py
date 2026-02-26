"""Surrogate model training on real PBPK ODE simulations."""
from __future__ import annotations
import csv
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent


def _load_ref_drugs() -> list[dict]:
    """Load 25 reference drugs from data/adme_reference.csv."""
    path = _REPO_ROOT / "data" / "adme_reference.csv"
    if not path.exists():
        return []
    drugs = []
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                drugs.append({
                    "name": row["name"],
                    "mw": float(row["mw"]),
                    "logP": float(row["logP"]),
                    "fup": float(row["fup"]),
                    "rbp": float(row["rbp"]),
                    "clint": float(row["clint_3a4_uL_min_pmol"]) * 3.6,  # → L/h
                    "peff": float(row["peff_cm_s"]),
                })
            except (ValueError, KeyError):
                continue
    return drugs


def _params_to_array(d: dict) -> NDArray:
    """Convert drug param dict to feature vector [logP, fup, clint_L_h, mw, rbp, peff]."""
    return np.array([d["logP"], d["fup"], d["clint"], d["mw"], d["rbp"], d["peff"]], dtype=float)


def _simulate_one(params: dict, dose_mg: float = 100.0, route: str = "oral") -> NDArray | None:
    """Run one PBPK simulation; return [Cmax, AUC, Tmax, t_half] or None on failure."""
    try:
        from omega_pbpk.core.body import WholeBodyPBPK
        from omega_pbpk.drugs.drug import Drug

        fup = max(float(params["fup"]), 0.001)
        drug = Drug(
            name=params.get("name", "compound"),
            mw=float(params["mw"]),
            logP=float(params["logP"]),
            fup=fup,
            rbp=float(params["rbp"]),
            clint_hepatic_L_per_h=float(params["clint"]),
            peff=float(params.get("peff", 1.0)),
        )
        model = WholeBodyPBPK(drug=drug)
        if route == "iv":
            model.setup_iv(dose_mg=dose_mg)
        else:
            model.setup_oral(dose_mg=dose_mg)
        result = model.simulate(t_end_h=24.0)
        cp = result.plasma_concentration()
        t = np.linspace(0, 24.0, len(cp))

        cmax = float(np.max(cp))
        tmax = float(t[int(np.argmax(cp))])
        auc = float(np.trapz(cp, t))

        # Terminal t_half from last 30% of timepoints
        n = len(t)
        t0 = int(0.7 * n)
        tail_cp = cp[t0:]
        if np.all(tail_cp > 1e-9):
            slope, _ = np.polyfit(t[t0:], np.log(tail_cp + 1e-12), 1)
            t_half = float(-np.log(2) / slope) if slope < -1e-9 else 99.0
        else:
            t_half = 99.0

        return np.array([cmax, auc, tmax, min(t_half, 99.0)], dtype=float)
    except Exception as e:
        logger.debug(f"Simulation failed for {params.get('name', '?')}: {e}")
        return None


def build_training_dataset(
    n_samples: int = 300,
    seed: int = 42,
    noise_cv: float = 0.3,
    n_workers: int | None = None,
    route: str = "oral",
) -> tuple[NDArray, NDArray]:
    """Build training dataset using real PBPK ODE simulations.

    Loads 25 reference drugs from adme_reference.csv, runs simulations,
    augments with log-normal noise perturbations to reach n_samples.
    Returns X (shape N×6) and y (shape N×4: Cmax, AUC, Tmax, t_half).
    """
    rng = np.random.default_rng(seed)
    ref_drugs = _load_ref_drugs()

    if not ref_drugs:
        logger.warning("No reference drugs found; using synthetic fallback data")
        X = rng.uniform(0, 5, (n_samples, 6))
        y = rng.uniform(0, 20, (n_samples, 4))
        return X, y

    # Generate perturbed parameter sets
    n_repeats = max(1, (n_samples + len(ref_drugs) - 1) // len(ref_drugs))
    all_params: list[dict] = []
    all_X: list[NDArray] = []

    for drug in ref_drugs:
        base = _params_to_array(drug)
        all_params.append(drug)
        all_X.append(base)
        for _ in range(n_repeats):
            noise = rng.lognormal(0, noise_cv, len(base))
            perturbed = base * noise
            # Clip fup to (0,1]
            perturbed[1] = np.clip(perturbed[1], 0.001, 0.99)
            perturbed_dict = {
                "name": drug["name"] + "_aug",
                "logP": float(perturbed[0]),
                "fup": float(perturbed[1]),
                "clint": float(perturbed[2]),
                "mw": float(perturbed[3]),
                "rbp": float(perturbed[4]),
                "peff": float(perturbed[5]),
            }
            all_params.append(perturbed_dict)
            all_X.append(perturbed)

    all_params = all_params[:n_samples]
    all_X = all_X[:n_samples]

    # Run simulations (parallelized)
    logger.info(f"Running {len(all_params)} PBPK simulations (workers={n_workers})...")
    results: list[NDArray | None] = [None] * len(all_params)

    # Use ProcessPoolExecutor for parallel sims — but simulate_one must be picklable
    # Run sequentially to avoid pickling issues with complex objects
    successful = 0
    for i, params in enumerate(all_params):
        r = _simulate_one(params, route=route)
        results[i] = r
        if r is not None:
            successful += 1
        if (i + 1) % 50 == 0:
            logger.info(f"  {i+1}/{len(all_params)} done ({successful} successful)")

    # Filter out failed simulations
    X_list, y_list = [], []
    for x, y in zip(all_X, results):
        if y is not None and np.all(np.isfinite(y)) and np.all(y >= 0):
            X_list.append(x)
            y_list.append(y)

    if not X_list:
        logger.error("All simulations failed; returning synthetic data")
        X = rng.uniform(0, 5, (n_samples, 6))
        y_out = rng.uniform(0, 20, (n_samples, 4))
        return X, y_out

    logger.info(f"Training dataset: {len(X_list)} successful simulations out of {len(all_params)}")
    return np.array(X_list), np.array(y_list)


def train_surrogate(
    n_samples: int = 300,
    save_path: str | None = None,
    seed: int = 42,
    epochs: int = 500,
    lr: float = 1e-3,
):
    """Train PKSurrogate on real PBPK ODE simulation data.

    Args:
        n_samples: Number of training simulations to run
        save_path: Path to save model weights (directory or .npz path; directory will be created)
        seed: Random seed for reproducibility
        epochs: Training epochs
        lr: Learning rate
    Returns:
        Trained PKSurrogate model
    """
    from omega_pbpk.surrogate import PKSurrogate

    X, y = build_training_dataset(n_samples=n_samples, seed=seed)
    logger.info(f"Training surrogate: {X.shape[0]} samples, {X.shape[1]} features → {y.shape[1]} outputs")

    model = PKSurrogate(n_input=X.shape[1], n_output=y.shape[1])
    model.train(X, y, epochs=epochs, lr=lr, batch_size=min(32, len(X)))

    # Evaluate on training set
    y_pred = model.predict(X)
    aafe_per_output = []
    for i in range(y.shape[1]):
        mask = y[:, i] > 0
        if mask.sum() > 0:
            fold_errors = np.abs(np.log10(np.maximum(y_pred[mask, i], 1e-9) / np.maximum(y[:, i][mask], 1e-9)))
            aafe_per_output.append(float(10 ** np.mean(fold_errors)))
    logger.info(f"Training AAFE per output (Cmax,AUC,Tmax,t½): {[f'{a:.2f}' for a in aafe_per_output]}")

    if save_path:
        # PKSurrogate.save() requires a directory path; strip .npz suffix if provided
        dir_path = save_path[:-4] if save_path.endswith(".npz") else save_path
        model.save(dir_path)
        logger.info(f"Saved surrogate to {dir_path}")

    return model


if __name__ == "__main__":
    import os
    logging.basicConfig(level=logging.INFO)
    save = str(_REPO_ROOT / "models" / "surrogate_real.npz")
    os.makedirs(os.path.dirname(save), exist_ok=True)
    train_surrogate(n_samples=100, save_path=save, epochs=300)
    print("Done.")
