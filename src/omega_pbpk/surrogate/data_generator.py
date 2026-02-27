"""PBPK parameter-space sampler for surrogate training data generation.

Generates diverse (drug_params, PK_output) pairs by sweeping the compound
parameter space and running the mechanistic PBPK solver. The resulting
dataset trains the neural surrogate.

Each simulation is independent and is dispatched to a worker process via
ProcessPoolExecutor for multi-core speed-up.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from omega_pbpk.core.body import WholeBodyPBPK
from omega_pbpk.drugs.drug import Drug

logger = logging.getLogger(__name__)

# Parameters to sweep and their (log-space) ranges
PARAM_RANGES: dict[str, tuple[float, float]] = {
    "clint_hepatic_L_per_h": (0.5, 200.0),
    "clint_gut_L_per_h": (0.0, 50.0),
    "fup": (0.005, 1.0),
    "rbp": (0.4, 2.0),
    "peff": (0.1, 50.0),
    "logP": (-1.0, 5.0),
}

# PK output feature names
PK_OUTPUT_NAMES = ["Cmax_mg_L", "AUC_mg_h_L", "Tmax_h", "half_life_h"]


@dataclass
class TrainingDataset:
    """Training dataset for surrogate model.

    Attributes:
        X: Input features, shape (n_samples, n_params).
        y: Output targets, shape (n_samples, n_outputs).
        param_names: Names of input parameters.
        output_names: Names of output PK metrics.
        dose_mg: Dose used for all samples.
        route: Administration route.
    """

    X: NDArray[np.float64]
    y: NDArray[np.float64]
    param_names: list[str] = field(default_factory=list)
    output_names: list[str] = field(default_factory=list)
    dose_mg: float = 0.0
    route: str = "oral"

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    @property
    def n_params(self) -> int:
        return self.X.shape[1]

    @property
    def n_outputs(self) -> int:
        return self.y.shape[1]


def _data_gen_worker(
    args: tuple[dict[str, float], float, str, float, float, int],
) -> tuple[int, list[float] | None]:
    """Worker: run one PBPK simulation for training data generation.

    Args:
        args: Tuple of (params, dose_mg, route, body_weight, t_end_h, index).

    Returns:
        (index, pk_row) where pk_row is None on failure.
    """
    params, dose_mg, route, body_weight, t_end_h, idx = args
    try:
        pk = _run_single_simulation(params, dose_mg, route, body_weight, t_end_h)
        return idx, [pk[k] for k in PK_OUTPUT_NAMES]
    except Exception as e:
        logger.debug("Sample %d failed: %s", idx, e)
        return idx, None


def generate_training_data(
    n_samples: int = 500,
    dose_mg: float = 10.0,
    route: str = "oral",
    body_weight: float = 70.0,
    t_end_h: float = 24.0,
    seed: int = 42,
    param_ranges: dict[str, tuple[float, float]] | None = None,
    n_workers: int | None = None,
) -> TrainingDataset:
    """Generate training data by sweeping PBPK parameter space.

    Uses Latin Hypercube Sampling (LHS) for efficient space coverage.
    Simulations are independent and run in parallel via ProcessPoolExecutor.

    Args:
        n_samples: Number of parameter combinations to sample.
        dose_mg: Dose (mg).
        route: 'oral' or 'iv'.
        body_weight: Subject body weight (kg).
        t_end_h: Simulation end time (h).
        seed: Random seed.
        param_ranges: Optional custom parameter ranges.
        n_workers: Number of worker processes.  None → os.cpu_count().
            Pass 1 to disable multiprocessing.

    Returns:
        TrainingDataset with input parameters and PK output targets.
    """
    if n_workers is None:
        n_workers = os.cpu_count() or 1

    rng = np.random.default_rng(seed)
    ranges = param_ranges or PARAM_RANGES
    param_names = list(ranges.keys())
    n_params = len(param_names)

    # Latin Hypercube Sampling
    X = _latin_hypercube(n_samples, n_params, rng)

    # Scale to parameter ranges
    for j, name in enumerate(param_names):
        lo, hi = ranges[name]
        if name in ("clint_hepatic_L_per_h", "clint_gut_L_per_h", "fup", "peff"):
            # Log-uniform sampling for positive parameters
            lo_safe = max(lo, 1e-6)
            X[:, j] = np.exp(np.log(lo_safe) + X[:, j] * (np.log(hi) - np.log(lo_safe)))
        else:
            # Uniform sampling
            X[:, j] = lo + X[:, j] * (hi - lo)

    # Build worker argument list (index preserved for result ordering)
    all_params = [
        {name: float(X[i, j]) for j, name in enumerate(param_names)} for i in range(n_samples)
    ]
    worker_args = [
        (params, dose_mg, route, body_weight, t_end_h, i)
        for i, params in enumerate(all_params)
    ]

    # Run simulations — parallel or serial
    pk_rows: list[list[float] | None] = [None] * n_samples
    valid_mask: list[bool] = [False] * n_samples

    if n_workers == 1 or n_samples <= 1:
        # Single-process path
        for args in worker_args:
            idx, row = _data_gen_worker(args)
            if row is not None:
                pk_rows[idx] = row
                valid_mask[idx] = True
            if (idx + 1) % 50 == 0:
                logger.info("Generated %d/%d training samples", idx + 1, n_samples)
    else:
        max_workers = min(n_workers, n_samples)
        completed = 0
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            future_to_idx = {pool.submit(_data_gen_worker, args): args[-1] for args in worker_args}
            for future in as_completed(future_to_idx):
                idx, row = future.result()
                if row is not None:
                    pk_rows[idx] = row
                    valid_mask[idx] = True
                completed += 1
                if completed % 50 == 0:
                    logger.info("Generated %d/%d training samples", completed, n_samples)

    mask = np.array(valid_mask)
    X_valid = X[mask]
    y_valid = np.array([r for r in pk_rows if r is not None], dtype=np.float64)

    logger.info(
        "Training data: %d/%d valid samples (%d params → %d outputs)",
        X_valid.shape[0],
        n_samples,
        n_params,
        len(PK_OUTPUT_NAMES),
    )

    return TrainingDataset(
        X=X_valid,
        y=y_valid,
        param_names=param_names,
        output_names=list(PK_OUTPUT_NAMES),
        dose_mg=dose_mg,
        route=route,
    )


def _latin_hypercube(n: int, d: int, rng: np.random.Generator) -> NDArray[np.float64]:
    """Generate Latin Hypercube Samples in [0, 1]^d."""
    result = np.zeros((n, d))
    for j in range(d):
        perm = rng.permutation(n)
        result[:, j] = (perm + rng.uniform(size=n)) / n
    return result


def _run_single_simulation(
    params: dict[str, float],
    dose_mg: float,
    route: str,
    body_weight: float,
    t_end_h: float,
) -> dict[str, float]:
    """Run one PBPK simulation with given parameters, return PK summary."""
    drug = Drug(
        name="surrogate_sample",
        logP=params.get("logP", 2.0),
        fup=params.get("fup", 0.5),
        rbp=params.get("rbp", 1.0),
        peff=params.get("peff", 1.0),
        clint_hepatic_L_per_h=params.get("clint_hepatic_L_per_h", 10.0),
        clint_gut_L_per_h=params.get("clint_gut_L_per_h", 0.0),
    )

    model = WholeBodyPBPK(drug, body_weight=body_weight)
    if route == "iv":
        model.setup_iv(dose_mg)
    else:
        model.setup_oral(dose_mg)

    result = model.simulate(t_end_h=t_end_h, dt_h=0.1)
    pk = result.pk_summary()

    # Clamp infinite half-life
    if pk["half_life_h"] == float("inf") or pk["half_life_h"] > 1e6:
        pk["half_life_h"] = t_end_h * 10.0

    return pk


__all__ = ["TrainingDataset", "generate_training_data", "PARAM_RANGES", "PK_OUTPUT_NAMES"]
