"""PBPK parameter-space sampler for surrogate training data generation.

Generates diverse (drug_params, PK_output) pairs by sweeping the compound
parameter space and running the mechanistic PBPK solver. The resulting
dataset trains the neural surrogate.
"""

from __future__ import annotations

import logging
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


def generate_training_data(
    n_samples: int = 500,
    dose_mg: float = 10.0,
    route: str = "oral",
    body_weight: float = 70.0,
    t_end_h: float = 24.0,
    seed: int = 42,
    param_ranges: dict[str, tuple[float, float]] | None = None,
) -> TrainingDataset:
    """Generate training data by sweeping PBPK parameter space.

    Uses Latin Hypercube Sampling (LHS) for efficient space coverage.

    Args:
        n_samples: Number of parameter combinations to sample.
        dose_mg: Dose (mg).
        route: 'oral' or 'iv'.
        body_weight: Subject body weight (kg).
        t_end_h: Simulation end time (h).
        seed: Random seed.
        param_ranges: Optional custom parameter ranges.

    Returns:
        TrainingDataset with input parameters and PK output targets.
    """
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

    # Run simulations
    y_list: list[list[float]] = []
    valid_mask: list[bool] = []

    for i in range(n_samples):
        params = {name: float(X[i, j]) for j, name in enumerate(param_names)}
        try:
            pk = _run_single_simulation(params, dose_mg, route, body_weight, t_end_h)
            y_list.append([pk[k] for k in PK_OUTPUT_NAMES])
            valid_mask.append(True)
        except Exception as e:
            logger.debug("Sample %d failed: %s", i, e)
            valid_mask.append(False)

        if (i + 1) % 50 == 0:
            logger.info("Generated %d/%d training samples", i + 1, n_samples)

    mask = np.array(valid_mask)
    X_valid = X[mask]
    y_valid = np.array(y_list, dtype=np.float64)

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
