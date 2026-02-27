"""Monte Carlo uncertainty propagation for PBPK parameters.

Propagates uncertainty in drug compound parameters (CLint, fup, etc.)
through the PBPK model using either the mechanistic solver or the fast
neural surrogate. Computes distributional PK metrics (CV, percentile bands).

Monte Carlo samples are drawn up-front in the main process (preserving
RNG reproducibility), then each mechanistic simulation is dispatched to a
worker process via ProcessPoolExecutor for multi-core speed-up.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DistributionSpec:
    """Parameter uncertainty specification.

    Attributes:
        param: Parameter name (must match Drug attribute).
        distribution: 'normal' or 'lognormal'.
        mean: Central value.
        cv: Coefficient of variation (SD/mean).
    """

    param: str
    distribution: str  # 'normal' or 'lognormal'
    mean: float
    cv: float


@dataclass(frozen=True)
class UncertaintyResult:
    """Monte Carlo uncertainty propagation results.

    Attributes:
        n_samples: Number of MC samples.
        cmax_values: Array of Cmax from each sample.
        auc_values: Array of AUC from each sample.
        sampled_params: Dict of parameter name → sampled values array.
        cmax_cv: Coefficient of variation of Cmax.
        auc_cv: Coefficient of variation of AUC.
        cmax_percentiles: Dict with p5, p25, p50, p75, p95.
        auc_percentiles: Dict with p5, p25, p50, p75, p95.
    """

    n_samples: int
    cmax_values: NDArray[np.float64]
    auc_values: NDArray[np.float64]
    sampled_params: dict[str, NDArray[np.float64]]
    cmax_cv: float
    auc_cv: float
    cmax_percentiles: dict[str, float] = field(default_factory=dict)
    auc_percentiles: dict[str, float] = field(default_factory=dict)


def _draw_parameter(spec: DistributionSpec, rng: np.random.Generator) -> float:
    """Draw a single sample from a parameter's uncertainty distribution."""
    if spec.distribution == "lognormal":
        sigma = np.sqrt(np.log1p(spec.cv**2))
        mu = np.log(spec.mean) - 0.5 * sigma**2
        return float(rng.lognormal(mu, sigma))
    else:
        # Normal, clamp to positive
        sd = spec.mean * spec.cv
        return float(max(rng.normal(spec.mean, sd), 1e-12))


def _percentiles(arr: NDArray[np.float64]) -> dict[str, float]:
    """Compute standard percentiles."""
    ps = [5, 25, 50, 75, 95]
    vals = np.percentile(arr, ps)
    return {f"p{p}": round(float(v), 6) for p, v in zip(ps, vals, strict=True)}


def _mc_worker(
    args: tuple[dict[str, float], float, str, float, float],
) -> dict[str, float] | None:
    """Worker: run one mechanistic PBPK simulation for Monte Carlo.

    Args:
        args: Tuple of (params, dose_mg, route, body_weight, t_end_h).

    Returns:
        PK summary dict or None on failure.
    """
    params, dose_mg, route, body_weight, t_end_h = args
    try:
        return _run_mechanistic(params, dose_mg, route, body_weight, t_end_h)
    except Exception:
        return None


def monte_carlo_propagation(
    drug_params: dict[str, float],
    uncertainty_specs: list[DistributionSpec],
    n_samples: int = 500,
    dose_mg: float = 10.0,
    route: str = "oral",
    body_weight: float = 70.0,
    t_end_h: float = 24.0,
    seed: int = 42,
    use_surrogate: Any | None = None,
    n_workers: int | None = None,
) -> UncertaintyResult:
    """Run Monte Carlo uncertainty propagation.

    All random samples are drawn up-front in the main process so that the
    results are identical regardless of the number of workers (reproducible).
    When use_surrogate is None, each mechanistic PBPK simulation is
    dispatched to a worker process via ProcessPoolExecutor.

    Args:
        drug_params: Baseline drug parameter dict.
        uncertainty_specs: List of parameter uncertainty specifications.
        n_samples: Number of MC samples.
        dose_mg: Dose (mg).
        route: 'oral' or 'iv'.
        body_weight: Subject body weight (kg).
        t_end_h: Simulation end time (h).
        seed: Random seed for reproducibility.
        use_surrogate: Optional PKSurrogate model for fast evaluation.
            When provided, parallelism is not applied (surrogate is already
            fast and is not picklable in general).
        n_workers: Number of worker processes for mechanistic path.
            None → os.cpu_count().  Pass 1 to disable multiprocessing.

    Returns:
        UncertaintyResult with distributional PK metrics.
    """
    if n_workers is None:
        n_workers = os.cpu_count() or 1

    rng = np.random.default_rng(seed)

    # --- Draw all samples up-front (preserves reproducibility) ---
    sampled: dict[str, list[float]] = {s.param: [] for s in uncertainty_specs}
    all_params: list[dict[str, float]] = []

    for _i in range(n_samples):
        params = dict(drug_params)
        for spec in uncertainty_specs:
            val = _draw_parameter(spec, rng)
            params[spec.param] = val
            sampled[spec.param].append(val)
        all_params.append(params)

    # --- Run simulations ---
    cmax_list: list[float] = []
    auc_list: list[float] = []

    if use_surrogate is not None:
        # Surrogate path: fast, no multiprocessing needed
        for params in all_params:
            try:
                pk = use_surrogate.predict_dict(params)
                cmax_list.append(pk["Cmax_mg_L"])
                auc_list.append(pk["AUC_mg_h_L"])
            except Exception:
                continue
    elif n_workers == 1 or n_samples <= 1:
        # Single-process mechanistic path
        for params in all_params:
            try:
                pk = _run_mechanistic(params, dose_mg, route, body_weight, t_end_h)
                cmax_list.append(pk["Cmax_mg_L"])
                auc_list.append(pk["AUC_mg_h_L"])
            except Exception:
                continue
    else:
        # Parallel mechanistic path — dispatch each sample to a worker
        worker_args = [(params, dose_mg, route, body_weight, t_end_h) for params in all_params]
        max_workers = min(n_workers, n_samples)
        # Collect results preserving insertion order to match sampled_params
        pk_results: list[dict[str, float] | None] = [None] * n_samples
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            future_to_idx = {pool.submit(_mc_worker, args): i for i, args in enumerate(worker_args)}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                pk_results[idx] = future.result()

        for pk in pk_results:
            if pk is not None:
                cmax_list.append(pk["Cmax_mg_L"])
                auc_list.append(pk["AUC_mg_h_L"])

    cmax_arr = np.array(cmax_list, dtype=np.float64)
    auc_arr = np.array(auc_list, dtype=np.float64)

    cmax_cv = float(np.std(cmax_arr) / max(np.mean(cmax_arr), 1e-12))
    auc_cv = float(np.std(auc_arr) / max(np.mean(auc_arr), 1e-12))

    return UncertaintyResult(
        n_samples=len(cmax_list),
        cmax_values=cmax_arr,
        auc_values=auc_arr,
        sampled_params={k: np.array(v) for k, v in sampled.items()},
        cmax_cv=round(cmax_cv, 4),
        auc_cv=round(auc_cv, 4),
        cmax_percentiles=_percentiles(cmax_arr),
        auc_percentiles=_percentiles(auc_arr),
    )


def _run_mechanistic(
    params: dict[str, float],
    dose_mg: float,
    route: str,
    body_weight: float,
    t_end_h: float,
) -> dict[str, float]:
    """Run one mechanistic PBPK simulation."""
    from omega_pbpk.core.body import WholeBodyPBPK
    from omega_pbpk.drugs.drug import Drug

    drug = Drug(
        name="mc_sample",
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

    result = model.simulate(t_end_h=t_end_h)
    return result.pk_summary()


__all__ = ["DistributionSpec", "UncertaintyResult", "monte_carlo_propagation"]
