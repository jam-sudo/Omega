"""Sparse PK sampling optimization for clinical studies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "SamplingResult",
    "_d_criterion",
    "optimize_sparse_sampling",
    "recommend_sampling_times",
]


@dataclass
class SamplingResult:
    drug_name: str
    n_subjects: int
    n_samples_per_subject: int
    optimal_times_h: list[float]
    predicted_cv_cl_pct: float
    predicted_cv_auc_pct: float
    d_optimality_criterion: float
    sampling_strategy: str


def _d_criterion(times: list[float], ka: float, ke: float, vd: float) -> float:
    """Approximate D-optimality as simplified FIM determinant.

    Uses sum of squared sensitivity terms from biexponential model:
    sum of [1/t * exp(-ke*t)]^2 for each time point.
    """
    total = 0.0
    for t in times:
        if t > 0:
            total += (np.exp(-ke * t) / t) ** 2
    return float(total)


def optimize_sparse_sampling(
    drug_name: str,
    n_samples: int,
    pk_params: dict | None = None,
    t_window_h: float = 24.0,
    seed: int = 42,
) -> SamplingResult:
    """Optimize sparse sampling times via grid search maximizing D-criterion.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    n_samples:
        Number of samples per subject (>=1).
    pk_params:
        Dict with keys ka, ke, vd. Missing keys use defaults (ka=1, ke=0.2, vd=50).
    t_window_h:
        Sampling window in hours.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    SamplingResult
    """
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")

    if pk_params is None:
        pk_params = {}

    ka = float(pk_params.get("ka", 1.0))
    ke = float(pk_params.get("ke", 0.2))
    vd = float(pk_params.get("vd", 50.0))

    n_subjects = 20
    rng = np.random.default_rng(seed)

    best_times: list[float] = []
    best_criterion = -1.0

    n_trials = 50
    for _ in range(n_trials):
        # Sample without replacement from (0, t_window_h]
        raw = rng.uniform(0.0, t_window_h, size=n_samples * 3)
        raw = raw[raw > 0]
        # Deduplicate approximately by rounding
        unique_times = sorted(set(round(float(v), 4) for v in raw))
        if len(unique_times) < n_samples:
            # Fallback: evenly spaced
            unique_times = list(np.linspace(t_window_h / (n_samples + 1), t_window_h, n_samples))
        candidate = unique_times[:n_samples]
        crit = _d_criterion(candidate, ka, ke, vd)
        if crit > best_criterion:
            best_criterion = crit
            best_times = candidate

    if n_samples <= 3:
        strategy = "ultra_sparse"
    elif n_samples <= 6:
        strategy = "sparse"
    else:
        strategy = "intensive"

    predicted_cv_cl = 100.0 / np.sqrt(n_samples * n_subjects)
    predicted_cv_auc = predicted_cv_cl * 0.8  # AUC typically estimated with less CV

    return SamplingResult(
        drug_name=drug_name,
        n_subjects=n_subjects,
        n_samples_per_subject=n_samples,
        optimal_times_h=sorted(best_times),
        predicted_cv_cl_pct=float(predicted_cv_cl),
        predicted_cv_auc_pct=float(predicted_cv_auc),
        d_optimality_criterion=float(best_criterion),
        sampling_strategy=strategy,
    )


def recommend_sampling_times(
    drug_name: str,
    n_samples: int,
    t_half_h: float,
    tmax_h: float = 2.0,
) -> list[float]:
    """Rule-based sampling time recommendations.

    Always includes:
    - Near Tmax (midpoint of 0.5*tmax to 1.5*tmax)
    - t_half, 2*t_half, 3*t_half
    Remaining slots filled with evenly spaced times.

    Parameters
    ----------
    drug_name:
        Name of the drug (unused in computation, kept for API consistency).
    n_samples:
        Number of sampling times to return (>=1).
    t_half_h:
        Half-life in hours.
    tmax_h:
        Expected time of maximum concentration in hours.

    Returns
    -------
    Sorted list of exactly n_samples time points.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")

    near_tmax = (0.5 * tmax_h + 1.5 * tmax_h) / 2.0  # = tmax_h
    anchors = [near_tmax, t_half_h, 2.0 * t_half_h, 3.0 * t_half_h]

    # Deduplicate and sort anchors
    seen: set[float] = set()
    priority: list[float] = []
    for t in anchors:
        rounded = round(t, 6)
        if rounded > 0 and rounded not in seen:
            seen.add(rounded)
            priority.append(rounded)

    if len(priority) >= n_samples:
        return sorted(priority[:n_samples])

    # Fill remaining slots with evenly spaced times over a window
    t_max_window = max(4.0 * t_half_h, 3.0 * tmax_h, 24.0)
    n_fill = n_samples - len(priority)
    fill_candidates = np.linspace(0.25, t_max_window, n_fill * 10)
    fill_times: list[float] = []
    for t in fill_candidates:
        rounded = round(float(t), 6)
        if rounded not in seen:
            seen.add(rounded)
            fill_times.append(rounded)
        if len(fill_times) >= n_fill:
            break

    combined = sorted(priority + fill_times[:n_fill])
    return combined[:n_samples]
