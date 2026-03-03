"""Sobol global sensitivity analysis for PBPK model parameters.

Implements variance-based Sobol indices (S1, ST) using the Saltelli 2010
sampling scheme with Jansen estimators. Bootstrap resampling provides
confidence intervals. Uses a fast 1-compartment analytical PK model for
evaluation speed.

References:
    Saltelli et al. (2010) Comp Phys Comm 181:259-270
    Jansen (1999) Comp Phys Comm 117:35-43
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from omega_pbpk.drugs.drug import Drug

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParameterRange:
    """Defines a uniform range for one sensitivity parameter."""

    name: str
    lower: float
    upper: float


@dataclass(frozen=True)
class SobolResult:
    """Results from Sobol global sensitivity analysis.

    Attributes:
        parameters: Names of the analysed parameters.
        S1: First-order Sobol indices.
        ST: Total-effect Sobol indices.
        S1_conf: Half-width of bootstrap CI for S1.
        ST_conf: Half-width of bootstrap CI for ST.
        metric: PK metric used (e.g. 'AUC', 'Cmax').
        n_samples: Base sample size *n*.
        n_total_evaluations: Total model evaluations = n*(k+2).
        convergence_flag: True if indices look well-converged.
        interaction_index: ST - S1 (parameter interactions).
    """

    parameters: list[str]
    S1: list[float]
    ST: list[float]
    S1_conf: list[float]
    ST_conf: list[float]
    metric: str
    n_samples: int
    n_total_evaluations: int
    convergence_flag: bool
    interaction_index: list[float]


# ---------------------------------------------------------------------------
# Default parameter ranges
# ---------------------------------------------------------------------------

DEFAULT_GSA_RANGES: list[ParameterRange] = [
    ParameterRange("fup", 0.01, 1.0),
    ParameterRange("rbp", 0.5, 2.0),
    ParameterRange("peff", 0.1, 10.0),
    ParameterRange("solubility_mg_mL", 0.001, 10.0),
    ParameterRange("clint_hepatic_L_per_h", 0.1, 50.0),
    ParameterRange("clint_gut_L_per_h", 0.0, 5.0),
    ParameterRange("clr_L_per_h", 0.0, 15.0),
]


# ---------------------------------------------------------------------------
# Sobol quasi-random sequence
# ---------------------------------------------------------------------------


def _sobol_sequence(n: int, dim: int, seed: int = 42) -> NDArray:
    """Generate *n* Sobol quasi-random points in [0,1]^dim."""
    from scipy.stats.qmc import Sobol

    sampler = Sobol(d=dim, scramble=True, seed=seed)
    # Sobol requires n to be a power of 2
    m = int(np.ceil(np.log2(max(n, 1))))
    n_pow2 = 2**m
    points = sampler.random(n_pow2)
    return points[:n]


# ---------------------------------------------------------------------------
# Saltelli sampling scheme
# ---------------------------------------------------------------------------


def _saltelli_sample(
    n: int,
    ranges: list[ParameterRange],
    seed: int = 42,
) -> tuple[NDArray, NDArray, list[NDArray]]:
    """Saltelli 2010 cross-sampling scheme.

    Returns:
        A: (n, k) base matrix scaled to parameter ranges.
        B: (n, k) complementary matrix scaled to parameter ranges.
        AB_list: k matrices each (n, k) where AB_i equals A with column i
                 replaced by column i from B.
    """
    k = len(ranges)
    # Generate 2k columns of Sobol points, then split into A and B
    raw = _sobol_sequence(n, 2 * k, seed=seed)
    unit_a = raw[:, :k]
    unit_b = raw[:, k:]

    # Scale to parameter ranges
    lowers = np.array([r.lower for r in ranges])
    uppers = np.array([r.upper for r in ranges])
    span = uppers - lowers

    A = lowers + unit_a * span
    B = lowers + unit_b * span

    AB_list: list[NDArray] = []
    for i in range(k):
        AB_i = A.copy()
        AB_i[:, i] = B[:, i]
        AB_list.append(AB_i)

    return A, B, AB_list


# ---------------------------------------------------------------------------
# 1-compartment PK model evaluation
# ---------------------------------------------------------------------------


def _evaluate_samples(
    samples: NDArray,
    ranges: list[ParameterRange],
    drug: Drug,
    dose_mg: float,
    route: str,
    body_weight: float,
    t_end_h: float,
    metric: str,
) -> NDArray:
    """Evaluate the 1-compartment PK model for each row of *samples*.

    Uses the analytical solution:
        ke = CL / Vd
        C(t) = (F * D * ka) / (Vd * (ka - ke)) * (exp(-ke*t) - exp(-ka*t))
        Cmax = D/Vd * ka/(ka-ke) * (ke/ka)^(ke/(ka-ke))   [oral]
        AUC  = F * D / CL
    """
    n_rows = samples.shape[0]
    results = np.empty(n_rows)
    param_names = [r.name for r in ranges]

    for i in range(n_rows):
        # Build parameter dict from sample row
        params: dict[str, float] = {}
        for j, name in enumerate(param_names):
            params[name] = float(samples[i, j])

        fup = params.get("fup", drug.fup)
        rbp = params.get("rbp", drug.rbp)
        peff = params.get("peff", drug.peff)
        clint_hepatic = params.get("clint_hepatic_L_per_h", drug.clint_hepatic_L_per_h)
        clint_gut = params.get("clint_gut_L_per_h", drug.clint_gut_L_per_h)
        clr = params.get("clr_L_per_h", drug.clr_L_per_h)

        # Derived PK parameters
        Vd = max(0.7 * rbp / max(fup, 1e-6) * 60.0, 1e-3)  # L
        CL = max(clint_hepatic * max(fup, 1e-6) + clr, 1e-6)  # L/h
        ke = CL / Vd

        if route == "iv":
            F = 1.0
        else:
            F = peff / (peff + 1e-5)
            # Gut first-pass reduces F
            gut_extraction = clint_gut / (clint_gut + 1.0) if clint_gut > 0 else 0.0
            F *= 1.0 - gut_extraction

        F = max(min(F, 1.0), 1e-6)

        if metric.upper() == "AUC":
            results[i] = F * dose_mg / CL
        else:
            # Cmax via analytical formula
            if route == "iv":
                results[i] = dose_mg / Vd
            else:
                ka = max(peff * 3600.0 / 0.01, 0.001)
                if abs(ka - ke) < 1e-12:
                    ka = ke + 1e-6
                ratio = ke / ka
                exponent = ke / (ka - ke)
                # Protect against numerical issues
                if ratio > 0 and abs(exponent) < 500:
                    results[i] = (F * dose_mg / Vd) * (ka / (ka - ke)) * (ratio**exponent)
                else:
                    results[i] = F * dose_mg / Vd
            results[i] = max(results[i], 0.0)

    return results


# ---------------------------------------------------------------------------
# Jansen estimators for S1 and ST
# ---------------------------------------------------------------------------


def _jansen_estimators(
    yA: NDArray,
    yB: NDArray,
    yAB_list: list[NDArray],
    n: int,
) -> tuple[NDArray, NDArray]:
    """Jansen (1999) estimators for first-order and total-effect indices.

    S1_i = 1 - (1/(2N)) * sum((yB - yAB_i)^2) / Var(Y)
    ST_i = (1/(2N)) * sum((yA - yAB_i)^2) / Var(Y)
    """
    all_y = np.concatenate([yA, yB])
    var_y = np.var(all_y)
    if var_y < 1e-30:
        k = len(yAB_list)
        return np.zeros(k), np.zeros(k)

    k = len(yAB_list)
    S1 = np.empty(k)
    ST = np.empty(k)

    for i in range(k):
        yABi = yAB_list[i]
        S1[i] = 1.0 - np.sum((yB - yABi) ** 2) / (2.0 * n * var_y)
        ST[i] = np.sum((yA - yABi) ** 2) / (2.0 * n * var_y)

    return S1, ST


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------


def _bootstrap_confidence(
    yA: NDArray,
    yB: NDArray,
    yAB_list: list[NDArray],
    n: int,
    n_bootstrap: int = 100,
    confidence: float = 0.95,
) -> tuple[NDArray, NDArray]:
    """Bootstrap CIs for S1 and ST.

    Returns half-widths of the confidence interval for each index.
    """
    rng = np.random.default_rng(seed=12345)
    k = len(yAB_list)
    s1_boot = np.empty((n_bootstrap, k))
    st_boot = np.empty((n_bootstrap, k))

    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        s1_b, st_b = _jansen_estimators(yA[idx], yB[idx], [yABi[idx] for yABi in yAB_list], n)
        s1_boot[b] = s1_b
        st_boot[b] = st_b

    alpha = (1.0 - confidence) / 2.0
    lo = alpha * 100
    hi = (1.0 - alpha) * 100

    s1_conf = (np.percentile(s1_boot, hi, axis=0) - np.percentile(s1_boot, lo, axis=0)) / 2.0
    st_conf = (np.percentile(st_boot, hi, axis=0) - np.percentile(st_boot, lo, axis=0)) / 2.0

    return s1_conf, st_conf


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def sobol_sensitivity(
    drug: Drug,
    dose_mg: float,
    route: str = "oral",
    body_weight: float = 70.0,
    t_end_h: float = 24.0,
    parameter_ranges: list[ParameterRange] | None = None,
    n_samples: int = 128,
    metric: str = "AUC",
    seed: int = 42,
    n_bootstrap: int = 100,
) -> SobolResult:
    """Run Sobol global sensitivity analysis.

    Args:
        drug: Base Drug compound.
        dose_mg: Dose in mg.
        route: 'oral' or 'iv'.
        body_weight: Body weight in kg.
        t_end_h: Simulation end time in hours.
        parameter_ranges: List of ParameterRange; defaults to DEFAULT_GSA_RANGES.
        n_samples: Base sample size (n). Total evaluations = n*(k+2).
        metric: PK output metric — 'AUC' or 'Cmax'.
        seed: Random seed for reproducibility.
        n_bootstrap: Number of bootstrap resamples for CIs.

    Returns:
        SobolResult with S1, ST, confidence intervals, and convergence flag.
    """
    if parameter_ranges is None:
        parameter_ranges = DEFAULT_GSA_RANGES

    k = len(parameter_ranges)
    n_total = n_samples * (k + 2)

    # 1. Sample
    A, B, AB_list = _saltelli_sample(n_samples, parameter_ranges, seed=seed)

    # 2. Evaluate
    eval_kwargs = dict(
        ranges=parameter_ranges,
        drug=drug,
        dose_mg=dose_mg,
        route=route,
        body_weight=body_weight,
        t_end_h=t_end_h,
        metric=metric,
    )
    yA = _evaluate_samples(A, **eval_kwargs)
    yB = _evaluate_samples(B, **eval_kwargs)
    yAB_list_eval = [_evaluate_samples(ab, **eval_kwargs) for ab in AB_list]

    # 3. Estimate indices
    S1_raw, ST_raw = _jansen_estimators(yA, yB, yAB_list_eval, n_samples)

    # 4. Bootstrap CIs
    S1_conf, ST_conf = _bootstrap_confidence(
        yA, yB, yAB_list_eval, n_samples, n_bootstrap=n_bootstrap
    )

    # 5. Clamp negative S1 to 0
    S1_clamped = np.maximum(S1_raw, 0.0)
    ST_clamped = np.maximum(ST_raw, 0.0)

    # 6. Interaction index
    interaction = ST_clamped - S1_clamped

    # 7. Convergence check: all CIs < 0.1 and sum(S1) <= 1.05
    ci_ok = bool(np.all(S1_conf < 0.1) and np.all(ST_conf < 0.1))
    sum_ok = float(np.sum(S1_clamped)) <= 1.05
    convergence_flag = ci_ok and sum_ok

    return SobolResult(
        parameters=[r.name for r in parameter_ranges],
        S1=[float(x) for x in S1_clamped],
        ST=[float(x) for x in ST_clamped],
        S1_conf=[float(x) for x in S1_conf],
        ST_conf=[float(x) for x in ST_conf],
        metric=metric,
        n_samples=n_samples,
        n_total_evaluations=n_total,
        convergence_flag=convergence_flag,
        interaction_index=[float(x) for x in interaction],
    )
