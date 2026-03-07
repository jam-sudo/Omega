"""PK variability analysis: inter-individual (IIV) and inter-occasion (IOV) variability."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class PKVariabilityResult:
    drug_name: str
    n_subjects: int
    n_occasions: int

    # Per-subject observed PK metrics
    auc_values: list[float]
    cmax_values: list[float]
    trough_values: list[float]

    # Summary statistics
    auc_mean: float
    auc_cv_pct: float
    auc_gm: float
    auc_gsd: float

    cmax_mean: float
    cmax_cv_pct: float

    trough_mean: float
    trough_cv_pct: float

    # Variability components (log-normal)
    omega_auc_pct: float
    omega_cmax_pct: float
    sigma_prop_pct: float

    # Range analysis
    auc_5th_pct: float
    auc_95th_pct: float
    fold_range: float

    variability_category: str  # "low", "moderate", "high"
    notes: str


def _cv_pct(values: list[float]) -> float:
    """Coefficient of variation as a percentage."""
    arr = np.array(values, dtype=float)
    mean = float(np.mean(arr))
    if mean == 0.0:
        return 0.0
    return float(np.std(arr, ddof=1) / mean * 100.0)


def _geometric_mean(values: list[float]) -> float:
    """Geometric mean via log-space."""
    log_vals = np.log(np.array(values, dtype=float))
    return float(np.exp(np.mean(log_vals)))


def _geometric_sd(values: list[float]) -> float:
    """Geometric standard deviation."""
    log_vals = np.log(np.array(values, dtype=float))
    return float(np.exp(np.std(log_vals, ddof=1)))


def _omega_cv_pct(values: list[float]) -> float:
    """IIV CV% using log-normal formula: sqrt(exp(var(log(values))) - 1) * 100."""
    log_vals = np.log(np.array(values, dtype=float))
    var_log = float(np.var(log_vals, ddof=1))
    return float(math.sqrt(max(0.0, math.exp(var_log) - 1.0)) * 100.0)


def _variability_category(cv_pct: float) -> str:
    if cv_pct < 30.0:
        return "low"
    if cv_pct > 60.0:
        return "high"
    return "moderate"


def analyze_pk_variability(
    drug_name: str,
    auc_values: list[float],
    cmax_values: list[float],
    trough_values: Optional[list[float]] = None,
    n_occasions: int = 1,
) -> PKVariabilityResult:
    """Analyze PK variability across a population.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    auc_values:
        Per-subject AUC values (same length as cmax_values). All must be > 0.
    cmax_values:
        Per-subject Cmax values. All must be > 0.
    trough_values:
        Optional per-subject trough concentrations. If None, trough stats are set to 0.
    n_occasions:
        Number of occasions per subject (used for IOV estimation context).

    Returns
    -------
    PKVariabilityResult
    """
    if len(auc_values) == 0:
        raise ValueError("auc_values must not be empty")
    if len(cmax_values) == 0:
        raise ValueError("cmax_values must not be empty")
    if len(auc_values) < 3:
        raise ValueError("At least 3 subjects required for variability analysis")
    if len(auc_values) != len(cmax_values):
        raise ValueError("auc_values and cmax_values must have the same length")
    if trough_values is not None and len(trough_values) != len(auc_values):
        raise ValueError("trough_values must have the same length as auc_values")
    if any(v <= 0 for v in auc_values):
        raise ValueError("All auc_values must be > 0")
    if any(v <= 0 for v in cmax_values):
        raise ValueError("All cmax_values must be > 0")
    if trough_values is not None and any(v <= 0 for v in trough_values):
        raise ValueError("All trough_values must be > 0")

    n_subjects = len(auc_values)

    # AUC statistics
    auc_arr = np.array(auc_values, dtype=float)
    auc_mean = float(np.mean(auc_arr))
    auc_cv = _cv_pct(auc_values)
    auc_gm = _geometric_mean(auc_values)
    auc_gsd = _geometric_sd(auc_values)
    omega_auc = _omega_cv_pct(auc_values)

    # Cmax statistics
    cmax_mean = float(np.mean(np.array(cmax_values, dtype=float)))
    cmax_cv = _cv_pct(cmax_values)
    omega_cmax = _omega_cv_pct(cmax_values)

    # Trough statistics
    if trough_values is not None:
        trough_mean = float(np.mean(np.array(trough_values, dtype=float)))
        trough_cv = _cv_pct(trough_values)
        trough_list = list(trough_values)
    else:
        trough_mean = 0.0
        trough_cv = 0.0
        trough_list = [0.0] * n_subjects

    # Residual / IOV: use 15% as placeholder when n_occasions == 1
    sigma_prop = 15.0 if n_occasions <= 1 else _cv_pct(auc_values) * 0.3

    # Percentiles
    auc_5th = float(np.percentile(auc_arr, 5))
    auc_95th = float(np.percentile(auc_arr, 95))
    fold_range = auc_95th / auc_5th if auc_5th > 0 else 1.0

    category = _variability_category(auc_cv)

    notes = (
        f"{drug_name}: n={n_subjects}, AUC CV%={auc_cv:.1f}% ({category}), "
        f"Cmax CV%={cmax_cv:.1f}%, fold_range={fold_range:.1f}x."
    )

    return PKVariabilityResult(
        drug_name=drug_name,
        n_subjects=n_subjects,
        n_occasions=n_occasions,
        auc_values=list(auc_values),
        cmax_values=list(cmax_values),
        trough_values=trough_list,
        auc_mean=auc_mean,
        auc_cv_pct=auc_cv,
        auc_gm=auc_gm,
        auc_gsd=auc_gsd,
        cmax_mean=cmax_mean,
        cmax_cv_pct=cmax_cv,
        trough_mean=trough_mean,
        trough_cv_pct=trough_cv,
        omega_auc_pct=omega_auc,
        omega_cmax_pct=omega_cmax,
        sigma_prop_pct=sigma_prop,
        auc_5th_pct=auc_5th,
        auc_95th_pct=auc_95th,
        fold_range=fold_range,
        variability_category=category,
        notes=notes,
    )


def simulate_population_variability(
    drug_name: str,
    cl_mean_L_per_h: float,
    vd_mean_L: float,
    dose_mg: float,
    n_subjects: int = 100,
    omega_cl_pct: float = 30.0,
    omega_vd_pct: float = 20.0,
    ka_per_h: float = 1.5,
    seed: int = 42,
) -> PKVariabilityResult:
    """Simulate PK variability across a virtual population.

    Uses 1-compartment oral PK with log-normally distributed CL and Vd.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    cl_mean_L_per_h:
        Population mean clearance (L/h).
    vd_mean_L:
        Population mean volume of distribution (L).
    dose_mg:
        Oral dose in mg.
    n_subjects:
        Number of virtual subjects to simulate.
    omega_cl_pct:
        IIV CV% for CL (log-normal).
    omega_vd_pct:
        IIV CV% for Vd (log-normal).
    ka_per_h:
        First-order absorption rate constant (1/h).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    PKVariabilityResult
    """
    if cl_mean_L_per_h <= 0:
        raise ValueError("cl_mean_L_per_h must be > 0")
    if vd_mean_L <= 0:
        raise ValueError("vd_mean_L must be > 0")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if n_subjects < 3:
        raise ValueError("n_subjects must be >= 3")
    if omega_cl_pct < 0:
        raise ValueError("omega_cl_pct must be >= 0")

    rng = np.random.default_rng(seed)

    # Log-normal parameters
    # CV = sqrt(exp(sigma^2) - 1) → sigma^2 = log(1 + CV^2)
    sigma_cl = math.sqrt(math.log(1.0 + (omega_cl_pct / 100.0) ** 2))
    sigma_vd = math.sqrt(math.log(1.0 + (omega_vd_pct / 100.0) ** 2))

    mu_cl = math.log(cl_mean_L_per_h) - 0.5 * sigma_cl**2
    mu_vd = math.log(vd_mean_L) - 0.5 * sigma_vd**2

    cl_i = rng.lognormal(mu_cl, sigma_cl, n_subjects)
    vd_i = rng.lognormal(mu_vd, sigma_vd, n_subjects)

    auc_list = []
    cmax_list = []

    for j in range(n_subjects):
        cl = cl_i[j]
        vd = vd_i[j]

        # 1-compartment oral PK analytics
        # AUC = F * dose / CL  (F=1 assumed)
        auc = dose_mg / cl
        auc_list.append(float(auc))

        # Cmax analytical for 1-cpt oral (F=1):
        # ke = CL/Vd; tmax = ln(ka/ke)/(ka-ke)
        ke = cl / vd
        if abs(ka_per_h - ke) < 1e-8:
            # Flip-flop: use approximation
            tmax = 1.0 / ka_per_h
        else:
            tmax = math.log(ka_per_h / ke) / (ka_per_h - ke)
        tmax = max(0.0, tmax)

        cmax = (
            (dose_mg / vd)
            * (ka_per_h / (ka_per_h - ke))
            * (math.exp(-ke * tmax) - math.exp(-ka_per_h * tmax))
            if abs(ka_per_h - ke) > 1e-8
            else (dose_mg / vd) * math.exp(-ke * tmax)
        )
        cmax_list.append(max(1e-12, float(cmax)))

    return analyze_pk_variability(
        drug_name=drug_name,
        auc_values=auc_list,
        cmax_values=cmax_list,
        trough_values=None,
        n_occasions=1,
    )


def identify_high_variability_drivers(
    auc_values: list[float],
    covariate_dict: dict[str, list[float]],
) -> dict[str, tuple[float, float]]:
    """Identify covariates most correlated with AUC variability.

    Computes Pearson correlation between log(AUC) and each covariate.

    Parameters
    ----------
    auc_values:
        Per-subject AUC values. All must be > 0.
    covariate_dict:
        Dictionary mapping covariate name to list of per-subject values.
        All lists must have the same length as auc_values.

    Returns
    -------
    dict mapping covariate name to (r, p_value), sorted by |r| descending.
    """
    if len(auc_values) == 0:
        raise ValueError("auc_values must not be empty")
    if any(v <= 0 for v in auc_values):
        raise ValueError("All auc_values must be > 0")

    log_auc = np.log(np.array(auc_values, dtype=float))
    results: dict[str, tuple[float, float]] = {}

    for cov_name, cov_vals in covariate_dict.items():
        if len(cov_vals) != len(auc_values):
            raise ValueError(f"Covariate '{cov_name}' length {len(cov_vals)} != {len(auc_values)}")
        cov_arr = np.array(cov_vals, dtype=float)
        if np.std(cov_arr) == 0 or np.std(log_auc) == 0:
            results[cov_name] = (0.0, 1.0)
        else:
            r, p = stats.pearsonr(log_auc, cov_arr)
            results[cov_name] = (float(r), float(p))

    # Sort by |r| descending
    sorted_results = dict(sorted(results.items(), key=lambda item: abs(item[1][0]), reverse=True))
    return sorted_results
