"""
Phase 930: Weight-based dosing strategy comparison.
Compares flat, mg/kg, and mg/m² dosing for PK variability using a virtual
patient population with allometric CL/Vd scaling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "WeightBasedDosingResult",
    "compare_dosing_strategies",
    "recommend_dosing_strategy",
]


@dataclass(frozen=True)
class WeightBasedDosingResult:
    """Result of dosing strategy comparison for one strategy."""

    drug_name: str
    strategy: str  # "flat" | "mg_per_kg" | "mg_per_m2"
    n_patients: int
    mean_auc: float
    cv_auc_pct: float  # coefficient of variation in %
    p5_auc: float
    p95_auc: float
    mean_cmax: float
    cv_cmax_pct: float
    recommended: bool  # True if lowest CV_auc among strategies
    notes: str


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------


def _lcg_normal(n: int, mu: float, sigma: float, seed: int = 42) -> list:
    """Generate n log-normal samples using LCG + Box-Muller."""
    a, c, m = 1664525, 1013904223, 2**32
    state = seed
    vals = []
    while len(vals) < n:
        state = (a * state + c) % m
        u1 = state / m
        state = (a * state + c) % m
        u2 = state / m
        z = math.sqrt(-2 * math.log(max(u1, 1e-10))) * math.cos(2 * math.pi * u2)
        vals.append(math.exp(mu + sigma * z))
    return vals[:n]


def _bsa(weight_kg: float) -> float:
    """Du Bois BSA simplified for average height 171 cm (m²)."""
    # BSA = 0.0167 * W^0.5
    return 0.0167 * math.sqrt(weight_kg)


def _mean(vals: list) -> float:
    return sum(vals) / len(vals)


def _std(vals: list, mu: float | None = None) -> float:
    if mu is None:
        mu = _mean(vals)
    n = len(vals)
    variance = sum((v - mu) ** 2 for v in vals) / n
    return math.sqrt(variance)


def _percentile(sorted_vals: list, pct: float) -> float:
    """Compute percentile (0–100) from a pre-sorted list."""
    n = len(sorted_vals)
    idx = pct / 100.0 * (n - 1)
    lo = int(idx)
    hi = lo + 1
    if hi >= n:
        return sorted_vals[-1]
    frac = idx - lo
    return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])


def _compute_strategy(
    weights: list,
    cl_ref: float,
    vd_ref: float,
    strategy: str,
    flat_dose_mg: float,
    dose_per_kg_mg: float,
    dose_per_m2_mg: float,
) -> tuple:
    """
    Compute AUC and Cmax lists for a given strategy.
    Returns (aucs, cmaxs).
    """
    aucs = []
    cmaxs = []
    ref_w = 70.0

    for w in weights:
        cl_i = cl_ref * (w / ref_w) ** 0.75
        vd_i = vd_ref * (w / ref_w) ** 1.0

        if strategy == "flat":
            dose_i = flat_dose_mg
        elif strategy == "mg_per_kg":
            dose_i = dose_per_kg_mg * w
        else:  # mg_per_m2
            dose_i = dose_per_m2_mg * _bsa(w)

        auc_i = dose_i / cl_i
        cmax_i = dose_i / vd_i

        aucs.append(auc_i)
        cmaxs.append(cmax_i)

    return aucs, cmaxs


def _summarise(
    drug_name: str,
    strategy: str,
    n_patients: int,
    aucs: list,
    cmaxs: list,
    recommended: bool,
) -> WeightBasedDosingResult:
    """Build a WeightBasedDosingResult from raw AUC/Cmax lists."""
    mu_auc = _mean(aucs)
    sd_auc = _std(aucs, mu_auc)
    cv_auc = (sd_auc / mu_auc * 100.0) if mu_auc > 0 else 0.0

    mu_cmax = _mean(cmaxs)
    sd_cmax = _std(cmaxs, mu_cmax)
    cv_cmax = (sd_cmax / mu_cmax * 100.0) if mu_cmax > 0 else 0.0

    sorted_aucs = sorted(aucs)
    p5_auc = _percentile(sorted_aucs, 5.0)
    p95_auc = _percentile(sorted_aucs, 95.0)

    notes = (
        f"Strategy={strategy}, N={n_patients}, "
        f"mean_AUC={mu_auc:.2f}, CV_AUC={cv_auc:.1f}%, "
        f"mean_Cmax={mu_cmax:.2f}, CV_Cmax={cv_cmax:.1f}%"
        + ("; RECOMMENDED (lowest CV_AUC)" if recommended else "")
    )

    return WeightBasedDosingResult(
        drug_name=drug_name,
        strategy=strategy,
        n_patients=n_patients,
        mean_auc=mu_auc,
        cv_auc_pct=cv_auc,
        p5_auc=p5_auc,
        p95_auc=p95_auc,
        mean_cmax=mu_cmax,
        cv_cmax_pct=cv_cmax,
        recommended=recommended,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compare_dosing_strategies(
    drug_name: str,
    cl_ref_L_per_h: float = 10.0,
    vd_ref_L: float = 50.0,
    flat_dose_mg: float = 100.0,
    dose_per_kg_mg: float = 1.5,
    dose_per_m2_mg: float = 50.0,
    n_patients: int = 100,
    weight_cv: float = 0.20,
    seed: int = 42,
) -> list:
    """
    Compare flat, mg/kg, and mg/m² dosing strategies in a virtual population.

    Returns list of 3 WeightBasedDosingResult, sorted by cv_auc_pct ascending.
    """
    # Validation
    if cl_ref_L_per_h <= 0:
        raise ValueError("cl_ref_L_per_h must be > 0")
    if vd_ref_L <= 0:
        raise ValueError("vd_ref_L must be > 0")
    if flat_dose_mg <= 0:
        raise ValueError("flat_dose_mg must be > 0")
    if dose_per_kg_mg <= 0:
        raise ValueError("dose_per_kg_mg must be > 0")
    if dose_per_m2_mg <= 0:
        raise ValueError("dose_per_m2_mg must be > 0")
    if n_patients < 10:
        raise ValueError("n_patients must be >= 10")
    if not (0 < weight_cv < 1):
        raise ValueError("weight_cv must be in (0, 1)")

    # Log-normal weight distribution: mean=70 kg, CV=weight_cv
    # For log-normal: mu_ln = log(mean) - 0.5*sigma_ln^2
    # CV^2 = exp(sigma_ln^2) - 1  =>  sigma_ln = sqrt(log(1 + CV^2))
    sigma_ln = math.sqrt(math.log(1.0 + weight_cv**2))
    mu_ln = math.log(70.0) - 0.5 * sigma_ln**2
    weights = _lcg_normal(n_patients, mu_ln, sigma_ln, seed=seed)

    strategies = ["flat", "mg_per_kg", "mg_per_m2"]
    raw: dict = {}
    for strat in strategies:
        aucs, cmaxs = _compute_strategy(
            weights, cl_ref_L_per_h, vd_ref_L, strat, flat_dose_mg, dose_per_kg_mg, dose_per_m2_mg
        )
        raw[strat] = (aucs, cmaxs)

    # Determine best strategy (lowest CV_AUC)
    cv_by_strat = {}
    for strat, (aucs, _) in raw.items():
        mu = _mean(aucs)
        sd = _std(aucs, mu)
        cv_by_strat[strat] = (sd / mu * 100.0) if mu > 0 else 0.0

    best_strat = min(cv_by_strat, key=lambda s: cv_by_strat[s])

    results = []
    for strat in strategies:
        aucs, cmaxs = raw[strat]
        result = _summarise(
            drug_name=drug_name,
            strategy=strat,
            n_patients=n_patients,
            aucs=aucs,
            cmaxs=cmaxs,
            recommended=(strat == best_strat),
        )
        results.append(result)

    results.sort(key=lambda r: r.cv_auc_pct)
    return results


def recommend_dosing_strategy(
    drug_name: str,
    cl_ref_L_per_h: float = 10.0,
    vd_ref_L: float = 50.0,
    flat_dose_mg: float = 100.0,
    dose_per_kg_mg: float = 1.5,
    dose_per_m2_mg: float = 50.0,
    n_patients: int = 100,
    weight_cv: float = 0.20,
    seed: int = 42,
) -> WeightBasedDosingResult:
    """
    Return the dosing strategy with the lowest AUC CV%.
    """
    results = compare_dosing_strategies(
        drug_name=drug_name,
        cl_ref_L_per_h=cl_ref_L_per_h,
        vd_ref_L=vd_ref_L,
        flat_dose_mg=flat_dose_mg,
        dose_per_kg_mg=dose_per_kg_mg,
        dose_per_m2_mg=dose_per_m2_mg,
        n_patients=n_patients,
        weight_cv=weight_cv,
        seed=seed,
    )
    # results are sorted by cv_auc_pct ascending, first is best
    return results[0]
