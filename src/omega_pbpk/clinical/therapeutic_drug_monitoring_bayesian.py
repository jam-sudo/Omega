"""
Phase 942: Bayesian dose individualization via therapeutic drug monitoring (TDM).

Uses MAP estimation with a 1-compartment IV bolus model and grid search
over CL and Vd space.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "BayesianTDMResult",
    "run_bayesian_tdm",
    "simulate_tdm_guided_dosing",
]

_GRID_POINTS = 50
_DOSE_ADJ_MIN = 0.25
_DOSE_ADJ_MAX = 4.0
_LN2 = math.log(2.0)


@dataclass(frozen=True)
class BayesianTDMResult:
    """Result of Bayesian TDM MAP estimation."""

    drug_name: str
    dose_mg: float
    cl_estimated_L_per_h: float
    vd_estimated_L: float
    t_half_estimated_h: float
    cl_fold_vs_pop: float
    vd_fold_vs_pop: float
    predicted_trough_at_interval_mg_L: float
    dose_adjustment_factor: float
    recommended_dose_mg: float
    objective_function_value: float
    notes: str


def _predict_conc_iv_bolus(dose_mg: float, vd: float, cl: float, t: float) -> float:
    """
    1-cpt IV bolus: C(t) = D/Vd * exp(-CL/Vd * t)

    Returns concentration in mg/L.
    """
    ke = cl / vd
    return (dose_mg / vd) * math.exp(-ke * t)


def _map_objective(
    cl: float,
    vd: float,
    dose_mg: float,
    observed_times: list,
    observed_conc: list,
    cl_pop: float,
    vd_pop: float,
    sigma_cl: float,
    sigma_vd: float,
    sigma_residual: float,
) -> float:
    """
    MAP objective function to minimize.

    Likelihood term: sum of squared proportional residuals.
    Prior terms: log-normal priors on CL and Vd.
    """
    # Likelihood
    likelihood_sum = 0.0
    for t_i, c_obs in zip(observed_times, observed_conc):
        c_pred = _predict_conc_iv_bolus(dose_mg, vd, cl, t_i)
        if c_obs > 0 and sigma_residual > 0:
            residual = (c_pred - c_obs) / (sigma_residual * c_obs)
            likelihood_sum += residual**2
        else:
            # Fallback: absolute residual
            likelihood_sum += (c_pred - c_obs) ** 2

    # Prior terms (log-normal)
    if cl > 0 and cl_pop > 0 and sigma_cl > 0:
        prior_cl = (math.log(cl / cl_pop) / sigma_cl) ** 2
    else:
        prior_cl = 1e10

    if vd > 0 and vd_pop > 0 and sigma_vd > 0:
        prior_vd = (math.log(vd / vd_pop) / sigma_vd) ** 2
    else:
        prior_vd = 1e10

    return likelihood_sum + prior_cl + prior_vd


def run_bayesian_tdm(
    drug_name: str,
    dose_mg: float,
    observed_times_h: list,
    observed_conc_mg_L: list,
    cl_pop_L_per_h: float = 10.0,
    vd_pop_L: float = 50.0,
    cv_cl: float = 0.3,
    cv_vd: float = 0.2,
    sigma_residual: float = 0.15,
    target_trough_mg_L: float = 1.0,
    dosing_interval_h: float = 12.0,
) -> BayesianTDMResult:
    """
    Run Bayesian MAP estimation for individual PK parameters.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Administered dose in mg.
    observed_times_h : list[float]
        Sample collection times (hours post-dose).
    observed_conc_mg_L : list[float]
        Measured drug concentrations (mg/L).
    cl_pop_L_per_h : float
        Population mean clearance (L/h). Default 10.0.
    vd_pop_L : float
        Population mean volume of distribution (L). Default 50.0.
    cv_cl : float
        Coefficient of variation for CL prior. Default 0.3.
    cv_vd : float
        Coefficient of variation for Vd prior. Default 0.2.
    sigma_residual : float
        Proportional residual error. Default 0.15.
    target_trough_mg_L : float
        Target trough concentration (mg/L). Default 1.0.
    dosing_interval_h : float
        Dosing interval in hours. Default 12.0.

    Returns
    -------
    BayesianTDMResult
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if len(observed_times_h) != len(observed_conc_mg_L):
        raise ValueError("times and conc lists must match in length")
    if len(observed_times_h) < 1:
        raise ValueError("at least 1 observation required")
    if cl_pop_L_per_h <= 0:
        raise ValueError("cl_pop_L_per_h must be > 0")
    if vd_pop_L <= 0:
        raise ValueError("vd_pop_L must be > 0")
    if target_trough_mg_L <= 0:
        raise ValueError("target_trough_mg_L must be > 0")

    # Grid search bounds
    cl_min = cl_pop_L_per_h * 0.1
    cl_max = cl_pop_L_per_h * 5.0
    vd_min = vd_pop_L * 0.1
    vd_max = vd_pop_L * 5.0

    # Build grids
    cl_step = (cl_max - cl_min) / (_GRID_POINTS - 1)
    vd_step = (vd_max - vd_min) / (_GRID_POINTS - 1)

    cl_grid = [cl_min + i * cl_step for i in range(_GRID_POINTS)]
    vd_grid = [vd_min + i * vd_step for i in range(_GRID_POINTS)]

    # Grid search for MAP optimum
    best_obj = float("inf")
    best_cl = cl_pop_L_per_h
    best_vd = vd_pop_L

    for cl in cl_grid:
        for vd in vd_grid:
            obj = _map_objective(
                cl=cl,
                vd=vd,
                dose_mg=dose_mg,
                observed_times=observed_times_h,
                observed_conc=observed_conc_mg_L,
                cl_pop=cl_pop_L_per_h,
                vd_pop=vd_pop_L,
                sigma_cl=cv_cl,
                sigma_vd=cv_vd,
                sigma_residual=sigma_residual,
            )
            if obj < best_obj:
                best_obj = obj
                best_cl = cl
                best_vd = vd

    # Derived PK parameters
    t_half = _LN2 * best_vd / best_cl
    cl_fold = best_cl / cl_pop_L_per_h
    vd_fold = best_vd / vd_pop_L

    # Predicted trough at dosing interval
    pred_trough = _predict_conc_iv_bolus(dose_mg, best_vd, best_cl, dosing_interval_h)

    # Dose adjustment
    if pred_trough > 0:
        raw_factor = target_trough_mg_L / pred_trough
    else:
        raw_factor = _DOSE_ADJ_MAX

    dose_adj = max(_DOSE_ADJ_MIN, min(_DOSE_ADJ_MAX, raw_factor))
    recommended_dose = dose_mg * dose_adj

    # Notes
    notes_parts = [
        f"Estimated CL={best_cl:.3f} L/h ({cl_fold:.2f}x pop), Vd={best_vd:.2f} L ({vd_fold:.2f}x pop).",
        f"t1/2={t_half:.2f} h, predicted trough={pred_trough:.4g} mg/L at {dosing_interval_h} h.",
        f"Dose adjustment factor={dose_adj:.3f} (target trough={target_trough_mg_L} mg/L).",
        f"Recommended dose={recommended_dose:.1f} mg.",
    ]
    notes = " ".join(notes_parts)

    return BayesianTDMResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        cl_estimated_L_per_h=best_cl,
        vd_estimated_L=best_vd,
        t_half_estimated_h=t_half,
        cl_fold_vs_pop=cl_fold,
        vd_fold_vs_pop=vd_fold,
        predicted_trough_at_interval_mg_L=pred_trough,
        dose_adjustment_factor=dose_adj,
        recommended_dose_mg=recommended_dose,
        objective_function_value=best_obj,
        notes=notes,
    )


def simulate_tdm_guided_dosing(
    drug_name: str,
    initial_dose_mg: float,
    observed_times_h: list,
    observed_conc_mg_L: list,
    n_adjustments: int = 3,
    cl_pop_L_per_h: float = 10.0,
    vd_pop_L: float = 50.0,
    cv_cl: float = 0.3,
    cv_vd: float = 0.2,
    sigma_residual: float = 0.15,
    target_trough_mg_L: float = 1.0,
    dosing_interval_h: float = 12.0,
) -> list:
    """
    Simulate sequential Bayesian TDM-guided dose adjustments.

    Each cycle uses the recommended dose from the previous cycle as the new dose,
    while using the same observed concentrations (relative to the current dose)
    to estimate PK parameters.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    initial_dose_mg : float
        Starting dose in mg.
    observed_times_h : list[float]
        Sample collection times post-dose.
    observed_conc_mg_L : list[float]
        Measured concentrations.
    n_adjustments : int
        Number of sequential Bayesian update cycles. Default 3.
    (other parameters as in run_bayesian_tdm)

    Returns
    -------
    list[BayesianTDMResult] of length n_adjustments.
    """
    results = []
    current_dose = initial_dose_mg
    current_conc = list(observed_conc_mg_L)

    for i in range(n_adjustments):
        result = run_bayesian_tdm(
            drug_name=drug_name,
            dose_mg=current_dose,
            observed_times_h=observed_times_h,
            observed_conc_mg_L=current_conc,
            cl_pop_L_per_h=cl_pop_L_per_h,
            vd_pop_L=vd_pop_L,
            cv_cl=cv_cl,
            cv_vd=cv_vd,
            sigma_residual=sigma_residual,
            target_trough_mg_L=target_trough_mg_L,
            dosing_interval_h=dosing_interval_h,
        )
        results.append(result)

        # For next cycle: use recommended dose, scale concentrations proportionally
        if i < n_adjustments - 1:
            dose_ratio = result.recommended_dose_mg / current_dose if current_dose > 0 else 1.0
            current_dose = result.recommended_dose_mg
            # Scale observed concentrations by dose ratio (linear PK assumption)
            current_conc = [c * dose_ratio for c in current_conc]

    return results
