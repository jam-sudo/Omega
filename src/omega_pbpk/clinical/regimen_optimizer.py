"""Dosing regimen optimizer.

Finds optimal dose and dosing interval to meet therapeutic targets
(Cmin, Cmax, AUC) at steady state using multi-dose PK simulation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar

from omega_pbpk.adapters import spec_to_drug
from omega_pbpk.clinical.dose_optimization import MultiDoseSimulator
from omega_pbpk.contracts.drug_spec import DrugSpec


@dataclass
class TherapeuticTarget:
    """Therapeutic target constraints for regimen optimization."""

    cmin_mg_L: float | None = None
    cmax_mg_L: float | None = None
    auc_min: float | None = None
    auc_max: float | None = None


@dataclass(frozen=True)
class RegimenResult:
    """Optimized dosing regimen result."""

    optimal_dose_mg: float
    optimal_interval_h: float
    css_max: float
    css_min: float
    css_avg: float
    auc_ss: float
    feasible: bool
    warnings: list[str] = field(default_factory=list)
    all_regimens: list[dict[str, Any]] = field(default_factory=list)


def _compute_regimen_metrics(
    sim: MultiDoseSimulator,
    drug: Any,
    dose_mg: float,
    interval_h: float,
    n_doses: int,
    body_weight: float,
) -> dict[str, float]:
    """Simulate multi-dose and extract steady-state metrics from last interval."""
    n_days = max(1, int(np.ceil(n_doses * interval_h / 24.0)))
    result = sim.simulate(
        drug=drug,
        dose_mg=dose_mg,
        interval_h=interval_h,
        n_days=n_days,
        body_weight=body_weight,
    )

    # Extract metrics from the last dosing interval
    css_max = float(result.css_max)
    css_min = float(result.css_min)
    css_avg = float(result.css_avg)

    # AUC at steady state ≈ Css_avg × interval
    auc_ss = css_avg * interval_h

    return {
        "css_max": css_max,
        "css_min": css_min,
        "css_avg": css_avg,
        "auc_ss": auc_ss,
    }


def _penalty(
    metrics: dict[str, float],
    target: TherapeuticTarget,
) -> float:
    """Compute penalty for deviation from therapeutic targets."""
    p = 0.0

    if target.cmin_mg_L is not None:
        shortfall = max(0.0, target.cmin_mg_L - metrics["css_min"])
        p += shortfall**2

    if target.cmax_mg_L is not None:
        excess = max(0.0, metrics["css_max"] - target.cmax_mg_L)
        p += excess**2

    if target.auc_min is not None:
        shortfall = max(0.0, target.auc_min - metrics["auc_ss"])
        p += shortfall**2

    if target.auc_max is not None:
        excess = max(0.0, metrics["auc_ss"] - target.auc_max)
        p += excess**2

    return p


def _check_feasibility(
    metrics: dict[str, float],
    target: TherapeuticTarget,
    tol: float = 0.10,
) -> bool:
    """Check if all constraints are satisfied within tolerance."""
    if target.cmin_mg_L is not None:
        if metrics["css_min"] < target.cmin_mg_L * (1 - tol):
            return False

    if target.cmax_mg_L is not None:
        if metrics["css_max"] > target.cmax_mg_L * (1 + tol):
            return False

    if target.auc_min is not None:
        if metrics["auc_ss"] < target.auc_min * (1 - tol):
            return False

    if target.auc_max is not None:
        if metrics["auc_ss"] > target.auc_max * (1 + tol):
            return False

    return True


def optimize_regimen(
    spec: DrugSpec,
    target: TherapeuticTarget,
    dose_range_mg: tuple[float, float] = (10.0, 1000.0),
    intervals_h: Sequence[float] = (6, 8, 12, 24),
    route: str = "oral",
    n_doses: int = 20,
    body_weight: float = 70.0,
) -> RegimenResult:
    """Find optimal dosing regimen to meet therapeutic targets.

    For each candidate interval, uses scipy.optimize.minimize_scalar to
    find the dose that minimizes penalty against Cmin/Cmax/AUC constraints.

    Args:
        spec: Drug specification.
        target: Therapeutic target constraints.
        dose_range_mg: (min, max) dose search range in mg.
        intervals_h: Candidate dosing intervals in hours.
        route: Administration route ('oral' or 'iv').
        n_doses: Number of doses to simulate for steady-state.
        body_weight: Body weight in kg.

    Returns:
        RegimenResult with optimal dose/interval and top-3 regimens.
    """
    drug = replace(spec_to_drug(spec), route=route)
    sim = MultiDoseSimulator()

    candidates: list[dict[str, Any]] = []
    warnings: list[str] = []

    for tau in intervals_h:
        tau = float(tau)

        def objective(dose: float, _tau: float = tau) -> float:
            try:
                m = _compute_regimen_metrics(sim, drug, dose, _tau, n_doses, body_weight)
                return _penalty(m, target)
            except Exception:
                return 1e12

        result = minimize_scalar(
            objective,
            bounds=dose_range_mg,
            method="bounded",
        )

        opt_dose = float(result.x)
        try:
            metrics = _compute_regimen_metrics(sim, drug, opt_dose, tau, n_doses, body_weight)
        except Exception as exc:
            warnings.append(f"Simulation failed for interval {tau}h: {exc}")
            continue

        pen = _penalty(metrics, target)
        feasible = _check_feasibility(metrics, target)

        candidates.append(
            {
                "dose_mg": round(opt_dose, 2),
                "interval_h": tau,
                "css_max": round(metrics["css_max"], 6),
                "css_min": round(metrics["css_min"], 6),
                "css_avg": round(metrics["css_avg"], 6),
                "auc_ss": round(metrics["auc_ss"], 4),
                "feasible": feasible,
                "penalty": pen,
            }
        )

    if not candidates:
        warnings.append("No valid regimens found for any interval")
        return RegimenResult(
            optimal_dose_mg=0.0,
            optimal_interval_h=0.0,
            css_max=0.0,
            css_min=0.0,
            css_avg=0.0,
            auc_ss=0.0,
            feasible=False,
            warnings=warnings,
            all_regimens=[],
        )

    # Sort by penalty (lower is better)
    candidates.sort(key=lambda c: c["penalty"])
    top3 = candidates[:3]
    best = candidates[0]

    if not best["feasible"]:
        warnings.append("Best regimen does not fully satisfy all constraints within 10% tolerance")

    return RegimenResult(
        optimal_dose_mg=best["dose_mg"],
        optimal_interval_h=best["interval_h"],
        css_max=best["css_max"],
        css_min=best["css_min"],
        css_avg=best["css_avg"],
        auc_ss=best["auc_ss"],
        feasible=best["feasible"],
        warnings=warnings,
        all_regimens=top3,
    )
