"""Dosing regimen optimizer.

Finds optimal dose and dosing interval to meet therapeutic targets
(Cmin, Cmax, AUC) at steady state using analytical 1-cpt approximations
for fast grid search, then evaluates top candidates with full simulation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

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


def _analytical_ss(
    spec: DrugSpec,
    dose_mg: float,
    interval_h: float,
    route: str,
    body_weight: float,
) -> dict[str, float]:
    """Fast analytical 1-compartment steady-state approximation.

    Uses well-stirred liver model and 1-cpt superposition to estimate
    Css_max/min/avg without running the full PBPK ODE.
    """
    # Estimate PK parameters from spec
    # Hepatic clearance via well-stirred model
    q_h = 80.0  # hepatic blood flow L/h (70 kg human)
    cl_h = (
        q_h * spec.fup * spec.clint_hepatic_L_per_h / (q_h + spec.fup * spec.clint_hepatic_L_per_h)
    )
    cl_r = spec.clr_L_per_h if spec.clr_L_per_h > 0 else 0.0
    cl_total = (cl_h + cl_r) * (body_weight / 70.0) ** 0.75

    # Apparent volume of distribution estimate (rough)
    # Using logP-based tissue partitioning
    log_p = spec.logP
    vd_L_kg = 0.5 + 3.5 * max(0.0, math.tanh(log_p / 2.0))
    vd = vd_L_kg * body_weight  # L

    # Half-life and rate constant
    ke = cl_total / max(vd, 0.001)  # h-1

    # Bioavailability for oral
    if route == "oral":
        # First-pass hepatic extraction
        e_h = cl_h / q_h
        fa = 1.0  # assume complete absorption for simplicity
        f = fa * (1.0 - e_h)
    else:
        f = 1.0

    effective_dose_mg = dose_mg * f

    # Steady-state metrics (1-cpt analytical)
    # Css_avg = dose * F / (CL * tau)
    css_avg = effective_dose_mg / max(cl_total * interval_h, 1e-12)

    # Css_max and Css_min via accumulation factor
    # For IV bolus-like: Css_max = D/(Vd * (1 - exp(-ke*tau)))
    # For oral (flip-flop or absorption-limited), use approximation
    if ke * interval_h < 0.01:
        # Extremely slow elimination — css_max ≈ css_avg
        css_max = css_avg * 1.05
        css_min = css_avg * 0.95
    else:
        acc_factor = 1.0 / (1.0 - math.exp(-ke * interval_h))
        if route == "iv":
            css_max = (effective_dose_mg / vd) * acc_factor
        else:
            # Oral: absorption-limited, ka >> ke assumed
            ka = 2.0  # typical ka estimate h-1
            css_max = effective_dose_mg / vd * (ka / (ka - ke)) * acc_factor
            # Clip to reasonable values
            css_max = min(css_max, effective_dose_mg / vd * acc_factor * 5)

        css_min = css_max * math.exp(-ke * interval_h)

    auc_ss = css_avg * interval_h

    return {
        "css_max": max(css_max, 0.0),
        "css_min": max(css_min, 0.0),
        "css_avg": max(css_avg, 0.0),
        "auc_ss": max(auc_ss, 0.0),
    }


def _golden_section_min(
    f: Any,
    lo: float,
    hi: float,
    tol: float = 5.0,
    max_iter: int = 20,
) -> float:
    """Pure-Python golden-section search for minimum of f on [lo, hi]."""
    gr = (math.sqrt(5.0) + 1.0) / 2.0
    a, b = lo, hi
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if f(c) < f(d):
            b = d
        else:
            a = c
        c = b - (b - a) / gr
        d = a + (b - a) / gr
    return (a + b) / 2.0


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

    For each candidate interval, uses a pure-Python golden-section search
    with fast analytical 1-cpt approximations to find the optimal dose,
    then evaluates top candidates with the full multi-dose simulator.

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

        def fast_objective(dose: float, _tau: float = tau) -> float:
            try:
                m = _analytical_ss(spec, dose, _tau, route, body_weight)
                return _penalty(m, target)
            except Exception:
                return 1e12

        opt_dose = _golden_section_min(
            fast_objective,
            dose_range_mg[0],
            dose_range_mg[1],
            tol=max(1.0, (dose_range_mg[1] - dose_range_mg[0]) / 100.0),
        )
        # Clip to bounds
        opt_dose = max(dose_range_mg[0], min(dose_range_mg[1], opt_dose))

        # Evaluate optimal dose with full simulation
        n_days_sim = max(1, int(math.ceil(n_doses * tau / 24.0)))
        try:
            ss = sim.simulate(
                drug=drug,
                dose_mg=opt_dose,
                interval_h=tau,
                n_days=n_days_sim,
                body_weight=body_weight,
            )
            metrics = {
                "css_max": float(ss.css_max),
                "css_min": float(ss.css_min),
                "css_avg": float(ss.css_avg),
                "auc_ss": float(ss.css_avg) * tau,
            }
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
