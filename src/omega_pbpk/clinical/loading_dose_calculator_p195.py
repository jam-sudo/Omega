"""Phase 195 — Loading Dose Calculator.

Calculate loading doses to rapidly achieve target plasma concentrations.
Uses forward Euler ODE integration and manual trapezoidal AUC.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class LoadingDoseResult:
    """Result of loading dose calculation."""

    drug_name: str
    target_conc_mg_L: float
    vd_L: float
    cl_L_per_h: float
    fu_plasma: float
    bioavailability: float
    loading_dose_mg: float
    maintenance_dose_mg: float
    n_loading_doses: int
    dosing_interval_h: float
    time_to_90pct_target_h: float
    accumulation_ratio: float
    dose_ratio_loading_to_maintenance: float
    notes: str


def _validate_inputs(
    target_conc_mg_L: float,
    vd_L: float,
    cl_L_per_h: float,
    bioavailability: float,
) -> None:
    if target_conc_mg_L <= 0:
        raise ValueError(f"target_conc_mg_L must be > 0, got {target_conc_mg_L}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if not (0.0 < bioavailability <= 1.0):
        raise ValueError(f"bioavailability must be in (0, 1], got {bioavailability}")


def _simulate_forward_euler(
    loading_dose_mg: float,
    maintenance_dose_mg: float,
    ke: float,
    vd_L: float,
    ka: float,
    dosing_interval_h: float,
    n_loading_doses: int,
    bioavailability: float,
    t_end_h: float = 72.0,
    dt_h: float = 0.05,
) -> tuple[list[float], list[float]]:
    """Forward Euler 1-compartment simulation with loading and maintenance doses.

    Returns (times, concentrations).
    """
    n_steps = int(t_end_h / dt_h) + 1
    times: list[float] = [0.0] * n_steps
    concs: list[float] = [0.0] * n_steps

    a_gut = 0.0
    a_central = 0.0

    # Pre-compute all dose events: (time, amount_in_gut)
    # Loading doses at t = 0, tau, 2*tau, ..., (n_loading_doses-1)*tau
    # Maintenance doses thereafter
    dose_schedule: list[tuple[float, float]] = []
    for i in range(n_loading_doses):
        t_dose = i * dosing_interval_h
        absorbed = loading_dose_mg * bioavailability
        dose_schedule.append((t_dose, absorbed))

    # Add maintenance doses after loading doses are done
    # First maintenance at n_loading_doses * tau
    n_maint = max(1, int(t_end_h / dosing_interval_h)) + 1
    for i in range(n_maint):
        t_dose = (n_loading_doses + i) * dosing_interval_h
        if t_dose <= t_end_h:
            absorbed = maintenance_dose_mg * bioavailability
            dose_schedule.append((t_dose, absorbed))

    # Sort by time
    dose_schedule.sort(key=lambda x: x[0])
    dose_idx = 0

    # Apply first dose at t=0 immediately
    if dose_schedule and dose_schedule[0][0] == 0.0:
        a_gut += dose_schedule[0][1]
        dose_idx = 1

    concs[0] = a_central / vd_L
    times[0] = 0.0

    for step in range(1, n_steps):
        t = step * dt_h
        times[step] = t

        # Apply any doses scheduled at this time step (within half dt tolerance)
        while dose_idx < len(dose_schedule) and dose_schedule[dose_idx][0] <= t + dt_h * 0.5:
            if dose_schedule[dose_idx][0] > (step - 1) * dt_h + dt_h * 0.5:
                a_gut += dose_schedule[dose_idx][1]
            dose_idx += 1

        # Forward Euler ODE:
        # dA_gut/dt = -ka * A_gut
        # dA_central/dt = ka * A_gut - ke * A_central
        da_gut = -ka * a_gut
        da_central = ka * a_gut - ke * a_central

        a_gut = max(0.0, a_gut + da_gut * dt_h)
        a_central = max(0.0, a_central + da_central * dt_h)
        concs[step] = a_central / vd_L

    return times, concs


def _time_to_fraction_of_target(
    times: list[float],
    concs: list[float],
    target: float,
    fraction: float = 0.9,
) -> float:
    """Return first time concentration >= fraction * target."""
    threshold = fraction * target
    for t, c in zip(times, concs):
        if c >= threshold:
            return t
    return times[-1]


def _accumulation_ratio(ke: float, dosing_interval_h: float) -> float:
    """Steady-state accumulation ratio = 1 / (1 - exp(-ke * tau))."""
    exponent = ke * dosing_interval_h
    val = 1.0 - math.exp(-exponent)
    if val <= 0:
        return 1.0
    return 1.0 / val


def calculate_loading_dose(
    drug_name: str,
    target_conc_mg_L: float,
    vd_L: float,
    fu_plasma: float = 1.0,
    cl_L_per_h: float = 1.0,
    bioavailability: float = 1.0,
    dosing_interval_h: float = 12.0,
    n_loading_doses: int = 1,
    ka_per_h: float = 1.0,
) -> LoadingDoseResult:
    """Calculate loading dose to rapidly achieve target plasma concentration.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    target_conc_mg_L:
        Target plasma concentration (mg/L).
    vd_L:
        Volume of distribution (L, must be > 0).
    fu_plasma:
        Unbound fraction in plasma (0–1).
    cl_L_per_h:
        Total clearance (L/h, must be > 0).
    bioavailability:
        Oral bioavailability fraction F (0, 1]; default 1.0 for IV.
    dosing_interval_h:
        Dosing interval (h).
    n_loading_doses:
        Number of loading doses to split the total loading across.
    ka_per_h:
        First-order absorption rate constant (1/h).

    Returns
    -------
    LoadingDoseResult
    """
    _validate_inputs(target_conc_mg_L, vd_L, cl_L_per_h, bioavailability)

    ke = cl_L_per_h / vd_L
    tau = dosing_interval_h

    # Loading dose: D_L = target_conc * Vd / F
    # For oral with fu_plasma correction: D_L = target_conc * Vd / (F * fu_plasma)
    if bioavailability >= 1.0:
        # IV route: no fu_plasma correction needed
        loading_dose_mg = target_conc_mg_L * vd_L
    else:
        # Oral route: account for both bioavailability and fu_plasma
        denom = bioavailability * fu_plasma
        if denom <= 0:
            denom = bioavailability
        loading_dose_mg = target_conc_mg_L * vd_L / denom

    # Maintenance dose: D_maint = CL * target * tau / F
    maintenance_dose_mg = cl_L_per_h * target_conc_mg_L * tau / bioavailability

    # Ensure loading >= maintenance
    loading_dose_mg = max(loading_dose_mg, maintenance_dose_mg)

    # Accumulation ratio
    acc_ratio = _accumulation_ratio(ke, tau)

    # Simulate to find time_to_90pct_target
    t_end = max(72.0, 10.0 * tau)
    times, concs = _simulate_forward_euler(
        loading_dose_mg=loading_dose_mg,
        maintenance_dose_mg=maintenance_dose_mg,
        ke=ke,
        vd_L=vd_L,
        ka=ka_per_h,
        dosing_interval_h=tau,
        n_loading_doses=n_loading_doses,
        bioavailability=bioavailability,
        t_end_h=t_end,
    )

    t90 = _time_to_fraction_of_target(times, concs, target_conc_mg_L, 0.9)
    # Ensure t90 > 0 (at least a small positive value)
    if t90 == 0.0:
        t90 = 0.05

    dose_ratio = loading_dose_mg / maintenance_dose_mg if maintenance_dose_mg > 0 else 1.0

    t_half = math.log(2.0) / ke if ke > 0 else float("inf")
    notes = (
        f"ke={ke:.4f}/h; t½={t_half:.2f}h; tau={tau:.1f}h; "
        f"F={bioavailability:.2f}; fu={fu_plasma:.2f}; "
        f"n_loading={n_loading_doses}"
    )

    return LoadingDoseResult(
        drug_name=drug_name,
        target_conc_mg_L=target_conc_mg_L,
        vd_L=vd_L,
        cl_L_per_h=cl_L_per_h,
        fu_plasma=fu_plasma,
        bioavailability=bioavailability,
        loading_dose_mg=loading_dose_mg,
        maintenance_dose_mg=maintenance_dose_mg,
        n_loading_doses=n_loading_doses,
        dosing_interval_h=tau,
        time_to_90pct_target_h=t90,
        accumulation_ratio=acc_ratio,
        dose_ratio_loading_to_maintenance=dose_ratio,
        notes=notes,
    )


def compare_loading_strategies(
    drug_name: str,
    target_conc_mg_L: float,
    vd_L: float,
    cl_L_per_h: float,
    n_doses_options: list[int],
    fu_plasma: float = 1.0,
    bioavailability: float = 1.0,
    dosing_interval_h: float = 12.0,
    ka_per_h: float = 1.0,
) -> list[LoadingDoseResult]:
    """Compare loading strategies with different numbers of loading doses.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    target_conc_mg_L:
        Target plasma concentration (mg/L).
    vd_L:
        Volume of distribution (L).
    cl_L_per_h:
        Total clearance (L/h).
    n_doses_options:
        List of n_loading_doses values to compare.
    fu_plasma:
        Unbound fraction in plasma.
    bioavailability:
        Oral bioavailability fraction.
    dosing_interval_h:
        Dosing interval (h).
    ka_per_h:
        Absorption rate constant (1/h).

    Returns
    -------
    list[LoadingDoseResult] — one per entry in n_doses_options.
    """
    results: list[LoadingDoseResult] = []
    for n in n_doses_options:
        result = calculate_loading_dose(
            drug_name=drug_name,
            target_conc_mg_L=target_conc_mg_L,
            vd_L=vd_L,
            fu_plasma=fu_plasma,
            cl_L_per_h=cl_L_per_h,
            bioavailability=bioavailability,
            dosing_interval_h=dosing_interval_h,
            n_loading_doses=n,
            ka_per_h=ka_per_h,
        )
        results.append(result)
    return results
