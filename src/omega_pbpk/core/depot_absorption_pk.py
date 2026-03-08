"""Phase 1097 — Drug Absorption from Depot Formulations (SC/IM).

Sustained release from subcutaneous/intramuscular depot formulations.
Covers: microspheres, implants, gel depots, oil solutions, suspensions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

KNOWN_FORMULATIONS = {
    "microsphere_PLGA",
    "oil_solution",
    "gel_depot",
    "suspension_SC",
    "long_acting_implant",
}


@dataclass
class DepotAbsorptionResult:
    """Result of depot absorption simulation."""

    drug_name: str
    dose_mg: float
    formulation: str
    times_h: list
    a_depot_mg: list
    c_plasma_mg_L: list
    cmax: float
    tmax_h: float
    auc: float
    f_absorbed_at_24h: float
    f_absorbed_at_7d: float
    f_absorbed_total: float
    t_half_depot_h: float
    release_duration_days: float
    notes: str


def _trapz(times: list, values: list) -> float:
    """Manual trapezoidal integration."""
    result = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        result += 0.5 * (values[i] + values[i - 1]) * dt
    return result


def _find_half_time(times: list, a_depot: list, dose_mg: float) -> float:
    """Find time when depot reaches 50% of original dose."""
    target = 0.5 * dose_mg
    for i in range(1, len(a_depot)):
        if a_depot[i] <= target:
            # Linear interpolation
            frac = (a_depot[i - 1] - target) / max(1e-12, a_depot[i - 1] - a_depot[i])
            return times[i - 1] + frac * (times[i] - times[i - 1])
    return times[-1]


def _find_release_duration_days(times: list, a_depot: list, dose_mg: float) -> float:
    """Find time (in days) when 90% of dose has been released (10% remaining)."""
    target = 0.10 * dose_mg
    for i in range(1, len(a_depot)):
        if a_depot[i] <= target:
            frac = (a_depot[i - 1] - target) / max(1e-12, a_depot[i - 1] - a_depot[i])
            t_h = times[i - 1] + frac * (times[i] - times[i - 1])
            return t_h / 24.0
    return times[-1] / 24.0


def simulate_depot_absorption(
    drug_name: str,
    dose_mg: float,
    formulation: str,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    t_end_h: float = 720.0,
    dt_h: float = 0.5,
) -> DepotAbsorptionResult:
    """Simulate drug absorption from a depot formulation.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Dose in milligrams.
    formulation:
        One of: microsphere_PLGA, oil_solution, gel_depot, suspension_SC,
        long_acting_implant.
    cl_L_per_h:
        Systemic clearance (L/h).
    vd_L:
        Volume of distribution (L).
    t_end_h:
        Simulation end time (hours). Default = 720 h (30 days).
    dt_h:
        Integration step size (hours).

    Returns
    -------
    DepotAbsorptionResult
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if formulation not in KNOWN_FORMULATIONS:
        raise ValueError(
            f"Unknown formulation '{formulation}'. Must be one of: {sorted(KNOWN_FORMULATIONS)}"
        )
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")

    ke = cl_L_per_h / vd_L  # elimination rate constant (1/h)

    times = []
    a_depot_traj = []
    c_plasma_traj = []

    t = 0.0
    n_steps = int(round(t_end_h / dt_h)) + 1

    # Track state for oil_solution
    a_fast_curr = 0.3 * dose_mg if formulation == "oil_solution" else 0.0
    a_slow_curr = 0.7 * dose_mg if formulation == "oil_solution" else 0.0
    a_depot_curr = dose_mg if formulation != "oil_solution" else (a_fast_curr + a_slow_curr)
    c_plasma_curr = 0.0

    # Parameters for formulations
    k_hig = 0.05  # Higuchi constant mg^0.5 / h
    k_fast_oil = 0.1  # fast pool for oil_solution
    k_slow_oil = 0.005  # slow pool for oil_solution
    k_suspension = 0.1  # suspension_SC first-order
    k_zo = 0.001  # long_acting_implant zero-order rate (mg/h) per mg... actually rate = k_zo * dose
    # Clarification: zero-order means constant release rate
    # rate_zo = k_zo * dose_mg  # we'll compute below
    koff_implant = 0.01  # first-order when depot < 10%

    # Precompute zero-order release rate for implant
    rate_zo = k_zo * dose_mg  # mg/h released

    for _step in range(n_steps):
        times.append(t)
        a_depot_traj.append(
            a_fast_curr + a_slow_curr if formulation == "oil_solution" else a_depot_curr
        )
        c_plasma_traj.append(c_plasma_curr)

        # Compute release rate from depot
        if formulation == "microsphere_PLGA":
            k_rel = 2.0 if t < 2.0 else 0.02
            d_a_depot = -k_rel * a_depot_curr
            release_rate = k_rel * a_depot_curr

        elif formulation == "oil_solution":
            d_a_fast = -k_fast_oil * a_fast_curr
            d_a_slow = -k_slow_oil * a_slow_curr
            release_rate = -(d_a_fast + d_a_slow)  # positive value released
            d_a_fast_curr = d_a_fast
            d_a_slow_curr = d_a_slow

        elif formulation == "gel_depot":
            a_pos = max(0.0, a_depot_curr)
            release_rate = k_hig * math.sqrt(a_pos)
            d_a_depot = -release_rate

        elif formulation == "suspension_SC":
            d_a_depot = -k_suspension * a_depot_curr
            release_rate = k_suspension * a_depot_curr

        elif formulation == "long_acting_implant":
            if a_depot_curr > 0.1 * dose_mg:
                # Zero-order region
                actual_rate = min(rate_zo, a_depot_curr / dt_h)
                release_rate = actual_rate
                d_a_depot = -actual_rate
            else:
                # Switch to first-order when < 10% remaining
                release_rate = koff_implant * max(0.0, a_depot_curr)
                d_a_depot = -release_rate

        # dC_plasma/dt = release_rate / Vd - ke * C_plasma
        d_c_plasma = release_rate / vd_L - ke * c_plasma_curr

        # Forward Euler update
        if formulation == "oil_solution":
            a_fast_new = max(0.0, a_fast_curr + d_a_fast_curr * dt_h)
            a_slow_new = max(0.0, a_slow_curr + d_a_slow_curr * dt_h)
            a_fast_curr = a_fast_new
            a_slow_curr = a_slow_new
        else:
            a_depot_curr = max(0.0, a_depot_curr + d_a_depot * dt_h)

        c_plasma_curr = max(0.0, c_plasma_curr + d_c_plasma * dt_h)
        t += dt_h

    # Compute metrics
    cmax = max(c_plasma_traj)
    tmax_h = times[c_plasma_traj.index(cmax)]
    auc = _trapz(times, c_plasma_traj)

    # f_absorbed = fraction of dose absorbed = 1 - remaining in depot
    # Use the depot trajectory
    def depot_at_time(target_h: float) -> float:
        """Interpolate depot amount at given time."""
        for i in range(1, len(times)):
            if times[i] >= target_h:
                frac = (target_h - times[i - 1]) / max(1e-12, times[i] - times[i - 1])
                return a_depot_traj[i - 1] + frac * (a_depot_traj[i] - a_depot_traj[i - 1])
        return a_depot_traj[-1]

    depot_24h = depot_at_time(24.0)
    depot_7d = depot_at_time(7.0 * 24.0)
    depot_end = a_depot_traj[-1]

    f_absorbed_at_24h = max(0.0, min(1.0, (dose_mg - depot_24h) / dose_mg))
    f_absorbed_at_7d = max(0.0, min(1.0, (dose_mg - depot_7d) / dose_mg))
    f_absorbed_total = max(0.0, min(1.0, (dose_mg - depot_end) / dose_mg))

    t_half_depot_h = _find_half_time(times, a_depot_traj, dose_mg)
    release_duration_days = _find_release_duration_days(times, a_depot_traj, dose_mg)

    notes_map = {
        "microsphere_PLGA": "Biphasic: burst release in first 2h then slow PLGA erosion",
        "oil_solution": "Dual-pool (30% fast, 70% slow); mimics oil vehicle absorption",
        "gel_depot": "Higuchi matrix release; rate proportional to sqrt(remaining amount)",
        "suspension_SC": "Simple first-order absorption (k=0.1/h)",
        "long_acting_implant": "Near zero-order for >90% dose, switches to first-order at end",
    }

    return DepotAbsorptionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        formulation=formulation,
        times_h=times,
        a_depot_mg=a_depot_traj,
        c_plasma_mg_L=c_plasma_traj,
        cmax=cmax,
        tmax_h=tmax_h,
        auc=auc,
        f_absorbed_at_24h=f_absorbed_at_24h,
        f_absorbed_at_7d=f_absorbed_at_7d,
        f_absorbed_total=f_absorbed_total,
        t_half_depot_h=t_half_depot_h,
        release_duration_days=release_duration_days,
        notes=notes_map[formulation],
    )


def compare_formulations(drug_name: str, dose_mg: float) -> list:
    """Simulate all 5 formulations and return sorted by release_duration_days descending.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Dose in milligrams.

    Returns
    -------
    List of DepotAbsorptionResult sorted longest release first.
    """
    results = []
    for formulation in KNOWN_FORMULATIONS:
        result = simulate_depot_absorption(
            drug_name=drug_name,
            dose_mg=dose_mg,
            formulation=formulation,
        )
        results.append(result)

    # Sort by release_duration_days descending (longest release first)
    results.sort(key=lambda r: r.release_duration_days, reverse=True)
    return results
