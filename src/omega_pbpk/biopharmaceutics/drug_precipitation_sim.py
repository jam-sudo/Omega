"""Drug precipitation simulation after oral dosing — Phase 491."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = [
    "PrecipitationSimResult",
    "simulate_precipitation",
    "estimate_absorption_loss",
    "supersaturation_maintenance_time",
    "compare_polymer_inhibition",
]

# Polymer inhibition factors: fraction of base k_precip retained
_POLYMER_FACTORS: dict[str, float] = {
    "HPMC": 0.3,
    "PVP": 0.5,
    "PVPVA": 0.4,
    "no_polymer": 1.0,
}

_DEFAULT_POLYMERS = ["HPMC", "PVP", "PVPVA", "no_polymer"]


@dataclass
class PrecipitationSimResult:
    """Time-course result for a drug precipitation simulation."""

    drug_name: str
    times_h: list[float] = field(default_factory=list)
    c_dissolved_mg_mL: list[float] = field(default_factory=list)
    c_precipitated_mg_mL: list[float] = field(default_factory=list)
    supersaturation_ratio: list[float] = field(default_factory=list)
    t_max_supersaturation_h: float = 0.0
    maintenance_time_h: float = 0.0
    fraction_precipitated: float = 0.0
    notes: str = ""


def simulate_precipitation(
    drug_name: str,
    initial_supersaturation: float,
    equilibrium_solubility_mg_mL: float,
    precipitation_rate_per_h: float = 0.5,
    t_end_h: float = 2.0,
    dt_h: float = 0.02,
    polymer: str = "no_polymer",
) -> PrecipitationSimResult:
    """Simulate time-dependent drug precipitation from supersaturated GI solution.

    ODE:
        dC_dissolved/dt = -k_precip * (C_dissolved - C_eq)  when C_dissolved > C_eq
        dC_precip/dt    =  k_precip * (C_dissolved - C_eq)  when C_dissolved > C_eq

    Parameters
    ----------
    drug_name:
        Name of the drug being simulated.
    initial_supersaturation:
        Initial supersaturation ratio S = C0 / C_eq (dimensionless, >= 1).
    equilibrium_solubility_mg_mL:
        Thermodynamic equilibrium solubility C_eq (mg/mL), must be > 0.
    precipitation_rate_per_h:
        First-order precipitation rate constant k_precip (1/h), must be > 0.
    t_end_h:
        Simulation end time (h), must be > 0.
    dt_h:
        Integration time step (h), must be > 0.
    polymer:
        Polymer inhibitor name ('HPMC', 'PVP', 'PVPVA', or 'no_polymer').

    Returns
    -------
    PrecipitationSimResult
    """
    if equilibrium_solubility_mg_mL <= 0:
        raise ValueError("equilibrium_solubility_mg_mL must be > 0")
    if precipitation_rate_per_h <= 0:
        raise ValueError("precipitation_rate_per_h must be > 0")
    if initial_supersaturation < 0:
        raise ValueError("initial_supersaturation must be >= 0")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be > 0")
    if dt_h <= 0:
        raise ValueError("dt_h must be > 0")
    if polymer not in _POLYMER_FACTORS:
        raise ValueError(f"Unknown polymer '{polymer}'. Choose from {list(_POLYMER_FACTORS)}")

    c_eq = equilibrium_solubility_mg_mL
    k = precipitation_rate_per_h * _POLYMER_FACTORS[polymer]
    c0 = initial_supersaturation * c_eq

    times: list[float] = []
    c_diss: list[float] = []
    c_prec: list[float] = []
    ss_ratio: list[float] = []

    c_dissolved = c0
    c_precipitated = 0.0
    t = 0.0
    n_steps = math.ceil(t_end_h / dt_h)

    for _ in range(n_steps + 1):
        times.append(round(t, 10))
        c_diss.append(c_dissolved)
        c_prec.append(c_precipitated)
        ratio = c_dissolved / c_eq if c_eq > 0 else 1.0
        ss_ratio.append(ratio)
        if t < t_end_h:
            # Forward Euler
            dc_dt = -k * max(c_dissolved - c_eq, 0.0)
            c_dissolved = max(c_dissolved + dc_dt * dt_h, c_eq)
            c_precipitated = c_precipitated + k * max(c_diss[-1] - c_eq, 0.0) * dt_h
            t = t + dt_h
            if t > t_end_h:
                t = t_end_h

    # t_max_supersaturation: time at which supersaturation is maximum (t=0)
    t_max_ss = 0.0

    # maintenance_time: duration C_dissolved >= 1.0 * C_eq (it always starts >= C_eq)
    # but interpreted here as hours above threshold=2.0 (default), computed in
    # supersaturation_maintenance_time; store raw maintenance at threshold=1.0
    maintenance_time = _calc_maintenance_time(times, ss_ratio, threshold_ratio=2.0)

    # fraction precipitated at end relative to initial dissolved
    frac_precip = 0.0 if c0 <= 0 else min(c_prec[-1] / c0, 1.0)

    notes_parts = []
    if initial_supersaturation <= 1.0:
        notes_parts.append("No precipitation: initial supersaturation <= 1.")
    if polymer != "no_polymer":
        notes_parts.append(f"Polymer {polymer} applied: effective k_precip = {k:.4f}/h.")
    notes = " ".join(notes_parts) if notes_parts else "Simulation completed normally."

    return PrecipitationSimResult(
        drug_name=drug_name,
        times_h=times,
        c_dissolved_mg_mL=c_diss,
        c_precipitated_mg_mL=c_prec,
        supersaturation_ratio=ss_ratio,
        t_max_supersaturation_h=t_max_ss,
        maintenance_time_h=maintenance_time,
        fraction_precipitated=frac_precip,
        notes=notes,
    )


def _calc_maintenance_time(
    times: list[float],
    ss_ratio: list[float],
    threshold_ratio: float,
) -> float:
    """Compute total time (h) that supersaturation ratio >= threshold_ratio."""
    if len(times) < 2:
        return 0.0
    total = 0.0
    for i in range(len(times) - 1):
        dt = times[i + 1] - times[i]
        if ss_ratio[i] >= threshold_ratio:
            total += dt
    return total


def estimate_absorption_loss(
    precipitation_profile: PrecipitationSimResult,
    absorption_rate_per_h: float = 0.3,
) -> float:
    """Estimate fraction of initial dissolved drug lost to precipitation before absorption.

    Uses trapezoidal integration of the competing processes:
        - Precipitation flux: k_precip * (C - C_eq)  [taken from stored profile]
        - Absorption flux: k_abs * C_dissolved

    Returns the fraction of total initial drug (dose proxy = C0) that precipitates
    before being absorbed, approximated as:

        loss_fraction = integral(precipitation_flux) /
                        (integral(precipitation_flux) + integral(absorption_flux))

    Parameters
    ----------
    precipitation_profile:
        Result from simulate_precipitation.
    absorption_rate_per_h:
        First-order absorption rate constant k_abs (1/h).

    Returns
    -------
    float
        Fraction in [0, 1] representing dose lost to precipitation.
    """
    if absorption_rate_per_h < 0:
        raise ValueError("absorption_rate_per_h must be >= 0")

    times = precipitation_profile.times_h
    c_diss = precipitation_profile.c_dissolved_mg_mL
    c_prec = precipitation_profile.c_precipitated_mg_mL

    if len(times) < 2:
        return 0.0

    # Cumulative precipitation = final precipitated amount relative to c0
    c0 = c_diss[0]
    if c0 <= 0:
        return 0.0

    # Integrate absorption flux: k_abs * C_dissolved(t) dt
    absorbed_integral = 0.0
    precip_integral = 0.0
    for i in range(len(times) - 1):
        dt = times[i + 1] - times[i]
        c_avg = (c_diss[i] + c_diss[i + 1]) / 2.0
        dp_avg = ((c_prec[i + 1] - c_prec[i]) / dt) if dt > 0 else 0.0
        absorbed_integral += absorption_rate_per_h * c_avg * dt
        precip_integral += max(dp_avg, 0.0) * dt

    total = absorbed_integral + precip_integral
    if total <= 0:
        return 0.0
    return min(precip_integral / total, 1.0)


def supersaturation_maintenance_time(
    precipitation_profile: PrecipitationSimResult,
    threshold_ratio: float = 2.0,
) -> float:
    """Calculate the time (h) that the supersaturation ratio is above a threshold.

    Parameters
    ----------
    precipitation_profile:
        Result from simulate_precipitation.
    threshold_ratio:
        Supersaturation ratio threshold (default 2.0, i.e. 2x equilibrium solubility).

    Returns
    -------
    float
        Time in hours above the threshold.
    """
    if threshold_ratio <= 0:
        raise ValueError("threshold_ratio must be > 0")

    return _calc_maintenance_time(
        precipitation_profile.times_h,
        precipitation_profile.supersaturation_ratio,
        threshold_ratio,
    )


def compare_polymer_inhibition(
    drug_name: str,
    initial_supersaturation: float,
    solubility: float,
    polymers: list[str] | None = None,
) -> list[dict]:
    """Compare effect of different polymer inhibitors on precipitation maintenance.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    initial_supersaturation:
        Initial supersaturation ratio S = C0 / C_eq.
    solubility:
        Equilibrium solubility (mg/mL).
    polymers:
        List of polymer names to compare. Defaults to
        ['HPMC', 'PVP', 'PVPVA', 'no_polymer'].

    Returns
    -------
    list[dict]
        Each dict contains:
        - 'polymer': str
        - 'maintenance_time_h': float
        - 'fraction_precipitated': float
        - 'effective_k_precip': float
        Sorted by maintenance_time_h descending (longest maintenance first).
    """
    if polymers is None:
        polymers = _DEFAULT_POLYMERS

    for p in polymers:
        if p not in _POLYMER_FACTORS:
            raise ValueError(f"Unknown polymer '{p}'. Choose from {list(_POLYMER_FACTORS)}")

    results = []
    for polymer in polymers:
        sim = simulate_precipitation(
            drug_name=drug_name,
            initial_supersaturation=initial_supersaturation,
            equilibrium_solubility_mg_mL=solubility,
            precipitation_rate_per_h=0.5,
            t_end_h=4.0,
            dt_h=0.02,
            polymer=polymer,
        )
        maint = supersaturation_maintenance_time(sim, threshold_ratio=2.0)
        effective_k = 0.5 * _POLYMER_FACTORS[polymer]
        results.append(
            {
                "polymer": polymer,
                "maintenance_time_h": maint,
                "fraction_precipitated": sim.fraction_precipitated,
                "effective_k_precip": effective_k,
            }
        )

    # Sort by maintenance time descending
    results.sort(key=lambda d: d["maintenance_time_h"], reverse=True)
    return results
