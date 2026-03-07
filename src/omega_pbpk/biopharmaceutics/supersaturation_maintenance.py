"""Supersaturation maintenance by polymeric precipitation inhibitors.

Models how polymers such as HPMC, PVP, and PVPVA maintain drug supersaturation
in the GI tract to improve oral absorption of poorly soluble compounds.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)

__all__ = [
    "SupersaturationMaintenanceResult",
    "simulate_supersaturation_maintenance",
    "compare_polymers",
]

_VALID_POLYMERS = ("none", "HPMC", "PVP", "PVPVA")

# Polymer k_precip inhibition factors (multiplied by no-polymer rate)
_PRECIP_FACTOR: dict[str, float] = {
    "none": 1.0,
    "HPMC": 0.15,
    "PVP": 0.25,
    "PVPVA": 0.20,
}

# Default induction delays per polymer (hours)
_INDUCTION_DELAY_H: dict[str, float] = {
    "none": 0.0,
    "HPMC": 0.5,
    "PVP": 0.3,
    "PVPVA": 0.4,
}


@dataclass(frozen=True)
class SupersaturationMaintenanceResult:
    """Result of supersaturation maintenance simulation."""

    drug_name: str
    polymer_name: str  # "HPMC", "PVP", "PVPVA", "none"
    dose_mg: float

    # Supersaturation profile
    times_h: list[float]
    c_dissolved_mg_mL: list[float]  # Drug in solution
    c_precipitated_mg: list[float]  # Precipitated drug
    supersaturation_ratio: list[float]  # C / crystalline_solubility

    # Maintenance metrics
    max_supersaturation: float  # Peak SS ratio
    duration_above_2x_ss_h: float  # Hours above 2x crystalline solubility
    mean_dissolved_mg_mL: float  # Time-averaged dissolved concentration

    # Absorption impact
    f_absorbed: float  # Total fraction absorbed
    f_absorbed_no_polymer: float  # Reference (no polymer)
    absorption_enhancement_ratio: float  # f_absorbed / f_absorbed_ref

    # Polymer effectiveness
    polymer_effective: bool  # If duration_above_2x_ss > 2 h
    induction_time_h: float  # Time before precipitation starts

    notes: str


def _run_simulation(
    dose_mg: float,
    crystalline_solubility_mg_mL: float,
    amorphous_solubility_mg_mL: float,
    k_precip: float,
    induction_delay_h: float,
    gi_volume_mL: float,
    ka_absorption_per_h: float,
    t_end_h: float,
    dt_h: float,
) -> tuple[list[float], list[float], list[float], float, float]:
    """Forward-Euler ODE simulation.

    Returns
    -------
    times, c_dissolved, c_precipitated, m_absorbed_total, induction_time_h
    """
    times: list[float] = []
    c_diss: list[float] = []
    c_prec: list[float] = []

    # Initial conditions
    initial_mass = min(dose_mg, amorphous_solubility_mg_mL * gi_volume_mL)
    c_d = initial_mass / gi_volume_mL  # mg/mL
    c_p = 0.0  # mg precipitated

    m_absorbed = 0.0
    induction_time = float("inf")
    t = 0.0

    times.append(t)
    c_diss.append(c_d)
    c_prec.append(c_p)

    while t < t_end_h - 1e-12:
        step = min(dt_h, t_end_h - t)

        # Before induction delay: no precipitation
        if t >= induction_delay_h:
            supersaturation = max(0.0, c_d - crystalline_solubility_mg_mL)
            precip_rate = k_precip * supersaturation * gi_volume_mL  # mg/h
            if precip_rate > 0 and math.isinf(induction_time):
                induction_time = t
        else:
            precip_rate = 0.0

        absorption_rate = ka_absorption_per_h * c_d * gi_volume_mL  # mg/h
        rediss_rate = 0.1 * c_p  # mg/h

        dc_d = (rediss_rate - precip_rate - absorption_rate) / gi_volume_mL
        dc_p = precip_rate - rediss_rate

        c_d = max(0.0, c_d + dc_d * step)
        c_p = max(0.0, c_p + dc_p * step)
        m_absorbed += absorption_rate * step
        t = t + step

        times.append(t)
        c_diss.append(c_d)
        c_prec.append(c_p)

    return times, c_diss, c_prec, m_absorbed, induction_time


def simulate_supersaturation_maintenance(
    drug_name: str,
    dose_mg: float,
    crystalline_solubility_mg_mL: float,
    amorphous_solubility_mg_mL: float,
    polymer_name: str = "HPMC",
    gi_volume_mL: float = 500.0,
    ka_absorption_per_h: float = 1.5,
    k_precip_per_h_no_polymer: float = 5.0,
    induction_delay_h: float | None = None,
    t_end_h: float = 4.0,
    dt_h: float = 0.05,
) -> SupersaturationMaintenanceResult:
    """Simulate supersaturation maintenance by a polymeric inhibitor.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Administered dose (mg).
    crystalline_solubility_mg_mL:
        Equilibrium (crystalline) solubility of the drug in GI fluid (mg/mL).
    amorphous_solubility_mg_mL:
        Apparent solubility of the amorphous form (mg/mL). Must be greater
        than crystalline_solubility_mg_mL.
    polymer_name:
        Precipitation inhibitor polymer: "none", "HPMC", "PVP", or "PVPVA".
    gi_volume_mL:
        Volume of GI fluid (mL). Default 500 mL.
    ka_absorption_per_h:
        First-order absorption rate constant (h^-1).
    k_precip_per_h_no_polymer:
        Precipitation rate constant without polymer (h^-1).
    induction_delay_h:
        Override the default induction delay for the polymer (h).
    t_end_h:
        Simulation end time (h).
    dt_h:
        Time step (h).

    Returns
    -------
    SupersaturationMaintenanceResult
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if crystalline_solubility_mg_mL <= 0:
        raise ValueError("crystalline_solubility_mg_mL must be > 0")
    if amorphous_solubility_mg_mL <= crystalline_solubility_mg_mL:
        raise ValueError("amorphous_solubility_mg_mL must be > crystalline_solubility_mg_mL")
    if polymer_name not in _VALID_POLYMERS:
        raise ValueError(f"polymer_name must be one of {_VALID_POLYMERS}, got '{polymer_name}'")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be > 0")
    if dt_h <= 0:
        raise ValueError("dt_h must be > 0")

    # Polymer-specific parameters
    k_precip = k_precip_per_h_no_polymer * _PRECIP_FACTOR[polymer_name]
    delay = induction_delay_h if induction_delay_h is not None else _INDUCTION_DELAY_H[polymer_name]

    # Run simulation with polymer
    times, c_diss, c_prec, m_absorbed, induction_time = _run_simulation(
        dose_mg=dose_mg,
        crystalline_solubility_mg_mL=crystalline_solubility_mg_mL,
        amorphous_solubility_mg_mL=amorphous_solubility_mg_mL,
        k_precip=k_precip,
        induction_delay_h=delay,
        gi_volume_mL=gi_volume_mL,
        ka_absorption_per_h=ka_absorption_per_h,
        t_end_h=t_end_h,
        dt_h=dt_h,
    )

    # Reference: no polymer
    _, _, _, m_absorbed_ref, _ = _run_simulation(
        dose_mg=dose_mg,
        crystalline_solubility_mg_mL=crystalline_solubility_mg_mL,
        amorphous_solubility_mg_mL=amorphous_solubility_mg_mL,
        k_precip=k_precip_per_h_no_polymer * _PRECIP_FACTOR["none"],
        induction_delay_h=_INDUCTION_DELAY_H["none"],
        gi_volume_mL=gi_volume_mL,
        ka_absorption_per_h=ka_absorption_per_h,
        t_end_h=t_end_h,
        dt_h=dt_h,
    )

    # ------------------------------------------------------------------ #
    # Derived metrics                                                     #
    # ------------------------------------------------------------------ #
    ss_ratio = [c / crystalline_solubility_mg_mL for c in c_diss]
    max_ss = max(ss_ratio) if ss_ratio else 1.0

    # Duration above 2x crystalline solubility
    threshold_2x = 2.0 * crystalline_solubility_mg_mL
    duration_2x = sum(dt_h for c in c_diss[:-1] if c > threshold_2x)

    # Mean dissolved concentration (simple trapezoidal via average)
    if len(c_diss) > 1:
        mean_dissolved = sum(c_diss) / len(c_diss)
    else:
        mean_dissolved = c_diss[0]

    f_absorbed = min(1.0, m_absorbed / dose_mg) if dose_mg > 0 else 0.0
    f_absorbed_ref = min(1.0, m_absorbed_ref / dose_mg) if dose_mg > 0 else 0.0

    if f_absorbed_ref > 0:
        enhancement = f_absorbed / f_absorbed_ref
    else:
        enhancement = 1.0 if f_absorbed == 0 else float("inf")

    polymer_effective = duration_2x > 2.0

    actual_induction = induction_time if not math.isinf(induction_time) else t_end_h

    notes = f"Polymer: {polymer_name}; k_precip={k_precip:.3f} h^-1; induction delay={delay:.2f} h"

    return SupersaturationMaintenanceResult(
        drug_name=drug_name,
        polymer_name=polymer_name,
        dose_mg=dose_mg,
        times_h=times,
        c_dissolved_mg_mL=c_diss,
        c_precipitated_mg=c_prec,
        supersaturation_ratio=ss_ratio,
        max_supersaturation=max_ss,
        duration_above_2x_ss_h=duration_2x,
        mean_dissolved_mg_mL=mean_dissolved,
        f_absorbed=f_absorbed,
        f_absorbed_no_polymer=f_absorbed_ref,
        absorption_enhancement_ratio=enhancement,
        polymer_effective=polymer_effective,
        induction_time_h=actual_induction,
        notes=notes,
    )


def compare_polymers(
    drug_name: str,
    dose_mg: float,
    crystalline_solubility_mg_mL: float,
    amorphous_solubility_mg_mL: float,
    **kwargs,
) -> list[SupersaturationMaintenanceResult]:
    """Simulate all polymers and rank by absorption enhancement.

    Parameters
    ----------
    drug_name, dose_mg, crystalline_solubility_mg_mL, amorphous_solubility_mg_mL:
        Passed to simulate_supersaturation_maintenance for each polymer.
    **kwargs:
        Additional keyword arguments forwarded to
        simulate_supersaturation_maintenance.

    Returns
    -------
    List of 4 SupersaturationMaintenanceResult objects (one per polymer),
    sorted by absorption_enhancement_ratio descending.
    """
    results: list[SupersaturationMaintenanceResult] = []
    for polymer in _VALID_POLYMERS:
        result = simulate_supersaturation_maintenance(
            drug_name=drug_name,
            dose_mg=dose_mg,
            crystalline_solubility_mg_mL=crystalline_solubility_mg_mL,
            amorphous_solubility_mg_mL=amorphous_solubility_mg_mL,
            polymer_name=polymer,
            **kwargs,
        )
        results.append(result)

    return sorted(results, key=lambda r: r.absorption_enhancement_ratio, reverse=True)
