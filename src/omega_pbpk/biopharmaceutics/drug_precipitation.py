"""Drug precipitation and supersaturation/precipitation kinetics."""
from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "PrecipitationResult",
    "simulate_precipitation",
    "supersaturation_index",
]


@dataclass(frozen=True)
class PrecipitationResult:
    drug_name: str
    initial_conc_mg_mL: float
    equilibrium_solubility_mg_mL: float
    supersaturation_ratio: float   # C / Cs
    precipitation_rate_mg_per_mL_h: float
    time_to_equilibrium_h: float
    absorbed_fraction_before_precip: float
    risk_category: str   # "none", "low", "moderate", "high"
    formulation_strategy: list[str]
    notes: str


def supersaturation_index(
    actual_conc_mg_mL: float,
    equilibrium_solubility_mg_mL: float,
) -> float:
    """Return supersaturation ratio C/Cs.

    Parameters
    ----------
    actual_conc_mg_mL : float
        Actual drug concentration (mg/mL).
    equilibrium_solubility_mg_mL : float
        Equilibrium solubility (mg/mL, must be > 0).

    Returns
    -------
    float
        C/Cs ratio (1.0 = saturated, <1 = unsaturated, >1 = supersaturated).
    """
    if equilibrium_solubility_mg_mL <= 0:
        raise ValueError("equilibrium_solubility_mg_mL must be > 0")
    return actual_conc_mg_mL / equilibrium_solubility_mg_mL


def simulate_precipitation(
    drug_name: str,
    initial_conc_mg_mL: float,
    equilibrium_solubility_mg_mL: float,
    k_precip_per_h: float = 0.5,
    k_abs_per_h: float = 1.0,
    t_end_h: float = 4.0,
    dt_h: float = 0.01,
) -> PrecipitationResult:
    """Simulate drug precipitation competing with absorption.

    Two competing processes:
      - Absorption (removes drug at rate k_abs * C)
      - Precipitation (removes supersaturated drug at rate k_precip * max(0, C - Cs))

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    initial_conc_mg_mL : float
        Initial drug concentration (mg/mL, must be >= 0).
    equilibrium_solubility_mg_mL : float
        Equilibrium solubility (mg/mL, must be > 0).
    k_precip_per_h : float
        First-order precipitation rate constant (1/h, must be >= 0).
    k_abs_per_h : float
        First-order absorption rate constant (1/h, must be > 0).
    t_end_h : float
        Simulation end time (h).
    dt_h : float
        Time step (h).

    Returns
    -------
    PrecipitationResult
    """
    if initial_conc_mg_mL < 0:
        raise ValueError("initial_conc_mg_mL must be >= 0")
    if equilibrium_solubility_mg_mL <= 0:
        raise ValueError("equilibrium_solubility_mg_mL must be > 0")
    if k_precip_per_h < 0:
        raise ValueError("k_precip_per_h must be >= 0")
    if k_abs_per_h <= 0:
        raise ValueError("k_abs_per_h must be > 0")

    cs = equilibrium_solubility_mg_mL
    c = initial_conc_mg_mL
    n_steps = int(t_end_h / dt_h)
    time_to_eq = t_end_h  # default if equilibrium never reached

    for i in range(n_steps):
        excess = max(0.0, c - cs)
        dc = (-k_abs_per_h * c - k_precip_per_h * excess) * dt_h
        c = max(0.0, c + dc)
        if cs > 0 and abs(c - cs) < 0.001 * cs:
            time_to_eq = (i + 1) * dt_h
            break

    # Precipitation rate at t=0
    precip_rate_0 = k_precip_per_h * max(0.0, initial_conc_mg_mL - cs)

    # Supersaturation ratio
    ss_ratio = initial_conc_mg_mL / max(cs, 1e-9)

    # Absorbed fraction approximation
    if ss_ratio <= 1.0:
        absorbed_fraction = 1.0 - math.exp(-k_abs_per_h * t_end_h)
    else:
        absorbed_fraction = k_abs_per_h / (k_abs_per_h + k_precip_per_h)
        absorbed_fraction = min(
            absorbed_fraction,
            1.0 - math.exp(-k_abs_per_h * t_end_h),
        )

    # Risk category
    if ss_ratio <= 1.0:
        risk_category = "none"
    elif ss_ratio < 2.0:
        risk_category = "low"
    elif ss_ratio < 5.0:
        risk_category = "moderate"
    else:
        risk_category = "high"

    # Formulation strategy
    if ss_ratio > 2.0:
        formulation_strategy = [
            "Use amorphous solid dispersion",
            "Add precipitation inhibitor (HPMC/PVP)",
        ]
    elif ss_ratio > 1.0:
        formulation_strategy = [
            "Monitor supersaturation in vivo",
            "Consider buffer optimization",
        ]
    else:
        formulation_strategy = ["No precipitation risk"]

    notes = (
        f"Forward Euler simulation with dt={dt_h} h. "
        f"Competing absorption (k_abs={k_abs_per_h}/h) and precipitation "
        f"(k_precip={k_precip_per_h}/h). "
        f"Equilibrium solubility={cs} mg/mL."
    )

    return PrecipitationResult(
        drug_name=drug_name,
        initial_conc_mg_mL=initial_conc_mg_mL,
        equilibrium_solubility_mg_mL=cs,
        supersaturation_ratio=ss_ratio,
        precipitation_rate_mg_per_mL_h=precip_rate_0,
        time_to_equilibrium_h=time_to_eq,
        absorbed_fraction_before_precip=absorbed_fraction,
        risk_category=risk_category,
        formulation_strategy=formulation_strategy,
        notes=notes,
    )
