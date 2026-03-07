"""Microbiome-mediated drug metabolism simulation (Phase 879).

Models gut microbiome-mediated drug metabolism for oral drugs.
The microbiome can metabolize drugs before systemic absorption
(e.g., sulfasalazine -> 5-ASA + sulfapyridine, prodrug activation,
or drug inactivation).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "MicrobiomePKResult",
    "simulate_microbiome_pk",
    "compare_microbiome_activities",
]

# Activity scaling factors relative to "normal" (1.0x)
_ACTIVITY_SCALE: dict[str, float] = {
    "none": 0.0,
    "low": 0.5,
    "normal": 1.0,
    "high": 2.0,
}

_VALID_ACTIVITIES = set(_ACTIVITY_SCALE.keys())


@dataclass
class MicrobiomePKResult:
    """Result of microbiome-mediated drug metabolism simulation.

    Attributes
    ----------
    drug_name : Drug identifier.
    dose_mg : Oral dose administered (mg).
    microbiome_activity : Activity level ("none", "low", "normal", "high").
    times_h : Simulation time points (h).
    c_parent_plasma_mg_L : Parent drug plasma concentration (mg/L).
    c_metabolite_plasma_mg_L : Microbiome metabolite plasma concentration (mg/L).
    cmax_parent : Peak parent plasma concentration (mg/L).
    tmax_parent_h : Time to peak parent concentration (h).
    auc_parent : AUC of parent drug (mg*h/L), trapezoidal.
    cmax_metabolite : Peak metabolite plasma concentration (mg/L).
    tmax_metabolite_h : Time to peak metabolite concentration (h).
    auc_metabolite : AUC of metabolite (mg*h/L), trapezoidal.
    f_microbiome_metabolism : Fraction of dose metabolized by microbiome (0-1).
    metabolite_ratio : AUC_metabolite / AUC_parent (0 if AUC_parent=0).
    notes : Interpretation text.
    """

    drug_name: str
    dose_mg: float
    microbiome_activity: str
    times_h: list[float]
    c_parent_plasma_mg_L: list[float]
    c_metabolite_plasma_mg_L: list[float]
    cmax_parent: float
    tmax_parent_h: float
    auc_parent: float
    cmax_metabolite: float
    tmax_metabolite_h: float
    auc_metabolite: float
    f_microbiome_metabolism: float
    metabolite_ratio: float
    notes: str


def _trapz(y: list[float], x: list[float]) -> float:
    """Manual trapezoidal integration."""
    total = 0.0
    for i in range(len(y) - 1):
        total += 0.5 * (y[i] + y[i + 1]) * (x[i + 1] - x[i])
    return total


def simulate_microbiome_pk(
    drug_name: str,
    dose_mg: float,
    microbiome_activity: str = "normal",
    k_abs_per_h: float = 1.0,
    k_microb_per_h: float = 0.5,
    f_microb: float = 0.3,
    cl_parent_L_per_h: float = 10.0,
    cl_met_L_per_h: float = 15.0,
    vd_L: float = 50.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> MicrobiomePKResult:
    """Simulate microbiome-mediated drug metabolism with 3-compartment ODE.

    Compartments: GI lumen (a_gut), parent plasma (A_parent), metabolite plasma
    (A_met). Forward Euler integration.

    Parameters
    ----------
    drug_name : Drug identifier string.
    dose_mg : Oral dose (mg). Must be > 0.
    microbiome_activity : One of "none", "low", "normal", "high".
    k_abs_per_h : First-order GI absorption rate constant (1/h).
    k_microb_per_h : Microbiome metabolism rate constant at normal activity (1/h).
    f_microb : Fraction of microbially metabolized drug appearing as metabolite
        in plasma (0-1). Scaled by activity.
    cl_parent_L_per_h : Systemic clearance of parent drug (L/h).
    cl_met_L_per_h : Systemic clearance of metabolite (L/h).
    vd_L : Volume of distribution (L) shared by parent and metabolite.
    t_end_h : Simulation end time (h).
    dt_h : Integration step size (h).
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if microbiome_activity not in _VALID_ACTIVITIES:
        raise ValueError(
            f"microbiome_activity must be one of {sorted(_VALID_ACTIVITIES)}, "
            f"got '{microbiome_activity}'"
        )

    scale = _ACTIVITY_SCALE[microbiome_activity]
    k_microb_eff = k_microb_per_h * scale
    f_microb_eff = f_microb * scale  # fraction metabolized by microbiome
    # Clamp f_microb_eff to [0, 1]
    f_microb_eff = max(0.0, min(1.0, f_microb_eff))

    ke_parent = cl_parent_L_per_h / vd_L
    ke_met = cl_met_L_per_h / vd_L

    n_steps = max(int(math.ceil(t_end_h / dt_h)), 1)
    times = [i * dt_h for i in range(n_steps + 1)]

    c_parent: list[float] = [0.0] * (n_steps + 1)
    c_met: list[float] = [0.0] * (n_steps + 1)

    # State: amounts in mg (gut lumen) and mg (central compartments)
    a_gut = float(dose_mg)
    a_parent = 0.0
    a_met = 0.0

    for i in range(n_steps):
        # ODEs
        # da_gut/dt = -(k_abs + k_microb_eff) * a_gut
        da_gut = -(k_abs_per_h + k_microb_eff) * a_gut

        # dA_parent/dt = (1 - f_microb_eff) * k_abs * a_gut - ke_parent * A_parent
        da_parent = (1.0 - f_microb_eff) * k_abs_per_h * a_gut - ke_parent * a_parent

        # dA_met/dt = f_microb_eff * k_microb_eff * a_gut - ke_met * A_met
        da_met = f_microb_eff * k_microb_eff * a_gut - ke_met * a_met

        # Forward Euler update
        a_gut = max(0.0, a_gut + da_gut * dt_h)
        a_parent = max(0.0, a_parent + da_parent * dt_h)
        a_met = max(0.0, a_met + da_met * dt_h)

        c_parent[i + 1] = a_parent / vd_L
        c_met[i + 1] = a_met / vd_L

    # PK metrics — parent
    cmax_parent = max(c_parent)
    tmax_parent_h = times[c_parent.index(cmax_parent)]
    auc_parent = _trapz(c_parent, times)

    # PK metrics — metabolite
    cmax_metabolite = max(c_met)
    tmax_metabolite_h = times[c_met.index(cmax_metabolite)]
    auc_metabolite = _trapz(c_met, times)

    # Fraction metabolized by microbiome
    # f_microbiome_metabolism = k_microb_eff / (k_abs + k_microb_eff) * f_microb_eff
    total_rate = k_abs_per_h + k_microb_eff
    if total_rate > 0:
        f_microbiome_metabolism = (k_microb_eff / total_rate) * f_microb_eff
    else:
        f_microbiome_metabolism = 0.0

    metabolite_ratio = auc_metabolite / auc_parent if auc_parent > 0 else 0.0

    # Notes
    if metabolite_ratio > 1.0:
        notes = "Extensive microbiome metabolism — metabolite dominant"
    elif metabolite_ratio > 0.2:
        notes = "Moderate microbiome metabolism"
    else:
        notes = "Minimal microbiome contribution"

    return MicrobiomePKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        microbiome_activity=microbiome_activity,
        times_h=times,
        c_parent_plasma_mg_L=c_parent,
        c_metabolite_plasma_mg_L=c_met,
        cmax_parent=cmax_parent,
        tmax_parent_h=tmax_parent_h,
        auc_parent=auc_parent,
        cmax_metabolite=cmax_metabolite,
        tmax_metabolite_h=tmax_metabolite_h,
        auc_metabolite=auc_metabolite,
        f_microbiome_metabolism=f_microbiome_metabolism,
        metabolite_ratio=metabolite_ratio,
        notes=notes,
    )


def compare_microbiome_activities(
    drug_name: str,
    dose_mg: float,
    k_abs_per_h: float = 1.0,
    k_microb_per_h: float = 0.5,
    f_microb: float = 0.3,
    cl_parent_L_per_h: float = 10.0,
    cl_met_L_per_h: float = 15.0,
    vd_L: float = 50.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> list[MicrobiomePKResult]:
    """Compare all 4 microbiome activity levels for a drug.

    Returns a list of 4 MicrobiomePKResult objects sorted by auc_parent
    descending (highest parent AUC first, corresponding to lowest microbiome
    activity).

    Parameters
    ----------
    drug_name : Drug identifier string.
    dose_mg : Oral dose (mg).
    """
    results = []
    for activity in _VALID_ACTIVITIES:
        result = simulate_microbiome_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            microbiome_activity=activity,
            k_abs_per_h=k_abs_per_h,
            k_microb_per_h=k_microb_per_h,
            f_microb=f_microb,
            cl_parent_L_per_h=cl_parent_L_per_h,
            cl_met_L_per_h=cl_met_L_per_h,
            vd_L=vd_L,
            t_end_h=t_end_h,
            dt_h=dt_h,
        )
        results.append(result)

    results.sort(key=lambda r: r.auc_parent, reverse=True)
    return results
