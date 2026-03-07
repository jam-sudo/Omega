"""Drug binding to red blood cells — blood-to-plasma ratio modelling.

Models drug partitioning into red blood cells (RBCs), which affects the
blood-to-plasma ratio (B/P). A B/P > 1 indicates drug accumulation in RBCs,
impacting interpretation of blood vs plasma PK and hepatic/renal CL scaling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz

__all__ = [
    "RBCPartitionResult",
    "RBCPKResult",
    "blood_to_plasma_ratio",
    "fu_blood_from_bp",
    "rbc_partitioning",
    "cl_blood_to_plasma",
    "simulate_rbc_pk",
]

# Kp_rbc clamp limits
_KP_RBC_MIN = 0.1
_KP_RBC_MAX = 20.0


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RBCPartitionResult:
    """RBC partitioning prediction from physicochemical properties."""

    logP: float
    charge: str
    hematocrit: float
    kp_rbc: float
    bp_ratio: float
    fu_blood: float
    notes: str


@dataclass(frozen=True)
class RBCPKResult:
    """PK simulation results incorporating RBC binding effects."""

    drug_name: str
    dose_mg: float
    bp_ratio: float
    times_h: list[float]
    c_plasma: list[float]
    c_blood: list[float]
    cmax_plasma: float
    cmax_blood: float
    auc_plasma: float
    auc_blood: float
    notes: str


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def blood_to_plasma_ratio(
    fu_plasma: float,
    fu_blood: float,
    hematocrit: float = 0.45,
) -> float:
    """Calculate blood-to-plasma concentration ratio from free fractions.

    Uses the first-principles partition approach:
        Kp_rbc = fu_plasma / fu_rbc  (ratio of free fractions)
        B/P = 1 - hct + hct * Kp_rbc

    Parameters
    ----------
    fu_plasma:
        Fraction unbound in plasma (0 < fu_plasma ≤ 1).
    fu_blood:
        Fraction unbound in blood (used as fu_rbc proxy) (0 < fu_blood ≤ 1).
    hematocrit:
        Hematocrit fraction (default 0.45). Must be in [0.3, 0.6].

    Returns
    -------
    float
        Blood-to-plasma ratio B/P.
    """
    if not (0.0 < fu_plasma <= 1.0):
        raise ValueError(f"fu_plasma must be in (0, 1], got {fu_plasma}")
    if not (0.0 < fu_blood <= 1.0):
        raise ValueError(f"fu_blood must be in (0, 1], got {fu_blood}")
    if not (0.3 <= hematocrit <= 0.6):
        raise ValueError(f"hematocrit must be in [0.3, 0.6], got {hematocrit}")

    kp_rbc = fu_plasma / fu_blood
    return (1.0 - hematocrit) + hematocrit * kp_rbc


def fu_blood_from_bp(
    bp_ratio: float,
    fu_plasma: float,
    hematocrit: float = 0.45,
) -> float:
    """Derive fraction unbound in blood from B/P ratio and fu_plasma.

    Simplified relationship: fu_blood = fu_plasma / bp_ratio

    Parameters
    ----------
    bp_ratio:
        Blood-to-plasma ratio. Must be > 0.
    fu_plasma:
        Fraction unbound in plasma (0 < fu_plasma ≤ 1).
    hematocrit:
        Hematocrit (not used in simplified formula; included for API consistency).

    Returns
    -------
    float
        Estimated fu_blood, clamped to (0, 1].
    """
    if bp_ratio <= 0:
        raise ValueError(f"bp_ratio must be > 0, got {bp_ratio}")
    if not (0.0 < fu_plasma <= 1.0):
        raise ValueError(f"fu_plasma must be in (0, 1], got {fu_plasma}")

    fu_blood = fu_plasma / bp_ratio
    return min(1.0, max(1e-6, fu_blood))


def rbc_partitioning(
    logP: float,
    pka: float | None = None,
    charge: str = "neutral",
    hematocrit: float = 0.45,
) -> RBCPartitionResult:
    """Predict RBC partitioning and B/P ratio from physicochemical properties.

    Empirical equations:
        Neutral : Kp_rbc = exp(0.3 * logP + 0.1)
        Basic   : Kp_rbc = exp(0.5 * logP + 0.5)  — hemoglobin binding
        Acidic  : Kp_rbc = exp(0.1 * logP - 0.2)  — RBC exclusion

    Kp_rbc is clamped to [0.1, 20.0].

    Parameters
    ----------
    logP:
        Octanol-water partition coefficient (log scale).
    pka:
        Most basic or acidic pKa (informational only).
    charge:
        Ionization character at physiological pH: 'neutral', 'basic', or 'acidic'.
    hematocrit:
        Hematocrit fraction (default 0.45). Must be in [0.3, 0.6].

    Returns
    -------
    RBCPartitionResult
    """
    if not (0.3 <= hematocrit <= 0.6):
        raise ValueError(f"hematocrit must be in [0.3, 0.6], got {hematocrit}")

    charge_lower = charge.lower()
    if charge_lower == "neutral":
        kp_raw = math.exp(0.3 * logP + 0.1)
        mechanism = "Passive partitioning into RBC lipid bilayer (neutral compound)."
    elif charge_lower == "basic":
        kp_raw = math.exp(0.5 * logP + 0.5)
        mechanism = "Enhanced RBC uptake via hemoglobin binding (basic compound)."
    elif charge_lower == "acidic":
        kp_raw = math.exp(0.1 * logP - 0.2)
        mechanism = "Reduced RBC entry due to electrostatic exclusion (acidic compound)."
    else:
        raise ValueError(f"charge must be 'neutral', 'basic', or 'acidic', got '{charge}'")

    kp_rbc = max(_KP_RBC_MIN, min(_KP_RBC_MAX, kp_raw))
    bp_ratio = (1.0 - hematocrit) + hematocrit * kp_rbc

    # Derive fu_blood from fu_plasma = 1.0 assumption (relative measure)
    # fu_blood = (1 - hct + hct * kp_rbc * fu_plasma) / bp_ratio  [simplified]
    # Using fu_plasma = 1.0 as a reference (relative fu_blood)
    fu_blood = 1.0 / bp_ratio  # assumes fu_plasma = 1 (relative)

    clamped_note = ""
    if kp_raw < _KP_RBC_MIN or kp_raw > _KP_RBC_MAX:
        clamped_note = f" (Kp_rbc clamped from {kp_raw:.3f})"

    notes = f"{mechanism} Kp_rbc={kp_rbc:.3f}, B/P={bp_ratio:.3f}.{clamped_note}"
    if pka is not None:
        notes = f"pKa={pka:.1f}. {notes}"

    return RBCPartitionResult(
        logP=logP,
        charge=charge,
        hematocrit=hematocrit,
        kp_rbc=kp_rbc,
        bp_ratio=bp_ratio,
        fu_blood=fu_blood,
        notes=notes,
    )


def cl_blood_to_plasma(
    cl_blood_L_per_h: float,
    bp_ratio: float,
) -> float:
    """Convert blood clearance to plasma clearance.

    CL_plasma = CL_blood / B/P

    Parameters
    ----------
    cl_blood_L_per_h:
        Clearance measured in whole blood (L/h). Must be > 0.
    bp_ratio:
        Blood-to-plasma concentration ratio. Must be > 0.

    Returns
    -------
    float
        Plasma clearance in L/h.
    """
    if cl_blood_L_per_h <= 0:
        raise ValueError(f"cl_blood_L_per_h must be > 0, got {cl_blood_L_per_h}")
    if bp_ratio <= 0:
        raise ValueError(f"bp_ratio must be > 0, got {bp_ratio}")

    return cl_blood_L_per_h / bp_ratio


def simulate_rbc_pk(
    drug_name: str,
    dose_mg: float,
    cl_plasma_L_per_h: float,
    vd_plasma_L: float,
    bp_ratio: float,
    ka_per_h: float = 1.5,
    route: str = "oral",
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> RBCPKResult:
    """Simulate plasma and blood PK incorporating RBC sequestration.

    Uses Forward Euler integration of a 1-compartment model. Blood concentration
    is derived from plasma concentration via B/P ratio.

    C_blood(t) = C_plasma(t) * B/P

    Parameters
    ----------
    drug_name:
        Drug identifier.
    dose_mg:
        Dose in mg.
    cl_plasma_L_per_h:
        Plasma clearance in L/h. Must be > 0.
    vd_plasma_L:
        Apparent volume of distribution (plasma-based) in L. Must be > 0.
    bp_ratio:
        Blood-to-plasma ratio. Must be > 0.
    ka_per_h:
        Oral absorption rate constant in h⁻¹ (ignored for IV route).
    route:
        Route of administration: 'oral' or 'iv'.
    t_end_h:
        Simulation end time in hours.
    dt_h:
        Integration step size in hours.

    Returns
    -------
    RBCPKResult
    """
    if cl_plasma_L_per_h <= 0:
        raise ValueError(f"cl_plasma_L_per_h must be > 0, got {cl_plasma_L_per_h}")
    if vd_plasma_L <= 0:
        raise ValueError(f"vd_plasma_L must be > 0, got {vd_plasma_L}")
    if bp_ratio <= 0:
        raise ValueError(f"bp_ratio must be > 0, got {bp_ratio}")

    ke = cl_plasma_L_per_h / vd_plasma_L
    n_steps = max(1, int(round(t_end_h / dt_h)))
    actual_dt = t_end_h / n_steps

    times: list[float] = []
    c_plasma_list: list[float] = []
    c_blood_list: list[float] = []

    route_lower = route.lower()

    if route_lower == "iv":
        c_plasma = dose_mg / vd_plasma_L
        c_gut = 0.0
    else:
        # Oral: initial gut load
        c_plasma = 0.0
        c_gut = dose_mg / vd_plasma_L  # store as mg/L equivalent

    t = 0.0
    for i in range(n_steps + 1):
        times.append(t)
        c_blood = c_plasma * bp_ratio
        c_plasma_list.append(max(0.0, c_plasma))
        c_blood_list.append(max(0.0, c_blood))

        if i < n_steps:
            if route_lower == "iv":
                dc_plasma = -ke * c_plasma * actual_dt
                c_plasma += dc_plasma
            else:
                dc_gut = -ka_per_h * c_gut * actual_dt
                dc_plasma = (ka_per_h * c_gut - ke * c_plasma) * actual_dt
                c_gut += dc_gut
                c_plasma += dc_plasma
            t += actual_dt

    c_plasma_arr = np.array(c_plasma_list)
    c_blood_arr = np.array(c_blood_list)
    times_arr = np.array(times)

    cmax_plasma = float(np.max(c_plasma_arr))
    cmax_blood = float(np.max(c_blood_arr))
    auc_plasma = float(np_trapz(c_plasma_arr, times_arr))
    auc_blood = float(np_trapz(c_blood_arr, times_arr))

    notes = (
        f"B/P={bp_ratio:.3f}; C_blood = C_plasma x B/P. "
        f"Route={route}; ke={ke:.4f} h⁻¹."
    )
    if bp_ratio > 1.5:
        notes += " High RBC sequestration: plasma may underestimate blood exposure."
    elif bp_ratio < 0.7:
        notes += " Low RBC partitioning: plasma overestimates blood exposure."

    return RBCPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        bp_ratio=bp_ratio,
        times_h=times,
        c_plasma=c_plasma_list,
        c_blood=c_blood_list,
        cmax_plasma=cmax_plasma,
        cmax_blood=cmax_blood,
        auc_plasma=auc_plasma,
        auc_blood=auc_blood,
        notes=notes,
    )
