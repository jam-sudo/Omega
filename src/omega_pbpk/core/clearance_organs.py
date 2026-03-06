"""Multi-organ clearance model separating hepatic, renal, and pulmonary elimination.

Uses the well-stirred (venous equilibrium) model for each eliminating organ to
compute organ-specific clearance and fractional contributions to total systemic
clearance. Supports full simulation with multi-organ PK.

Typical human blood flow values:
- Hepatic Q: ~90 L/h
- Renal Q: ~72 L/h
- Pulmonary Q: ~360 L/h (cardiac output; entire CO passes through lungs)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz


@dataclass(frozen=True)
class ClearanceBreakdown:
    """Organ-specific clearance contributions.

    Attributes
    ----------
    cl_hepatic : float
        Hepatic clearance via well-stirred model (L/h).
    cl_renal : float
        Renal clearance via well-stirred model (L/h).
    cl_pulmonary : float
        Pulmonary clearance via well-stirred model (L/h).
    cl_total : float
        Sum of all organ clearances (L/h).
    fe_hepatic : float
        Fraction of total clearance attributable to liver (0–1).
    fe_renal : float
        Fraction of total clearance attributable to kidney (0–1).
    fe_pulmonary : float
        Fraction of total clearance attributable to lung (0–1).
    eh : float
        Hepatic extraction ratio (0–1).
    er : float
        Renal extraction ratio (0–1).
    ep : float
        Pulmonary extraction ratio (0–1).
    """

    cl_hepatic: float
    cl_renal: float
    cl_pulmonary: float
    cl_total: float
    fe_hepatic: float
    fe_renal: float
    fe_pulmonary: float
    eh: float
    er: float
    ep: float


@dataclass(frozen=True)
class MultiOrganCLResult:
    """Result of a multi-organ clearance simulation.

    Attributes
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Administered dose (mg).
    fup : float
        Fraction unbound in plasma.
    route : str
        Administration route ("iv" or "oral").
    clearance : ClearanceBreakdown
        Organ-level clearance breakdown.
    times_h : list[float]
        Simulation time points (h).
    c_plasma_mg_L : list[float]
        Plasma drug concentrations over time (mg/L).
    cmax : float
        Peak plasma concentration (mg/L).
    tmax_h : float
        Time to peak plasma concentration (h).
    auc : float
        Area under the plasma concentration-time curve (mg/L·h).
    t_half_h : float
        Terminal half-life (h); 0.693 * Vd / CL_total.
    notes : str
        Summary of dominant elimination pathway.
    """

    drug_name: str
    dose_mg: float
    fup: float
    route: str
    clearance: ClearanceBreakdown
    times_h: list[float]
    c_plasma_mg_L: list[float]
    cmax: float
    tmax_h: float
    auc: float
    t_half_h: float
    notes: str


def organ_clearance(
    q_L_per_h: float,
    cl_int_L_per_h: float,
    fup: float,
) -> float:
    """Compute organ clearance using the well-stirred (venous equilibrium) model.

    CL_organ = Q * fu * CLint / (Q + fu * CLint)

    Parameters
    ----------
    q_L_per_h : float
        Organ blood flow (L/h). Must be > 0.
    cl_int_L_per_h : float
        Intrinsic clearance for this organ (L/h). Must be >= 0.
    fup : float
        Fraction unbound in plasma (0, 1].

    Returns
    -------
    float
        Organ clearance (L/h).

    Raises
    ------
    ValueError
        If parameters are out of valid range.
    """
    if q_L_per_h <= 0:
        raise ValueError(f"q_L_per_h must be > 0, got {q_L_per_h}")
    if cl_int_L_per_h < 0:
        raise ValueError(f"cl_int_L_per_h must be >= 0, got {cl_int_L_per_h}")
    if fup <= 0 or fup > 1:
        raise ValueError(f"fup must be in (0, 1], got {fup}")

    fu_clint = fup * cl_int_L_per_h
    return q_L_per_h * fu_clint / (q_L_per_h + fu_clint)


def total_clearance(
    hepatic_q: float,
    hepatic_clint: float,
    renal_q: float,
    renal_clint: float,
    pulmonary_q: float,
    pulmonary_clint: float,
    fup: float,
) -> ClearanceBreakdown:
    """Compute total clearance and fractional organ contributions.

    Each organ uses the well-stirred model. Total clearance is the sum of
    individual organ clearances.

    Parameters
    ----------
    hepatic_q : float
        Hepatic blood flow (L/h).
    hepatic_clint : float
        Hepatic intrinsic clearance (L/h).
    renal_q : float
        Renal blood flow (L/h).
    renal_clint : float
        Renal intrinsic clearance (L/h).
    pulmonary_q : float
        Pulmonary blood flow (L/h).
    pulmonary_clint : float
        Pulmonary intrinsic clearance (L/h).
    fup : float
        Fraction unbound in plasma (0, 1].

    Returns
    -------
    ClearanceBreakdown
        Detailed breakdown of organ clearances and fractions.
    """
    cl_h = organ_clearance(hepatic_q, hepatic_clint, fup)
    cl_r = organ_clearance(renal_q, renal_clint, fup)
    cl_p = organ_clearance(pulmonary_q, pulmonary_clint, fup)

    cl_tot = cl_h + cl_r + cl_p

    if cl_tot > 0:
        fe_h = cl_h / cl_tot
        fe_r = cl_r / cl_tot
        fe_p = cl_p / cl_tot
    else:
        fe_h = fe_r = fe_p = 0.0

    # Extraction ratios
    eh = cl_h / hepatic_q if hepatic_q > 0 else 0.0
    er = cl_r / renal_q if renal_q > 0 else 0.0
    ep = cl_p / pulmonary_q if pulmonary_q > 0 else 0.0

    return ClearanceBreakdown(
        cl_hepatic=cl_h,
        cl_renal=cl_r,
        cl_pulmonary=cl_p,
        cl_total=cl_tot,
        fe_hepatic=fe_h,
        fe_renal=fe_r,
        fe_pulmonary=fe_p,
        eh=eh,
        er=er,
        ep=ep,
    )


def simulate_multiorgancl(
    drug_name: str,
    dose_mg: float,
    hepatic_q_L_per_h: float,
    hepatic_clint_L_per_h: float,
    renal_q_L_per_h: float,
    renal_clint_L_per_h: float,
    pulmonary_q_L_per_h: float,
    pulmonary_clint_L_per_h: float,
    vd_L: float,
    fup: float,
    route: str = "iv",
    ka_per_h: float = 1.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> MultiOrganCLResult:
    """Simulate 1-compartment PK with multi-organ clearance contributions.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Administered dose (mg). Must be > 0.
    hepatic_q_L_per_h : float
        Hepatic blood flow (L/h). Must be > 0.
    hepatic_clint_L_per_h : float
        Hepatic intrinsic clearance (L/h). Must be >= 0.
    renal_q_L_per_h : float
        Renal blood flow (L/h). Must be > 0.
    renal_clint_L_per_h : float
        Renal intrinsic clearance (L/h). Must be >= 0.
    pulmonary_q_L_per_h : float
        Pulmonary blood flow (L/h). Must be > 0.
    pulmonary_clint_L_per_h : float
        Pulmonary intrinsic clearance (L/h). Must be >= 0.
    vd_L : float
        Volume of distribution (L). Must be > 0.
    fup : float
        Fraction unbound in plasma (0, 1].
    route : str
        Administration route: "iv" or "oral".
    ka_per_h : float
        Oral absorption rate constant (h⁻¹). Only used when route="oral".
    t_end_h : float
        Simulation duration (h).
    dt_h : float
        Forward Euler time step (h).

    Returns
    -------
    MultiOrganCLResult
        Simulation results with PK metrics and clearance breakdown.

    Raises
    ------
    ValueError
        If any parameter is out of valid range.
    """
    # Input validation
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if hepatic_q_L_per_h <= 0:
        raise ValueError(f"hepatic_q_L_per_h must be > 0, got {hepatic_q_L_per_h}")
    if renal_q_L_per_h <= 0:
        raise ValueError(f"renal_q_L_per_h must be > 0, got {renal_q_L_per_h}")
    if pulmonary_q_L_per_h <= 0:
        raise ValueError(f"pulmonary_q_L_per_h must be > 0, got {pulmonary_q_L_per_h}")
    if hepatic_clint_L_per_h < 0:
        raise ValueError(f"hepatic_clint_L_per_h must be >= 0, got {hepatic_clint_L_per_h}")
    if renal_clint_L_per_h < 0:
        raise ValueError(f"renal_clint_L_per_h must be >= 0, got {renal_clint_L_per_h}")
    if pulmonary_clint_L_per_h < 0:
        raise ValueError(f"pulmonary_clint_L_per_h must be >= 0, got {pulmonary_clint_L_per_h}")
    if fup <= 0 or fup > 1:
        raise ValueError(f"fup must be in (0, 1], got {fup}")

    # Compute clearance breakdown
    cl_breakdown = total_clearance(
        hepatic_q=hepatic_q_L_per_h,
        hepatic_clint=hepatic_clint_L_per_h,
        renal_q=renal_q_L_per_h,
        renal_clint=renal_clint_L_per_h,
        pulmonary_q=pulmonary_q_L_per_h,
        pulmonary_clint=pulmonary_clint_L_per_h,
        fup=fup,
    )

    cl_total = cl_breakdown.cl_total
    kel = cl_total / vd_L  # elimination rate constant

    n_steps = int(math.ceil(t_end_h / dt_h)) + 1
    times = [i * dt_h for i in range(n_steps)]

    c_plasma = 0.0
    depot = 0.0

    if route.lower() == "iv":
        c_plasma = dose_mg / vd_L
    elif route.lower() == "oral":
        depot = dose_mg / vd_L
    else:
        raise ValueError(f"route must be 'iv' or 'oral', got '{route}'")

    out_plasma = [c_plasma]

    for _ in range(1, n_steps):
        if route.lower() == "oral":
            dDepot = -ka_per_h * depot
            dC = ka_per_h * depot - kel * c_plasma
            depot += dDepot * dt_h
            depot = max(depot, 0.0)
        else:
            dC = -kel * c_plasma

        c_plasma = max(c_plasma + dC * dt_h, 0.0)
        out_plasma.append(c_plasma)

    # PK metrics
    cmax = float(max(out_plasma))
    tmax_idx = out_plasma.index(cmax)
    tmax_h = times[tmax_idx]

    t_arr = np.array(times)
    c_arr = np.array(out_plasma)
    auc = float(np_trapz(c_arr, t_arr))

    # Half-life
    t_half = (0.693 / kel) if kel > 0 else float("inf")

    # Notes
    dominant = max(
        [
            ("hepatic", cl_breakdown.fe_hepatic),
            ("renal", cl_breakdown.fe_renal),
            ("pulmonary", cl_breakdown.fe_pulmonary),
        ],
        key=lambda x: x[1],
    )
    notes = (
        f"Dominant elimination: {dominant[0]} ({dominant[1] * 100:.1f}% of total CL). "
        f"Total CL = {cl_total:.2f} L/h, t½ = {t_half:.2f} h."
    )

    return MultiOrganCLResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        fup=fup,
        route=route,
        clearance=cl_breakdown,
        times_h=times,
        c_plasma_mg_L=out_plasma,
        cmax=cmax,
        tmax_h=tmax_h,
        auc=auc,
        t_half_h=t_half,
        notes=notes,
    )
