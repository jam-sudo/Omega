"""Phase 727 — Intranasal Drug Delivery PK.

3-compartment model: nasal mucosa → systemic plasma + swallowed fraction (oral).

Compartments:
  A_nasal  (mg): drug in nasal cavity
  A_gut    (mg): swallowed drug in GI tract (from mucociliary clearance)
  C_plasma (mg/L): systemic plasma concentration

ODEs (forward Euler):
  dA_nasal/dt  = -(kabs_nasal + kclear) * A_nasal
  dA_gut/dt    = kclear * (1 - f_nasal) * A_nasal - ka_oral * A_gut
  dC_plasma/dt = (kabs_nasal * f_nasal * A_nasal
                  + ka_oral * f_swallowed_oral * A_gut) / Vd
                 - ke * C_plasma
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class IntranasalPKResult:
    """Results from intranasal PK simulation."""

    drug_name: str
    dose_mg: float
    times_h: list
    c_plasma_mg_L: list
    a_nasal_mg: list
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    f_systemic_nasal: float
    f_systemic_total: float
    t_half_h: float
    notes: str


def _validate_params(
    dose_mg: float,
    f_nasal: float,
    cl_L_per_h: float,
    vd_L: float,
    kabs_nasal_per_h: float,
) -> None:
    """Validate simulation parameters."""
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if not (0 < f_nasal < 1):
        raise ValueError(f"f_nasal must be in (0, 1), got {f_nasal}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if kabs_nasal_per_h <= 0:
        raise ValueError(f"kabs_nasal_per_h must be > 0, got {kabs_nasal_per_h}")


def simulate_intranasal_pk(
    drug_name: str = "drug",
    dose_mg: float = 1.0,
    kabs_nasal_per_h: float = 1.5,
    kclear_per_h: float = 0.5,
    f_nasal: float = 0.6,
    f_swallowed_oral: float = 0.3,
    ka_oral_per_h: float = 0.5,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    t_end_h: float = 12.0,
    dt_h: float = 0.05,
) -> IntranasalPKResult:
    """Simulate intranasal drug delivery PK.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Administered intranasal dose (mg).
    kabs_nasal_per_h:
        Nasal epithelial absorption rate constant (1/h). Typically faster than
        oral due to rich vascularization.
    kclear_per_h:
        Mucociliary clearance rate from nasal cavity (1/h). Default 0.5/h
        corresponds to ~30 min half-life in nasal cavity.
    f_nasal:
        Fraction of nasal drug that is absorbed systemically (vs swallowed).
        Must be in (0, 1).
    f_swallowed_oral:
        Oral bioavailability of the swallowed fraction.
    ka_oral_per_h:
        GI absorption rate constant for swallowed drug (1/h).
    cl_L_per_h:
        Systemic clearance (L/h).
    vd_L:
        Volume of distribution (L).
    t_end_h:
        Simulation end time (h).
    dt_h:
        Forward Euler time step (h).

    Returns
    -------
    IntranasalPKResult
    """
    _validate_params(dose_mg, f_nasal, cl_L_per_h, vd_L, kabs_nasal_per_h)

    ke = cl_L_per_h / vd_L  # elimination rate constant (1/h)

    # Initial conditions
    a_nasal = dose_mg
    a_gut = 0.0
    c_plasma = 0.0

    times_h: list = []
    c_plasma_list: list = []
    a_nasal_list: list = []

    t = 0.0
    n_steps = int(round(t_end_h / dt_h))

    for _ in range(n_steps + 1):
        times_h.append(t)
        c_plasma_list.append(c_plasma)
        a_nasal_list.append(a_nasal)

        # ODEs
        da_nasal = -(kabs_nasal_per_h + kclear_per_h) * a_nasal
        da_gut = kclear_per_h * (1.0 - f_nasal) * a_nasal - ka_oral_per_h * a_gut
        dc_plasma = (
            kabs_nasal_per_h * f_nasal * a_nasal + ka_oral_per_h * f_swallowed_oral * a_gut
        ) / vd_L - ke * c_plasma

        # Forward Euler update
        a_nasal = max(0.0, a_nasal + da_nasal * dt_h)
        a_gut = max(0.0, a_gut + da_gut * dt_h)
        c_plasma = max(0.0, c_plasma + dc_plasma * dt_h)

        t = round(t + dt_h, 10)

    # PK metrics
    cmax_mg_L = max(c_plasma_list)
    tmax_idx = c_plasma_list.index(cmax_mg_L)
    tmax_h = times_h[tmax_idx]

    # Trapezoidal AUC
    auc = 0.0
    for i in range(1, len(times_h)):
        auc += 0.5 * (c_plasma_list[i - 1] + c_plasma_list[i]) * (times_h[i] - times_h[i - 1])

    t_half_h = math.log(2.0) / ke

    # Bioavailability estimates
    f_systemic_nasal = f_nasal  # fraction absorbed via nasal mucosa
    swallowed_frac = 1.0 - f_nasal
    f_systemic_total = f_nasal + swallowed_frac * f_swallowed_oral

    notes = (
        f"Intranasal PK for {drug_name}. "
        f"Nasal absorption fraction: {f_nasal:.2f}, "
        f"mucociliary clearance: {kclear_per_h:.2f}/h, "
        f"total systemic bioavailability: {f_systemic_total:.2f}."
    )

    return IntranasalPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_list,
        a_nasal_mg=a_nasal_list,
        cmax_mg_L=cmax_mg_L,
        tmax_h=tmax_h,
        auc_mg_h_per_L=auc,
        f_systemic_nasal=f_systemic_nasal,
        f_systemic_total=f_systemic_total,
        t_half_h=t_half_h,
        notes=notes,
    )


def _simulate_oral_reference(
    drug_name: str,
    dose_mg: float,
    ka_oral_per_h: float,
    f_oral: float,
    cl_L_per_h: float,
    vd_L: float,
    t_end_h: float,
    dt_h: float,
) -> IntranasalPKResult:
    """Simulate a simple oral 1-compartment reference model.

    Uses IntranasalPKResult for unified comparison; a_nasal_mg holds gut amount.
    """
    ke = cl_L_per_h / vd_L

    a_gut = dose_mg
    c_plasma = 0.0

    times_h: list = []
    c_plasma_list: list = []
    a_gut_list: list = []

    t = 0.0
    n_steps = int(round(t_end_h / dt_h))

    for _ in range(n_steps + 1):
        times_h.append(t)
        c_plasma_list.append(c_plasma)
        a_gut_list.append(a_gut)

        da_gut = -ka_oral_per_h * a_gut
        dc_plasma = ka_oral_per_h * f_oral * a_gut / vd_L - ke * c_plasma

        a_gut = max(0.0, a_gut + da_gut * dt_h)
        c_plasma = max(0.0, c_plasma + dc_plasma * dt_h)
        t = round(t + dt_h, 10)

    cmax_mg_L = max(c_plasma_list)
    tmax_idx = c_plasma_list.index(cmax_mg_L)
    tmax_h = times_h[tmax_idx]

    auc = 0.0
    for i in range(1, len(times_h)):
        auc += 0.5 * (c_plasma_list[i - 1] + c_plasma_list[i]) * (times_h[i] - times_h[i - 1])

    t_half_h = math.log(2.0) / ke

    return IntranasalPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_list,
        a_nasal_mg=a_gut_list,
        cmax_mg_L=cmax_mg_L,
        tmax_h=tmax_h,
        auc_mg_h_per_L=auc,
        f_systemic_nasal=f_oral,
        f_systemic_total=f_oral,
        t_half_h=t_half_h,
        notes=f"Oral reference PK for {drug_name}, F={f_oral:.2f}.",
    )


def compare_intranasal_vs_oral(
    drug_name: str = "drug",
    dose_mg: float = 1.0,
    kabs_nasal_per_h: float = 1.5,
    kclear_per_h: float = 0.5,
    f_nasal: float = 0.6,
    f_swallowed_oral: float = 0.3,
    ka_oral_per_h: float = 0.5,
    f_oral: float = 0.5,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    t_end_h: float = 12.0,
    dt_h: float = 0.05,
) -> dict:
    """Compare intranasal vs oral delivery for the same drug.

    Parameters
    ----------
    drug_name:
        Drug name.
    dose_mg:
        Dose in mg (same for both routes).
    kabs_nasal_per_h:
        Nasal absorption rate constant (1/h).
    kclear_per_h:
        Mucociliary clearance rate (1/h).
    f_nasal:
        Nasal absorption fraction.
    f_swallowed_oral:
        BA of swallowed fraction.
    ka_oral_per_h:
        Oral absorption rate constant (1/h).
    f_oral:
        Oral bioavailability for the oral reference simulation.
    cl_L_per_h:
        Systemic clearance (L/h).
    vd_L:
        Volume of distribution (L).
    t_end_h:
        Simulation end time (h).
    dt_h:
        Time step (h).

    Returns
    -------
    dict with keys:
        intranasal_result, oral_result, auc_ratio, cmax_ratio, notes
    """
    intranasal_result = simulate_intranasal_pk(
        drug_name=drug_name,
        dose_mg=dose_mg,
        kabs_nasal_per_h=kabs_nasal_per_h,
        kclear_per_h=kclear_per_h,
        f_nasal=f_nasal,
        f_swallowed_oral=f_swallowed_oral,
        ka_oral_per_h=ka_oral_per_h,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
        t_end_h=t_end_h,
        dt_h=dt_h,
    )

    oral_result = _simulate_oral_reference(
        drug_name=drug_name,
        dose_mg=dose_mg,
        ka_oral_per_h=ka_oral_per_h,
        f_oral=f_oral,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
        t_end_h=t_end_h,
        dt_h=dt_h,
    )

    auc_ratio = (
        intranasal_result.auc_mg_h_per_L / oral_result.auc_mg_h_per_L
        if oral_result.auc_mg_h_per_L > 0
        else float("inf")
    )
    cmax_ratio = (
        intranasal_result.cmax_mg_L / oral_result.cmax_mg_L
        if oral_result.cmax_mg_L > 0
        else float("inf")
    )

    notes = (
        f"Intranasal vs oral comparison for {drug_name}. "
        f"AUC ratio (IN/oral): {auc_ratio:.2f}, "
        f"Cmax ratio: {cmax_ratio:.2f}. "
        f"Intranasal total F: {intranasal_result.f_systemic_total:.2f}, "
        f"Oral F: {f_oral:.2f}."
    )

    return {
        "intranasal_result": intranasal_result,
        "oral_result": oral_result,
        "auc_ratio": auc_ratio,
        "cmax_ratio": cmax_ratio,
        "notes": notes,
    }


__all__ = [
    "IntranasalPKResult",
    "simulate_intranasal_pk",
    "compare_intranasal_vs_oral",
]
