"""Phase 1010: Drug delivery through tumor vasculature (EPR effect + IFP barrier).

Three-compartment Forward-Euler model:
    Plasma → Tumor interstitium → Tumor cells
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class TumorVasculaturePKResult:
    """Result of tumor vasculature PK simulation."""

    drug_name: str
    mw_kDa: float
    dose_mg: float
    times_h: list
    c_plasma_mg_L: list
    c_tumor_interstitium_mg_L: list
    c_tumor_cell_mg_L: list
    cmax_plasma_mg_L: float
    cmax_tumor_mg_L: float
    tmax_tumor_h: float
    auc_plasma_mg_h_per_L: float
    auc_tumor_mg_h_per_L: float
    tumor_to_plasma_ratio: float
    epr_effect_factor: float
    ifp_barrier_factor: float
    notes: str


def simulate_tumor_vasculature_pk(
    drug_name: str,
    dose_mg: float,
    mw_kDa: float = 0.3,
    logp: float = 2.0,
    v_plasma_L: float = 3.0,
    v_tumor_interstitium_L: float = 0.05,
    v_tumor_cell_L: float = 0.03,
    cl_plasma_L_per_h: float = 5.0,
    t_end_h: float = 72.0,
    dt_h: float = 0.5,
) -> TumorVasculaturePKResult:
    """Simulate drug delivery through tumor vasculature using a 3-compartment model.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    dose_mg:
        IV bolus dose in mg.
    mw_kDa:
        Molecular weight in kDa (default 0.3, typical small molecule ~300 Da).
    logp:
        Octanol-water partition coefficient.
    v_plasma_L:
        Plasma volume in litres.
    v_tumor_interstitium_L:
        Tumor interstitial volume in litres.
    v_tumor_cell_L:
        Tumor cell volume in litres.
    cl_plasma_L_per_h:
        Systemic plasma clearance in L/h.
    t_end_h:
        Simulation end time in hours.
    dt_h:
        Forward Euler step size in hours.

    Returns
    -------
    TumorVasculaturePKResult
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if v_plasma_L <= 0:
        raise ValueError(f"v_plasma_L must be > 0, got {v_plasma_L}")
    if v_tumor_interstitium_L <= 0:
        raise ValueError(f"v_tumor_interstitium_L must be > 0, got {v_tumor_interstitium_L}")
    if cl_plasma_L_per_h <= 0:
        raise ValueError(f"cl_plasma_L_per_h must be > 0, got {cl_plasma_L_per_h}")

    # --- EPR effect factor ---
    if mw_kDa > 0.5:
        epr_factor = min(10.0, 1.0 + 2.0 * math.log10(mw_kDa / 0.5))
    else:
        epr_factor = 1.0

    # --- IFP barrier factor (capped at 1.0 for small molecules with negative log10) ---
    ifp_factor = max(0.1, min(1.0, 1.0 - 0.05 * math.log10(mw_kDa)))

    # --- Rate constants ---
    k_extr = 0.01 * epr_factor * ifp_factor * (1.0 / (mw_kDa**0.3))  # /h
    k_uptake = max(0.001, 0.05 * (1.0 + logp * 0.1))  # /h
    k_cell_elim = 0.02  # /h
    ke = cl_plasma_L_per_h / v_plasma_L  # plasma elimination rate /h

    # --- Initial conditions (IV bolus) ---
    n_steps = int(t_end_h / dt_h) + 1
    c_plasma = dose_mg / v_plasma_L
    c_inter = 0.0
    c_cell = 0.0

    times_h: list = []
    c_plasma_list: list = []
    c_inter_list: list = []
    c_cell_list: list = []

    t = 0.0
    for _i in range(n_steps):
        times_h.append(t)
        c_plasma_list.append(c_plasma)
        c_inter_list.append(c_inter)
        c_cell_list.append(c_cell)

        # ODEs (Forward Euler)
        dcp = -ke * c_plasma - k_extr * c_plasma
        dci = k_extr * c_plasma * v_plasma_L / v_tumor_interstitium_L - k_uptake * c_inter
        dcc = k_uptake * c_inter * v_tumor_interstitium_L / v_tumor_cell_L - k_cell_elim * c_cell

        c_plasma = max(0.0, c_plasma + dcp * dt_h)
        c_inter = max(0.0, c_inter + dci * dt_h)
        c_cell = max(0.0, c_cell + dcc * dt_h)
        t += dt_h

    # --- Derived metrics ---
    cmax_plasma = max(c_plasma_list)
    cmax_tumor = max(c_inter_list)
    tmax_tumor_h = times_h[c_inter_list.index(cmax_tumor)]

    # Manual trapezoidal AUC
    auc_plasma = 0.0
    auc_inter = 0.0
    for idx in range(len(times_h) - 1):
        h = times_h[idx + 1] - times_h[idx]
        auc_plasma += 0.5 * (c_plasma_list[idx] + c_plasma_list[idx + 1]) * h
        auc_inter += 0.5 * (c_inter_list[idx] + c_inter_list[idx + 1]) * h

    tumor_to_plasma_ratio = auc_inter / auc_plasma if auc_plasma > 0.0 else 0.0

    notes = f"3-cpt Forward Euler; k_extr={k_extr:.4f}/h; k_uptake={k_uptake:.4f}/h; ke={ke:.4f}/h"

    return TumorVasculaturePKResult(
        drug_name=drug_name,
        mw_kDa=mw_kDa,
        dose_mg=dose_mg,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_list,
        c_tumor_interstitium_mg_L=c_inter_list,
        c_tumor_cell_mg_L=c_cell_list,
        cmax_plasma_mg_L=cmax_plasma,
        cmax_tumor_mg_L=cmax_tumor,
        tmax_tumor_h=tmax_tumor_h,
        auc_plasma_mg_h_per_L=auc_plasma,
        auc_tumor_mg_h_per_L=auc_inter,
        tumor_to_plasma_ratio=tumor_to_plasma_ratio,
        epr_effect_factor=epr_factor,
        ifp_barrier_factor=ifp_factor,
        notes=notes,
    )


__all__ = [
    "TumorVasculaturePKResult",
    "simulate_tumor_vasculature_pk",
]
