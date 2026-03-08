"""Phase 619 — Drug Bone Marrow Distribution.

Models drug distribution into bone marrow using a 2-compartment
(plasma + bone marrow) ODE system, important for hematotoxicity assessment.

References
----------
- Brunton LL et al., Goodman & Gilman's Pharmacological Basis (bone marrow perfusion)
- Chabner BA & Roberts TG, Nat Rev Cancer. 2005 (hematotoxicity)
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class BoneMarrowPKResult:
    """Result of bone marrow PK simulation (Phase 619)."""

    drug_name: str
    dose_mg: float
    times_h: list
    c_plasma_mg_L: list
    c_marrow_mg_g: list
    kp_marrow: float
    cmax_plasma_mg_L: float
    cmax_marrow_mg_g: float
    auc_plasma_mg_h_per_L: float
    auc_marrow_mg_h_per_g: float
    marrow_to_plasma_ratio: float
    hematotoxicity_index: float
    t_half_marrow_h: float
    notes: str


def _trapezoidal_auc(x: list, y: list) -> float:
    """Manual trapezoidal AUC calculation."""
    return sum(0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]) for i in range(1, len(x)))


def simulate_bone_marrow_pk(
    drug_name: str,
    dose_mg: float,
    logP: float,
    mw_Da: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    marrow_weight_g: float = 1500.0,
    t_end_h: float = 72.0,
    dt_h: float = 0.1,
) -> BoneMarrowPKResult:
    """Simulate drug distribution into bone marrow.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    dose_mg:
        Oral dose in milligrams.
    logP:
        Octanol-water partition coefficient.
    mw_Da:
        Molecular weight in Daltons.
    cl_sys_L_per_h:
        Systemic clearance (L/h).
    vd_sys_L:
        Systemic volume of distribution (L).
    marrow_weight_g:
        Weight of bone marrow tissue (g). Default 1500 g.
    t_end_h:
        Simulation end time (h). Default 72 h.
    dt_h:
        Forward Euler step size (h). Default 0.1 h.

    Returns
    -------
    BoneMarrowPKResult
        Simulation results including plasma and marrow concentration profiles.
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    if marrow_weight_g <= 0:
        raise ValueError("marrow_weight_g must be > 0")

    # --- Derived parameters ---
    # Bone marrow partition coefficient: lipophilicity favors uptake, large MW reduces it
    kp_raw = (logP + 2.0) * 0.5 / max(1.0, mw_Da / 300.0)
    kp_marrow = max(0.1, min(15.0, max(0.3, kp_raw)))

    # Bone marrow blood flow ~0.8% of cardiac output
    Qm = 3.0  # L/h

    # Effective volume of distribution for marrow compartment
    Vd_marrow_L = marrow_weight_g * kp_marrow / 1000.0

    # Rate constants
    k_in = Qm / vd_sys_L  # plasma → marrow (/h)
    k_out = Qm / Vd_marrow_L  # marrow → plasma (/h)
    ke = cl_sys_L_per_h / vd_sys_L  # elimination rate constant (/h)

    # Oral absorption parameters
    ka = 1.2  # /h
    F = 0.85  # bioavailability factor

    # --- Forward Euler simulation ---
    n_steps = int(t_end_h / dt_h) + 1
    times_h: list = []
    c_plasma_mg_L: list = []
    c_marrow_mg_g: list = []

    # Initial conditions
    A_gut = dose_mg * F  # drug amount in gut (mg)
    C_plasma = 0.0  # plasma concentration (mg/L)
    C_marrow = 0.0  # marrow concentration in volume units (mg/L)

    for step in range(n_steps):
        t = step * dt_h
        times_h.append(t)

        # Convert C_marrow from mg/L to mg/g for output
        c_marrow_g = C_marrow * Vd_marrow_L * 1000.0 / marrow_weight_g
        c_plasma_mg_L.append(C_plasma)
        c_marrow_mg_g.append(c_marrow_g)

        # ODE right-hand sides
        dA_gut = -ka * A_gut
        dC_plasma = (
            ka * A_gut / vd_sys_L
            - ke * C_plasma
            - k_in * C_plasma
            + k_out * C_marrow * (Vd_marrow_L / vd_sys_L)
        )
        dC_marrow = k_in * C_plasma * (vd_sys_L / Vd_marrow_L) - k_out * C_marrow

        # Forward Euler update
        A_gut = max(0.0, A_gut + dt_h * dA_gut)
        C_plasma = max(0.0, C_plasma + dt_h * dC_plasma)
        C_marrow = max(0.0, C_marrow + dt_h * dC_marrow)

    # --- Summary statistics ---
    cmax_plasma_mg_L = max(c_plasma_mg_L)
    cmax_marrow_mg_g = max(c_marrow_mg_g)

    auc_plasma = _trapezoidal_auc(times_h, c_plasma_mg_L)
    auc_marrow = _trapezoidal_auc(times_h, c_marrow_mg_g)

    marrow_to_plasma_ratio = auc_marrow / auc_plasma if auc_plasma > 0 else 0.0

    # Hematotoxicity index: high marrow concentrations + lipophilicity → risk
    logP_factor = max(1.0, logP)
    htox_raw = cmax_marrow_mg_g * logP_factor / 5.0
    hematotoxicity_index = max(0.01, min(100.0, htox_raw))

    # Terminal half-life of marrow compartment
    t_half_marrow_h = math.log(2.0) / k_out

    notes = (
        f"kp_marrow={kp_marrow:.3f}; "
        f"Vd_marrow={Vd_marrow_L:.3f} L; "
        f"k_in={k_in:.4f}/h; k_out={k_out:.4f}/h; "
        f"t_half_marrow={t_half_marrow_h:.2f} h"
    )

    return BoneMarrowPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_mg_L,
        c_marrow_mg_g=c_marrow_mg_g,
        kp_marrow=kp_marrow,
        cmax_plasma_mg_L=cmax_plasma_mg_L,
        cmax_marrow_mg_g=cmax_marrow_mg_g,
        auc_plasma_mg_h_per_L=auc_plasma,
        auc_marrow_mg_h_per_g=auc_marrow,
        marrow_to_plasma_ratio=marrow_to_plasma_ratio,
        hematotoxicity_index=hematotoxicity_index,
        t_half_marrow_h=t_half_marrow_h,
        notes=notes,
    )
