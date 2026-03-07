"""
Phase 731 — Drug Efflux Transporter PK
Models P-gp (ABCB1), BCRP (ABCG2), and MRP2 (ABCC2) efflux transporter kinetics
using a 2-compartment model (systemic plasma + tissue) with Michaelis-Menten efflux.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log

# ---------------------------------------------------------------------------
# Result dataclass (plain — has list fields)
# ---------------------------------------------------------------------------


@dataclass
class EfluxTransporterPKResult:
    """Result of efflux transporter PK simulation."""

    drug_name: str
    dose_mg: float
    times_h: list
    c_plasma_mg_L: list
    c_tissue_mg_L: list
    cmax_plasma: float
    cmax_tissue: float
    auc_plasma: float
    auc_tissue: float
    tissue_to_plasma_auc_ratio: float
    efflux_saturation_index: float
    t_half_plasma_h: float
    notes: str


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------


def simulate_efflux_transporter_pk(
    drug_name: str,
    dose_mg: float,
    cl_sys_L_per_h: float = 5.0,
    vd_plasma_L: float = 10.0,
    vd_tissue_L: float = 40.0,
    q_L_per_h: float = 10.0,
    kp_tissue: float = 1.0,
    vmax_efflux_mg_per_h: float = 5.0,
    km_efflux_mg_L: float = 0.5,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> EfluxTransporterPKResult:
    """
    Simulate 2-compartment efflux transporter PK (IV bolus).

    ODEs (Forward Euler):
      dC_plasma/dt = -(CL_sys/Vd_plasma)*C_plasma - Q*(C_plasma - C_tissue*Kp)/Vd_plasma
      dC_tissue/dt = Q*(C_plasma - C_tissue*Kp)/Vd_tissue
                     - (Vmax_efflux * C_tissue) / (Km_efflux + C_tissue) / Vd_tissue
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive")
    if vd_plasma_L <= 0:
        raise ValueError("vd_plasma_L must be positive")
    if vd_tissue_L <= 0:
        raise ValueError("vd_tissue_L must be positive")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be positive")
    if km_efflux_mg_L <= 0:
        raise ValueError("km_efflux_mg_L must be positive")
    if vmax_efflux_mg_per_h <= 0:
        raise ValueError("vmax_efflux_mg_per_h must be positive")

    # --- Initial conditions: IV bolus into plasma ---
    c_plasma = dose_mg / vd_plasma_L
    c_tissue = 0.0

    times_h: list = []
    c_plasma_list: list = []
    c_tissue_list: list = []

    n_steps = int(t_end_h / dt_h) + 1
    t = 0.0

    for _step in range(n_steps):
        times_h.append(t)
        c_plasma_list.append(c_plasma)
        c_tissue_list.append(c_tissue)

        # Forward Euler derivatives
        exchange = q_L_per_h * (c_plasma - c_tissue * kp_tissue)
        efflux_rate = (vmax_efflux_mg_per_h * c_tissue) / (km_efflux_mg_L + c_tissue)

        dc_plasma = -(cl_sys_L_per_h / vd_plasma_L) * c_plasma - exchange / vd_plasma_L
        dc_tissue = exchange / vd_tissue_L - efflux_rate / vd_tissue_L

        c_plasma = max(0.0, c_plasma + dc_plasma * dt_h)
        c_tissue = max(0.0, c_tissue + dc_tissue * dt_h)
        t = round(t + dt_h, 10)

    # --- Derived metrics ---
    cmax_plasma = max(c_plasma_list)
    cmax_tissue = max(c_tissue_list)

    # Trapezoidal AUC
    auc_plasma = _trapz(times_h, c_plasma_list)
    auc_tissue = _trapz(times_h, c_tissue_list)

    tissue_to_plasma_auc_ratio = auc_tissue / auc_plasma if auc_plasma > 0 else 0.0
    efflux_saturation_index = cmax_tissue / km_efflux_mg_L

    # Estimate t½ from plasma: find time where concentration drops to half of Cmax
    t_half_plasma_h = _estimate_half_life(times_h, c_plasma_list, cmax_plasma)

    notes = (
        f"2-compartment efflux model for {drug_name}. "
        f"Vmax={vmax_efflux_mg_per_h} mg/h, Km={km_efflux_mg_L} mg/L. "
        f"Saturation index={efflux_saturation_index:.2f} "
        f"({'saturated' if efflux_saturation_index > 1 else 'unsaturated'})."
    )

    return EfluxTransporterPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_list,
        c_tissue_mg_L=c_tissue_list,
        cmax_plasma=cmax_plasma,
        cmax_tissue=cmax_tissue,
        auc_plasma=auc_plasma,
        auc_tissue=auc_tissue,
        tissue_to_plasma_auc_ratio=tissue_to_plasma_auc_ratio,
        efflux_saturation_index=efflux_saturation_index,
        t_half_plasma_h=t_half_plasma_h,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Assessment helper
# ---------------------------------------------------------------------------


def assess_efflux_saturation(
    cmax_tissue_mg_L: float,
    km_efflux_mg_L: float,
) -> dict:
    """
    Assess efflux transporter saturation risk.

    Returns dict with:
      - saturation_index: Cmax_tissue / Km_efflux
      - risk_level: "low" / "moderate" / "high"
    """
    if km_efflux_mg_L <= 0:
        raise ValueError("km_efflux_mg_L must be positive")

    saturation_index = cmax_tissue_mg_L / km_efflux_mg_L

    if saturation_index < 0.1:
        risk_level = "low"
    elif saturation_index < 1.0:
        risk_level = "moderate"
    else:
        risk_level = "high"

    return {
        "saturation_index": saturation_index,
        "risk_level": risk_level,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _trapz(x: list, y: list) -> float:
    """Manual trapezoidal integration."""
    total = 0.0
    for i in range(1, len(x)):
        total += 0.5 * (y[i - 1] + y[i]) * (x[i] - x[i - 1])
    return total


def _estimate_half_life(times_h: list, conc: list, cmax: float) -> float:
    """Estimate terminal half-life from plasma concentration profile."""
    half_max = cmax * 0.5
    # Find first time after Cmax where concentration falls to half
    cmax_idx = conc.index(cmax)
    for i in range(cmax_idx + 1, len(conc)):
        if conc[i] <= half_max:
            # Linear interpolation
            if i > 0 and conc[i - 1] > conc[i]:
                frac = (conc[i - 1] - half_max) / (conc[i - 1] - conc[i])
                return times_h[i - 1] + frac * (times_h[i] - times_h[i - 1])
            return times_h[i]
    # Fallback: use log-linear terminal slope if we have enough points
    n = len(conc)
    if n >= 4:
        try:
            c_end = conc[-1]
            c_start = conc[cmax_idx + 1] if cmax_idx + 1 < n else cmax
            t_end = times_h[-1]
            t_start = times_h[cmax_idx + 1] if cmax_idx + 1 < len(times_h) else times_h[0]
            if c_start > 0 and c_end > 0 and t_end > t_start:
                lam_z = log(c_start / c_end) / (t_end - t_start)
                if lam_z > 0:
                    return log(2.0) / lam_z
        except (ValueError, ZeroDivisionError):
            pass
    return times_h[-1]
