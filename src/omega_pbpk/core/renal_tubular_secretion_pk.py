"""
Phase 969 — Mechanistic renal tubular secretion PK model.

Models OAT/OCT/P-gp transporter-mediated secretion with saturable kinetics
alongside glomerular filtration and passive reabsorption.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["RenalTubularResult", "simulate_renal_tubular_pk", "screen_dose_nonlinearity"]


@dataclass
class RenalTubularResult:
    drug_name: str
    dose_mg: float
    logp: float
    times_h: list[float]
    c_plasma_mg_L: list[float]
    cmax_mg_L: float
    auc_mg_h_per_L: float
    cl_filtration_L_per_h: float
    cl_secretion_at_cmax_L_per_h: float
    cl_reabsorption_L_per_h: float
    cl_renal_total_L_per_h: float
    fe_renal: float
    saturation_at_cmax_pct: float
    notes: str


def _validate_params(
    dose_mg: float,
    fu_plasma: float,
    gfr_L_per_h: float,
    cl_hepatic_L_per_h: float,
    vd_L: float,
    route: str,
) -> None:
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if fu_plasma <= 0 or fu_plasma > 1:
        raise ValueError("fu_plasma must be in (0, 1]")
    if gfr_L_per_h <= 0:
        raise ValueError("gfr_L_per_h must be > 0")
    if cl_hepatic_L_per_h < 0:
        raise ValueError("cl_hepatic_L_per_h must be >= 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if route not in {"iv", "oral"}:
        raise ValueError("route must be 'iv' or 'oral'")


def _compute_cl_reabsorption(logp: float, vd_L: float) -> float:
    """Passive reabsorption CL (L/h). More lipophilic → more reabsorption."""
    kreabs = 0.01 * max(0.0, logp) * 0.1  # /h
    return kreabs * vd_L


def _active_secretion_cl(
    c_mg_L: float,
    fu_plasma: float,
    vmax_sec_mg_per_h: float,
    km_sec_mg_per_L: float,
) -> float:
    """Concentration-dependent active secretion CL (L/h)."""
    return fu_plasma * vmax_sec_mg_per_h / (km_sec_mg_per_L + c_mg_L)


def _trapezoidal_auc(times: list[float], concs: list[float]) -> float:
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        auc += 0.5 * (concs[i - 1] + concs[i]) * dt
    return auc


def simulate_renal_tubular_pk(
    drug_name: str,
    dose_mg: float,
    logp: float = 1.0,
    fu_plasma: float = 0.5,
    gfr_L_per_h: float = 7.5,
    vmax_sec_mg_per_h: float = 50.0,
    km_sec_mg_per_L: float = 5.0,
    cl_hepatic_L_per_h: float = 3.0,
    vd_L: float = 30.0,
    ka_per_h: float = 1.0,
    route: str = "iv",
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> RenalTubularResult:
    """Simulate 1-cpt PK with mechanistic renal tubular secretion.

    Parameters
    ----------
    drug_name : str
    dose_mg : float
    logp : float
    fu_plasma : float  fraction unbound in plasma (0, 1]
    gfr_L_per_h : float  glomerular filtration rate
    vmax_sec_mg_per_h : float  maximal secretion rate
    km_sec_mg_per_L : float  Michaelis constant for secretion
    cl_hepatic_L_per_h : float
    vd_L : float  volume of distribution
    ka_per_h : float  absorption rate (oral)
    route : str  "iv" or "oral"
    t_end_h : float
    dt_h : float  forward Euler step

    Returns
    -------
    RenalTubularResult
    """
    _validate_params(dose_mg, fu_plasma, gfr_L_per_h, cl_hepatic_L_per_h, vd_L, route)

    # Fixed CL components
    cl_filt = gfr_L_per_h * fu_plasma  # L/h — filtration
    cl_reabs = _compute_cl_reabsorption(logp, vd_L)  # L/h — reabsorption (removes drug from urine back to plasma)
    ke_hepatic = cl_hepatic_L_per_h / vd_L  # /h

    # Initial conditions
    n_steps = int(math.ceil(t_end_h / dt_h)) + 1
    times: list[float] = []
    c_plasma: list[float] = []

    if route == "iv":
        C = dose_mg / vd_L
        A_gut = 0.0
    else:
        C = 0.0
        A_gut = dose_mg

    for i in range(n_steps):
        t = i * dt_h
        times.append(t)
        c_plasma.append(max(0.0, C))

        # Concentration-dependent active secretion CL
        c_curr = max(0.0, C)
        cl_sec = _active_secretion_cl(c_curr, fu_plasma, vmax_sec_mg_per_h, km_sec_mg_per_L)

        # Net renal ke
        cl_renal_net = cl_filt + cl_sec - cl_reabs
        ke_renal = max(0.0, cl_renal_net) / vd_L

        # Total elimination ke
        ke_total = ke_hepatic + ke_renal

        if route == "oral":
            # Absorption input
            dA_gut = -ka_per_h * A_gut
            dC = (ka_per_h * A_gut / vd_L) - ke_total * C
            A_gut = max(0.0, A_gut + dA_gut * dt_h)
        else:
            dC = -ke_total * C

        C = max(0.0, C + dC * dt_h)

    # Summaries
    cmax = max(c_plasma)
    auc = _trapezoidal_auc(times, c_plasma)

    # Compute CL at Cmax
    cl_sec_cmax = _active_secretion_cl(cmax, fu_plasma, vmax_sec_mg_per_h, km_sec_mg_per_L)
    cl_renal_total = max(0.0, cl_filt + cl_sec_cmax - cl_reabs)

    # Total apparent CL (from dose / AUC)
    cl_total = dose_mg / auc if auc > 0 else (cl_hepatic_L_per_h + cl_renal_total)

    fe_renal = cl_renal_total / cl_total if cl_total > 0 else 0.0
    fe_renal = max(0.0, min(1.0, fe_renal))

    saturation_pct = cmax / (km_sec_mg_per_L + cmax) * 100.0

    notes = (
        f"Route={route}; CLfilt={cl_filt:.2f} L/h; "
        f"CLsec(Cmax)={cl_sec_cmax:.2f} L/h; "
        f"CLreabs={cl_reabs:.3f} L/h; "
        f"CLrenal_net={cl_renal_total:.2f} L/h; "
        f"saturation@Cmax={saturation_pct:.1f}%"
    )

    return RenalTubularResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        logp=logp,
        times_h=times,
        c_plasma_mg_L=c_plasma,
        cmax_mg_L=cmax,
        auc_mg_h_per_L=auc,
        cl_filtration_L_per_h=cl_filt,
        cl_secretion_at_cmax_L_per_h=cl_sec_cmax,
        cl_reabsorption_L_per_h=cl_reabs,
        cl_renal_total_L_per_h=cl_renal_total,
        fe_renal=fe_renal,
        saturation_at_cmax_pct=saturation_pct,
        notes=notes,
    )


def screen_dose_nonlinearity(
    drug_name: str,
    logp: float,
    doses_mg: list = None,
    fu_plasma: float = 0.5,
    gfr_L_per_h: float = 7.5,
    vmax_sec_mg_per_h: float = 50.0,
    km_sec_mg_per_L: float = 5.0,
    cl_hepatic_L_per_h: float = 3.0,
    vd_L: float = 30.0,
    ka_per_h: float = 1.0,
    route: str = "iv",
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> list[RenalTubularResult]:
    """Screen multiple doses to demonstrate secretion saturation nonlinearity.

    Returns list of RenalTubularResult sorted by dose ascending.
    """
    if doses_mg is None:
        doses_mg = [1.0, 5.0, 10.0, 50.0, 100.0]

    results = []
    for dose in sorted(doses_mg):
        r = simulate_renal_tubular_pk(
            drug_name=drug_name,
            dose_mg=dose,
            logp=logp,
            fu_plasma=fu_plasma,
            gfr_L_per_h=gfr_L_per_h,
            vmax_sec_mg_per_h=vmax_sec_mg_per_h,
            km_sec_mg_per_L=km_sec_mg_per_L,
            cl_hepatic_L_per_h=cl_hepatic_L_per_h,
            vd_L=vd_L,
            ka_per_h=ka_per_h,
            route=route,
            t_end_h=t_end_h,
            dt_h=dt_h,
        )
        results.append(r)

    return results
