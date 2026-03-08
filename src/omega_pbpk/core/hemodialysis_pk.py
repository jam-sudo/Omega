"""
Phase 1137 — Hemodialysis PK model.

Models drug removal kinetics during hemodialysis (HD) sessions,
critical for dosing in ESRD patients.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class HemodialysisPKResult:
    drug_name: str
    dose_mg: float
    mw_Da: float
    fu_plasma: float
    times_h: list
    c_plasma_mg_L: list
    cmax_plasma_mg_L: float
    auc_plasma_mg_h_per_L: float
    auc_during_dialysis_mg_h_per_L: float
    hd_clearance_L_per_h: float
    fraction_removed_by_hd: float
    dose_supplement_recommended: bool
    supplemental_dose_mg: float
    mw_class: str
    notes: str


def _compute_koa_hd(mw_Da: float, fu_plasma: float) -> float:
    """Compute mass transfer-area coefficient KoA in L/min."""
    mw_kDa = mw_Da / 1000.0
    return fu_plasma * max(0.5, 3.0 * math.exp(-mw_kDa / 5.0))


def _compute_cl_dialysis(mw_Da: float, fu_plasma: float, qd_L_per_min: float = 0.5) -> float:
    """Compute dialysis clearance in L/h."""
    koa_hd_L_per_min = _compute_koa_hd(mw_Da, fu_plasma)
    e = 1.0 - math.exp(-koa_hd_L_per_min / qd_L_per_min)
    e = max(0.0, min(1.0, e))
    cl_dialysis_L_per_min = qd_L_per_min * e
    return cl_dialysis_L_per_min * 60.0  # convert to L/h


def _mw_class(mw_Da: float) -> str:
    if mw_Da < 500:
        return "small"
    elif mw_Da <= 5000:
        return "medium"
    return "large"


def _trapz(times: list, values: list) -> float:
    """Manual trapezoidal integration."""
    area = 0.0
    for i in range(1, len(times)):
        area += 0.5 * (values[i - 1] + values[i]) * (times[i] - times[i - 1])
    return area


def simulate_hemodialysis_pk(
    drug_name: str,
    dose_mg: float,
    mw_Da: float,
    fu_plasma: float = 0.5,
    cl_residual_L_per_h: float = 0.5,
    vd_L: float = 40.0,
    dialysis_duration_h: float = 4.0,
    dialysis_start_h: float = 0.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> HemodialysisPKResult:
    """Simulate plasma PK during a hemodialysis session.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    dose_mg:
        IV bolus dose administered at t=0.
    mw_Da:
        Molecular weight in Daltons.
    fu_plasma:
        Unbound fraction in plasma (0, 1].
    cl_residual_L_per_h:
        Residual (non-HD) clearance in L/h.
    vd_L:
        Volume of distribution in L.
    dialysis_duration_h:
        Duration of the HD session in hours.
    dialysis_start_h:
        Time after dose when HD begins (h).
    t_end_h:
        Simulation end time (h).
    dt_h:
        Forward Euler step size (h).
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_residual_L_per_h < 0:
        raise ValueError("cl_residual_L_per_h must be >= 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if mw_Da <= 0:
        raise ValueError("mw_Da must be > 0")
    if not (0 < fu_plasma <= 1):
        raise ValueError("fu_plasma must be in (0, 1]")
    if dialysis_duration_h <= 0:
        raise ValueError("dialysis_duration_h must be > 0")

    cl_hd = _compute_cl_dialysis(mw_Da, fu_plasma)
    ke_residual = cl_residual_L_per_h / vd_L

    dialysis_end_h = dialysis_start_h + dialysis_duration_h

    # Initial conditions: bolus at t=0 → C_plasma = dose/Vd
    c = dose_mg / vd_L
    t = 0.0

    times_h: list = [t]
    c_plasma: list = [c]

    n_steps = int(round(t_end_h / dt_h))

    for _ in range(n_steps):
        in_dialysis = dialysis_start_h <= t < dialysis_end_h
        cl_active = cl_hd if in_dialysis else 0.0
        ke_total = ke_residual + cl_active / vd_L

        dc_dt = -ke_total * c
        c = max(0.0, c + dc_dt * dt_h)
        t = t + dt_h

        times_h.append(t)
        c_plasma.append(c)

    auc_total = _trapz(times_h, c_plasma)

    # AUC during dialysis window
    dial_times: list = []
    dial_conc: list = []
    for ti, ci in zip(times_h, c_plasma):
        if dialysis_start_h <= ti <= dialysis_end_h:
            dial_times.append(ti)
            dial_conc.append(ci)
    auc_dial = _trapz(dial_times, dial_conc) if len(dial_times) >= 2 else 0.0

    cmax = max(c_plasma)

    # Fraction removed = mass removed by HD / initial dose
    # Mass removed = integral of CL_hd * C(t) dt over dialysis window
    mass_removed = auc_dial * cl_hd
    fraction_removed = mass_removed / dose_mg
    fraction_removed = max(0.0, min(1.0, fraction_removed))

    supplement_recommended = fraction_removed > 0.30
    supplemental_dose_mg = dose_mg * fraction_removed if supplement_recommended else 0.0

    mw_cls = _mw_class(mw_Da)
    notes = (
        f"KoA_HD={_compute_koa_hd(mw_Da, fu_plasma):.3f} L/min; "
        f"E={cl_hd / 30.0:.3f}; "  # E = CLdial/(QD in L/h) = CLdial/30
        f"Dialysis: {dialysis_start_h:.1f}–{dialysis_end_h:.1f} h; "
        f"MW class: {mw_cls}"
    )

    return HemodialysisPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        mw_Da=mw_Da,
        fu_plasma=fu_plasma,
        times_h=times_h,
        c_plasma_mg_L=c_plasma,
        cmax_plasma_mg_L=cmax,
        auc_plasma_mg_h_per_L=auc_total,
        auc_during_dialysis_mg_h_per_L=auc_dial,
        hd_clearance_L_per_h=cl_hd,
        fraction_removed_by_hd=fraction_removed,
        dose_supplement_recommended=supplement_recommended,
        supplemental_dose_mg=supplemental_dose_mg,
        mw_class=mw_cls,
        notes=notes,
    )
