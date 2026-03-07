"""Peritoneal dialysis PK model.

Models drug removal kinetics during peritoneal dialysis (PD), where the
peritoneum acts as a semipermeable membrane allowing drug to diffuse from
blood into dialysate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class PeritonealDialysisPKResult:
    """Result of peritoneal dialysis PK simulation."""

    drug_name: str
    dose_mg: float
    mw_Da: float
    fu_plasma: float
    times_h: list
    c_plasma_mg_L: list
    c_dialysate_mg_L: list
    cmax_plasma_mg_L: float
    auc_plasma_mg_h_per_L: float
    pd_clearance_L_per_h: float
    fraction_removed_by_pd: float
    dose_supplement_needed: bool
    mw_class: str
    notes: str


def _mw_class(mw_Da: float) -> str:
    if mw_Da < 500:
        return "small"
    if mw_Da <= 5000:
        return "medium"
    return "large"


def _koa(mw_Da: float) -> float:
    """MW-dependent mass transfer coefficient (L/h)."""
    mw_kDa = mw_Da / 1000.0
    return max(0.1, 2.0 * math.exp(-mw_kDa / 10.0))


def _trapz_auc(times: list, concs: list) -> float:
    """Manual trapezoidal AUC."""
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        auc += 0.5 * (concs[i - 1] + concs[i]) * dt
    return auc


def simulate_peritoneal_dialysis_pk(
    drug_name: str,
    dose_mg: float,
    mw_Da: float,
    fu_plasma: float = 0.5,
    cl_residual_L_per_h: float = 1.0,
    vd_L: float = 30.0,
    dwell_time_h: float = 4.0,
    n_exchanges: int = 4,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> PeritonealDialysisPKResult:
    """Simulate drug PK during peritoneal dialysis.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Administered dose in mg.
    mw_Da:
        Molecular weight in Daltons.
    fu_plasma:
        Unbound fraction in plasma (0 < fu_plasma <= 1).
    cl_residual_L_per_h:
        Residual renal/hepatic clearance in L/h.
    vd_L:
        Volume of distribution in L.
    dwell_time_h:
        PD dwell time in hours (informational; model uses continuous drain).
    n_exchanges:
        Number of PD exchanges per day (informational).
    t_end_h:
        Simulation end time in hours.
    dt_h:
        ODE step size in hours.

    Returns
    -------
    PeritonealDialysisPKResult
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_residual_L_per_h < 0:
        raise ValueError(f"cl_residual_L_per_h must be >= 0, got {cl_residual_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if mw_Da <= 0:
        raise ValueError(f"mw_Da must be > 0, got {mw_Da}")
    if not (0 < fu_plasma <= 1):
        raise ValueError(f"fu_plasma must be in (0, 1], got {fu_plasma}")
    if dwell_time_h <= 0:
        raise ValueError(f"dwell_time_h must be > 0, got {dwell_time_h}")

    # Derived parameters
    ke = cl_residual_L_per_h / vd_L
    koa = _koa(mw_Da)
    kpd = koa * fu_plasma / vd_L  # plasma → dialysate transfer rate constant (1/h)
    vd_dial = 2.0  # L, standard 2L fill volume
    k_drain = 0.1  # 1/h continuous slow drain approximation

    # Initial conditions
    c_plasma = dose_mg / vd_L
    c_dialysate = 0.0

    times: list = []
    cp_list: list = []
    cd_list: list = []

    t = 0.0
    steps = int(round(t_end_h / dt_h)) + 1

    for _ in range(steps):
        times.append(t)
        cp_list.append(c_plasma)
        cd_list.append(c_dialysate)

        # ODEs (Forward Euler)
        dcp = -ke * c_plasma - kpd * (c_plasma - c_dialysate)
        dcd = kpd * (vd_L / vd_dial) * (c_plasma - c_dialysate) - k_drain * c_dialysate

        c_plasma = max(0.0, c_plasma + dcp * dt_h)
        c_dialysate = max(0.0, c_dialysate + dcd * dt_h)

        t = round(t + dt_h, 10)

    # Derived outputs
    cmax = max(cp_list)
    auc_plasma = _trapz_auc(times, cp_list)
    auc_dialysate = _trapz_auc(times, cd_list)

    # PD clearance estimated from dialysate AUC / plasma AUC
    pd_clearance = (auc_dialysate * k_drain * vd_dial) / max(auc_plasma, 1e-12)
    total_cl = cl_residual_L_per_h + pd_clearance
    fraction_removed = pd_clearance / max(total_cl, 1e-12)
    fraction_removed = min(1.0, max(0.0, fraction_removed))

    dose_supplement_needed = fraction_removed > 0.25

    mw_cls = _mw_class(mw_Da)
    koa_val = koa

    notes = (
        f"KoA={koa_val:.3f} L/h, kpd={kpd:.4f} 1/h, "
        f"n_exchanges={n_exchanges}/day, dwell={dwell_time_h}h. "
        f"MW class: {mw_cls}. "
        f"{'Supplemental dose recommended.' if dose_supplement_needed else 'No supplemental dose needed.'}"
    )

    return PeritonealDialysisPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        mw_Da=mw_Da,
        fu_plasma=fu_plasma,
        times_h=times,
        c_plasma_mg_L=cp_list,
        c_dialysate_mg_L=cd_list,
        cmax_plasma_mg_L=cmax,
        auc_plasma_mg_h_per_L=auc_plasma,
        pd_clearance_L_per_h=pd_clearance,
        fraction_removed_by_pd=fraction_removed,
        dose_supplement_needed=dose_supplement_needed,
        mw_class=mw_cls,
        notes=notes,
    )
