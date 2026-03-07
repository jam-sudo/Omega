"""Phase 735 — Dialysis Drug Clearance.

Models hemodialysis (HD) and peritoneal dialysis (PD) drug removal with
1-compartment ODE during the dialysis session and post-dialysis decay.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class DialysisPKResult:
    """Result of a dialysis PK simulation."""

    drug_name: str
    dose_mg: float
    times_h: list[float]
    c_plasma_mg_L: list[float]
    cl_dialysis_L_per_h: float
    cl_total_L_per_h: float
    cmax_mg_L: float
    auc_mg_h_per_L: float
    t_half_with_dialysis_h: float
    t_half_without_dialysis_h: float
    fraction_removed_by_dialysis: float
    post_dialysis_supplement_mg: float
    dialysis_type: str
    notes: str


_MW_CUTOFF_HD = 500.0  # Da — effective small-molecule dialysis membrane cutoff
_LN2 = math.log(2.0)


def estimate_dialysis_clearance(
    dialysis_type: str,
    mw_da: float,
    fu_plasma: float,
    qb_mL_min: float = 300.0,
    sieving_coeff: float | None = None,
    session_duration_h: float = 4.0,
    exchanges_per_day: float = 4.0,
    clearance_per_exchange_L: float = 2.0,
) -> float:
    """Estimate dialysis clearance in L/h.

    Parameters
    ----------
    dialysis_type:
        "hd" for hemodialysis, "pd" for peritoneal dialysis.
    mw_da:
        Molecular weight in Daltons.
    fu_plasma:
        Unbound fraction in plasma (0–1).
    qb_mL_min:
        Blood flow rate for HD in mL/min.
    sieving_coeff:
        Override for the sieving coefficient (0–1). If None, computed from
        fu_plasma and MW vs. MW_cutoff.
    session_duration_h:
        Duration of a single HD session in hours (unused in CL calc, kept for
        signature symmetry).
    exchanges_per_day:
        Number of PD exchanges per day.
    clearance_per_exchange_L:
        Dialysate volume cleared per PD exchange (L).

    Returns
    -------
    float
        Clearance in L/h (>= 0).
    """
    if dialysis_type == "hd":
        if sieving_coeff is not None:
            sc = max(0.0, min(1.0, sieving_coeff))
        else:
            if mw_da < _MW_CUTOFF_HD:
                sc = fu_plasma * math.sqrt(_MW_CUTOFF_HD / mw_da)
            else:
                sc = 0.0
            sc = max(0.0, min(1.0, sc))
        # CL_HD in L/h: sieving_coeff * Qb_mL_min * 0.06
        cl_L_h = sc * qb_mL_min * 0.06
        return cl_L_h

    elif dialysis_type == "pd":
        # CL_PD = clearance_per_exchange_L * exchanges_per_day / 24
        cl_L_h = clearance_per_exchange_L * exchanges_per_day / 24.0
        return cl_L_h

    else:
        raise ValueError(f"dialysis_type must be 'hd' or 'pd', got '{dialysis_type}'")


def simulate_dialysis_pk(
    drug_name: str,
    dose_mg: float,
    vd_L: float = 50.0,
    cl_normal_L_per_h: float = 1.0,
    mw_da: float = 300.0,
    fu_plasma: float = 0.5,
    dialysis_type: str = "hd",
    qb_mL_min: float = 300.0,
    sieving_coeff: float | None = None,
    session_duration_h: float = 4.0,
    exchanges_per_day: float = 4.0,
    clearance_per_exchange_L: float = 2.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> DialysisPKResult:
    """Simulate drug PK with active dialysis.

    A dialysis session runs from t=0 to t=session_duration_h (HD) or is
    assumed continuous for PD. After the session the drug continues to
    eliminate via normal CL only.

    Parameters
    ----------
    drug_name:
        Identifier string for the drug.
    dose_mg:
        Initial IV bolus dose (mg).
    vd_L:
        Volume of distribution (L).
    cl_normal_L_per_h:
        Residual (non-dialysis) clearance in L/h.
    mw_da:
        Molecular weight (Da).
    fu_plasma:
        Unbound fraction in plasma.
    dialysis_type:
        "hd" or "pd".
    qb_mL_min:
        Blood flow rate for HD (mL/min).
    sieving_coeff:
        Override sieving coefficient; if None, computed from fu_plasma and MW.
    session_duration_h:
        Duration of one HD session in hours.
    exchanges_per_day:
        PD exchanges per day.
    clearance_per_exchange_L:
        Volume cleared per PD exchange (L).
    t_end_h:
        Total simulation time (h).
    dt_h:
        Forward-Euler step size (h).

    Returns
    -------
    DialysisPKResult
    """
    # --- validation ---
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if cl_normal_L_per_h < 0:
        raise ValueError(f"cl_normal_L_per_h must be >= 0, got {cl_normal_L_per_h}")
    if dialysis_type not in ("hd", "pd"):
        raise ValueError(f"dialysis_type must be 'hd' or 'pd', got '{dialysis_type}'")

    cl_dial = estimate_dialysis_clearance(
        dialysis_type=dialysis_type,
        mw_da=mw_da,
        fu_plasma=fu_plasma,
        qb_mL_min=qb_mL_min,
        sieving_coeff=sieving_coeff,
        session_duration_h=session_duration_h,
        exchanges_per_day=exchanges_per_day,
        clearance_per_exchange_L=clearance_per_exchange_L,
    )
    cl_total = cl_normal_L_per_h + cl_dial

    ke_normal = cl_normal_L_per_h / vd_L if vd_L > 0 else 0.0
    ke_dial = cl_dial / vd_L if vd_L > 0 else 0.0
    ke_total = ke_normal + ke_dial

    # Initial concentration
    c0 = dose_mg / vd_L

    # Forward Euler ODE
    times: list[float] = []
    concs: list[float] = []

    t = 0.0
    c = c0
    n_steps = max(1, int(round(t_end_h / dt_h)))

    # For PD, dialysis is continuous (treated as constant extra CL throughout).
    # For HD, dialysis is active only during [0, session_duration_h].
    for i in range(n_steps + 1):
        times.append(t)
        concs.append(c)
        if i < n_steps:
            if dialysis_type == "hd":
                ke = ke_total if t < session_duration_h else ke_normal
            else:
                # PD: continuous extra clearance
                ke = ke_total
            dc_dt = -ke * c
            c = max(0.0, c + dc_dt * dt_h)
            t = round(t + dt_h, 10)

    # AUC via trapezoidal rule
    auc = 0.0
    for i in range(len(times) - 1):
        auc += 0.5 * (concs[i] + concs[i + 1]) * (times[i + 1] - times[i])

    cmax = c0

    # Half-lives
    if ke_total > 0:
        t_half_with = _LN2 / ke_total
    else:
        t_half_with = float("inf")

    if ke_normal > 0:
        t_half_without = _LN2 / ke_normal
    else:
        t_half_without = float("inf")

    # Fraction of elimination attributable to dialysis
    fraction_removed = cl_dial / cl_total if cl_total > 0 else 0.0

    # Post-dialysis supplement to restore Cmax
    post_dial_supplement = dose_mg * fraction_removed

    notes_parts = [
        f"dialysis_type={dialysis_type}",
        f"cl_dialysis={cl_dial:.3f} L/h",
        f"cl_normal={cl_normal_L_per_h:.3f} L/h",
        f"sieving_coeff_effective={cl_dial / (qb_mL_min * 0.06 + 1e-12):.3f}"
        if dialysis_type == "hd"
        else f"exchanges={exchanges_per_day}/day",
        f"fraction_removed={fraction_removed:.3f}",
    ]
    notes = "; ".join(notes_parts)

    return DialysisPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times,
        c_plasma_mg_L=concs,
        cl_dialysis_L_per_h=cl_dial,
        cl_total_L_per_h=cl_total,
        cmax_mg_L=cmax,
        auc_mg_h_per_L=auc,
        t_half_with_dialysis_h=t_half_with,
        t_half_without_dialysis_h=t_half_without,
        fraction_removed_by_dialysis=fraction_removed,
        post_dialysis_supplement_mg=post_dial_supplement,
        dialysis_type=dialysis_type,
        notes=notes,
    )
