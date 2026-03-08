"""
Phase 1028 — Hepatic Blood Flow PK

Model the effect of hepatic blood flow changes on drug PK using the well-stirred
model. Includes ODE simulation and flow-condition comparison.
"""

from __future__ import annotations

from dataclasses import dataclass


def _trapz(times: list[float], values: list[float]) -> float:
    """Manual trapezoidal integration."""
    if len(times) < 2:
        return 0.0
    area = 0.0
    for i in range(len(times) - 1):
        dt = times[i + 1] - times[i]
        area += 0.5 * (values[i] + values[i + 1]) * dt
    return area


@dataclass
class HepaticBloodFlowResult:
    drug_name: str
    dose_mg: float
    hepatic_blood_flow_L_h: float
    times_h: list
    c_plasma_mg_L: list
    c_liver_mg_L: list
    extraction_ratio: float
    hepatic_clearance_L_h: float
    oral_bioavailability: float
    auc_plasma: float
    cmax_plasma: float
    tmax_plasma_h: float
    flow_sensitivity: str
    notes: str


def simulate_hepatic_blood_flow_pk(
    drug_name: str,
    dose_mg: float,
    fu_blood: float = 0.1,
    clint_L_per_h: float = 50.0,
    hepatic_blood_flow_L_h: float = 90.0,
    vd_L: float = 50.0,
    route: str = "oral",
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> HepaticBloodFlowResult:
    """
    Simulate hepatic blood flow PK using the well-stirred model.

    Parameters
    ----------
    drug_name : str
        Drug name.
    dose_mg : float
        Dose in mg. Must be > 0.
    fu_blood : float
        Unbound fraction in blood (0 < fu_blood <= 1).
    clint_L_per_h : float
        Intrinsic clearance (L/h). Must be > 0.
    hepatic_blood_flow_L_h : float
        Hepatic blood flow (L/h). Normal ~90 L/h (1.5 L/min). Must be > 0.
    vd_L : float
        Volume of distribution (L). Must be > 0.
    route : str
        "oral" or "iv".
    t_end_h : float
        Simulation end time (h).
    dt_h : float
        Time step (h).

    Returns
    -------
    HepaticBloodFlowResult
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if fu_blood <= 0 or fu_blood > 1:
        raise ValueError(f"fu_blood must be in (0, 1], got {fu_blood}")
    if clint_L_per_h <= 0:
        raise ValueError(f"clint_L_per_h must be > 0, got {clint_L_per_h}")
    if hepatic_blood_flow_L_h <= 0:
        raise ValueError(f"hepatic_blood_flow_L_h must be > 0, got {hepatic_blood_flow_L_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if route not in {"oral", "iv"}:
        raise ValueError(f"route must be 'oral' or 'iv', got '{route}'")

    q_h = hepatic_blood_flow_L_h

    # --- Well-stirred model ---
    fu_clint = fu_blood * clint_L_per_h
    cl_h = q_h * fu_clint / (q_h + fu_clint)
    e = cl_h / q_h
    f_oral = 1.0 - e

    cl_total = cl_h
    ke = cl_total / vd_L
    ka = 1.0  # absorption rate constant /h

    # --- ODE simulation (Forward Euler) ---
    n_steps = max(1, int(round(t_end_h / dt_h)))
    times: list[float] = []
    c_plasma: list[float] = []
    c_liver: list[float] = []

    if route == "iv":
        c = dose_mg / vd_L
        a_gut = 0.0
    else:
        c = 0.0
        a_gut = dose_mg

    times.append(0.0)
    c_plasma.append(c)
    c_liver.append(c * (1.0 + e))

    for _i in range(n_steps):
        if route == "iv":
            dc_dt = -ke * c
            c = max(0.0, c + dc_dt * dt_h)
            a_gut = 0.0
        else:
            # oral: gut absorption with F_oral applied to absorption flux
            da_gut_dt = -ka * a_gut
            dc_dt = ka * a_gut * f_oral / vd_L - ke * c
            a_gut = max(0.0, a_gut + da_gut_dt * dt_h)
            c = max(0.0, c + dc_dt * dt_h)

        t = times[-1] + dt_h
        times.append(t)
        c_plasma.append(c)
        c_liver.append(c * (1.0 + e))

    # --- PK metrics ---
    auc = _trapz(times, c_plasma)
    cmax = max(c_plasma)
    tmax = times[c_plasma.index(cmax)]

    # --- Flow sensitivity classification ---
    if e > 0.7:
        flow_sensitivity = "flow-limited"
    elif e < 0.3:
        flow_sensitivity = "capacity-limited"
    else:
        flow_sensitivity = "intermediate"

    notes = (
        f"Drug: {drug_name}; route={route}; Q_h={q_h:.1f} L/h; "
        f"CLint={clint_L_per_h:.1f} L/h; E={e:.3f}; F_oral={f_oral:.3f}; "
        f"CL_h={cl_h:.3f} L/h; {flow_sensitivity}"
    )

    return HepaticBloodFlowResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        hepatic_blood_flow_L_h=q_h,
        times_h=times,
        c_plasma_mg_L=c_plasma,
        c_liver_mg_L=c_liver,
        extraction_ratio=e,
        hepatic_clearance_L_h=cl_h,
        oral_bioavailability=f_oral,
        auc_plasma=auc,
        cmax_plasma=cmax,
        tmax_plasma_h=tmax,
        flow_sensitivity=flow_sensitivity,
        notes=notes,
    )


def compare_flow_conditions(
    drug_name: str,
    dose_mg: float,
    fu_blood: float = 0.1,
    clint_L_per_h: float = 50.0,
    flows_L_h: list | None = None,
) -> list:
    """
    Compare PK across different hepatic blood flow conditions.

    Parameters
    ----------
    drug_name : str
        Drug name.
    dose_mg : float
        Dose in mg.
    fu_blood : float
        Unbound fraction in blood.
    clint_L_per_h : float
        Intrinsic clearance (L/h).
    flows_L_h : list or None
        List of hepatic blood flows (L/h) to compare.
        Default: [45.0, 90.0, 135.0] (reduced, normal, increased).

    Returns
    -------
    list[HepaticBloodFlowResult]
    """
    if flows_L_h is None:
        flows_L_h = [45.0, 90.0, 135.0]

    results = []
    for flow in flows_L_h:
        result = simulate_hepatic_blood_flow_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            fu_blood=fu_blood,
            clint_L_per_h=clint_L_per_h,
            hepatic_blood_flow_L_h=flow,
        )
        results.append(result)
    return results
