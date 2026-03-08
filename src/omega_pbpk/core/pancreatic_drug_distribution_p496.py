"""Pancreatic drug distribution (Phase 496): 2-compartment plasma <-> pancreas model.

Kp_pancreas = 0.5 + 0.2 * max(0, logP), clamped [0.1, 8.0].
"""

from __future__ import annotations

from dataclasses import dataclass


def _trapz(times: list, values: list) -> float:
    """Manual trapezoidal AUC integration."""
    if len(times) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(times)):
        total += 0.5 * (values[i] + values[i - 1]) * (times[i] - times[i - 1])
    return total


def _compute_kp_pancreas(logP: float) -> float:
    """Compute pancreatic Kp from logP, clamped to [0.1, 8.0]."""
    kp = 0.5 + 0.2 * max(0.0, logP)
    return max(0.1, min(8.0, kp))


def _classify_therapeutic_adequacy(panc_to_plasma: float) -> str:
    """Classify therapeutic adequacy based on pancreas/plasma ratio."""
    if panc_to_plasma < 0.5:
        return "insufficient"
    if panc_to_plasma <= 1.5:
        return "adequate"
    return "high"


@dataclass
class PancreaticDistributionResult:
    """Result of pancreatic drug distribution simulation (Phase 496)."""

    drug_name: str
    dose_mg: float
    logP: float
    times_h: list
    c_plasma_mg_L: list
    c_pancreas_mg_L: list
    cmax_pancreas_mg_L: float
    tmax_pancreas_h: float
    auc_pancreas_mg_h_per_L: float
    auc_plasma_mg_h_per_L: float
    kp_pancreas: float
    panc_to_plasma_ratio: float
    therapeutic_adequacy: str
    notes: str


def simulate_pancreatic_distribution(
    drug_name: str,
    dose_mg: float,
    logP: float,
    cl_sys_L_per_h: float,
    vd_plasma_L: float = 5.0,
    vd_panc_L: float = 1.0,
    Q_panc_L_per_h: float = 1.2,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> PancreaticDistributionResult:
    """Simulate 2-compartment plasma-pancreas drug distribution using Forward Euler.

    ODE:
        dC_plasma/dt = -Q_p/Vp*(C_plasma - C_panc/Kp) - k_el*C_plasma
        dC_panc/dt  =  Q_p/V_panc*(C_plasma - C_panc/Kp)

    Args:
        drug_name: Drug identifier.
        dose_mg: IV bolus dose in mg.
        logP: Octanol-water partition coefficient.
        cl_sys_L_per_h: Systemic clearance (L/h).
        vd_plasma_L: Plasma volume of distribution (L).
        vd_panc_L: Pancreatic volume of distribution (L).
        Q_panc_L_per_h: Pancreatic blood flow (L/h).
        t_end_h: Simulation end time (h).
        dt_h: Time step (h).

    Returns:
        PancreaticDistributionResult with PK profiles and derived metrics.
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_plasma_L <= 0:
        raise ValueError("vd_plasma_L must be > 0")
    if vd_panc_L <= 0:
        raise ValueError("vd_panc_L must be > 0")
    if Q_panc_L_per_h <= 0:
        raise ValueError("Q_panc_L_per_h must be > 0")

    kp = _compute_kp_pancreas(logP)
    k_el = cl_sys_L_per_h / vd_plasma_L

    # Initial conditions: IV bolus
    c_plasma = dose_mg / vd_plasma_L
    c_panc = 0.0

    times: list = []
    c_plasma_list: list = []
    c_panc_list: list = []

    t = 0.0
    n_steps = int(round(t_end_h / dt_h)) + 1

    for _i in range(n_steps):
        times.append(t)
        c_plasma_list.append(c_plasma)
        c_panc_list.append(c_panc)

        # ODE: plasma
        dcp_dt = -(Q_panc_L_per_h / vd_plasma_L) * (c_plasma - c_panc / kp) - k_el * c_plasma
        # ODE: pancreas
        dct_dt = (Q_panc_L_per_h / vd_panc_L) * (c_plasma - c_panc / kp)

        c_plasma = max(0.0, c_plasma + dcp_dt * dt_h)
        c_panc = max(0.0, c_panc + dct_dt * dt_h)

        t += dt_h

    # Derived metrics
    auc_plasma = _trapz(times, c_plasma_list)
    auc_pancreas = _trapz(times, c_panc_list)
    cmax_pancreas = max(c_panc_list) if c_panc_list else 0.0
    tmax_idx = c_panc_list.index(cmax_pancreas)
    tmax_pancreas_h = times[tmax_idx]

    # Ratio at t_end (last time point)
    c_plasma_end = c_plasma_list[-1]
    c_panc_end = c_panc_list[-1]
    if c_plasma_end > 0:
        panc_to_plasma_ratio = c_panc_end / c_plasma_end
    else:
        panc_to_plasma_ratio = 0.0

    adequacy = _classify_therapeutic_adequacy(panc_to_plasma_ratio)

    notes = (
        f"Kp_pancreas={kp:.3f}; AUC_ratio={auc_pancreas / auc_plasma:.3f}"
        if auc_plasma > 0
        else f"Kp_pancreas={kp:.3f}; AUC_ratio=NA"
    )
    notes += f"; therapeutic_adequacy={adequacy}"

    return PancreaticDistributionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        logP=logP,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        c_pancreas_mg_L=c_panc_list,
        cmax_pancreas_mg_L=cmax_pancreas,
        tmax_pancreas_h=tmax_pancreas_h,
        auc_pancreas_mg_h_per_L=auc_pancreas,
        auc_plasma_mg_h_per_L=auc_plasma,
        kp_pancreas=kp,
        panc_to_plasma_ratio=panc_to_plasma_ratio,
        therapeutic_adequacy=adequacy,
        notes=notes,
    )


def screen_antibiotics(
    dose_mg: float,
    logp_values: list,
    cl_sys_L_per_h: float,
) -> list:
    """Screen multiple compounds by logP for pancreatic distribution.

    Args:
        dose_mg: IV bolus dose in mg.
        logp_values: List of logP values to screen.
        cl_sys_L_per_h: Systemic clearance (L/h).

    Returns:
        List of PancreaticDistributionResult sorted by kp_pancreas descending.
    """
    results = []
    for lp in logp_values:
        name = f"Drug_logP{lp}"
        res = simulate_pancreatic_distribution(
            drug_name=name,
            dose_mg=dose_mg,
            logP=lp,
            cl_sys_L_per_h=cl_sys_L_per_h,
        )
        results.append(res)

    results.sort(key=lambda r: r.kp_pancreas, reverse=True)
    return results
