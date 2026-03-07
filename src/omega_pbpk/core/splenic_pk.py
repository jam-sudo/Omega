"""
Phase 971 — Splenic drug distribution PBPK model.

Models the spleen as a filter/reservoir organ with quasi-steady-state
partitioning. Uses full 2-compartment ODE with sub-stepping for stability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["SplenicPKResult", "simulate_splenic_pk", "screen_spleen_trapping"]


@dataclass
class SplenicPKResult:
    drug_name: str
    dose_mg: float
    logp: float
    times_h: list
    c_plasma_mg_L: list
    c_spleen_mg_L: list
    cmax_plasma: float
    cmax_spleen: float
    auc_plasma: float
    auc_spleen: float
    spleen_to_plasma_ratio: float
    kp_spleen_used: float
    spleen_exposure_index: float
    notes: str


def _kp_spleen_from_logp(logp: float) -> float:
    """Estimate Kp_spleen from logP using empirical relationship."""
    return max(0.5, 10.0 ** (0.3 * logp))


def _trapezoidal_auc(times: list, concs: list) -> float:
    """Manual trapezoidal AUC."""
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        auc += 0.5 * (concs[i] + concs[i - 1]) * dt
    return auc


def simulate_splenic_pk(
    drug_name: str,
    dose_mg: float,
    logp: float = 1.0,
    route: str = "iv",
    ka_per_h: float = 1.0,
    cl_hepatic_L_per_h: float = 10.0,
    v_plasma_L: float = 4.0,
    v_spleen_L: float = 0.18,
    q_spleen_L_per_h: float = 18.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> SplenicPKResult:
    """Simulate splenic drug distribution using 2-compartment ODE with sub-stepping."""
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_hepatic_L_per_h <= 0:
        raise ValueError("cl_hepatic_L_per_h must be > 0")
    if v_plasma_L <= 0:
        raise ValueError("v_plasma_L must be > 0")
    if v_spleen_L <= 0:
        raise ValueError("v_spleen_L must be > 0")
    if q_spleen_L_per_h <= 0:
        raise ValueError("q_spleen_L_per_h must be > 0")
    if route not in {"iv", "oral"}:
        raise ValueError("route must be 'iv' or 'oral'")

    kp_spleen = _kp_spleen_from_logp(logp)

    # Rate constants
    # k_ps_eff: plasma -> spleen (units: /h), from Q/Vpl
    k_ps_eff = q_spleen_L_per_h / v_plasma_L
    # k_sp_eff: spleen back-transfer rate (/h), from Q/(Kp*Vsp)
    k_sp_eff = q_spleen_L_per_h / (kp_spleen * v_spleen_L)
    # Hepatic elimination rate (/h)
    ke_hep = cl_hepatic_L_per_h / v_plasma_L

    # Sub-stepping for ODE stability
    dt_internal = min(0.001, dt_h)

    n_steps = max(1, int(round(t_end_h / dt_h)))
    times = []
    c_plasma_list = []
    c_spleen_list = []

    # Initial conditions
    if route == "iv":
        # Bolus: all dose in plasma at t=0
        c_plasma = dose_mg / v_plasma_L  # mg/L
        c_spleen = kp_spleen * c_plasma  # quasi-steady initial distribution
        depot = 0.0
    else:  # oral
        c_plasma = 0.0
        c_spleen = 0.0
        depot = dose_mg  # depot in mg

    t = 0.0
    times.append(t)
    c_plasma_list.append(c_plasma)
    c_spleen_list.append(c_spleen)

    for step in range(n_steps):
        t_target = (step + 1) * dt_h
        # Sub-step integration
        n_sub = max(1, int(math.ceil(dt_h / dt_internal)))
        dt_sub = dt_h / n_sub

        for _ in range(n_sub):
            if route == "oral":
                # Depot absorption
                absorption_rate = ka_per_h * depot / v_plasma_L  # mg/h / L = mg/(L*h)
                d_depot = -ka_per_h * depot * dt_sub
                depot += d_depot
                if depot < 0.0:
                    depot = 0.0
            else:
                absorption_rate = 0.0

            # Spleen mass in mg: c_spleen * v_spleen_L
            # dC_plasma/dt = absorption - ke_hep*C_plasma - k_ps_eff*C_plasma + k_sp_eff*(C_spleen*Vsp/Vpl)
            # Note: k_sp_eff * C_spleen gives spleen-to-plasma flux per unit plasma volume
            dC_plasma = (
                absorption_rate
                - ke_hep * c_plasma
                - k_ps_eff * c_plasma
                + k_sp_eff * c_spleen * (v_spleen_L / v_plasma_L)
            ) * dt_sub

            # dC_spleen/dt = k_ps_eff * (Vpl/Vsp) * C_plasma - k_sp_eff * C_spleen
            dC_spleen = (
                k_ps_eff * (v_plasma_L / v_spleen_L) * c_plasma
                - k_sp_eff * c_spleen
            ) * dt_sub

            c_plasma = max(0.0, c_plasma + dC_plasma)
            c_spleen = max(0.0, c_spleen + dC_spleen)

        t = t_target
        times.append(t)
        c_plasma_list.append(c_plasma)
        c_spleen_list.append(c_spleen)

    auc_plasma = _trapezoidal_auc(times, c_plasma_list)
    auc_spleen = _trapezoidal_auc(times, c_spleen_list)
    cmax_plasma = max(c_plasma_list)
    cmax_spleen = max(c_spleen_list)
    spleen_to_plasma_ratio = auc_spleen / auc_plasma if auc_plasma > 0 else 0.0
    spleen_exposure_index = auc_spleen / dose_mg if dose_mg > 0 else 0.0

    notes = (
        f"Kp_spleen={kp_spleen:.2f} (logP={logp:.1f}); "
        f"k_ps_eff={k_ps_eff:.2f}/h, k_sp_eff={k_sp_eff:.2f}/h; "
        f"route={route}"
    )

    return SplenicPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        logp=logp,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        c_spleen_mg_L=c_spleen_list,
        cmax_plasma=cmax_plasma,
        cmax_spleen=cmax_spleen,
        auc_plasma=auc_plasma,
        auc_spleen=auc_spleen,
        spleen_to_plasma_ratio=spleen_to_plasma_ratio,
        kp_spleen_used=kp_spleen,
        spleen_exposure_index=spleen_exposure_index,
        notes=notes,
    )


def screen_spleen_trapping(
    drug_name: str,
    logp_values: list,
    dose_mg: float = 10.0,
    route: str = "iv",
    ka_per_h: float = 1.0,
    cl_hepatic_L_per_h: float = 10.0,
    v_plasma_L: float = 4.0,
    v_spleen_L: float = 0.18,
    q_spleen_L_per_h: float = 18.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> list:
    """Screen a series of logP values and rank by spleen-to-plasma AUC ratio."""
    results = []
    for logp in logp_values:
        result = simulate_splenic_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            logp=logp,
            route=route,
            ka_per_h=ka_per_h,
            cl_hepatic_L_per_h=cl_hepatic_L_per_h,
            v_plasma_L=v_plasma_L,
            v_spleen_L=v_spleen_L,
            q_spleen_L_per_h=q_spleen_L_per_h,
            t_end_h=t_end_h,
            dt_h=dt_h,
        )
        results.append(result)
    results.sort(key=lambda r: r.spleen_to_plasma_ratio, reverse=True)
    return results
