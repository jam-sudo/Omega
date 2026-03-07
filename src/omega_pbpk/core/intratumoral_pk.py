"""
Phase 961 — Intratumoral drug distribution model.

3-compartment Forward Euler ODE:
  1. Plasma (systemic)
  2. Tumor vascular/interstitial (periphery)
  3. Tumor core/avascular
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["IntratumoralPKResult", "simulate_intratumoral_pk", "compare_routes"]


@dataclass
class IntratumoralPKResult:
    drug_name: str
    dose_mg: float
    route: str
    logp: float
    times_h: list
    c_plasma_mg_L: list
    c_tv_mg_L: list
    c_tc_mg_L: list
    cmax_plasma: float
    cmax_tv: float
    cmax_tc: float
    auc_plasma: float
    auc_tv: float
    auc_tc: float
    tumor_to_plasma_ratio: float
    core_periphery_ratio: float
    notes: str


def _trapezoidal_auc(times: list, concs: list) -> float:
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        auc += 0.5 * (concs[i - 1] + concs[i]) * dt
    return auc


def simulate_intratumoral_pk(
    drug_name: str,
    dose_mg: float,
    logp: float = 2.0,
    route: str = "systemic_iv",
    cl_sys_L_per_h: float = 10.0,
    v_plasma_L: float = 4.0,
    q_tumor_L_per_h: float = 0.5,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
) -> IntratumoralPKResult:
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be positive")
    if v_plasma_L <= 0:
        raise ValueError("v_plasma_L must be positive")
    if q_tumor_L_per_h <= 0:
        raise ValueError("q_tumor_L_per_h must be positive")
    if route not in {"systemic_iv", "intratumoral"}:
        raise ValueError(f"route must be 'systemic_iv' or 'intratumoral', got '{route}'")

    # Derived parameters
    kp_tumor = max(0.5, logp * 0.5)
    v_tv = 0.05  # L, tumor periphery
    _v_tc = 0.02  # L, tumor core

    k_plasma_to_tv = q_tumor_L_per_h / v_plasma_L
    k_tv_to_plasma_base = q_tumor_L_per_h / (kp_tumor * v_tv)

    k_tv_to_tc = 0.05  # /h
    k_tc_to_tv = 0.02  # /h
    ke_sys = cl_sys_L_per_h / v_plasma_L

    # For intratumoral route: elevated IFP greatly reduces outward convection from
    # tumor to plasma, and local matrix binding/metabolism retains most drug in the
    # tumor depot. Use reduced back-transfer and high local elimination.
    if route == "intratumoral":
        k_tv_to_plasma = k_tv_to_plasma_base * 0.01
        k_elim_tv = 2.0  # /h — dominant local retention/degradation
    else:
        k_tv_to_plasma = k_tv_to_plasma_base
        k_elim_tv = 0.01  # /h

    # Initial concentrations (mg/L)
    if route == "systemic_iv":
        cp = dose_mg / v_plasma_L
        ctv = 0.0
        ctc = 0.0
    else:  # intratumoral
        cp = 0.0
        ctv = dose_mg / v_tv
        ctc = 0.0

    # Forward Euler simulation
    n_steps = int(t_end_h / dt_h) + 1
    times = []
    c_plasma = []
    c_tv = []
    c_tc = []

    t = 0.0
    for _ in range(n_steps):
        times.append(t)
        c_plasma.append(cp)
        c_tv.append(ctv)
        c_tc.append(ctc)

        # Volume scaling: mass flow rate from tumor to plasma must be divided by V_plasma
        # Q_tumor * C_tv/Kp is mass/h, so dC_plasma/dt += Q_tumor * C_tv/(Kp * V_plasma)
        #   = k_tv_to_plasma * (V_tv / V_plasma) * C_tv
        dcp_dt = -ke_sys * cp - k_plasma_to_tv * cp + k_tv_to_plasma * (v_tv / v_plasma_L) * ctv
        # Volume scaling: plasma input to tumor divided by V_tv
        # Q_tumor * C_plasma is mass/h entering tumor, so dC_tv/dt += Q_tumor * C_plasma / V_tv
        #   = k_plasma_to_tv * (V_plasma / V_tv) * C_plasma
        dctv_dt = (
            k_plasma_to_tv * (v_plasma_L / v_tv) * cp
            - k_tv_to_plasma * ctv
            - k_tv_to_tc * ctv
            + k_tc_to_tv * ctc
            - k_elim_tv * ctv
        )
        dctc_dt = k_tv_to_tc * ctv - k_tc_to_tv * ctc

        cp = max(0.0, cp + dcp_dt * dt_h)
        ctv = max(0.0, ctv + dctv_dt * dt_h)
        ctc = max(0.0, ctc + dctc_dt * dt_h)

        t += dt_h

    auc_p = _trapezoidal_auc(times, c_plasma)
    auc_tv_val = _trapezoidal_auc(times, c_tv)
    auc_tc_val = _trapezoidal_auc(times, c_tc)

    cmax_p = max(c_plasma)
    cmax_tv = max(c_tv)
    cmax_tc = max(c_tc)

    tumor_to_plasma = auc_tv_val / auc_p if auc_p > 0 else 0.0
    core_peri = auc_tc_val / auc_tv_val if auc_tv_val > 0 else 0.0

    notes = (
        f"Route={route}; logP={logp:.2f}; Kp_tumor={kp_tumor:.2f}; "
        f"tumor-to-plasma AUC ratio={tumor_to_plasma:.3f}; "
        f"core/periphery ratio={core_peri:.3f}"
    )

    return IntratumoralPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route=route,
        logp=logp,
        times_h=times,
        c_plasma_mg_L=c_plasma,
        c_tv_mg_L=c_tv,
        c_tc_mg_L=c_tc,
        cmax_plasma=cmax_p,
        cmax_tv=cmax_tv,
        cmax_tc=cmax_tc,
        auc_plasma=auc_p,
        auc_tv=auc_tv_val,
        auc_tc=auc_tc_val,
        tumor_to_plasma_ratio=tumor_to_plasma,
        core_periphery_ratio=core_peri,
        notes=notes,
    )


def compare_routes(
    drug_name: str,
    dose_mg: float,
    logp: float = 2.0,
    cl_sys_L_per_h: float = 10.0,
    v_plasma_L: float = 4.0,
    q_tumor_L_per_h: float = 0.5,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
) -> list:
    """Simulate both routes and return sorted by auc_tc descending."""
    results = []
    for route in ("systemic_iv", "intratumoral"):
        r = simulate_intratumoral_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            logp=logp,
            route=route,
            cl_sys_L_per_h=cl_sys_L_per_h,
            v_plasma_L=v_plasma_L,
            q_tumor_L_per_h=q_tumor_L_per_h,
            t_end_h=t_end_h,
            dt_h=dt_h,
        )
        results.append(r)
    results.sort(key=lambda x: x.auc_tc, reverse=True)
    return results
