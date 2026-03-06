"""Intravenous infusion PK — constant rate and variable rate infusions."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz


@dataclass(frozen=True)
class InfusionResult:
    drug_name: str
    infusion_rate_mg_h: float
    infusion_duration_h: float
    times_h: list[float]
    conc_mg_L: list[float]
    css_mg_L: float  # steady-state concentration (theoretical)
    css_90pct_h: float  # time to reach 90% of Css
    auc_0_t: float
    auc_0_inf: float
    cmax: float
    tmax_h: float
    t_half_h: float
    post_infusion_auc: float  # AUC after infusion stops


def simulate_constant_infusion(
    drug_name: str,
    dose_mg: float,
    infusion_duration_h: float,
    cl_L_per_h: float,
    vd_L: float,
    t_end_h: float | None = None,
    dt_h: float = 0.05,
) -> InfusionResult:
    """Simulate constant-rate IV infusion followed by post-infusion decay.

    During infusion: C(t) = (R/CL) * (1 - exp(-ke*t))
    Post-infusion: C(t) = C(T) * exp(-ke*(t-T))

    Parameters
    ----------
    dose_mg : float
        Total dose (mg).
    infusion_duration_h : float
        Infusion duration (h). Rate = dose / duration.
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if infusion_duration_h <= 0:
        raise ValueError("infusion_duration_h must be > 0")

    rate_mg_h = dose_mg / infusion_duration_h
    ke = cl_L_per_h / vd_L
    t_half = math.log(2) / ke

    if t_end_h is None:
        t_end_h = infusion_duration_h + 5.0 * t_half

    css = rate_mg_h / cl_L_per_h

    n = max(int(t_end_h / dt_h), 1)
    t = np.linspace(0.0, t_end_h, n + 1)
    conc = np.zeros(n + 1)

    T = infusion_duration_h

    for i, ti in enumerate(t):
        if ti <= T:
            conc[i] = css * (1.0 - math.exp(-ke * ti))
        else:
            c_at_T = css * (1.0 - math.exp(-ke * T))
            conc[i] = c_at_T * math.exp(-ke * (ti - T))

    auc_0_t = float(np_trapz(conc, t))
    cmax = float(np.max(conc))
    tmax_h = float(t[int(np.argmax(conc))])

    # AUC 0-inf
    last_c = float(conc[-1])
    auc_0_inf = auc_0_t + last_c / ke if ke > 0 else auc_0_t

    # Post-infusion AUC
    post_mask = t > T
    if np.any(post_mask):
        t_post = t[post_mask]
        c_post = conc[post_mask]
        post_auc = float(np_trapz(c_post, t_post))
    else:
        post_auc = 0.0

    # Time to 90% Css
    css_90_h = -math.log(1.0 - 0.9) / ke  # ≈ 3.32 half-lives

    return InfusionResult(
        drug_name=drug_name,
        infusion_rate_mg_h=rate_mg_h,
        infusion_duration_h=infusion_duration_h,
        times_h=t.tolist(),
        conc_mg_L=conc.tolist(),
        css_mg_L=css,
        css_90pct_h=css_90_h,
        auc_0_t=auc_0_t,
        auc_0_inf=auc_0_inf,
        cmax=cmax,
        tmax_h=tmax_h,
        t_half_h=t_half,
        post_infusion_auc=post_auc,
    )


def simulate_loading_then_maintenance(
    drug_name: str,
    loading_dose_mg: float,
    maintenance_dose_mg: float,
    maintenance_duration_h: float,
    cl_L_per_h: float,
    vd_L: float,
    loading_duration_h: float = 0.5,
    dt_h: float = 0.05,
) -> InfusionResult:
    """Simulate loading dose (short infusion) followed by maintenance infusion."""
    ke = cl_L_per_h / vd_L
    t_half = math.log(2) / ke

    t_end_h = loading_duration_h + maintenance_duration_h + 3.0 * t_half
    n = max(int(t_end_h / dt_h), 1)
    t = np.linspace(0.0, t_end_h, n + 1)
    conc = np.zeros(n + 1)

    rate_load = loading_dose_mg / max(loading_duration_h, 1e-9)
    rate_maint = maintenance_dose_mg / maintenance_duration_h
    css_maint = rate_maint / cl_L_per_h
    css_load = rate_load / cl_L_per_h

    T_load = loading_duration_h
    T_maint_end = T_load + maintenance_duration_h

    for i, ti in enumerate(t):
        if ti <= T_load:
            # Loading infusion
            conc[i] = css_load * (1.0 - math.exp(-ke * ti))
        elif ti <= T_maint_end:
            # Maintenance (superpose on post-loading decay)
            c_at_Tload = css_load * (1.0 - math.exp(-ke * T_load))
            t_since = ti - T_load
            conc[i] = c_at_Tload * math.exp(-ke * t_since) + css_maint * (
                1.0 - math.exp(-ke * t_since)
            )
        else:
            # Post-maintenance
            c_at_Tmaint = 0.0
            if T_maint_end > 0:
                c_at_Tload = css_load * (1.0 - math.exp(-ke * T_load))
                t_maint = T_maint_end - T_load
                c_at_Tmaint = c_at_Tload * math.exp(-ke * t_maint) + css_maint * (
                    1.0 - math.exp(-ke * t_maint)
                )
            conc[i] = c_at_Tmaint * math.exp(-ke * (ti - T_maint_end))

    auc = float(np_trapz(conc, t))
    cmax = float(np.max(conc))
    tmax_h = float(t[int(np.argmax(conc))])
    auc_inf = auc + float(conc[-1]) / ke

    post_mask = t > T_maint_end
    post_auc = float(np_trapz(conc[post_mask], t[post_mask])) if np.any(post_mask) else 0.0

    return InfusionResult(
        drug_name=drug_name,
        infusion_rate_mg_h=rate_maint,
        infusion_duration_h=maintenance_duration_h,
        times_h=t.tolist(),
        conc_mg_L=conc.tolist(),
        css_mg_L=css_maint,
        css_90pct_h=-math.log(0.1) / ke,
        auc_0_t=auc,
        auc_0_inf=auc_inf,
        cmax=cmax,
        tmax_h=tmax_h,
        t_half_h=t_half,
        post_infusion_auc=post_auc,
    )


def compare_bolus_vs_infusion(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    infusion_durations_h: list[float] | None = None,
) -> dict:
    """Compare IV bolus vs different infusion durations."""
    if infusion_durations_h is None:
        infusion_durations_h = [0.5, 1.0, 2.0]

    results = {}
    # Bolus: treat as 5-min infusion
    results["bolus_5min"] = simulate_constant_infusion(
        drug_name, dose_mg, 5.0 / 60.0, cl_L_per_h, vd_L
    )
    for dur in infusion_durations_h:
        results[f"{dur}h_infusion"] = simulate_constant_infusion(
            drug_name, dose_mg, dur, cl_L_per_h, vd_L
        )

    return results


__all__ = [
    "InfusionResult",
    "simulate_constant_infusion",
    "simulate_loading_then_maintenance",
    "compare_bolus_vs_infusion",
]
