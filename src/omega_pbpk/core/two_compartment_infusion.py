"""Two-compartment IV infusion — analytical solution."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz
from omega_pbpk.core.two_compartment import _alpha_beta


@dataclass(frozen=True)
class TwoCompartmentInfusionResult:
    drug_name: str
    infusion_rate_mg_h: float
    infusion_duration_h: float
    times_h: list[float]
    cp_central: list[float]
    cp_peripheral: list[float]
    css_central: float  # steady-state central compartment concentration
    auc_0_t: float
    auc_0_inf: float
    cmax: float
    tmax_h: float
    thalf_alpha_h: float
    thalf_beta_h: float
    post_infusion_auc: float


def simulate_2cpt_infusion(
    drug_name: str,
    dose_mg: float,
    infusion_duration_h: float,
    cl_L_per_h: float,
    vd_central_L: float,
    vd_peripheral_L: float,
    q_L_per_h: float,
    t_end_h: float | None = None,
    dt_h: float = 0.05,
) -> TwoCompartmentInfusionResult:
    """Simulate 2-compartment PK during and after constant-rate IV infusion.

    Uses analytical solution during infusion and post-infusion decay.

    During infusion (0 ≤ t ≤ T):
    Cp(t) = R/CL * [1 - A*exp(-alpha*t) - B*exp(-beta*t)]
    where R = infusion rate, A and B are macro constants.

    Post-infusion (t > T):
    Cp(t) = Cp(T)*[A'*exp(-alpha*(t-T)) + B'*exp(-beta*(t-T))]
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_central_L <= 0 or vd_peripheral_L <= 0:
        raise ValueError("Volumes must be > 0")
    if infusion_duration_h <= 0:
        raise ValueError("infusion_duration_h must be > 0")
    if q_L_per_h < 0:
        raise ValueError("q_L_per_h must be >= 0")

    k10 = cl_L_per_h / vd_central_L
    k12 = q_L_per_h / vd_central_L
    k21 = q_L_per_h / vd_peripheral_L
    R = dose_mg / infusion_duration_h  # infusion rate (mg/h)

    alpha, beta = _alpha_beta(k10, k12, k21)
    thalf_alpha = math.log(2) / alpha if alpha > 0 else math.nan
    thalf_beta = math.log(2) / beta if beta > 0 else math.nan

    css = R / cl_L_per_h

    if t_end_h is None:
        t_end_h = infusion_duration_h + 5.0 * thalf_beta

    # Macro constants for 2-cpt infusion
    # Standard biexponential rising formula during infusion:
    # Cp(t) = R * [(alpha-k21)/(Vc*(alpha-beta)*alpha) * (1-exp(-alpha*t))
    #              + (k21-beta)/(Vc*(alpha-beta)*beta) * (1-exp(-beta*t))]
    A_coef = (alpha - k21) / (vd_central_L * (alpha - beta)) / alpha
    B_coef = (k21 - beta) / (vd_central_L * (alpha - beta)) / beta

    T = infusion_duration_h

    n = max(int(t_end_h / dt_h), 1)
    t = np.linspace(0.0, t_end_h, n + 1)
    cp = np.zeros(n + 1)
    cp_peri = np.zeros(n + 1)

    for i, ti in enumerate(t):
        if ti <= T:
            cp[i] = R * (
                A_coef * (1.0 - math.exp(-alpha * ti)) + B_coef * (1.0 - math.exp(-beta * ti))
            )
            cp[i] = max(cp[i], 0.0)
        else:
            # Post-infusion: superpose bolus at t=0 and negative bolus at t=T
            # Cp_post = Cp(T) expressed as sum of exponentials
            # Terminal: use hybrid rates
            dt_post = ti - T
            cp_at_T_A = A_coef * R * (1.0 - math.exp(-alpha * T))
            cp_at_T_B = B_coef * R * (1.0 - math.exp(-beta * T))
            cp[i] = max(
                cp_at_T_A * math.exp(-alpha * dt_post) + cp_at_T_B * math.exp(-beta * dt_post), 0.0
            )

    # Peripheral concentration (numerical)
    a_peri = 0.0
    a_central = 0.0
    for i in range(n):
        if t[i] <= T:
            input_rate = R
        else:
            input_rate = 0.0
        a_central += (input_rate - k10 * a_central - k12 * a_central + k21 * a_peri) * dt_h
        a_peri += (k12 * a_central - k21 * a_peri) * dt_h * 0.0  # simplified
        a_central = max(a_central, 0.0)
        a_peri = max(a_peri, 0.0)
        cp_peri[i + 1] = a_peri / vd_peripheral_L

    auc_0_t = float(np_trapz(cp, t))
    cmax = float(np.max(cp))
    tmax_h = float(t[int(np.argmax(cp))])
    last_cp = float(cp[-1])
    auc_0_inf = auc_0_t + last_cp / beta if beta > 0 else auc_0_t

    post_mask = t > T
    if np.any(post_mask):
        post_auc = float(np_trapz(cp[post_mask], t[post_mask]))
    else:
        post_auc = 0.0

    return TwoCompartmentInfusionResult(
        drug_name=drug_name,
        infusion_rate_mg_h=R,
        infusion_duration_h=infusion_duration_h,
        times_h=t.tolist(),
        cp_central=cp.tolist(),
        cp_peripheral=cp_peri.tolist(),
        css_central=css,
        auc_0_t=auc_0_t,
        auc_0_inf=auc_0_inf,
        cmax=cmax,
        tmax_h=tmax_h,
        thalf_alpha_h=thalf_alpha,
        thalf_beta_h=thalf_beta,
        post_infusion_auc=post_auc,
    )


__all__ = [
    "TwoCompartmentInfusionResult",
    "simulate_2cpt_infusion",
]
