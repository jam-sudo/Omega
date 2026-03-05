"""Two-compartment PK model — central + peripheral compartments."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz


@dataclass(frozen=True)
class TwoCompartmentResult:
    times_h: list[float]
    cp_central: list[float]     # concentration in central compartment (mg/L)
    cp_peripheral: list[float]  # concentration in peripheral compartment (mg/L)
    auc_0_inf: float            # AUC 0→∞ (mg·h/L)
    auc_0_t: float              # AUC 0→last time point
    cmax: float
    tmax_h: float
    thalf_alpha_h: float        # distribution half-life
    thalf_beta_h: float         # elimination half-life
    vd_ss_L: float              # volume of distribution at steady state
    cl_L_per_h: float           # systemic clearance
    vd_central_L: float
    vd_peripheral_L: float


@dataclass(frozen=True)
class TwoCompartmentParams:
    """Two-compartment micro-rate constants."""
    k10: float  # elimination rate (1/h)
    k12: float  # central → peripheral (1/h)
    k21: float  # peripheral → central (1/h)
    vd_central_L: float
    ka_per_h: float = 0.0   # absorption rate (oral), 0 for IV


def params_from_macro(
    cl_L_per_h: float,
    vd_central_L: float,
    vd_peripheral_L: float,
    q_L_per_h: float,  # intercompartmental clearance
    ka_per_h: float = 0.0,
) -> TwoCompartmentParams:
    """Convert macro PK parameters to micro-rate constants."""
    k10 = cl_L_per_h / vd_central_L
    k12 = q_L_per_h / vd_central_L
    k21 = q_L_per_h / vd_peripheral_L
    return TwoCompartmentParams(k10=k10, k12=k12, k21=k21,
                                 vd_central=vd_central_L, ka_per_h=ka_per_h)


def _alpha_beta(k10: float, k12: float, k21: float) -> tuple[float, float]:
    """Compute hybrid rate constants α and β."""
    s = k10 + k12 + k21
    d = math.sqrt(max(s ** 2 - 4 * k10 * k21, 0.0))
    alpha = (s + d) / 2.0
    beta = (s - d) / 2.0
    return alpha, beta


def simulate_two_compartment(
    dose_mg: float,
    cl_L_per_h: float,
    vd_central_L: float,
    vd_peripheral_L: float,
    q_L_per_h: float,
    ka_per_h: float = 0.0,
    route: str = "iv",
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
    f: float = 1.0,  # bioavailability
) -> TwoCompartmentResult:
    """Simulate 2-compartment PK using analytical solution (IV) or numerical (oral).

    Parameters
    ----------
    cl_L_per_h : float
        Systemic clearance (L/h).
    vd_central_L : float
        Central compartment volume (L).
    vd_peripheral_L : float
        Peripheral compartment volume (L).
    q_L_per_h : float
        Intercompartmental clearance (L/h).
    ka_per_h : float
        Absorption rate constant (1/h), oral only.
    route : str
        "iv" or "oral".
    f : float
        Oral bioavailability (0–1).
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_central_L <= 0 or vd_peripheral_L <= 0:
        raise ValueError("Volumes must be > 0")
    if q_L_per_h < 0:
        raise ValueError("q_L_per_h must be >= 0")

    k10 = cl_L_per_h / vd_central_L
    k12 = q_L_per_h / vd_central_L
    k21 = q_L_per_h / vd_peripheral_L

    alpha, beta = _alpha_beta(k10, k12, k21)

    n = max(int(t_end_h / dt_h), 1)
    t = np.linspace(0.0, t_end_h, n + 1)

    is_oral = route.lower() == "oral"

    if not is_oral:
        # Analytical IV bolus solution
        A = dose_mg / vd_central_L * (alpha - k21) / (alpha - beta)
        B = dose_mg / vd_central_L * (k21 - beta) / (alpha - beta)
        cp = A * np.exp(-alpha * t) + B * np.exp(-beta * t)

        # Peripheral concentration (numerical from mass balance)
        cp_peri = np.zeros(n + 1)
        a_peri = 0.0
        a_central = dose_mg
        for i in range(n):
            flux_12 = k12 * (a_central / vd_central_L) * vd_central_L * dt_h
            flux_21 = k21 * a_peri * dt_h
            a_central -= (k10 * a_central + flux_12 - flux_21) * dt_h
            a_peri += (flux_12 - flux_21)
            a_central = max(a_central, 0.0)
            a_peri = max(a_peri, 0.0)
            cp_peri[i + 1] = a_peri / vd_peripheral_L
    else:
        # Numerical oral solution
        cp = np.zeros(n + 1)
        cp_peri = np.zeros(n + 1)
        a_gut = dose_mg * f
        a_central = 0.0
        a_peri = 0.0
        for i in range(n):
            abs_rate = ka_per_h * a_gut * dt_h
            a_gut -= abs_rate
            flux_12 = k12 * a_central * dt_h
            flux_21 = k21 * a_peri * dt_h
            a_central += abs_rate - k10 * a_central * dt_h - flux_12 + flux_21
            a_peri += flux_12 - flux_21
            a_gut = max(a_gut, 0.0)
            a_central = max(a_central, 0.0)
            a_peri = max(a_peri, 0.0)
            cp[i + 1] = a_central / vd_central_L
            cp_peri[i + 1] = a_peri / vd_peripheral_L

        if not is_oral:
            cp[0] = dose_mg / vd_central_L

    # PK metrics
    auc_0_t = float(np_trapz(cp, t))
    cmax = float(np.max(cp))
    tmax_h = float(t[int(np.argmax(cp))])

    # AUC0-inf (biexponential extrapolation)
    last_cp = float(cp[-1])
    auc_0_inf = auc_0_t + last_cp / beta if beta > 0 else auc_0_t

    # Half-lives
    thalf_alpha = math.log(2) / alpha if alpha > 0 else math.nan
    thalf_beta = math.log(2) / beta if beta > 0 else math.nan

    # Vdss = Vc + Vp
    vdss = vd_central_L + vd_peripheral_L

    return TwoCompartmentResult(
        times_h=t.tolist(),
        cp_central=cp.tolist(),
        cp_peripheral=cp_peri.tolist(),
        auc_0_inf=auc_0_inf,
        auc_0_t=auc_0_t,
        cmax=cmax,
        tmax_h=tmax_h,
        thalf_alpha_h=thalf_alpha,
        thalf_beta_h=thalf_beta,
        vd_ss_L=vdss,
        cl_L_per_h=cl_L_per_h,
        vd_central_L=vd_central_L,
        vd_peripheral_L=vd_peripheral_L,
    )


def fit_biexponential(
    times: list[float],
    conc: list[float],
) -> dict[str, float]:
    """Fit C(t) = A*exp(-alpha*t) + B*exp(-beta*t) by log-linear stripping.

    Returns dict with keys: A, alpha, B, beta, thalf_alpha, thalf_beta.
    Uses method of residuals (feathering) — requires at least 4 points.
    """
    t = np.asarray(times, dtype=float)
    c = np.asarray(conc, dtype=float)
    if len(t) < 4:
        raise ValueError("Need at least 4 time points for biexponential fitting")

    # Fit terminal phase (last 40% of points) for beta
    n_terminal = max(int(0.4 * len(t)), 2)
    t_term = t[-n_terminal:]
    c_term = c[-n_terminal:]
    log_c_term = np.log(np.maximum(c_term, 1e-12))
    coeffs_beta = np.polyfit(t_term, log_c_term, 1)
    beta = -float(coeffs_beta[0])
    B = math.exp(float(coeffs_beta[1]))

    # Subtract terminal component to get residuals
    c_residual = c - B * np.exp(-beta * t)
    c_residual = np.maximum(c_residual, 1e-12)

    # Fit early phase for alpha
    n_early = max(int(0.5 * len(t)), 2)
    log_resid = np.log(c_residual[:n_early])
    coeffs_alpha = np.polyfit(t[:n_early], log_resid, 1)
    alpha = -float(coeffs_alpha[0])
    A = math.exp(float(coeffs_alpha[1]))

    return {
        "A": A, "alpha": max(alpha, beta + 1e-6),
        "B": B, "beta": max(beta, 1e-6),
        "thalf_alpha_h": math.log(2) / max(alpha, 1e-6),
        "thalf_beta_h": math.log(2) / max(beta, 1e-6),
    }


__all__ = [
    "TwoCompartmentResult",
    "TwoCompartmentParams",
    "params_from_macro",
    "simulate_two_compartment",
    "fit_biexponential",
]
