"""IV bolus pharmacokinetics — Phase 648.

Analytical and numerical (Forward Euler) solutions for 1- and 2-compartment
IV bolus models. Instantaneous injection (no infusion rate).

Pure Python only: stdlib math — no numpy, no scipy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "IVBolusPKResult",
    "simulate_iv_bolus_1cpt",
    "simulate_iv_bolus_2cpt",
    "analytical_1cpt_iv_bolus",
    "analytical_2cpt_iv_bolus",
]


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class IVBolusPKResult:
    """Result of an IV bolus PK simulation.

    Attributes
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Administered dose in mg.
    n_compartments : int
        1 or 2.
    times_h : list[float]
        Time points in hours.
    c_plasma_mg_L : list[float]
        Central compartment concentrations (mg/L).
    c_peripheral_mg_L : list[float]
        Peripheral compartment concentrations (mg/L). Empty for 1-cpt.
    cmax_mg_L : float
        Peak plasma concentration at t=0 (= dose/Vd1).
    t_half_h : float
        Terminal elimination half-life (h).
    auc_mg_h_per_L : float
        AUC(0→t_end) by trapezoidal rule (mg·h/L).
    vss_L : float
        Volume of distribution at steady state (L).
    cl_L_per_h : float
        Total clearance (L/h).
    residual_error_pct : float
        Mean absolute % difference between numerical and analytical profiles.
    notes : str
        Descriptive summary.
    """

    drug_name: str
    dose_mg: float
    n_compartments: int
    times_h: list
    c_plasma_mg_L: list
    c_peripheral_mg_L: list
    cmax_mg_L: float
    t_half_h: float
    auc_mg_h_per_L: float
    vss_L: float
    cl_L_per_h: float
    residual_error_pct: float
    notes: str


# ---------------------------------------------------------------------------
# Analytical solutions
# ---------------------------------------------------------------------------


def analytical_1cpt_iv_bolus(
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    times_h: list[float],
) -> list[float]:
    """Analytical 1-compartment IV bolus concentration-time profile.

    C(t) = C0 * exp(-ke * t)
    C0 = dose / Vd
    ke = CL / Vd

    Parameters
    ----------
    dose_mg : float
        Administered dose (mg).
    cl_L_per_h : float
        Total clearance (L/h).
    vd_L : float
        Volume of distribution (L).
    times_h : list[float]
        Time points (h).

    Returns
    -------
    list[float]
        Plasma concentrations (mg/L) at each time point.
    """
    c0 = dose_mg / vd_L
    ke = cl_L_per_h / vd_L
    return [c0 * math.exp(-ke * t) for t in times_h]


def analytical_2cpt_iv_bolus(
    dose_mg: float,
    cl_L_per_h: float,
    vd1_L: float,
    vd2_L: float,
    q12_L_per_h: float,
    times_h: list[float],
) -> list[float]:
    """Analytical 2-compartment IV bolus biexponential concentration-time profile.

    C(t) = A * exp(-alpha * t) + B * exp(-beta * t)

    Micro-rate constants:
        k10 = CL / Vd1
        k12 = Q12 / Vd1
        k21 = Q12 / Vd2

    Hybrid rate constants:
        alpha = ((k10+k12+k21) + sqrt((k10+k12+k21)^2 - 4*k10*k21)) / 2
        beta  = k10 * k21 / alpha

    Coefficients:
        A = (dose/Vd1) * (alpha - k21) / (alpha - beta)
        B = (dose/Vd1) * (k21 - beta) / (alpha - beta)

    Parameters
    ----------
    dose_mg : float
        Administered dose (mg).
    cl_L_per_h : float
        Total clearance (L/h).
    vd1_L : float
        Central volume (L).
    vd2_L : float
        Peripheral volume (L).
    q12_L_per_h : float
        Inter-compartmental clearance (L/h).
    times_h : list[float]
        Time points (h).

    Returns
    -------
    list[float]
        Central compartment concentrations (mg/L) at each time point.
    """
    k10 = cl_L_per_h / vd1_L
    k12 = q12_L_per_h / vd1_L
    k21 = q12_L_per_h / vd2_L

    s = k10 + k12 + k21
    discriminant = s * s - 4.0 * k10 * k21
    if discriminant < 0.0:
        discriminant = 0.0
    sqrt_disc = math.sqrt(discriminant)

    alpha = (s + sqrt_disc) / 2.0
    beta = (s - sqrt_disc) / 2.0

    # Guard against degenerate case (alpha == beta)
    if abs(alpha - beta) < 1e-12:
        alpha += 1e-12

    c0 = dose_mg / vd1_L
    a_coef = c0 * (alpha - k21) / (alpha - beta)
    b_coef = c0 * (k21 - beta) / (alpha - beta)

    return [a_coef * math.exp(-alpha * t) + b_coef * math.exp(-beta * t) for t in times_h]


# ---------------------------------------------------------------------------
# Internal: trapezoidal AUC
# ---------------------------------------------------------------------------


def _trapz_auc(times: list[float], concs: list[float]) -> float:
    """Manual trapezoidal AUC calculation."""
    auc = 0.0
    for i in range(len(times) - 1):
        auc += (concs[i] + concs[i + 1]) / 2.0 * (times[i + 1] - times[i])
    return auc


def _build_time_vector(t_end_h: float, dt_h: float) -> list[float]:
    """Build uniform time vector from 0 to t_end_h (inclusive)."""
    times = []
    t = 0.0
    while t <= t_end_h + 1e-12:
        times.append(t)
        t += dt_h
    if times[-1] < t_end_h - 1e-12:
        times.append(t_end_h)
    return times


def _mean_abs_pct_error(analytical: list[float], numerical: list[float]) -> float:
    """Mean absolute percentage error between two profiles."""
    errors = []
    for a, n in zip(analytical, numerical):
        if abs(a) > 1e-15:
            errors.append(abs(n - a) / abs(a) * 100.0)
    if not errors:
        return 0.0
    return sum(errors) / len(errors)


# ---------------------------------------------------------------------------
# Simulation functions
# ---------------------------------------------------------------------------


def simulate_iv_bolus_1cpt(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> IVBolusPKResult:
    """Simulate 1-compartment IV bolus PK.

    Parameters
    ----------
    drug_name : str
        Drug identifier.
    dose_mg : float
        Administered dose (mg). Must be > 0.
    cl_L_per_h : float
        Total clearance (L/h). Must be > 0.
    vd_L : float
        Volume of distribution (L). Must be > 0.
    t_end_h : float
        Simulation end time (h). Default 24 h.
    dt_h : float
        Integration step (h). Default 0.1 h.

    Returns
    -------
    IVBolusPKResult
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0; got {dose_mg}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0; got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0; got {vd_L}")

    ke = cl_L_per_h / vd_L
    c0 = dose_mg / vd_L

    times = _build_time_vector(t_end_h, dt_h)
    n = len(times)

    # --- Numerical (Forward Euler) ---
    c_num = [0.0] * n
    c_num[0] = c0
    for i in range(n - 1):
        dt_i = times[i + 1] - times[i]
        dcdt = -ke * c_num[i]
        c_num[i + 1] = max(0.0, c_num[i] + dcdt * dt_i)

    # --- Analytical ---
    c_ana = analytical_1cpt_iv_bolus(dose_mg, cl_L_per_h, vd_L, times)

    # --- Derived PK ---
    cmax = c0
    t_half = math.log(2.0) / ke
    auc = _trapz_auc(times, c_num)
    residual_error_pct = _mean_abs_pct_error(c_ana, c_num)

    notes = (
        f"1-cpt IV bolus: {drug_name}, dose={dose_mg} mg, "
        f"CL={cl_L_per_h} L/h, Vd={vd_L} L, ke={ke:.4f} /h, "
        f"t½={t_half:.2f} h, Cmax={cmax:.4f} mg/L, AUC={auc:.4f} mg·h/L"
    )

    return IVBolusPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        n_compartments=1,
        times_h=times,
        c_plasma_mg_L=c_num,
        c_peripheral_mg_L=[],
        cmax_mg_L=cmax,
        t_half_h=t_half,
        auc_mg_h_per_L=auc,
        vss_L=vd_L,
        cl_L_per_h=cl_L_per_h,
        residual_error_pct=residual_error_pct,
        notes=notes,
    )


def simulate_iv_bolus_2cpt(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd1_L: float,
    vd2_L: float,
    q12_L_per_h: float,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> IVBolusPKResult:
    """Simulate 2-compartment IV bolus PK.

    Parameters
    ----------
    drug_name : str
        Drug identifier.
    dose_mg : float
        Administered dose (mg). Must be > 0.
    cl_L_per_h : float
        Total clearance (L/h). Must be > 0.
    vd1_L : float
        Central volume (L). Must be > 0.
    vd2_L : float
        Peripheral volume (L). Must be > 0.
    q12_L_per_h : float
        Inter-compartmental clearance (L/h). Must be > 0.
    t_end_h : float
        Simulation end time (h). Default 24 h.
    dt_h : float
        Integration step (h). Default 0.1 h.

    Returns
    -------
    IVBolusPKResult
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0; got {dose_mg}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0; got {cl_L_per_h}")
    if vd1_L <= 0:
        raise ValueError(f"vd1_L must be > 0; got {vd1_L}")
    if vd2_L <= 0:
        raise ValueError(f"vd2_L must be > 0; got {vd2_L}")
    if q12_L_per_h <= 0:
        raise ValueError(f"q12_L_per_h must be > 0; got {q12_L_per_h}")

    k10 = cl_L_per_h / vd1_L
    k12 = q12_L_per_h / vd1_L
    k21 = q12_L_per_h / vd2_L

    times = _build_time_vector(t_end_h, dt_h)
    n = len(times)

    # --- Numerical (Forward Euler, 2-cpt ODE) ---
    # dA1/dt = -(k10 + k12)*A1 + k21*A2
    # dA2/dt = k12*A1 - k21*A2
    # C1 = A1/Vd1, C2 = A2/Vd2
    a1 = dose_mg  # initial amount in central (mg)
    a2 = 0.0  # initial amount in peripheral

    c1_list = [0.0] * n
    c2_list = [0.0] * n
    c1_list[0] = a1 / vd1_L
    c2_list[0] = a2 / vd2_L

    for i in range(n - 1):
        dt_i = times[i + 1] - times[i]
        da1 = (-(k10 + k12) * a1 + k21 * a2) * dt_i
        da2 = (k12 * a1 - k21 * a2) * dt_i
        a1 = max(0.0, a1 + da1)
        a2 = max(0.0, a2 + da2)
        c1_list[i + 1] = a1 / vd1_L
        c2_list[i + 1] = a2 / vd2_L

    # --- Analytical ---
    c_ana = analytical_2cpt_iv_bolus(dose_mg, cl_L_per_h, vd1_L, vd2_L, q12_L_per_h, times)

    # --- Derived PK ---
    cmax = dose_mg / vd1_L  # at t=0

    # Terminal half-life from beta (slower hybrid rate constant)
    s = k10 + k12 + k21
    discriminant = max(0.0, s * s - 4.0 * k10 * k21)
    sqrt_disc = math.sqrt(discriminant)
    beta = (s - sqrt_disc) / 2.0
    t_half = math.log(2.0) / beta if beta > 0 else float("inf")

    auc = _trapz_auc(times, c1_list)
    vss = vd1_L + vd2_L

    # Residual error vs analytical
    residual_error_pct = _mean_abs_pct_error(c_ana, c1_list)

    notes = (
        f"2-cpt IV bolus: {drug_name}, dose={dose_mg} mg, "
        f"CL={cl_L_per_h} L/h, Vd1={vd1_L} L, Vd2={vd2_L} L, "
        f"Q12={q12_L_per_h} L/h, Vss={vss} L, "
        f"t½(terminal)={t_half:.2f} h, Cmax={cmax:.4f} mg/L"
    )

    return IVBolusPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        n_compartments=2,
        times_h=times,
        c_plasma_mg_L=c1_list,
        c_peripheral_mg_L=c2_list,
        cmax_mg_L=cmax,
        t_half_h=t_half,
        auc_mg_h_per_L=auc,
        vss_L=vss,
        cl_L_per_h=cl_L_per_h,
        residual_error_pct=residual_error_pct,
        notes=notes,
    )
