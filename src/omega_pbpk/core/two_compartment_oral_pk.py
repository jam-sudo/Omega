"""Two-compartment oral PK model with Forward Euler numerical solution."""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["TwoCptOralResult", "simulate_two_cpt_oral", "compare_one_vs_two_cpt"]


@dataclass
class TwoCptOralResult:
    """Result from two-compartment oral PK simulation."""

    drug_name: str
    dose_mg: float
    route: str
    times_h: list
    c1_mg_L: list
    c2_mg_L: list
    cmax_c1_mg_L: float
    tmax_c1_h: float
    cmax_c2_mg_L: float
    tmax_c2_h: float
    auc_c1: float
    auc_c2: float
    t_half_alpha_h: float
    t_half_beta_h: float
    vss_L: float
    notes: str


def _validate_params(
    cl_L_per_h: float, v1_L: float, v2_L: float, q_L_per_h: float, ka_per_h: float, f_oral: float
) -> None:
    if v1_L <= 0:
        raise ValueError(f"v1_L must be > 0, got {v1_L}")
    if v2_L <= 0:
        raise ValueError(f"v2_L must be > 0, got {v2_L}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if q_L_per_h <= 0:
        raise ValueError(f"q_L_per_h must be > 0, got {q_L_per_h}")
    if ka_per_h <= 0:
        raise ValueError(f"ka_per_h must be > 0, got {ka_per_h}")
    if not (0 < f_oral <= 1):
        raise ValueError(f"f_oral must be in (0, 1], got {f_oral}")


def _trapezoidal_auc(times: list, concentrations: list) -> float:
    """Compute AUC using trapezoidal rule."""
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        auc += 0.5 * (concentrations[i] + concentrations[i - 1]) * dt
    return auc


def _terminal_half_life(times: list, concentrations: list) -> float:
    """Estimate terminal half-life from last 30% of profile using log-linear regression."""
    n = len(concentrations)
    start_idx = int(n * 0.70)

    # Collect points with positive concentrations
    log_c = []
    t_pts = []
    for i in range(start_idx, n):
        if concentrations[i] > 0:
            log_c.append(math.log(concentrations[i]))
            t_pts.append(times[i])

    if len(t_pts) < 2:
        return float("nan")

    # Linear regression: log(C) = log(C0) - lambda_z * t
    n_pts = len(t_pts)
    sum_t = sum(t_pts)
    sum_lc = sum(log_c)
    sum_t2 = sum(t * t for t in t_pts)
    sum_tlc = sum(t_pts[i] * log_c[i] for i in range(n_pts))

    denom = n_pts * sum_t2 - sum_t * sum_t
    if abs(denom) < 1e-12:
        return float("nan")

    slope = (n_pts * sum_tlc - sum_t * sum_lc) / denom  # negative for elimination

    if slope >= 0:
        return float("nan")

    lambda_z = -slope
    return math.log(2) / lambda_z


def simulate_two_cpt_oral(
    drug_name: str,
    dose_mg: float,
    ka_per_h: float = 1.0,
    f_oral: float = 0.8,
    cl_L_per_h: float = 5.0,
    v1_L: float = 10.0,
    v2_L: float = 40.0,
    q_L_per_h: float = 5.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> TwoCptOralResult:
    """Simulate two-compartment oral PK using Forward Euler.

    Model:
        dA_gut/dt = -ka * A_gut
        dC1/dt = ka * f_oral * A_gut / V1 - (k10 + k12) * C1 + k21 * C2 * V2/V1
        dC2/dt = k12 * C1 * V1/V2 - k21 * C2

    where k10 = CL/V1, k12 = Q/V1, k21 = Q/V2
    """
    _validate_params(cl_L_per_h, v1_L, v2_L, q_L_per_h, ka_per_h, f_oral)

    k10 = cl_L_per_h / v1_L
    k12 = q_L_per_h / v1_L
    k21 = q_L_per_h / v2_L

    # Initial conditions
    a_gut = dose_mg  # amount in gut (mg)
    c1 = 0.0  # central concentration (mg/L)
    c2 = 0.0  # peripheral concentration (mg/L)

    times = [0.0]
    c1_list = [0.0]
    c2_list = [0.0]

    n_steps = int(t_end_h / dt_h)
    t = 0.0

    for _ in range(n_steps):
        da_gut = -ka_per_h * a_gut
        dc1 = ka_per_h * f_oral * a_gut / v1_L - (k10 + k12) * c1 + k21 * c2 * v2_L / v1_L
        dc2 = k12 * c1 * v1_L / v2_L - k21 * c2

        a_gut += da_gut * dt_h
        c1 += dc1 * dt_h
        c2 += dc2 * dt_h

        # Clamp to zero (numerical artifact)
        if a_gut < 0:
            a_gut = 0.0
        if c1 < 0:
            c1 = 0.0
        if c2 < 0:
            c2 = 0.0

        t += dt_h
        times.append(t)
        c1_list.append(c1)
        c2_list.append(c2)

    # Find Cmax/Tmax for central
    cmax_c1 = max(c1_list)
    tmax_c1 = times[c1_list.index(cmax_c1)]

    # Find Cmax/Tmax for peripheral
    cmax_c2 = max(c2_list)
    tmax_c2 = times[c2_list.index(cmax_c2)]

    # AUC via trapezoidal rule
    auc_c1 = _trapezoidal_auc(times, c1_list)
    auc_c2 = _trapezoidal_auc(times, c2_list)

    # Terminal half-life (beta)
    t_half_beta = _terminal_half_life(times, c1_list)
    if math.isnan(t_half_beta) or t_half_beta <= 0:
        t_half_beta = math.log(2) / k10

    # Alpha phase: approximate from rate constants
    # k_alpha + k_beta = k10 + k12 + k21, k_alpha * k_beta = k10 * k21
    sum_k = k10 + k12 + k21
    prod_k = k10 * k21
    discriminant = sum_k * sum_k - 4 * prod_k
    if discriminant > 0:
        lambda_alpha = (sum_k + math.sqrt(discriminant)) / 2
        t_half_alpha = math.log(2) / lambda_alpha
        if t_half_alpha >= t_half_beta:
            t_half_alpha = t_half_beta * 0.5
    else:
        t_half_alpha = 0.5

    # Vss = V1 + V2 * (k12/k21)
    vss = v1_L + v2_L * (k12 / k21)

    notes = (
        f"2-cpt oral PK | ka={ka_per_h}/h, F={f_oral}, CL={cl_L_per_h} L/h, "
        f"V1={v1_L} L, V2={v2_L} L, Q={q_L_per_h} L/h | "
        f"Vss={vss:.1f} L, t½β={t_half_beta:.2f} h"
    )

    return TwoCptOralResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route="oral",
        times_h=times,
        c1_mg_L=c1_list,
        c2_mg_L=c2_list,
        cmax_c1_mg_L=cmax_c1,
        tmax_c1_h=tmax_c1,
        cmax_c2_mg_L=cmax_c2,
        tmax_c2_h=tmax_c2,
        auc_c1=auc_c1,
        auc_c2=auc_c2,
        t_half_alpha_h=t_half_alpha,
        t_half_beta_h=t_half_beta,
        vss_L=vss,
        notes=notes,
    )


def _simulate_one_cpt_oral(
    dose_mg: float,
    ka_per_h: float,
    f_oral: float,
    cl_L_per_h: float,
    vd_L: float,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> dict:
    """Simulate one-compartment oral PK (Forward Euler) and return summary."""
    k_el = cl_L_per_h / vd_L

    a_gut = dose_mg
    c = 0.0

    times = [0.0]
    c_list = [0.0]

    n_steps = int(t_end_h / dt_h)
    t = 0.0

    for _ in range(n_steps):
        da_gut = -ka_per_h * a_gut
        dc = ka_per_h * f_oral * a_gut / vd_L - k_el * c

        a_gut += da_gut * dt_h
        c += dc * dt_h

        if a_gut < 0:
            a_gut = 0.0
        if c < 0:
            c = 0.0

        t += dt_h
        times.append(t)
        c_list.append(c)

    cmax = max(c_list)
    auc = _trapezoidal_auc(times, c_list)
    t_half = _terminal_half_life(times, c_list)
    if math.isnan(t_half) or t_half <= 0:
        t_half = math.log(2) / k_el

    return {"cmax": cmax, "auc": auc, "t_half": t_half}


def compare_one_vs_two_cpt(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float = 5.0,
    v1_L: float = 10.0,
    v2_L: float = 40.0,
    q_L_per_h: float = 5.0,
    ka_per_h: float = 1.0,
    f_oral: float = 0.8,
) -> dict:
    """Compare 1-cpt vs 2-cpt oral PK models.

    Uses the same CL and Vd_total = V1 + V2 for the 1-cpt model.

    Returns:
        dict with keys: two_cpt, one_cpt (each with cmax, auc, t_half),
        auc_difference_pct, notes
    """
    _validate_params(cl_L_per_h, v1_L, v2_L, q_L_per_h, ka_per_h, f_oral)

    vd_total = v1_L + v2_L

    result_2cpt = simulate_two_cpt_oral(
        drug_name=drug_name,
        dose_mg=dose_mg,
        ka_per_h=ka_per_h,
        f_oral=f_oral,
        cl_L_per_h=cl_L_per_h,
        v1_L=v1_L,
        v2_L=v2_L,
        q_L_per_h=q_L_per_h,
        t_end_h=24.0,
        dt_h=0.05,
    )

    result_1cpt = _simulate_one_cpt_oral(
        dose_mg=dose_mg,
        ka_per_h=ka_per_h,
        f_oral=f_oral,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_total,
        t_end_h=24.0,
        dt_h=0.05,
    )

    two_cpt_summary = {
        "cmax": result_2cpt.cmax_c1_mg_L,
        "auc": result_2cpt.auc_c1,
        "t_half": result_2cpt.t_half_beta_h,
    }

    auc_diff_pct = 0.0
    if result_1cpt["auc"] > 0:
        auc_diff_pct = abs(two_cpt_summary["auc"] - result_1cpt["auc"]) / result_1cpt["auc"] * 100

    notes = (
        f"Comparison for {drug_name}: 2-cpt Cmax={two_cpt_summary['cmax']:.3f} mg/L vs "
        f"1-cpt Cmax={result_1cpt['cmax']:.3f} mg/L | "
        f"AUC difference: {auc_diff_pct:.1f}%"
    )

    return {
        "two_cpt": two_cpt_summary,
        "one_cpt": result_1cpt,
        "auc_difference_pct": auc_diff_pct,
        "notes": notes,
    }
