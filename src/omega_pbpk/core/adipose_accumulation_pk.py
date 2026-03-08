"""Phase 463 — Drug Accumulation in Adipose Tissue.

2-compartment PK model for lipophilic drug accumulation in adipose tissue.
Compartments: plasma + adipose tissue.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class AdiposeAccumulationResult:
    """Result of adipose accumulation simulation."""

    drug_name: str
    dose_mg: float
    logP: float
    fat_fraction: float
    times_h: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    c_adipose_mg_L: list = field(default_factory=list)
    cmax_plasma_mg_L: float = 0.0
    cmax_adipose_mg_L: float = 0.0
    tmax_adipose_h: float = 0.0
    auc_plasma_mg_h_per_L: float = 0.0
    auc_adipose_mg_h_per_L: float = 0.0
    kp_adipose: float = 0.0
    accumulation_ratio: float = 0.0
    t_half_terminal_h: float = 0.0
    notes: str = ""


def _calc_kp_adipose(logP: float) -> float:
    """Calculate adipose tissue partition coefficient from logP."""
    if logP > 0:
        raw = logP**1.5 / 3.0
    else:
        raw = 0.1
    # Clamp to [0.1, 50.0]
    return max(0.1, min(50.0, raw))


def _trapezoidal_auc(times: list, concs: list) -> float:
    """Manual trapezoidal AUC calculation."""
    return sum(
        0.5 * (concs[i] + concs[i - 1]) * (times[i] - times[i - 1]) for i in range(1, len(times))
    )


def simulate_adipose_accumulation(
    drug_name: str,
    dose_mg: float,
    logP: float,
    cl_sys_L_per_h: float,
    vd_plasma_L: float = 10.0,
    fat_fraction: float = 0.25,
    body_weight_kg: float = 70.0,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
) -> AdiposeAccumulationResult:
    """Simulate drug accumulation in adipose tissue (2-compartment, IV bolus).

    Args:
        drug_name: Name of the drug.
        dose_mg: IV bolus dose in mg.
        logP: Octanol-water partition coefficient (log).
        cl_sys_L_per_h: Systemic clearance in L/h.
        vd_plasma_L: Volume of distribution for plasma compartment in L.
        fat_fraction: Body fat fraction (0-1).
        body_weight_kg: Body weight in kg.
        t_end_h: Simulation end time in hours.
        dt_h: Time step in hours.

    Returns:
        AdiposeAccumulationResult with concentration-time profiles and derived PK metrics.
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_plasma_L <= 0:
        raise ValueError("vd_plasma_L must be > 0")
    if not (0 < fat_fraction < 1):
        raise ValueError("fat_fraction must be between 0 and 1 (exclusive)")
    if body_weight_kg <= 0:
        raise ValueError("body_weight_kg must be > 0")

    # Derived parameters
    kp_adipose = _calc_kp_adipose(logP)
    fat_mass_kg = fat_fraction * body_weight_kg
    vd_adipose_L = fat_mass_kg * 0.9  # density correction

    # Transfer rates
    k_adipose_in = kp_adipose * 0.5  # plasma -> adipose (1/h)
    k_adipose_out = 0.5  # adipose -> plasma (1/h)

    # Elimination rate from plasma
    k_el = cl_sys_L_per_h / vd_plasma_L  # 1/h

    # Initial conditions: IV bolus
    c_plasma = dose_mg / vd_plasma_L  # mg/L
    c_adipose = 0.0  # mg/L

    times: list = []
    c_plasma_list: list = []
    c_adipose_list: list = []

    t = 0.0
    n_steps = int(round(t_end_h / dt_h)) + 1

    for step in range(n_steps):
        t = step * dt_h
        times.append(t)
        c_plasma_list.append(c_plasma)
        c_adipose_list.append(c_adipose)

        if step < n_steps - 1:
            # Flux terms
            flux_in = k_adipose_in * c_plasma  # plasma -> adipose (mg/L/h based on plasma conc)
            flux_out_mass = k_adipose_out * c_adipose * vd_adipose_L  # mg/h leaving adipose

            # dC_plasma/dt = -k_el * C_plasma - k_adipose_in * C_plasma + flux_out_mass/vd_plasma_L
            dc_plasma = -k_el * c_plasma - flux_in + flux_out_mass / vd_plasma_L
            # dC_adipose/dt = flux_in * vd_plasma_L/vd_adipose_L - k_adipose_out * c_adipose
            dc_adipose = flux_in * vd_plasma_L / vd_adipose_L - k_adipose_out * c_adipose

            c_plasma = max(0.0, c_plasma + dc_plasma * dt_h)
            c_adipose = max(0.0, c_adipose + dc_adipose * dt_h)

    # Derived PK metrics
    cmax_plasma = max(c_plasma_list)
    cmax_adipose = max(c_adipose_list)

    tmax_adipose = 0.0
    if cmax_adipose > 0:
        tmax_adipose = times[c_adipose_list.index(cmax_adipose)]

    auc_plasma = _trapezoidal_auc(times, c_plasma_list)
    auc_adipose = _trapezoidal_auc(times, c_adipose_list)

    # Accumulation ratio
    acc_ratio = 0.0
    if cmax_plasma > 0:
        acc_ratio = cmax_adipose / cmax_plasma

    # Terminal half-life estimation from plasma (use last 30% of time points)
    t_half_terminal = 0.0
    n = len(c_plasma_list)
    start_idx = int(0.7 * n)
    # Find a pair with sufficient concentration to estimate lambda_z
    for i in range(start_idx, n - 1):
        c1 = c_plasma_list[i]
        c2 = c_plasma_list[i + 1]
        if c1 > 1e-12 and c2 > 1e-12 and c1 > c2:
            lambda_z = math.log(c1 / c2) / (times[i + 1] - times[i])
            if lambda_z > 0:
                t_half_terminal = math.log(2) / lambda_z
                break

    notes = (
        f"Kp_adipose={kp_adipose:.3f}, "
        f"Vd_adipose={vd_adipose_L:.1f}L, "
        f"k_in={k_adipose_in:.3f}/h, k_out={k_adipose_out:.3f}/h"
    )

    return AdiposeAccumulationResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        logP=logP,
        fat_fraction=fat_fraction,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        c_adipose_mg_L=c_adipose_list,
        cmax_plasma_mg_L=cmax_plasma,
        cmax_adipose_mg_L=cmax_adipose,
        tmax_adipose_h=tmax_adipose,
        auc_plasma_mg_h_per_L=auc_plasma,
        auc_adipose_mg_h_per_L=auc_adipose,
        kp_adipose=kp_adipose,
        accumulation_ratio=acc_ratio,
        t_half_terminal_h=t_half_terminal,
        notes=notes,
    )


def compare_logp_accumulation(
    drug_name: str,
    dose_mg: float,
    logp_values: list,
    cl_sys_L_per_h: float,
    vd_plasma_L: float = 10.0,
) -> list:
    """Compare adipose accumulation across different logP values.

    Returns list of AdiposeAccumulationResult sorted by kp_adipose descending.
    """
    results = []
    for logp in logp_values:
        res = simulate_adipose_accumulation(
            drug_name=drug_name,
            dose_mg=dose_mg,
            logP=logp,
            cl_sys_L_per_h=cl_sys_L_per_h,
            vd_plasma_L=vd_plasma_L,
        )
        results.append(res)
    results.sort(key=lambda r: r.kp_adipose, reverse=True)
    return results
