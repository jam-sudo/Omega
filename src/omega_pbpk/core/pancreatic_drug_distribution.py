"""Pancreatic drug distribution: 2-compartment plasma ↔ pancreas tissue model."""

from __future__ import annotations

from dataclasses import dataclass


def _trapz(times: list, values: list) -> float:
    """Manual trapezoidal AUC integration."""
    if len(times) < 2:
        return 0.0
    total = 0.0
    for i in range(len(times) - 1):
        dt = times[i + 1] - times[i]
        total += 0.5 * (values[i] + values[i + 1]) * dt
    return total


@dataclass
class PancreaticDistributionResult:
    drug_name: str
    dose_mg: float
    times_h: list
    c_plasma_mg_L: list
    c_pancreas_mg_L: list
    auc_plasma: float
    auc_pancreas: float
    kp_pancreas: float
    tmax_pancreas_h: float
    cmax_pancreas_mg_L: float
    pancreas_to_plasma_ratio: float
    notes: str


def simulate_pancreatic_distribution(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float = 5.0,
    vd_plasma_L: float = 20.0,
    vd_pancreas_L: float = 1.0,
    q_pancreas_L_per_h: float = 0.8,
    kp_pancreas: float = 1.5,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> PancreaticDistributionResult:
    """Simulate 2-compartment plasma-pancreas drug distribution using Forward Euler."""
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_plasma_L <= 0:
        raise ValueError("vd_plasma_L must be > 0")
    if vd_pancreas_L <= 0:
        raise ValueError("vd_pancreas_L must be > 0")
    if q_pancreas_L_per_h <= 0:
        raise ValueError("q_pancreas_L_per_h must be > 0")
    if kp_pancreas <= 0:
        raise ValueError("kp_pancreas must be > 0")

    ke = cl_L_per_h / vd_plasma_L
    k12 = q_pancreas_L_per_h / vd_plasma_L
    k21 = q_pancreas_L_per_h / (vd_pancreas_L * kp_pancreas)

    c_plasma = dose_mg / vd_plasma_L
    c_pancreas = 0.0

    times: list = []
    plasma_conc: list = []
    pancreas_conc: list = []

    t = 0.0
    n_steps = int(round(t_end_h / dt_h)) + 1

    for _i in range(n_steps):
        times.append(t)
        plasma_conc.append(c_plasma)
        pancreas_conc.append(c_pancreas)

        dcp_dt = -ke * c_plasma - k12 * c_plasma + k21 * c_pancreas * (vd_pancreas_L / vd_plasma_L)
        dct_dt = k12 * c_plasma * (vd_plasma_L / vd_pancreas_L) - k21 * c_pancreas

        c_plasma = max(0.0, c_plasma + dcp_dt * dt_h)
        c_pancreas = max(0.0, c_pancreas + dct_dt * dt_h)

        t += dt_h

    auc_plasma = _trapz(times, plasma_conc)
    auc_pancreas = _trapz(times, pancreas_conc)

    cmax_pancreas = max(pancreas_conc)
    tmax_pancreas_h = times[pancreas_conc.index(cmax_pancreas)]

    if auc_plasma > 0:
        ratio = auc_pancreas / auc_plasma
    else:
        ratio = 0.0

    notes = (
        f"Kp_pancreas={kp_pancreas:.2f}; "
        f"AUC ratio={ratio:.3f}; "
        f"Tmax_pancreas={tmax_pancreas_h:.1f}h"
    )

    return PancreaticDistributionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times,
        c_plasma_mg_L=plasma_conc,
        c_pancreas_mg_L=pancreas_conc,
        auc_plasma=auc_plasma,
        auc_pancreas=auc_pancreas,
        kp_pancreas=kp_pancreas,
        tmax_pancreas_h=tmax_pancreas_h,
        cmax_pancreas_mg_L=cmax_pancreas,
        pancreas_to_plasma_ratio=ratio,
        notes=notes,
    )
