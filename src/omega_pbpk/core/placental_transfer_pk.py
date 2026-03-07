"""Placental transfer PK model (Phase 906).

Models drug transfer across the placenta from maternal to fetal circulation.
Uses a simple 2-compartment ODE model (maternal + fetal) with oral absorption
from a gut compartment. Distinct from maternal_fetal.py (full PBPK) —
this module focuses on placental transfer kinetics.

Key safety metric: fetal:maternal AUC ratio.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PlacentalTransferResult:
    """Result of placental transfer PK simulation."""

    drug_name: str
    dose_mg: float
    times_h: list
    c_maternal_mg_L: list
    c_fetal_mg_L: list
    cmax_maternal: float
    tmax_maternal_h: float
    auc_maternal: float
    cmax_fetal: float
    tmax_fetal_h: float
    auc_fetal: float
    fetal_maternal_ratio: float
    placental_clearance_mL_min: float
    fetal_safety_concern: bool
    notes: str


def _trapezoidal_auc(times: list, values: list) -> float:
    """Compute AUC via trapezoidal rule."""
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        auc += 0.5 * (values[i - 1] + values[i]) * dt
    return auc


def simulate_placental_transfer(
    drug_name: str = "Drug",
    dose_mg: float = 100.0,
    k_abs_per_h: float = 1.0,
    k_placenta_transfer_per_h: float = 0.1,
    k_fetal_elimination_per_h: float = 0.05,
    cl_maternal_L_per_h: float = 10.0,
    vd_maternal_L: float = 30.0,
    vd_fetal_L: float = 3.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> PlacentalTransferResult:
    """Simulate placental drug transfer using forward Euler.

    Three compartments: gut (absorption) → maternal plasma → fetal plasma.
    One-directional placental transfer driven by k_placenta_transfer_per_h.

    Args:
        drug_name: Name of the drug.
        dose_mg: Oral dose in mg (absorbed from gut compartment).
        k_abs_per_h: First-order absorption rate constant (per h).
        k_placenta_transfer_per_h: First-order placental transfer rate (per h).
        k_fetal_elimination_per_h: First-order fetal elimination rate (per h).
        cl_maternal_L_per_h: Maternal systemic clearance in L/h.
        vd_maternal_L: Maternal volume of distribution in L.
        vd_fetal_L: Fetal volume of distribution in L.
        t_end_h: Simulation end time in hours.
        dt_h: Forward Euler time step in hours.

    Returns:
        PlacentalTransferResult with maternal/fetal profiles and safety assessment.

    Raises:
        ValueError: If dose_mg <= 0, cl_maternal_L_per_h <= 0, vd_maternal_L <= 0,
                    or vd_fetal_L <= 0.
    """
    if dose_mg <= 0.0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_maternal_L_per_h <= 0.0:
        raise ValueError(f"cl_maternal_L_per_h must be > 0, got {cl_maternal_L_per_h}")
    if vd_maternal_L <= 0.0:
        raise ValueError(f"vd_maternal_L must be > 0, got {vd_maternal_L}")
    if vd_fetal_L <= 0.0:
        raise ValueError(f"vd_fetal_L must be > 0, got {vd_fetal_L}")

    ke_maternal = cl_maternal_L_per_h / vd_maternal_L

    # Initial conditions
    a_gut = dose_mg
    a_maternal = 0.0
    a_fetal = 0.0

    times_h: list = []
    c_maternal_list: list = []
    c_fetal_list: list = []

    n_steps = int(round(t_end_h / dt_h)) + 1
    t = 0.0

    cmax_maternal = 0.0
    tmax_maternal_h = 0.0
    cmax_fetal = 0.0
    tmax_fetal_h = 0.0

    for _ in range(n_steps):
        c_maternal = a_maternal / vd_maternal_L
        c_fetal = a_fetal / vd_fetal_L

        times_h.append(t)
        c_maternal_list.append(c_maternal)
        c_fetal_list.append(c_fetal)

        if c_maternal > cmax_maternal:
            cmax_maternal = c_maternal
            tmax_maternal_h = t
        if c_fetal > cmax_fetal:
            cmax_fetal = c_fetal
            tmax_fetal_h = t

        # ODEs (forward Euler on amounts)
        da_gut = -k_abs_per_h * a_gut
        da_maternal = (
            k_abs_per_h * a_gut - ke_maternal * a_maternal - k_placenta_transfer_per_h * a_maternal
        )
        da_fetal = k_placenta_transfer_per_h * a_maternal - k_fetal_elimination_per_h * a_fetal

        a_gut = max(0.0, a_gut + da_gut * dt_h)
        a_maternal = max(0.0, a_maternal + da_maternal * dt_h)
        a_fetal = max(0.0, a_fetal + da_fetal * dt_h)

        t += dt_h

    auc_maternal = _trapezoidal_auc(times_h, c_maternal_list)
    auc_fetal = _trapezoidal_auc(times_h, c_fetal_list)

    fetal_maternal_ratio = auc_fetal / auc_maternal if auc_maternal > 0.0 else 0.0

    # Placental clearance: k_placenta * Vd_maternal converted to mL/min
    # k_placenta_transfer_per_h [1/h] * vd_maternal_L [L] = L/h
    # L/h * 1000 mL/L / 60 min/h = mL/min
    placental_clearance_mL_min = k_placenta_transfer_per_h * vd_maternal_L * 1000.0 / 60.0
    placental_clearance_mL_min = max(0.001, min(100.0, placental_clearance_mL_min))

    fetal_safety_concern = fetal_maternal_ratio > 0.5

    if fetal_safety_concern:
        notes = (
            "High placental transfer — fetal exposure is >50% of maternal. "
            "Contraindicated in pregnancy without careful benefit-risk analysis."
        )
    elif fetal_maternal_ratio > 0.2:
        notes = "Moderate placental transfer — use in pregnancy with caution"
    else:
        notes = "Limited fetal exposure — relatively safer for use in pregnancy"

    return PlacentalTransferResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times_h,
        c_maternal_mg_L=c_maternal_list,
        c_fetal_mg_L=c_fetal_list,
        cmax_maternal=cmax_maternal,
        tmax_maternal_h=tmax_maternal_h,
        auc_maternal=auc_maternal,
        cmax_fetal=cmax_fetal,
        tmax_fetal_h=tmax_fetal_h,
        auc_fetal=auc_fetal,
        fetal_maternal_ratio=fetal_maternal_ratio,
        placental_clearance_mL_min=placental_clearance_mL_min,
        fetal_safety_concern=fetal_safety_concern,
        notes=notes,
    )


def compare_placental_transfer(
    drug_name: str,
    dose_mg: float,
    scenarios: list,
) -> list:
    """Compare placental transfer across different scenarios.

    Each scenario dict may contain optional keys:
        - "k_placenta_transfer_per_h": float
        - "name": str (used to label the result via drug_name)

    Args:
        drug_name: Base drug name (used if scenario has no "name").
        dose_mg: Dose in mg.
        scenarios: List of dicts with optional simulation parameters.

    Returns:
        List of PlacentalTransferResult sorted by fetal_maternal_ratio descending.
    """
    results = []
    for scenario in scenarios:
        scenario_name = scenario.get("name", drug_name)
        k_transfer = scenario.get("k_placenta_transfer_per_h", 0.1)
        result = simulate_placental_transfer(
            drug_name=scenario_name,
            dose_mg=dose_mg,
            k_placenta_transfer_per_h=k_transfer,
        )
        results.append(result)

    results.sort(key=lambda r: r.fetal_maternal_ratio, reverse=True)
    return results
