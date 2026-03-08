"""Phase 540 — Drug Distribution in Amniotic Fluid.

3-compartment model: maternal plasma → fetal plasma → amniotic fluid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class AmnioticFluidPKResult:
    """Result of amniotic fluid PK simulation."""

    drug_name: str
    dose_mg: float
    ps_placenta_L_per_h: float
    gestational_week: int
    times_h: list = field(default_factory=list)
    c_maternal_mg_L: list = field(default_factory=list)
    c_fetal_mg_L: list = field(default_factory=list)
    c_amniotic_mg_L: list = field(default_factory=list)
    cmax_maternal: float = 0.0
    cmax_fetal: float = 0.0
    cmax_amniotic: float = 0.0
    auc_maternal: float = 0.0
    auc_fetal: float = 0.0
    auc_amniotic: float = 0.0
    fetal_maternal_ratio: float = 0.0
    amniotic_maternal_ratio: float = 0.0
    fetal_exposure_concern: bool = False
    t_half_maternal_h: float = 0.0
    notes: str = ""


def simulate_amniotic_fluid_pk(
    drug_name: str,
    dose_mg: float,
    cl_maternal_L_per_h: float,
    vd_maternal_L: float,
    ps_placenta_L_per_h: float = 0.05,
    vd_fetal_L: float = 2.0,
    v_amniotic_L: float = 1.0,
    cl_fetal_renal_L_per_h: float = 0.1,
    k_swallow_per_h: float = 0.05,
    gestational_week: int = 36,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
) -> AmnioticFluidPKResult:
    """Simulate drug distribution in amniotic fluid using 3-compartment model.

    Args:
        drug_name: Name of the drug.
        dose_mg: IV bolus dose in mg.
        cl_maternal_L_per_h: Maternal systemic clearance in L/h.
        vd_maternal_L: Maternal volume of distribution in L.
        ps_placenta_L_per_h: Placental permeability-surface area product in L/h.
        vd_fetal_L: Fetal volume of distribution in L.
        v_amniotic_L: Volume of amniotic fluid in L.
        cl_fetal_renal_L_per_h: Fetal renal clearance in L/h.
        k_swallow_per_h: Rate constant for fetal swallowing of amniotic fluid in /h.
        gestational_week: Gestational week (20–42).
        t_end_h: Simulation end time in hours.
        dt_h: Forward Euler time step in hours.

    Returns:
        AmnioticFluidPKResult with simulated concentration-time profiles.
    """
    # Validate inputs
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive")
    if cl_maternal_L_per_h <= 0:
        raise ValueError("cl_maternal_L_per_h must be positive")
    if vd_maternal_L <= 0:
        raise ValueError("vd_maternal_L must be positive")
    if not (20 <= gestational_week <= 42):
        raise ValueError("gestational_week must be between 20 and 42")

    # Derived rate constants
    k_el_m = cl_maternal_L_per_h / vd_maternal_L
    k_m_to_f = ps_placenta_L_per_h / vd_maternal_L
    k_f_to_m = ps_placenta_L_per_h / vd_fetal_L
    k_fetal_renal = cl_fetal_renal_L_per_h / vd_fetal_L
    t_half_maternal = (
        math.log(2.0) / (k_el_m + k_m_to_f) if (k_el_m + k_m_to_f) > 0 else float("inf")
    )

    # Initial conditions (IV bolus into maternal compartment)
    c_maternal = dose_mg / vd_maternal_L
    c_fetal = 0.0
    c_amniotic = 0.0

    times_h: list[float] = []
    c_mat_list: list[float] = []
    c_fet_list: list[float] = []
    c_amn_list: list[float] = []

    n_steps = int(round(t_end_h / dt_h))

    for step in range(n_steps + 1):
        t = step * dt_h
        times_h.append(t)
        c_mat_list.append(c_maternal)
        c_fet_list.append(c_fetal)
        c_amn_list.append(c_amniotic)

        if step < n_steps:
            # Forward Euler derivatives
            dc_mat_dt = -k_el_m * c_maternal - k_m_to_f * c_maternal + k_f_to_m * c_fetal
            dc_fet_dt = k_m_to_f * c_maternal - k_f_to_m * c_fetal - k_fetal_renal * c_fetal
            dc_amn_dt = (
                k_fetal_renal * c_fetal * vd_fetal_L / v_amniotic_L - k_swallow_per_h * c_amniotic
            )

            c_maternal += dc_mat_dt * dt_h
            c_fetal += dc_fet_dt * dt_h
            c_amniotic += dc_amn_dt * dt_h

            # Guard against negatives
            if c_maternal < 0.0:
                c_maternal = 0.0
            if c_fetal < 0.0:
                c_fetal = 0.0
            if c_amniotic < 0.0:
                c_amniotic = 0.0

    # Peak values
    cmax_maternal = max(c_mat_list)
    cmax_fetal = max(c_fet_list)
    cmax_amniotic = max(c_amn_list)

    # Trapezoidal AUC
    def _trapz(x: list[float], y: list[float]) -> float:
        return sum(0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]) for i in range(1, len(x)))

    auc_maternal = _trapz(times_h, c_mat_list)
    auc_fetal = _trapz(times_h, c_fet_list)
    auc_amniotic = _trapz(times_h, c_amn_list)

    # Ratios based on AUC (more robust than Cmax for ratio)
    fetal_maternal_ratio = auc_fetal / auc_maternal if auc_maternal > 0 else 0.0
    amniotic_maternal_ratio = auc_amniotic / auc_maternal if auc_maternal > 0 else 0.0
    fetal_exposure_concern = fetal_maternal_ratio > 0.1

    notes = (
        f"Gestational week {gestational_week}; "
        f"PS_placenta={ps_placenta_L_per_h} L/h; "
        f"fetal/maternal AUC ratio={fetal_maternal_ratio:.3f}; "
        f"exposure concern={'YES' if fetal_exposure_concern else 'NO'}"
    )

    return AmnioticFluidPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        ps_placenta_L_per_h=ps_placenta_L_per_h,
        gestational_week=gestational_week,
        times_h=times_h,
        c_maternal_mg_L=c_mat_list,
        c_fetal_mg_L=c_fet_list,
        c_amniotic_mg_L=c_amn_list,
        cmax_maternal=cmax_maternal,
        cmax_fetal=cmax_fetal,
        cmax_amniotic=cmax_amniotic,
        auc_maternal=auc_maternal,
        auc_fetal=auc_fetal,
        auc_amniotic=auc_amniotic,
        fetal_maternal_ratio=fetal_maternal_ratio,
        amniotic_maternal_ratio=amniotic_maternal_ratio,
        fetal_exposure_concern=fetal_exposure_concern,
        t_half_maternal_h=t_half_maternal,
        notes=notes,
    )


def compare_placental_permeability(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_maternal_L: float,
    ps_values: list,
) -> list[AmnioticFluidPKResult]:
    """Compare fetal exposure across different placental permeability values.

    Args:
        drug_name: Name of the drug.
        dose_mg: IV bolus dose in mg.
        cl_L_per_h: Maternal systemic clearance in L/h.
        vd_maternal_L: Maternal volume of distribution in L.
        ps_values: List of PS_placenta values (L/h) to compare.

    Returns:
        List of AmnioticFluidPKResult sorted by fetal_maternal_ratio ascending (lower = safer).
    """
    results = []
    for ps in ps_values:
        result = simulate_amniotic_fluid_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            cl_maternal_L_per_h=cl_L_per_h,
            vd_maternal_L=vd_maternal_L,
            ps_placenta_L_per_h=ps,
        )
        results.append(result)

    results.sort(key=lambda r: r.fetal_maternal_ratio)
    return results
