"""Phase 539 — Subcutaneous Drug Absorption PK.

PK model for subcutaneous (SC) drug administration with depot-dependent absorption.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class SubcutaneousAbsorptionResult:
    """Result of subcutaneous absorption simulation."""

    drug_name: str
    dose_mg: float
    mw_Da: float
    formulation_type: str  # "solution" / "suspension" / "biologic"
    ka_sc_per_h: float
    f_sc: float
    f_lymph: float
    t_half_absorption_h: float
    times_h: list = field(default_factory=list)
    c_systemic_mg_L: list = field(default_factory=list)
    cmax_mg_L: float = 0.0
    tmax_h: float = 0.0
    auc_mg_h_per_L: float = 0.0
    bioavailability_pct: float = 0.0
    relative_to_iv_auc: float = 0.0
    notes: str = ""


def _get_sc_params(formulation_type: str, mw_Da: float) -> tuple[float, float, float]:
    """Return (ka_sc_per_h, f_sc, f_lymph) for given formulation type."""
    ftype = formulation_type.lower()
    if ftype == "biologic" or mw_Da > 5000:
        return 0.02, 0.6, 0.2
    elif ftype == "suspension":
        return 0.05, 0.8, 0.05
    else:  # solution
        return 0.3, 0.8, 0.05


def simulate_subcutaneous_absorption(
    drug_name: str,
    dose_mg: float,
    mw_Da: float,
    formulation_type: str,
    cl_L_per_h: float,
    vd_sys_L: float,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
) -> SubcutaneousAbsorptionResult:
    """Simulate SC absorption using a 2-compartment depot model.

    Args:
        drug_name: Name of the drug.
        dose_mg: Dose in mg.
        mw_Da: Molecular weight in Daltons.
        formulation_type: One of "solution", "suspension", "biologic".
        cl_L_per_h: Systemic clearance in L/h.
        vd_sys_L: Volume of distribution (systemic) in L.
        t_end_h: Simulation end time in hours.
        dt_h: Forward Euler time step in hours.

    Returns:
        SubcutaneousAbsorptionResult with simulated concentration-time profile.
    """
    # Validate inputs
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive")
    if mw_Da <= 0:
        raise ValueError("mw_Da must be positive")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be positive")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be positive")
    valid_formulations = {"solution", "suspension", "biologic"}
    if formulation_type.lower() not in valid_formulations:
        raise ValueError(
            f"formulation_type must be one of {valid_formulations}, got '{formulation_type}'"
        )

    ka_sc, f_sc, f_lymph = _get_sc_params(formulation_type, mw_Da)
    k_el = cl_L_per_h / vd_sys_L
    t_half_abs = math.log(2.0) / ka_sc

    # Initial conditions
    a_depot = dose_mg  # amount in SC depot [mg]
    c_sys = 0.0  # systemic concentration [mg/L]

    times_h: list[float] = []
    c_systemic: list[float] = []

    t = 0.0
    n_steps = int(round(t_end_h / dt_h))

    for step in range(n_steps + 1):
        t = step * dt_h
        times_h.append(t)
        c_systemic.append(c_sys)

        if step < n_steps:
            # Forward Euler
            da_depot_dt = -ka_sc * a_depot
            dc_sys_dt = (ka_sc * a_depot * f_sc / vd_sys_L) - k_el * c_sys

            a_depot += da_depot_dt * dt_h
            c_sys += dc_sys_dt * dt_h

            # Guard against negative values
            if a_depot < 0.0:
                a_depot = 0.0
            if c_sys < 0.0:
                c_sys = 0.0

    # Compute Cmax and Tmax
    cmax_mg_L = max(c_systemic)
    tmax_h = times_h[c_systemic.index(cmax_mg_L)]

    # Manual trapezoidal AUC
    auc = sum(
        0.5 * (c_systemic[i] + c_systemic[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )

    bioavailability_pct = f_sc * 100.0
    # relative_to_iv_auc = AUC * CL / dose; for perfect IV bolus F=1 → AUC_iv = dose/CL
    relative_to_iv_auc = auc * cl_L_per_h / dose_mg

    notes = (
        f"Formulation: {formulation_type}; "
        f"ka_sc={ka_sc}/h; F_sc={f_sc}; f_lymph={f_lymph}; "
        f"t½_abs={t_half_abs:.2f} h"
    )

    return SubcutaneousAbsorptionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        mw_Da=mw_Da,
        formulation_type=formulation_type.lower(),
        ka_sc_per_h=ka_sc,
        f_sc=f_sc,
        f_lymph=f_lymph,
        t_half_absorption_h=t_half_abs,
        times_h=times_h,
        c_systemic_mg_L=c_systemic,
        cmax_mg_L=cmax_mg_L,
        tmax_h=tmax_h,
        auc_mg_h_per_L=auc,
        bioavailability_pct=bioavailability_pct,
        relative_to_iv_auc=relative_to_iv_auc,
        notes=notes,
    )


def compare_sc_formulations(
    drug_name: str,
    dose_mg: float,
    mw_Da: float,
    cl_L_per_h: float,
    vd_sys_L: float,
) -> list[SubcutaneousAbsorptionResult]:
    """Compare all 3 SC formulation types sorted by Cmax descending.

    Args:
        drug_name: Name of the drug.
        dose_mg: Dose in mg.
        mw_Da: Molecular weight in Daltons.
        cl_L_per_h: Systemic clearance in L/h.
        vd_sys_L: Volume of distribution (systemic) in L.

    Returns:
        List of 3 SubcutaneousAbsorptionResult objects sorted by cmax_mg_L descending.
    """
    results = []
    for ftype in ("solution", "suspension", "biologic"):
        result = simulate_subcutaneous_absorption(
            drug_name=drug_name,
            dose_mg=dose_mg,
            mw_Da=mw_Da,
            formulation_type=ftype,
            cl_L_per_h=cl_L_per_h,
            vd_sys_L=vd_sys_L,
        )
        results.append(result)

    results.sort(key=lambda r: r.cmax_mg_L, reverse=True)
    return results
