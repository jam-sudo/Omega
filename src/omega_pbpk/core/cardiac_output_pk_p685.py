"""Phase 685 — Cardiac Output Effects on PK model."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class CardiacOutputPKResult:
    """Result of cardiac output PK simulation."""

    drug_name: str
    dose_mg: float
    cardiac_output_L_per_min: float
    condition: str
    times_h: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    cmax_mg_L: float = 0.0
    tmax_h: float = 0.0
    auc_mg_h_per_L: float = 0.0
    cl_effective_L_per_h: float = 0.0
    vd_effective_L: float = 0.0
    t_half_h: float = 0.0
    notes: str = ""


_VALID_CONDITIONS = {"normal", "heart_failure", "exercise", "sepsis"}

_CONDITION_CO = {
    "normal": 5.0,
    "heart_failure": 3.0,
    "exercise": 15.0,
    "sepsis": 7.5,
}


def simulate_cardiac_output_pk(
    drug_name: str,
    dose_mg: float,
    logP: float,
    cl_base_L_per_h: float,
    vd_base_L: float,
    cardiac_output_L_per_min: float = 5.0,
    condition: str = "normal",
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> CardiacOutputPKResult:
    """Simulate cardiac output effects on drug PK."""

    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive")
    if cl_base_L_per_h <= 0:
        raise ValueError("cl_base_L_per_h must be positive")
    if vd_base_L <= 0:
        raise ValueError("vd_base_L must be positive")
    if cardiac_output_L_per_min <= 0:
        raise ValueError("cardiac_output_L_per_min must be positive")
    if condition not in _VALID_CONDITIONS:
        raise ValueError(f"condition must be one of {sorted(_VALID_CONDITIONS)}, got '{condition}'")

    co = cardiac_output_L_per_min

    # Hepatic extraction and effective clearance
    hepatic_extraction = min(0.9, max(0.1, cl_base_L_per_h / (co * 60 * 0.25)))
    if hepatic_extraction > 0.5:
        cl_effective = cl_base_L_per_h * (co / 5.0)
    else:
        cl_effective = cl_base_L_per_h

    # CO effect on Vd
    vd_effective = vd_base_L * min(1.5, max(0.5, co / 5.0))

    # Derived PK parameters
    ke = cl_effective / vd_effective
    t_half = math.log(2) / ke
    ka = 1.2  # absorption rate constant, /h
    f_oral = 0.85
    a_gut_0 = dose_mg * f_oral

    # Forward Euler ODE simulation
    n_steps = int(t_end_h / dt_h) + 1
    times_h = []
    c_plasma_mg_L = []

    a_gut = a_gut_0
    a_central = 0.0
    cmax = 0.0
    tmax = 0.0

    for i in range(n_steps):
        t = i * dt_h
        conc = a_central / vd_effective
        if conc < 0:
            conc = 0.0

        times_h.append(round(t, 6))
        c_plasma_mg_L.append(conc)

        if conc > cmax:
            cmax = conc
            tmax = t

        # Euler step
        da_gut = -ka * a_gut
        da_central = ka * a_gut - ke * a_central
        a_gut += da_gut * dt_h
        a_central += da_central * dt_h

    # Manual trapezoidal AUC
    auc = sum(
        0.5 * (c_plasma_mg_L[i] + c_plasma_mg_L[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )

    notes_parts = [
        f"Condition: {condition}",
        f"CO: {co} L/min",
        f"Hepatic extraction: {hepatic_extraction:.3f}",
    ]
    if hepatic_extraction > 0.5:
        notes_parts.append("High-extraction drug: CL scales with CO")
    else:
        notes_parts.append("Low-extraction drug: CL independent of CO")

    return CardiacOutputPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        cardiac_output_L_per_min=co,
        condition=condition,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_mg_L,
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc_mg_h_per_L=auc,
        cl_effective_L_per_h=cl_effective,
        vd_effective_L=vd_effective,
        t_half_h=t_half,
        notes="; ".join(notes_parts),
    )
