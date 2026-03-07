"""Phase 983 — Digoxin PK model for heart failure patients.

Two-compartment model with renal elimination.
Target therapeutic range: 0.5–2.0 ng/mL.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class DigoxinPKResult:
    """Result of digoxin PK simulation."""

    drug_name: str = "Digoxin"
    dose_mg: float = 0.0
    times_h: list = field(default_factory=list)
    c_plasma_ng_mL: list = field(default_factory=list)
    c_tissue_ng_mL: list = field(default_factory=list)
    cmax_ng_mL: float = 0.0
    tmax_h: float = 0.0
    auc_ng_h_per_mL: float = 0.0
    c_at_6h_ng_mL: float = 0.0
    within_therapeutic_range: bool = False
    toxicity_risk: str = "safe"
    dose_recommendation: str = "maintain"
    notes: str = ""


def simulate_digoxin_pk(
    dose_mg: float = 0.25,
    patient_weight_kg: float = 70.0,
    gfr_mL_min: float = 80.0,
    t_end_h: float = 72.0,
    dt_h: float = 0.1,
    route: str = "oral",
) -> DigoxinPKResult:
    """Simulate digoxin PK using a two-compartment model.

    Args:
        dose_mg: Dose in milligrams.
        patient_weight_kg: Patient body weight in kg.
        gfr_mL_min: Glomerular filtration rate in mL/min.
        t_end_h: Simulation end time in hours.
        dt_h: Time step in hours.
        route: Administration route ("oral" or "iv").

    Returns:
        DigoxinPKResult with full PK profile.
    """
    # Validate inputs
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if patient_weight_kg <= 0:
        raise ValueError(f"patient_weight_kg must be > 0, got {patient_weight_kg}")
    if gfr_mL_min <= 0:
        raise ValueError(f"gfr_mL_min must be > 0, got {gfr_mL_min}")
    if route not in {"oral", "iv"}:
        raise ValueError(f"route must be 'oral' or 'iv', got {route!r}")

    # PK parameters
    v1_L = 0.5 * patient_weight_kg  # central volume (L)
    v2_L = 6.5 * patient_weight_kg  # tissue volume (L)
    q_L_per_h = 5.0  # inter-compartmental clearance (L/h); adjusted for realistic digoxin tissue distribution
    cl_renal_L_per_h = gfr_mL_min * 60.0 / 1000.0 * 0.7  # L/h
    cl_nonrenal_L_per_h = 0.2  # L/h fixed
    cl_total_L_per_h = cl_renal_L_per_h + cl_nonrenal_L_per_h

    f_bioavailability = 0.75 if route == "oral" else 1.0
    ka_per_h = 0.5  # oral absorption rate constant

    # Micro-rate constants
    ke10 = cl_total_L_per_h / v1_L
    ke12 = q_L_per_h / v1_L
    ke21 = q_L_per_h / v2_L

    # Initial conditions (mg/L)
    if route == "oral":
        a_gut_mg = f_bioavailability * dose_mg
        c1_mg_L = 0.0
    else:
        a_gut_mg = 0.0
        c1_mg_L = f_bioavailability * dose_mg / v1_L

    c2_mg_L = 0.0

    # Time integration
    n_steps = int(math.ceil(t_end_h / dt_h))
    times_h: list[float] = []
    c_plasma_mg_L: list[float] = []
    c_tissue_mg_L: list[float] = []

    t = 0.0
    for _i in range(n_steps + 1):
        times_h.append(t)
        c_plasma_mg_L.append(c1_mg_L)
        c_tissue_mg_L.append(c2_mg_L)

        if _i == n_steps:
            break

        # Oral absorption input
        if route == "oral":
            oral_input = ka_per_h * a_gut_mg / v1_L
            da_gut = -ka_per_h * a_gut_mg * dt_h
        else:
            oral_input = 0.0
            da_gut = 0.0

        # Two-compartment ODEs (Forward Euler)
        dc1 = (oral_input + ke21 * c2_mg_L * v2_L / v1_L - (ke10 + ke12) * c1_mg_L) * dt_h
        dc2 = (ke12 * c1_mg_L * v1_L / v2_L - ke21 * c2_mg_L) * dt_h

        a_gut_mg += da_gut
        c1_mg_L += dc1
        c2_mg_L += dc2
        if c1_mg_L < 0.0:
            c1_mg_L = 0.0
        if c2_mg_L < 0.0:
            c2_mg_L = 0.0

        t += dt_h

    # Convert to ng/mL (multiply by 1000)
    c_plasma_ng_mL = [c * 1000.0 for c in c_plasma_mg_L]
    c_tissue_ng_mL = [c * 1000.0 for c in c_tissue_mg_L]

    # Derive PK metrics
    cmax_ng_mL = max(c_plasma_ng_mL)
    tmax_h = times_h[c_plasma_ng_mL.index(cmax_ng_mL)]

    # Trapezoidal AUC
    auc = 0.0
    for i in range(1, len(times_h)):
        auc += 0.5 * (c_plasma_ng_mL[i - 1] + c_plasma_ng_mL[i]) * (times_h[i] - times_h[i - 1])

    # c_at_6h: find value nearest to 6 h
    target_t = 6.0
    idx_6h = min(range(len(times_h)), key=lambda i: abs(times_h[i] - target_t))
    c_at_6h = c_plasma_ng_mL[idx_6h]

    # Therapeutic assessment
    within_range = 0.5 <= c_at_6h <= 2.0

    if c_at_6h > 2.0:
        toxicity_risk = "toxic"
    elif c_at_6h > 1.5:
        toxicity_risk = "borderline"
    else:
        toxicity_risk = "safe"

    if c_at_6h < 0.5:
        dose_recommendation = "increase"
    elif c_at_6h > 2.0:
        dose_recommendation = "decrease"
    else:
        dose_recommendation = "maintain"

    notes_parts = [
        f"CL_total={cl_total_L_per_h:.3f} L/h",
        f"V1={v1_L:.1f} L",
        f"V2={v2_L:.1f} L",
        f"Route={route}",
        f"F={f_bioavailability}",
    ]
    notes = "; ".join(notes_parts)

    return DigoxinPKResult(
        drug_name="Digoxin",
        dose_mg=dose_mg,
        times_h=times_h,
        c_plasma_ng_mL=c_plasma_ng_mL,
        c_tissue_ng_mL=c_tissue_ng_mL,
        cmax_ng_mL=cmax_ng_mL,
        tmax_h=tmax_h,
        auc_ng_h_per_mL=auc,
        c_at_6h_ng_mL=c_at_6h,
        within_therapeutic_range=within_range,
        toxicity_risk=toxicity_risk,
        dose_recommendation=dose_recommendation,
        notes=notes,
    )
