"""
Phase 975 — Methotrexate PK Model

2-compartment IV infusion model for high-dose MTX therapy monitoring.
Includes leucovorin rescue trigger and toxicity risk assessment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class MethotrexatePKResult:
    """Result from high-dose methotrexate PK simulation."""

    drug_name: str = "Methotrexate"
    dose_mg: float = 0.0
    infusion_duration_h: float = 0.0
    times_h: list = field(default_factory=list)
    c_central_mg_L: list = field(default_factory=list)
    c_central_umol_L: list = field(default_factory=list)
    cmax_mg_L: float = 0.0
    auc_mg_h_per_L: float = 0.0
    c_at_24h_umol_L: float = 0.0
    c_at_48h_umol_L: float = 0.0
    leucovorin_rescue_required: bool = False
    toxicity_risk: str = "low"
    notes: str = ""


# MW of methotrexate in g/mol
_MW_MTX = 454.4  # g/mol → 1 µmol/L = 0.4544 mg/L
_MG_PER_UMOL = _MW_MTX / 1000.0  # 0.4544 mg/µmol·L


def simulate_methotrexate_pk(
    dose_mg: float,
    infusion_duration_h: float = 24.0,
    gfr_mL_min: float = 100.0,
    fup: float = 0.5,
    v1_L: float = 18.0,
    v2_L: float = 8.0,
    q_L_per_h: float = 1.5,
    t_end_h: float = 120.0,
    dt_h: float = 0.1,
) -> MethotrexatePKResult:
    """
    Simulate high-dose methotrexate IV infusion PK.

    Parameters
    ----------
    dose_mg : float
        Total dose in mg.
    infusion_duration_h : float
        Duration of IV infusion in hours.
    gfr_mL_min : float
        Glomerular filtration rate in mL/min (default 100 mL/min).
    fup : float
        Fraction unbound in plasma (0-1).
    v1_L : float
        Volume of central compartment (L).
    v2_L : float
        Volume of peripheral compartment (L).
    q_L_per_h : float
        Inter-compartmental clearance (L/h).
    t_end_h : float
        Simulation end time (h).
    dt_h : float
        Forward Euler step size (h).

    Returns
    -------
    MethotrexatePKResult
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if infusion_duration_h <= 0:
        raise ValueError(f"infusion_duration_h must be > 0, got {infusion_duration_h}")
    if gfr_mL_min <= 0:
        raise ValueError(f"gfr_mL_min must be > 0, got {gfr_mL_min}")

    # --- PK parameters ---
    k_inf = dose_mg / infusion_duration_h  # mg/h

    # Renal clearance: GFR * fup (convert mL/min → L/h)
    cl_renal = gfr_mL_min * 60.0 / 1000.0 * fup  # L/h

    k10 = cl_renal / v1_L
    k12 = q_L_per_h / v1_L
    k21 = q_L_per_h / v2_L

    # --- Forward Euler simulation ---
    n_steps = int(math.ceil(t_end_h / dt_h)) + 1
    times_h: list[float] = []
    c1_list: list[float] = []  # central compartment concentration mg/L
    c2: float = 0.0  # peripheral compartment concentration mg/L
    c1: float = 0.0

    for step in range(n_steps):
        t = step * dt_h
        times_h.append(t)
        c1_list.append(c1)

        # Infusion rate: active during [0, infusion_duration_h]
        infusion_rate = k_inf if t <= infusion_duration_h else 0.0

        dc1_dt = infusion_rate / v1_L - (k10 + k12) * c1 + k21 * c2 * v2_L / v1_L
        dc2_dt = k12 * c1 * v1_L / v2_L - k21 * c2

        c1 = c1 + dc1_dt * dt_h
        c2 = c2 + dc2_dt * dt_h

        # Clamp to avoid negative concentrations from numerical issues
        if c1 < 0.0:
            c1 = 0.0
        if c2 < 0.0:
            c2 = 0.0

    # --- Derived metrics ---
    cmax_mg_L = max(c1_list) if c1_list else 0.0

    # Trapezoidal AUC
    auc = 0.0
    for i in range(1, len(times_h)):
        auc += 0.5 * (c1_list[i - 1] + c1_list[i]) * (times_h[i] - times_h[i - 1])

    # Concentration at 24h and 48h (nearest time point)
    def _nearest_conc(target_h: float) -> float:
        best_idx = 0
        best_diff = abs(times_h[0] - target_h)
        for idx in range(1, len(times_h)):
            diff = abs(times_h[idx] - target_h)
            if diff < best_diff:
                best_diff = diff
                best_idx = idx
        return c1_list[best_idx]

    c_at_24h_mg_L = _nearest_conc(24.0)
    c_at_48h_mg_L = _nearest_conc(48.0)

    # Convert mg/L → µmol/L  (divide by _MG_PER_UMOL = 0.4544)
    c_umol_L = [c / _MG_PER_UMOL for c in c1_list]
    c_at_24h_umol_L = c_at_24h_mg_L / _MG_PER_UMOL
    c_at_48h_umol_L = c_at_48h_mg_L / _MG_PER_UMOL

    # --- Leucovorin rescue ---
    leucovorin_rescue_required = c_at_24h_umol_L > 1.0 or c_at_48h_umol_L > 0.1

    # --- Toxicity risk ---
    if leucovorin_rescue_required:
        toxicity_risk = "high"
    elif auc > 1000.0:
        toxicity_risk = "moderate"
    else:
        toxicity_risk = "low"

    # --- Notes ---
    notes_parts: list[str] = []
    notes_parts.append(f"CL_renal={cl_renal:.2f} L/h (GFR={gfr_mL_min} mL/min, fup={fup})")
    notes_parts.append(f"C_24h={c_at_24h_umol_L:.3f} µmol/L, C_48h={c_at_48h_umol_L:.3f} µmol/L")
    if leucovorin_rescue_required:
        notes_parts.append("Leucovorin rescue indicated per institutional thresholds.")
    notes = "; ".join(notes_parts)

    return MethotrexatePKResult(
        drug_name="Methotrexate",
        dose_mg=dose_mg,
        infusion_duration_h=infusion_duration_h,
        times_h=times_h,
        c_central_mg_L=c1_list,
        c_central_umol_L=c_umol_L,
        cmax_mg_L=cmax_mg_L,
        auc_mg_h_per_L=auc,
        c_at_24h_umol_L=c_at_24h_umol_L,
        c_at_48h_umol_L=c_at_48h_umol_L,
        leucovorin_rescue_required=leucovorin_rescue_required,
        toxicity_risk=toxicity_risk,
        notes=notes,
    )
