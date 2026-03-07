"""Biliary excretion and enterohepatic circulation model."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class EHCResult:
    """Result from enterohepatic circulation simulation."""

    drug_name: str
    times_h: list[float] = field(default_factory=list)
    c_plasma_mg_L: list[float] = field(default_factory=list)
    a_bile_mg: list[float] = field(default_factory=list)
    a_gi_mg: list[float] = field(default_factory=list)
    secondary_peaks: list[float] = field(default_factory=list)
    auc: float = 0.0
    notes: str = ""


def biliary_clearance(
    cl_renal_L_per_h: float,
    cl_total_L_per_h: float,
    fe_urine: float | None = None,
) -> dict:
    """Calculate biliary clearance from total and renal clearance.

    Args:
        cl_renal_L_per_h: Renal clearance [L/h]
        cl_total_L_per_h: Total clearance [L/h]
        fe_urine: Fraction excreted in urine (optional, for validation)

    Returns:
        dict with cl_biliary_L_per_h, fe_bile, fe_urine, cl_total_L_per_h
    """
    if cl_total_L_per_h <= 0:
        raise ValueError("cl_total_L_per_h must be > 0")
    if cl_renal_L_per_h < 0:
        raise ValueError("cl_renal_L_per_h must be >= 0")
    if cl_renal_L_per_h > cl_total_L_per_h:
        raise ValueError("cl_renal_L_per_h cannot exceed cl_total_L_per_h")

    cl_biliary = cl_total_L_per_h - cl_renal_L_per_h
    fe_bile = cl_biliary / cl_total_L_per_h
    fe_urine_calc = cl_renal_L_per_h / cl_total_L_per_h

    return {
        "cl_biliary_L_per_h": cl_biliary,
        "fe_bile": fe_bile,
        "fe_urine": fe_urine_calc,
        "cl_total_L_per_h": cl_total_L_per_h,
    }


def enterohepatic_circulation_factor(
    f_reabsorbed: float,
    t_recirculation_h: float,
    t_half_h: float,
) -> float:
    """Compute effective half-life extended by enterohepatic circulation.

    Args:
        f_reabsorbed: Fraction reabsorbed each cycle (0-1)
        t_recirculation_h: Transit time for recirculation cycle [h]
        t_half_h: Intrinsic terminal half-life [h]

    Returns:
        effective_t_half_h
    """
    if not (0.0 <= f_reabsorbed <= 1.0):
        raise ValueError("f_reabsorbed must be in [0, 1]")
    if t_recirculation_h <= 0:
        raise ValueError("t_recirculation_h must be > 0")
    if t_half_h <= 0:
        raise ValueError("t_half_h must be > 0")

    if f_reabsorbed == 0.0:
        return t_half_h

    # EHC factor: fraction reabsorbed that arrives back before the drug would be gone
    recirculation_fraction = f_reabsorbed * math.exp(-math.log(2) * t_recirculation_h / t_half_h)

    # Avoid division by zero / negative denominator
    denominator = 1.0 - recirculation_fraction
    if denominator <= 0:
        return float("inf")

    effective_t_half = t_half_h / denominator
    return effective_t_half


def secondary_peak_detection(
    times_h: list[float],
    concentrations: list[float],
    min_prominence: float = 0.05,
) -> list[float]:
    """Detect secondary peaks from enterohepatic recirculation.

    A peak at index i: c[i] > c[i-1] and c[i] > c[i+1] and c[i] > min_prominence * max(c)

    Args:
        times_h: Time points [h]
        concentrations: Concentration at each time point
        min_prominence: Minimum peak height as fraction of max concentration

    Returns:
        List of time values where secondary peaks occur
    """
    if len(times_h) != len(concentrations):
        raise ValueError("times_h and concentrations must have same length")
    if len(concentrations) < 3:
        return []

    c = concentrations
    max_c = max(c)
    if max_c <= 0:
        return []

    threshold = min_prominence * max_c
    peaks = []

    # Skip initial peak - only detect peaks that occur after the initial decline
    # Find the first local minimum (after initial peak) to skip the first peak
    first_peak_idx = 0
    for i in range(1, len(c)):
        if c[i] > c[first_peak_idx]:
            first_peak_idx = i
        else:
            break

    # Now detect peaks after the first one
    for i in range(first_peak_idx + 1, len(c) - 1):
        if c[i] > c[i - 1] and c[i] > c[i + 1] and c[i] > threshold:
            peaks.append(times_h[i])

    return peaks


def simulate_enterohepatic_circulation(
    drug_name: str,
    dose_mg: float,
    cl_hepatic_L_per_h: float,
    vd_L: float,
    f_biliary: float = 0.3,
    f_reabsorbed: float = 0.7,
    t_transit_h: float = 6.0,
    t_end_h: float = 72.0,
    dt_h: float = 0.1,
    route: str = "iv",
) -> EHCResult:
    """Simulate 3-compartment enterohepatic circulation model.

    Compartments:
    - Plasma (Cp): central
    - Bile (A_bile): biliary excretion pool
    - GI (A_gi): intestinal pool for reabsorption

    ODEs:
        dCp/dt = -(cl_total/Vd)*Cp + reabsorption_rate/Vd
        dA_bile/dt = cl_biliary * Cp - A_bile / t_transit_h
        dA_gi/dt = A_bile/t_transit_h - reabsorption_rate
        reabsorption_rate = f_reabsorbed * A_gi / t_transit_h

    Args:
        drug_name: Drug identifier
        dose_mg: Dose [mg]
        cl_hepatic_L_per_h: Hepatic clearance [L/h]
        vd_L: Volume of distribution [L]
        f_biliary: Fraction of hepatic CL that is biliary (0-1)
        f_reabsorbed: Fraction of bile reabsorbed from GI (0-1)
        t_transit_h: Bile transit time [h]
        t_end_h: Simulation end time [h]
        dt_h: Time step [h]
        route: 'iv' only for this model

    Returns:
        EHCResult with plasma concentrations and secondary peak information
    """
    if not (0.0 <= f_biliary <= 1.0):
        raise ValueError("f_biliary must be in [0, 1]")
    if not (0.0 <= f_reabsorbed <= 1.0):
        raise ValueError("f_reabsorbed must be in [0, 1]")
    if cl_hepatic_L_per_h <= 0:
        raise ValueError("cl_hepatic_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")

    cl_total = cl_hepatic_L_per_h
    cl_biliary = f_biliary * cl_total

    # Initial conditions: IV bolus
    cp = dose_mg / vd_L  # mg/L
    a_bile = 0.0  # mg
    a_gi = 0.0  # mg

    n_steps = int(t_end_h / dt_h) + 1
    times_h = []
    c_plasma = []
    a_bile_list = []
    a_gi_list = []

    for step in range(n_steps):
        t = step * dt_h
        times_h.append(t)
        c_plasma.append(cp)
        a_bile_list.append(a_bile)
        a_gi_list.append(a_gi)

        reabsorption_rate = f_reabsorbed * a_gi / t_transit_h  # mg/h

        dcp_dt = -(cl_total / vd_L) * cp + reabsorption_rate / vd_L
        da_bile_dt = cl_biliary * cp - a_bile / t_transit_h
        da_gi_dt = a_bile / t_transit_h - reabsorption_rate

        cp = max(0.0, cp + dcp_dt * dt_h)
        a_bile = max(0.0, a_bile + da_bile_dt * dt_h)
        a_gi = max(0.0, a_gi + da_gi_dt * dt_h)

    # Trapezoidal AUC
    auc = 0.0
    for i in range(1, len(times_h)):
        auc += 0.5 * (c_plasma[i] + c_plasma[i - 1]) * (times_h[i] - times_h[i - 1])

    secondary_peaks = secondary_peak_detection(times_h, c_plasma)

    notes = (
        f"EHC model: f_biliary={f_biliary:.2f}, f_reabsorbed={f_reabsorbed:.2f}, "
        f"t_transit={t_transit_h:.1f}h. "
        f"Secondary peaks detected: {len(secondary_peaks)}"
    )

    return EHCResult(
        drug_name=drug_name,
        times_h=times_h,
        c_plasma_mg_L=c_plasma,
        a_bile_mg=a_bile_list,
        a_gi_mg=a_gi_list,
        secondary_peaks=secondary_peaks,
        auc=auc,
        notes=notes,
    )
