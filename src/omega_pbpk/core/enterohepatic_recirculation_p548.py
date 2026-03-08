"""Phase 548 - Enterohepatic Recirculation PK Model.

Forward Euler simulation of oral drug absorption with enterohepatic
recirculation producing secondary concentration peaks.
"""

import math
from dataclasses import dataclass, field


@dataclass
class EHCResult:
    """Result of enterohepatic recirculation simulation."""

    drug_name: str = ""
    dose_mg: float = 0.0
    f_biliary: float = 0.0
    times_h: list = field(default_factory=list)
    c_systemic_mg_L: list = field(default_factory=list)
    cmax_primary_mg_L: float = 0.0
    tmax_primary_h: float = 0.0
    cmax_secondary_mg_L: float = 0.0
    tmax_secondary_h: float = 0.0
    auc_total_mg_h_per_L: float = 0.0
    secondary_peak_ratio: float = 0.0
    notes: str = ""


def simulate_ehc(
    drug_name: str,
    dose_mg: float,
    ka_per_h: float,
    cl_L_per_h: float,
    vd_L: float,
    f_biliary: float,
    bile_release_h: float = 4.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> EHCResult:
    """Simulate enterohepatic recirculation using Forward Euler.

    Args:
        drug_name: Name of the drug.
        dose_mg: Oral dose in mg.
        ka_per_h: Absorption rate constant (1/h).
        cl_L_per_h: Systemic clearance (L/h).
        vd_L: Volume of distribution (L).
        f_biliary: Fraction of eliminated drug excreted into bile (0-1).
        bile_release_h: Time of bile release into GI tract (h).
        t_end_h: Simulation end time (h).
        dt_h: Time step for Forward Euler (h).

    Returns:
        EHCResult with concentration-time profile and PK parameters.
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if ka_per_h <= 0:
        raise ValueError(f"ka_per_h must be > 0, got {ka_per_h}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if not (0.0 <= f_biliary <= 1.0):
        raise ValueError(f"f_biliary must be in [0, 1], got {f_biliary}")

    k_el = cl_L_per_h / vd_L

    # State variables
    a_gi = dose_mg  # Drug in GI tract (mg)
    c_sys = 0.0  # Systemic concentration (mg/L)
    a_bile = 0.0  # Drug accumulated in bile (mg)

    times = []
    concentrations = []

    bile_released = False
    n_steps = int(math.ceil(t_end_h / dt_h))

    for i in range(n_steps + 1):
        t = i * dt_h
        times.append(t)
        concentrations.append(c_sys)

        # Check bile release
        if not bile_released and t >= bile_release_h:
            a_gi += a_bile
            a_bile = 0.0
            bile_released = True

        # Forward Euler derivatives
        da_gi = -ka_per_h * a_gi
        dc_sys = ka_per_h * a_gi / vd_L - k_el * c_sys

        # Bile accumulation (before release)
        if not bile_released:
            da_bile = f_biliary * k_el * vd_L * c_sys
        else:
            da_bile = 0.0

        # Update state
        a_gi += da_gi * dt_h
        c_sys += dc_sys * dt_h
        a_bile += da_bile * dt_h

        # Clamp negatives
        if a_gi < 0:
            a_gi = 0.0
        if c_sys < 0:
            c_sys = 0.0

    # Find primary peak (before bile_release_h)
    cmax_primary = 0.0
    tmax_primary = 0.0
    for j, t in enumerate(times):
        if t < bile_release_h:
            if concentrations[j] > cmax_primary:
                cmax_primary = concentrations[j]
                tmax_primary = t

    # Find secondary peak (after bile_release_h + 0.5h)
    cmax_secondary = 0.0
    tmax_secondary = 0.0
    for j, t in enumerate(times):
        if t > bile_release_h + 0.5:
            if concentrations[j] > cmax_secondary:
                cmax_secondary = concentrations[j]
                tmax_secondary = t

    # Manual trapezoidal AUC
    auc = sum(
        0.5 * (concentrations[j] + concentrations[j - 1]) * (times[j] - times[j - 1])
        for j in range(1, len(times))
    )

    secondary_peak_ratio = cmax_secondary / cmax_primary if cmax_primary > 0 else 0.0

    notes_parts = []
    if f_biliary > 0.5:
        notes_parts.append("High biliary excretion fraction")
    if secondary_peak_ratio > 0.3:
        notes_parts.append("Prominent secondary peak")
    if not notes_parts:
        notes_parts.append("Standard EHC profile")

    return EHCResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        f_biliary=f_biliary,
        times_h=times,
        c_systemic_mg_L=concentrations,
        cmax_primary_mg_L=cmax_primary,
        tmax_primary_h=tmax_primary,
        cmax_secondary_mg_L=cmax_secondary,
        tmax_secondary_h=tmax_secondary,
        auc_total_mg_h_per_L=auc,
        secondary_peak_ratio=secondary_peak_ratio,
        notes="; ".join(notes_parts),
    )
