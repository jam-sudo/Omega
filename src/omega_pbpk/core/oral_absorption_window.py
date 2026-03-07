"""
Oral Absorption Window Simulator.

Models GI transit and absorption window concept for drugs with limited
absorption sites (e.g., modified-release formulations that only absorb
in specific GI regions).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class AbsorptionWindowResult:
    """Result from absorption window simulation."""

    drug_name: str
    times_h: list[float] = field(default_factory=list)
    c_plasma_mg_L: list[float] = field(default_factory=list)
    fa: float = 0.0  # fraction absorbed (0–1)
    absorption_window_h: float = 3.0
    cmax: float = 0.0
    tmax_h: float = 0.0
    auc: float = 0.0
    notes: str = ""


def simulate_absorption_window(
    drug_name: str,
    dose_mg: float,
    peff_cm_s: float,
    solubility_mg_mL: float,
    dose_volume_mL: float = 250.0,
    absorption_window_h: float = 3.0,
    transit_time_h: list[float] | None = None,
    ka_background_per_h: float = 0.0,
    t_end_h: float = 12.0,
    dt_h: float = 0.05,
    cl_L_per_h: float = 10.0,
    vd_L: float = 50.0,
) -> AbsorptionWindowResult:
    """
    Simulate oral absorption with an absorption window constraint.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Oral dose in mg.
    peff_cm_s:
        Effective permeability in cm/s.
    solubility_mg_mL:
        Aqueous solubility in mg/mL.
    dose_volume_mL:
        Co-administered fluid volume (default 250 mL).
    absorption_window_h:
        Duration over which absorption can occur (hours).
    transit_time_h:
        List of cumulative transit times (h) for GI segments:
        [stomach, SI1, SI2, SI3, colon]. Defaults to [0.25, 1.0, 3.0, 4.0, 6.0].
    ka_background_per_h:
        Background first-order absorption rate outside window (h⁻¹).
    t_end_h:
        Simulation end time (hours).
    dt_h:
        Integration step size (hours).
    cl_L_per_h:
        Systemic clearance (L/h).
    vd_L:
        Volume of distribution (L).

    Returns
    -------
    AbsorptionWindowResult
    """
    # Input validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if peff_cm_s <= 0:
        raise ValueError("peff_cm_s must be > 0")
    if solubility_mg_mL <= 0:
        raise ValueError("solubility_mg_mL must be > 0")
    if absorption_window_h < 0:
        raise ValueError("absorption_window_h must be >= 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")

    if transit_time_h is None:
        transit_time_h = [0.25, 1.0, 3.0, 4.0, 6.0]

    # Maximum dissolved dose limited by solubility
    max_dissolved_mg = solubility_mg_mL * dose_volume_mL
    dissolved_dose_mg = min(dose_mg, max_dissolved_mg)
    undissolved_dose_mg = dose_mg - dissolved_dose_mg

    notes_parts: list[str] = []
    if undissolved_dose_mg > 0:
        notes_parts.append(f"Solubility-limited: {undissolved_dose_mg:.1f} mg undissolved")

    # Convert peff (cm/s) to an absorption rate coefficient (h⁻¹)
    # Surface area approximation: intestinal radius ~1.75 cm, length 250 cm
    # Absorption clearance = peff [cm/s] * 2*pi*r*L [cm²]
    # = peff * 2 * 3.14159 * 1.75 * 250 * 3600 [cm³/h = mL/h]
    # Convert to L/h by dividing by 1000
    intestinal_surface_cm2 = 2 * math.pi * 1.75 * 250  # ~2749 cm²
    abs_CL_mL_per_h = peff_cm_s * intestinal_surface_cm2 * 3600.0
    abs_CL_L_per_h = abs_CL_mL_per_h / 1000.0

    # Lumen volume (L) — treat dose_volume_mL as the lumen volume
    lumen_vol_L = dose_volume_mL / 1000.0

    # ka effective = abs_CL / lumen_vol (h⁻¹)
    ka_window = abs_CL_L_per_h / lumen_vol_L  # h⁻¹

    # Forward Euler simulation
    n_steps = int(t_end_h / dt_h) + 1
    times = [i * dt_h for i in range(n_steps)]

    # State variables
    a_gut = dissolved_dose_mg  # drug in gut lumen (mg)
    a_plasma = 0.0  # drug amount in plasma (mg)
    a_absorbed = 0.0  # cumulative absorbed (mg)

    c_plasma_list: list[float] = []
    prev_c = 0.0
    prev_t = 0.0
    auc = 0.0
    cmax = 0.0
    tmax_h = 0.0

    for i, t in enumerate(times):
        c_plasma = a_plasma / vd_L
        c_plasma_list.append(c_plasma)

        # Trapezoidal AUC
        if i > 0:
            auc += 0.5 * (prev_c + c_plasma) * (t - prev_t)
        prev_c = c_plasma
        prev_t = t

        if c_plasma > cmax:
            cmax = c_plasma
            tmax_h = t

        # Determine absorption rate
        # Use strict less-than so window_h=0 means no absorption at any t>=0
        in_window = absorption_window_h > 0 and t < absorption_window_h
        if in_window:
            ka = ka_window
        else:
            ka = ka_background_per_h

        # dA_gut/dt = -ka * A_gut
        dA_gut = -ka * a_gut * dt_h
        a_gut = max(0.0, a_gut + dA_gut)

        # Absorbed amount
        absorbed_step = -dA_gut  # positive
        a_absorbed += absorbed_step

        # dA_plasma/dt = absorbed - CL*C_plasma
        elim = cl_L_per_h * c_plasma * dt_h
        a_plasma = max(0.0, a_plasma + absorbed_step - elim)

    fa = min(1.0, a_absorbed / dose_mg) if dose_mg > 0 else 0.0

    if absorption_window_h == 0.0:
        notes_parts.append("Zero absorption window: fa=0")

    return AbsorptionWindowResult(
        drug_name=drug_name,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        fa=fa,
        absorption_window_h=absorption_window_h,
        cmax=cmax,
        tmax_h=tmax_h,
        auc=auc,
        notes="; ".join(notes_parts),
    )


def window_sensitivity(
    drug_name: str,
    dose_mg: float,
    peff_cm_s: float,
    solubility_mg_mL: float,
    window_hours: list[float] | None = None,
    cl_L_per_h: float = 10.0,
    vd_L: float = 50.0,
) -> list[AbsorptionWindowResult]:
    """
    Run absorption window simulation across multiple window sizes.

    Useful for showing the impact of modified-release formulation design
    on bioavailability.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Oral dose in mg.
    peff_cm_s:
        Effective permeability in cm/s.
    solubility_mg_mL:
        Aqueous solubility in mg/mL.
    window_hours:
        List of absorption window durations (h). Defaults to [1, 2, 3, 4, 6].
    cl_L_per_h:
        Systemic clearance (L/h).
    vd_L:
        Volume of distribution (L).

    Returns
    -------
    list[AbsorptionWindowResult] sorted by absorption_window_h ascending.
    """
    if window_hours is None:
        window_hours = [1.0, 2.0, 3.0, 4.0, 6.0]

    results = []
    for wh in sorted(window_hours):
        result = simulate_absorption_window(
            drug_name=drug_name,
            dose_mg=dose_mg,
            peff_cm_s=peff_cm_s,
            solubility_mg_mL=solubility_mg_mL,
            absorption_window_h=wh,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
        )
        results.append(result)

    return results
