"""
Phase 571 — Nasal drug absorption and delivery modeling.
"""

from __future__ import annotations

import math

# Human nasal cavity surface area (cm^2)
_NASAL_SURFACE_AREA_CM2 = 150.0


def nasal_deposition_fraction(
    particle_size_um: float,
    flow_rate_L_min: float = 15.0,
) -> dict:
    """
    Estimate nasal and lung deposition fractions based on particle size and
    inspiratory flow rate using a simplified Stokes-number approach.

    Parameters
    ----------
    particle_size_um : float
        Aerodynamic particle size in micrometres.
    flow_rate_L_min : float
        Inspiratory flow rate in L/min (default 15 L/min for nasal breathing).

    Returns
    -------
    dict with keys:
        deposit_frac_nasal, deposit_frac_lung, exhaled_frac, notes
    """
    if particle_size_um <= 0:
        raise ValueError("particle_size_um must be > 0")
    if flow_rate_L_min <= 0:
        raise ValueError("flow_rate_L_min must be > 0")

    # Simplified dimensionless Stokes number
    stk = particle_size_um**2 * flow_rate_L_min / 1000.0

    # Empirical S-curve for nasal deposition
    deposit_frac_nasal = stk / (stk + 0.5)

    # A fixed 5 % fraction is always exhaled (undeposited)
    exhaled_frac = 0.05
    deposit_frac_lung = max(0.0, 1.0 - deposit_frac_nasal - exhaled_frac)

    notes = []
    if particle_size_um > 10:
        notes.append("Large particles (>10 um): predominantly nasal deposition.")
    elif particle_size_um < 2:
        notes.append("Small particles (<2 um): significant pulmonary penetration.")
    else:
        notes.append("Intermediate particle size: mixed deposition.")

    return {
        "deposit_frac_nasal": deposit_frac_nasal,
        "deposit_frac_lung": deposit_frac_lung,
        "exhaled_frac": exhaled_frac,
        "notes": " ".join(notes),
    }


def nasal_absorption_model(
    drug_name: str,
    deposited_dose_mg: float,
    peff_cm_s: float,
    mucociliary_t_half_h: float = 0.25,
    nasal_volume_mL: float = 15.0,
    t_end_h: float = 4.0,
    dt_h: float = 0.05,
) -> dict:
    """
    Simple 1-compartment model for drug absorption from the nasal mucosa.

    Amount on mucosa: dA/dt = -(kabs + kmc) * A
    kabs = peff_cm_s * surface_area_cm2 / nasal_volume_mL * 3600  (h^-1)
    kmc  = ln(2) / mucociliary_t_half_h

    Parameters
    ----------
    drug_name : str
    deposited_dose_mg : float   Dose deposited on nasal mucosa (mg).
    peff_cm_s : float           Effective permeability (cm/s).
    mucociliary_t_half_h : float Half-life of mucociliary clearance (h).
    nasal_volume_mL : float     Mucosal fluid volume (mL).
    t_end_h : float             Simulation end time (h).
    dt_h : float                Forward-Euler step size (h).

    Returns
    -------
    dict with keys:
        drug_name, times_h, amounts_on_mucosa_mg, f_absorbed,
        cmax_approx_mg_L, notes
    """
    if deposited_dose_mg < 0:
        raise ValueError("deposited_dose_mg must be >= 0")
    if peff_cm_s < 0:
        raise ValueError("peff_cm_s must be >= 0")
    if mucociliary_t_half_h <= 0:
        raise ValueError("mucociliary_t_half_h must be > 0")
    if nasal_volume_mL <= 0:
        raise ValueError("nasal_volume_mL must be > 0")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be > 0")
    if dt_h <= 0:
        raise ValueError("dt_h must be > 0")

    # Rate constants (h^-1)
    kabs = peff_cm_s * _NASAL_SURFACE_AREA_CM2 / nasal_volume_mL * 3600.0
    kmc = math.log(2) / mucociliary_t_half_h
    k_total = kabs + kmc

    # Fraction absorbed (analytical)
    if k_total > 0:
        f_absorbed = kabs / k_total
    else:
        f_absorbed = 0.0

    # Forward-Euler simulation
    times_h: list[float] = []
    amounts_mg: list[float] = []

    t = 0.0
    A = deposited_dose_mg
    while t <= t_end_h + 1e-9:
        times_h.append(round(t, 10))
        amounts_mg.append(A)
        dA = -k_total * A * dt_h
        A = max(0.0, A + dA)
        t += dt_h

    # Approximate Cmax at the nasal mucosa surface (mg/L)
    # = peak concentration in mucosal fluid from absorbed portion
    cmax_approx_mg_L = (deposited_dose_mg * f_absorbed / nasal_volume_mL) * 1000.0

    notes = (
        f"kabs={kabs:.4f} h-1, kmc={kmc:.4f} h-1, "
        f"f_absorbed={f_absorbed:.3f}, Cmax~{cmax_approx_mg_L:.2f} mg/L"
    )

    return {
        "drug_name": drug_name,
        "times_h": times_h,
        "amounts_on_mucosa_mg": amounts_mg,
        "f_absorbed": f_absorbed,
        "cmax_approx_mg_L": cmax_approx_mg_L,
        "notes": notes,
    }


def intranasal_bioavailability(
    f_nasal: float,
    f_first_pass: float = 0.0,
) -> float:
    """
    Compute intranasal bioavailability.

    F_nasal = f_nasal * (1 - f_first_pass)

    First-pass is generally negligible for nasal delivery (default 0).
    Return value clamped to [0, 1].
    """
    if not (0.0 <= f_nasal <= 1.0):
        raise ValueError("f_nasal must be in [0, 1]")
    if not (0.0 <= f_first_pass <= 1.0):
        raise ValueError("f_first_pass must be in [0, 1]")

    F = f_nasal * (1.0 - f_first_pass)
    return max(0.0, min(1.0, F))


def compare_nasal_vs_oral(
    nasal_f: float,
    oral_f: float,
    nasal_tmax_h: float,
    oral_tmax_h: float,
) -> dict:
    """
    Compare nasal vs. oral delivery for a drug.

    Returns
    -------
    dict with keys:
        nasal_advantage (bool), bioavailability_ratio, tmax_ratio, notes
    """
    if not (0.0 <= nasal_f <= 1.0):
        raise ValueError("nasal_f must be in [0, 1]")
    if not (0.0 <= oral_f <= 1.0):
        raise ValueError("oral_f must be in [0, 1]")
    if nasal_tmax_h <= 0:
        raise ValueError("nasal_tmax_h must be > 0")
    if oral_tmax_h <= 0:
        raise ValueError("oral_tmax_h must be > 0")

    nasal_advantage = nasal_f > oral_f

    if oral_f > 0:
        bioavailability_ratio = nasal_f / oral_f
    else:
        bioavailability_ratio = float("inf") if nasal_f > 0 else 1.0

    tmax_ratio = nasal_tmax_h / oral_tmax_h

    parts = []
    if nasal_advantage:
        parts.append("Nasal delivery provides higher bioavailability than oral.")
    else:
        parts.append("Oral delivery provides equal or higher bioavailability.")
    if tmax_ratio < 1.0:
        parts.append("Nasal route achieves faster absorption (lower Tmax).")
    elif tmax_ratio > 1.0:
        parts.append("Oral route achieves faster peak (lower Tmax).")
    else:
        parts.append("Similar Tmax for both routes.")

    return {
        "nasal_advantage": nasal_advantage,
        "bioavailability_ratio": bioavailability_ratio,
        "tmax_ratio": tmax_ratio,
        "notes": " ".join(parts),
    }
