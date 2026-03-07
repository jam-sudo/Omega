"""Dual absorption pathway pharmacokinetics.

Models drugs absorbed via both passive diffusion and active transport
(e.g., OATP1B1-mediated hepatic uptake, PEPT1 intestinal uptake).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = ["DualAbsorptionResult", "simulate_dual_absorption", "assess_transporter_saturation"]


@dataclass
class DualAbsorptionResult:
    """Result of dual absorption PK simulation."""

    drug_name: str
    dose_mg: float
    mw: float
    times_h: list = field(default_factory=list)
    a_lumen_mg: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    cmax_mg_L: float = 0.0
    tmax_h: float = 0.0
    auc_mg_h_per_L: float = 0.0
    t_half_h: float = 0.0
    f_passive: float = 0.7
    f_active: float = 0.3
    f_bioavailability: float = 0.8
    transporter_contribution_pct: float = 0.0
    saturation_observed: bool = False
    notes: str = ""


def _validate_params(
    dose_mg: float,
    f_passive: float,
    f_active: float,
    vmax_nmol_per_h: float,
    km_uM: float,
    mw: float,
) -> None:
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if f_passive <= 0 or f_passive > 1:
        raise ValueError("f_passive must be in (0, 1]")
    if f_active <= 0 or f_active > 1:
        raise ValueError("f_active must be in (0, 1]")
    if f_passive + f_active > 1.0 + 1e-6:
        raise ValueError("f_passive + f_active must be <= 1.0")
    if vmax_nmol_per_h <= 0:
        raise ValueError("vmax_nmol_per_h must be > 0")
    if km_uM <= 0:
        raise ValueError("km_uM must be > 0")
    if mw <= 0:
        raise ValueError("mw must be > 0")


def simulate_dual_absorption(
    drug_name: str,
    dose_mg: float,
    ka_passive_per_h: float = 0.5,
    f_passive: float = 0.7,
    f_active: float = 0.3,
    vmax_nmol_per_h: float = 500.0,
    km_uM: float = 50.0,
    f_bioavailability: float = 0.8,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    mw: float = 400.0,
    t_end_h: float = 12.0,
    dt_h: float = 0.05,
) -> DualAbsorptionResult:
    """Simulate dual absorption pathway PK.

    Two parallel absorption paths:
    1. Passive: ka_passive × A_lumen × f_passive
    2. Active (transporter): Vmax × A_lumen / (Km + A_lumen) × f_active

    Args:
        drug_name: Compound identifier.
        dose_mg: Oral dose in mg.
        ka_passive_per_h: Passive absorption rate constant (1/h).
        f_passive: Fraction of absorption via passive route.
        f_active: Fraction via active transport.
        vmax_nmol_per_h: Transporter max velocity (nmol/h).
        km_uM: Michaelis constant for transporter (µM).
        f_bioavailability: Combined bioavailability after absorption.
        cl_L_per_h: Total clearance (L/h).
        vd_L: Volume of distribution (L).
        mw: Molecular weight (g/mol) for unit conversion.
        t_end_h: Simulation end time (h).
        dt_h: Time step (h).

    Returns:
        DualAbsorptionResult with PK metrics.
    """
    _validate_params(dose_mg, f_passive, f_active, vmax_nmol_per_h, km_uM, mw)

    ke = cl_L_per_h / vd_L

    # Convert dose to nmol for transporter calculations
    dose_nmol = dose_mg / mw * 1e6

    # Convert Km from µM to nmol (in 1L lumen volume reference)
    # µM = nmol/mL = µmol/L, so km_nmol is used in nmol units for A_lumen in nmol
    # Actually Km in µM means µmol/L = nmol/mL
    # We track A_lumen in nmol, so use Km in nmol (= km_uM × 1000 if volume is 1L)
    # Use a reference volume of 0.25L (250 mL gastric + intestinal)
    ref_vol_L = 0.25
    km_nmol = km_uM * ref_vol_L * 1000  # µM × L × 1000 nmol/µmol = nmol

    # State: A_lumen_mg, C_plasma_mg_L
    a_lumen_mg = dose_mg
    a_lumen_nmol = dose_nmol
    c_plasma = 0.0

    times = []
    a_lumen_list = []
    c_plasma_list = []

    n_steps = int(t_end_h / dt_h) + 1

    auc_passive = 0.0
    auc_active = 0.0
    prev_c = 0.0

    for i in range(n_steps):
        t = i * dt_h
        times.append(t)
        a_lumen_list.append(a_lumen_mg)
        c_plasma_list.append(c_plasma)

        # Passive absorption rate (mg/h)
        passive_rate_mg = ka_passive_per_h * a_lumen_mg * f_passive

        # Active transport rate in nmol/h (MM kinetics)
        active_rate_nmol = vmax_nmol_per_h * a_lumen_nmol / (km_nmol + a_lumen_nmol) * f_active

        # Convert active rate to mg/h
        active_rate_mg = active_rate_nmol * mw / 1e6

        # Total absorption into plasma (with bioavailability)
        total_absorbed_mg_h = (passive_rate_mg + active_rate_mg) * f_bioavailability

        # ODE: dA_lumen/dt
        d_a_lumen_mg = -(passive_rate_mg + active_rate_mg)
        d_a_lumen_nmol = -(ka_passive_per_h * a_lumen_nmol * f_passive + active_rate_nmol)

        # ODE: dC_plasma/dt
        d_c_plasma = total_absorbed_mg_h / vd_L - ke * c_plasma

        # Accumulate AUC contributions by pathway
        if i > 0:
            # Weight by contribution
            total_rate = passive_rate_mg + active_rate_mg
            if total_rate > 0:
                auc_passive += passive_rate_mg / total_rate * (c_plasma + prev_c) / 2 * dt_h
                auc_active += active_rate_mg / total_rate * (c_plasma + prev_c) / 2 * dt_h
            else:
                # No absorption but still distribution
                auc_passive += (c_plasma + prev_c) / 2 * dt_h * f_passive
                auc_active += (c_plasma + prev_c) / 2 * dt_h * f_active

        prev_c = c_plasma

        # Forward Euler update
        a_lumen_mg = max(0.0, a_lumen_mg + d_a_lumen_mg * dt_h)
        a_lumen_nmol = max(0.0, a_lumen_nmol + d_a_lumen_nmol * dt_h)
        c_plasma = max(0.0, c_plasma + d_c_plasma * dt_h)

    # Compute PK metrics
    cmax = max(c_plasma_list)
    tmax = times[c_plasma_list.index(cmax)]

    # Trapezoidal AUC
    auc = 0.0
    for i in range(1, len(times)):
        auc += (c_plasma_list[i] + c_plasma_list[i - 1]) / 2 * dt_h

    t_half = math.log(2) / ke if ke > 0 else 0.0

    # Transporter contribution estimate
    total_auc_attributed = auc_passive + auc_active
    if total_auc_attributed > 0:
        transporter_pct = auc_active / total_auc_attributed * 100.0
    else:
        transporter_pct = f_active / (f_passive + f_active) * 100.0

    # Check saturation by simulating 2× dose
    dose_2x_mg = dose_mg * 2
    dose_2x_nmol = dose_2x_mg / mw * 1e6

    a2_mg = dose_2x_mg
    a2_nmol = dose_2x_nmol
    c2 = 0.0
    cmax_2x = 0.0

    for _ in range(n_steps):
        passive_r2 = ka_passive_per_h * a2_mg * f_passive
        active_r2_nmol = vmax_nmol_per_h * a2_nmol / (km_nmol + a2_nmol) * f_active
        active_r2_mg = active_r2_nmol * mw / 1e6
        total_r2 = (passive_r2 + active_r2_mg) * f_bioavailability
        d_a2_mg = -(passive_r2 + active_r2_mg)
        d_a2_nmol = -(ka_passive_per_h * a2_nmol * f_passive + active_r2_nmol)
        d_c2 = total_r2 / vd_L - ke * c2
        a2_mg = max(0.0, a2_mg + d_a2_mg * dt_h)
        a2_nmol = max(0.0, a2_nmol + d_a2_nmol * dt_h)
        c2 = max(0.0, c2 + d_c2 * dt_h)
        if c2 > cmax_2x:
            cmax_2x = c2

    # Saturation: if 2× dose doesn't give close to 2× Cmax
    expected_cmax_2x = cmax * 2.0
    saturation_observed = False
    if cmax > 0 and cmax_2x < expected_cmax_2x * 0.9:
        saturation_observed = True

    notes = (
        f"Passive route: {f_passive * 100:.0f}%, "
        f"Active route: {f_active * 100:.0f}%. "
        f"Transporter Vmax={vmax_nmol_per_h:.0f} nmol/h, Km={km_uM:.0f} µM. "
    )
    if saturation_observed:
        notes += "Nonlinear kinetics (saturation) observed at 2x dose."
    else:
        notes += "Linear kinetics at tested dose range."

    return DualAbsorptionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        mw=mw,
        times_h=times,
        a_lumen_mg=a_lumen_list,
        c_plasma_mg_L=c_plasma_list,
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc_mg_h_per_L=auc,
        t_half_h=t_half,
        f_passive=f_passive,
        f_active=f_active,
        f_bioavailability=f_bioavailability,
        transporter_contribution_pct=transporter_pct,
        saturation_observed=saturation_observed,
        notes=notes,
    )


def assess_transporter_saturation(
    drug_name: str,
    doses_mg: list,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    mw: float = 400.0,
    km_uM: float = 50.0,
    vmax_nmol_per_h: float = 500.0,
) -> dict:
    """Simulate at multiple doses and check for dose-AUC nonlinearity.

    Args:
        drug_name: Compound identifier.
        doses_mg: List of doses to simulate (mg).
        cl_L_per_h: Total clearance (L/h).
        vd_L: Volume of distribution (L).
        mw: Molecular weight (g/mol).
        km_uM: Michaelis constant for transporter (µM).
        vmax_nmol_per_h: Transporter max velocity (nmol/h).

    Returns:
        dict with doses, aucs, auc_ratios_per_dose, is_saturating, notes.
    """
    if not doses_mg:
        return {
            "doses": [],
            "aucs": [],
            "auc_ratios_per_dose": [],
            "is_saturating": False,
            "notes": "No doses provided.",
        }

    aucs = []
    for dose in doses_mg:
        result = simulate_dual_absorption(
            drug_name=drug_name,
            dose_mg=dose,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            mw=mw,
            km_uM=km_uM,
            vmax_nmol_per_h=vmax_nmol_per_h,
        )
        aucs.append(result.auc_mg_h_per_L)

    # Calculate AUC/dose ratios (normalized to first dose)
    ref_ratio = aucs[0] / doses_mg[0] if doses_mg[0] > 0 and aucs[0] > 0 else 1.0
    auc_ratios = []
    for dose, auc in zip(doses_mg, aucs):
        if dose > 0:
            ratio = (auc / dose) / ref_ratio if ref_ratio > 0 else 1.0
        else:
            ratio = 1.0
        auc_ratios.append(ratio)

    # Check if saturating: any ratio < 0.9 (AUC/dose decreases with dose)
    is_saturating = any(r < 0.9 for r in auc_ratios[1:]) if len(auc_ratios) > 1 else False

    notes = (
        f"Assessed {len(doses_mg)} dose levels for {drug_name}. "
        f"Km={km_uM:.0f} µM, Vmax={vmax_nmol_per_h:.0f} nmol/h. "
    )
    if is_saturating:
        notes += "Dose-dependent decrease in AUC/dose ratio indicates transporter saturation."
    else:
        notes += "AUC/dose ratio approximately constant — no saturation detected."

    return {
        "doses": doses_mg,
        "aucs": aucs,
        "auc_ratios_per_dose": auc_ratios,
        "is_saturating": is_saturating,
        "notes": notes,
    }
