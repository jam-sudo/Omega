"""Transdermal drug permeation through explicit skin layers (SC, VE, dermis).

Models Fick's-law diffusion across stratum corneum → viable epidermis → dermis →
capillaries using layer-specific diffusion coefficients and a plasma compartment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class TransdermalLayersPKResult:
    """PK result for transdermal layer permeation simulation."""

    # Inputs
    drug_name: str
    dose_mg_per_cm2: float
    patch_area_cm2: float
    logp: float
    mw_da: float

    # Trajectories
    times_h: list
    c_plasma_mg_L: list
    a_sc_mg_cm2: list
    a_ve_mg_cm2: list

    # Summary metrics
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    total_dose_mg: float
    fraction_absorbed: float
    p_sc_cm_per_h: float
    lag_time_h: float
    notes: str


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------


def _compute_diffusion_coefficients(
    logp: float,
    mw_da: float,
    d_sc_base_cm2_per_s: float,
    d_aq_cm2_per_s: float,
) -> tuple[float, float, float]:
    """Return (D_sc, D_ve, D_d) all in cm²/s."""
    d_sc = d_sc_base_cm2_per_s * math.exp(logp * 0.5 - mw_da / 1000.0)
    d_sc = max(d_sc, 1e-20)  # prevent underflow

    scale_aq = math.sqrt(500.0 / max(mw_da, 1.0))
    d_ve = d_aq_cm2_per_s * scale_aq
    d_d = d_aq_cm2_per_s * scale_aq

    return d_sc, d_ve, d_d


def simulate_transdermal_layers_pk(
    drug_name: str,
    dose_mg_per_cm2: float = 0.5,
    patch_area_cm2: float = 10.0,
    logp: float = 2.0,
    mw_da: float = 300.0,
    d_sc_base_cm2_per_s: float = 1e-14,
    d_aq_cm2_per_s: float = 5e-6,
    k_removal_per_h: float = 5.0,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> TransdermalLayersPKResult:
    """Simulate drug permeation through skin layers using a 3-compartment ODE model.

    Compartments (amount per unit area, mg/cm²):
        A_sc  — stratum corneum
        A_ve  — viable epidermis
        A_d   — dermis

    Plasma compartment tracks systemic concentration (mg/L).

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    dose_mg_per_cm2:
        Applied surface dose density (mg/cm²).
    patch_area_cm2:
        Occlusive patch area (cm²).
    logp:
        Octanol-water log partition coefficient.
    mw_da:
        Molecular weight (Da).
    d_sc_base_cm2_per_s:
        Base SC diffusion coefficient (cm²/s).
    d_aq_cm2_per_s:
        Aqueous diffusion coefficient used for VE and dermis (cm²/s).
    k_removal_per_h:
        First-order blood-removal rate from dermis (/h).
    cl_L_per_h:
        Systemic clearance (L/h).
    vd_L:
        Volume of distribution (L).
    t_end_h:
        Simulation end time (h).
    dt_h:
        Forward-Euler step size (h).

    Returns
    -------
    TransdermalLayersPKResult
    """
    # Validation
    if dose_mg_per_cm2 <= 0:
        raise ValueError("dose_mg_per_cm2 must be positive")
    if patch_area_cm2 <= 0:
        raise ValueError("patch_area_cm2 must be positive")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be positive")
    if vd_L <= 0:
        raise ValueError("vd_L must be positive")

    # Layer thicknesses (cm)
    L_sc_cm = 20e-4  # 20 µm → cm
    L_ve_cm = 100e-4  # 100 µm → cm
    L_d_cm = 1000e-4  # 1000 µm → cm

    # Diffusion coefficients (cm²/s)
    d_sc, d_ve, _d_d = _compute_diffusion_coefficients(
        logp, mw_da, d_sc_base_cm2_per_s, d_aq_cm2_per_s
    )

    # Permeabilities (cm/s) → convert to cm/h
    _S_PER_H = 3600.0
    P_sc_cm_per_h = (d_sc / L_sc_cm) * _S_PER_H
    P_ve_cm_per_h = (d_ve / L_ve_cm) * _S_PER_H

    # Elimination rate constant (1/h)

    # Initial conditions
    a_sc = dose_mg_per_cm2  # all dose in SC at t=0
    a_ve = 0.0
    a_d = 0.0
    c_plasma = 0.0

    # Storage
    times_h: list[float] = []
    c_plasma_list: list[float] = []
    a_sc_list: list[float] = []
    a_ve_list: list[float] = []

    n_steps = max(1, int(round(t_end_h / dt_h)))
    t = 0.0

    for _ in range(n_steps + 1):
        times_h.append(t)
        c_plasma_list.append(c_plasma)
        a_sc_list.append(a_sc)
        a_ve_list.append(a_ve)

        if _ == n_steps:
            break

        # Concentration in each layer (mg/cm³ = mg/mL, but we work in mg/cm²)
        # Flux SC→VE (mg/cm²/h)
        flux_sc_ve = P_sc_cm_per_h * (a_sc / L_sc_cm - a_ve / L_ve_cm)
        # Flux VE→dermis (mg/cm²/h)
        flux_ve_d = P_ve_cm_per_h * (a_ve / L_ve_cm - a_d / L_d_cm)

        # ODEs (mg/cm²/h)
        da_sc = -flux_sc_ve
        da_ve = flux_sc_ve - flux_ve_d
        da_d = flux_ve_d - k_removal_per_h * a_d

        # Absorbed flux from dermis into systemic: k_removal × A_d × patch_area (mg/h)
        # d(C_plasma × Vd) / dt = absorbed_rate - CL × C_plasma
        absorbed_rate_mg_per_h = k_removal_per_h * a_d * patch_area_cm2
        dc_plasma = (absorbed_rate_mg_per_h - cl_L_per_h * c_plasma) / vd_L

        # Forward Euler update
        a_sc = max(0.0, a_sc + da_sc * dt_h)
        a_ve = max(0.0, a_ve + da_ve * dt_h)
        a_d = max(0.0, a_d + da_d * dt_h)
        c_plasma = max(0.0, c_plasma + dc_plasma * dt_h)

        t += dt_h

    # Summary metrics
    cmax_mg_L = max(c_plasma_list) if c_plasma_list else 0.0
    tmax_h = times_h[c_plasma_list.index(cmax_mg_L)] if cmax_mg_L > 0 else 0.0

    # Trapezoidal AUC
    auc = 0.0
    for i in range(1, len(times_h)):
        auc += 0.5 * (c_plasma_list[i - 1] + c_plasma_list[i]) * (times_h[i] - times_h[i - 1])

    total_dose_mg = dose_mg_per_cm2 * patch_area_cm2
    fraction_absorbed = min(1.0, auc * cl_L_per_h / total_dose_mg) if total_dose_mg > 0 else 0.0

    # Lag time: first time C_plasma >= 1% of Cmax
    lag_time_h = 0.0
    threshold = 0.01 * cmax_mg_L
    if cmax_mg_L > 0:
        for t_val, c_val in zip(times_h, c_plasma_list):
            if c_val >= threshold:
                lag_time_h = t_val
                break

    notes = (
        f"Transdermal layers PK: SC permeability {P_sc_cm_per_h:.3e} cm/h; "
        f"logP={logp:.1f}, MW={mw_da:.0f} Da; patch {patch_area_cm2:.1f} cm²; "
        f"total dose {total_dose_mg:.2f} mg; FA={fraction_absorbed:.3f}"
    )

    return TransdermalLayersPKResult(
        drug_name=drug_name,
        dose_mg_per_cm2=dose_mg_per_cm2,
        patch_area_cm2=patch_area_cm2,
        logp=logp,
        mw_da=mw_da,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_list,
        a_sc_mg_cm2=a_sc_list,
        a_ve_mg_cm2=a_ve_list,
        cmax_mg_L=cmax_mg_L,
        tmax_h=tmax_h,
        auc_mg_h_per_L=auc,
        total_dose_mg=total_dose_mg,
        fraction_absorbed=fraction_absorbed,
        p_sc_cm_per_h=P_sc_cm_per_h,
        lag_time_h=lag_time_h,
        notes=notes,
    )


def compare_patch_areas(
    drug_name: str,
    dose_mg_per_cm2: float,
    areas_cm2: list[float],
    logp: float = 2.0,
    mw_da: float = 300.0,
    d_sc_base_cm2_per_s: float = 1e-14,
    d_aq_cm2_per_s: float = 5e-6,
    k_removal_per_h: float = 5.0,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> list[TransdermalLayersPKResult]:
    """Simulate transdermal PK for multiple patch areas.

    Parameters
    ----------
    drug_name:
        Drug identifier.
    dose_mg_per_cm2:
        Applied dose density (mg/cm²).
    areas_cm2:
        List of patch areas to evaluate (cm²).

    Returns
    -------
    List of TransdermalLayersPKResult, one per area in the same order as ``areas_cm2``.
    """
    results = []
    for area in areas_cm2:
        result = simulate_transdermal_layers_pk(
            drug_name=drug_name,
            dose_mg_per_cm2=dose_mg_per_cm2,
            patch_area_cm2=area,
            logp=logp,
            mw_da=mw_da,
            d_sc_base_cm2_per_s=d_sc_base_cm2_per_s,
            d_aq_cm2_per_s=d_aq_cm2_per_s,
            k_removal_per_h=k_removal_per_h,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            t_end_h=t_end_h,
            dt_h=dt_h,
        )
        results.append(result)
    return results
