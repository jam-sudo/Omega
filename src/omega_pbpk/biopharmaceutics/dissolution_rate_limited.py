"""Dissolution rate limited absorption model for BCS Class II/IV drugs.

Models coupled Noyes-Whitney dissolution + GI permeation + 1-compartment plasma PK
using forward Euler integration (pure stdlib, no numpy/scipy).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "DissolutionAbsorptionResult",
    "dissolution_rate",
    "simulate_dissolution_absorption",
    "particle_size_effect",
]


@dataclass
class DissolutionAbsorptionResult:
    """Result of coupled dissolution-absorption simulation."""

    drug_name: str
    times_h: list[float]
    m_undissolved_mg: list[float]
    c_lumen_mg_mL: list[float]
    c_plasma_mg_L: list[float]
    fa: float  # fraction absorbed from dose
    cmax: float  # mg/L
    tmax_h: float
    auc: float  # mg/L·h (trapezoidal)
    dissolution_limited: bool  # fa < 0.8
    notes: str


def dissolution_rate(
    cs_mg_mL: float,
    c_dissolved_mg_mL: float,
    surface_area_cm2: float,
    D_cm2_h: float,
    h_cm: float = 0.005,
) -> float:
    """Noyes-Whitney dissolution rate.

    dM/dt = D * A * (Cs - C) / h  [mg/h]

    Parameters
    ----------
    cs_mg_mL:
        Saturation solubility in mg/mL.
    c_dissolved_mg_mL:
        Current dissolved drug concentration in mg/mL.
    surface_area_cm2:
        Total particle surface area in cm².
    D_cm2_h:
        Diffusion coefficient in cm²/h.
    h_cm:
        Diffusion layer thickness in cm (default 0.005 cm = 50 µm).

    Returns
    -------
    float
        Dissolution rate in mg/h (non-negative).
    """
    driving_force = max(0.0, cs_mg_mL - c_dissolved_mg_mL)
    if surface_area_cm2 <= 0.0 or h_cm <= 0.0:
        return 0.0
    return D_cm2_h * surface_area_cm2 * driving_force / h_cm


def simulate_dissolution_absorption(
    drug_name: str,
    dose_mg: float,
    cs_mg_mL: float,
    particle_radius_um: float,
    rho_g_cm3: float = 1.2,
    peff_cm_s: float = 1e-4,
    vd_L: float = 50.0,
    cl_L_per_h: float = 5.0,
    v_gi_mL: float = 250.0,
    t_end_h: float = 12.0,
    dt_h: float = 0.05,
) -> DissolutionAbsorptionResult:
    """Simulate dissolution-rate limited absorption with coupled 1-cpt plasma PK.

    Three coupled ODEs (forward Euler):
    1. dm_undissolved/dt = -dissolution_rate  (shrinking sphere)
    2. dC_lumen/dt = (dissolution_rate - absorption_rate) / V_gi
    3. dCp/dt = (absorption_rate - ke * Cp * Vd) / Vd

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Total dose in mg (> 0).
    cs_mg_mL:
        Saturation solubility in mg/mL (> 0).
    particle_radius_um:
        Initial particle radius in µm (> 0).
    rho_g_cm3:
        Particle density in g/cm³.
    peff_cm_s:
        Effective intestinal permeability in cm/s.
    vd_L:
        Volume of distribution in L.
    cl_L_per_h:
        Systemic clearance in L/h.
    v_gi_mL:
        GI lumen volume in mL.
    t_end_h:
        Simulation duration in hours.
    dt_h:
        Time step in hours.

    Returns
    -------
    DissolutionAbsorptionResult
    """
    if dose_mg <= 0.0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cs_mg_mL <= 0.0:
        raise ValueError(f"cs_mg_mL must be > 0, got {cs_mg_mL}")
    if particle_radius_um <= 0.0:
        raise ValueError(f"particle_radius_um must be > 0, got {particle_radius_um}")
    if v_gi_mL <= 0.0:
        raise ValueError(f"v_gi_mL must be > 0, got {v_gi_mL}")

    # --- Pre-compute constants ---
    # Diffusion coefficient: use Stokes-Einstein estimate in water at 37°C
    # D ~ 5e-6 cm²/s → convert to cm²/h
    D_cm2_h = 5e-6 * 3600.0  # 0.018 cm²/h

    # Diffusion layer thickness
    h_cm = 0.005  # 50 µm

    # Density in mg/cm³
    rho_mg_cm3 = rho_g_cm3 * 1000.0

    # Particle radius in cm
    r0_cm = particle_radius_um * 1e-4  # cm

    # Volume and mass of a single particle
    vol_per_particle_cm3 = (4.0 / 3.0) * math.pi * r0_cm**3
    mass_per_particle_mg = vol_per_particle_cm3 * rho_mg_cm3

    if mass_per_particle_mg <= 0.0:
        raise ValueError("Computed mass_per_particle_mg is zero (particle_radius_um too small).")

    # Number of particles
    n_particles = dose_mg / mass_per_particle_mg

    # Absorption rate constant from GI lumen: peff * absorption_area / V_gi
    # absorption area ~ 200 cm² (small intestinal effective absorptive area)
    absorption_area_cm2 = 200.0
    # peff in cm/s → cm/h
    peff_cm_h = peff_cm_s * 3600.0
    # ka_eff [1/h] such that absorption_rate = ka_eff * M_lumen (mg/h)
    # C_lumen [mg/mL], absorption_rate [mg/h] = peff_cm_h * absorption_area_cm2 * C_lumen
    # (peff in cm/h, area in cm², C in mg/mL = mg/cm³ → rate in mg/h)

    # Elimination rate constant
    ke = cl_L_per_h / vd_L  # /h

    # --- State initialisation ---
    m_undissolved = dose_mg  # mg solid
    c_lumen = 0.0  # mg/mL dissolved in GI
    c_plasma = 0.0  # mg/L in plasma
    m_absorbed = 0.0  # cumulative absorbed mg

    n_steps = max(int(round(t_end_h / dt_h)), 1)

    times_h: list[float] = [0.0] * (n_steps + 1)
    m_undiss_arr: list[float] = [0.0] * (n_steps + 1)
    c_lumen_arr: list[float] = [0.0] * (n_steps + 1)
    c_plasma_arr: list[float] = [0.0] * (n_steps + 1)

    m_undiss_arr[0] = m_undissolved
    c_lumen_arr[0] = c_lumen
    c_plasma_arr[0] = c_plasma

    for i in range(n_steps):
        # --- Current particle radius (shrinking sphere) ---
        if m_undissolved > 1e-9:
            mass_per_p = m_undissolved / n_particles
            # r = (3 * m / (4*pi*rho)) ^ (1/3)
            r_cm = (mass_per_p / (rho_mg_cm3 * (4.0 / 3.0) * math.pi)) ** (1.0 / 3.0)
            surface_area_cm2 = 4.0 * math.pi * r_cm**2 * n_particles
        else:
            surface_area_cm2 = 0.0

        # --- Dissolution rate [mg/h] ---
        diss_rate_mg_h = dissolution_rate(cs_mg_mL, c_lumen, surface_area_cm2, D_cm2_h, h_cm)

        # --- Absorption rate from GI lumen [mg/h] ---
        # rate = peff * area * C_lumen
        abs_rate_mg_h = peff_cm_h * absorption_area_cm2 * c_lumen

        # --- Plasma ODE: dCp/dt = (abs_rate / Vd_mL_equiv) - ke * Cp ---
        # abs_rate_mg_h / vd_L = mg/(L·h) = concentration rate in plasma (mg/L/h)
        # But Cp is in mg/L, so:
        dCp_dt = abs_rate_mg_h / vd_L - ke * c_plasma

        # --- Forward Euler updates ---
        diss_mg = min(diss_rate_mg_h * dt_h, m_undissolved)
        abs_mg = min(abs_rate_mg_h * dt_h, c_lumen * v_gi_mL)
        abs_mg = max(0.0, abs_mg)

        m_undissolved = max(0.0, m_undissolved - diss_mg)
        # c_lumen update: gain from dissolution, loss from absorption
        c_lumen = max(0.0, c_lumen + (diss_mg - abs_mg) / v_gi_mL)
        c_plasma = max(0.0, c_plasma + dCp_dt * dt_h)
        m_absorbed += abs_mg

        times_h[i + 1] = (i + 1) * dt_h
        m_undiss_arr[i + 1] = m_undissolved
        c_lumen_arr[i + 1] = c_lumen
        c_plasma_arr[i + 1] = c_plasma

    # --- Derived PK metrics ---
    cmax = max(c_plasma_arr)
    tmax_h = times_h[c_plasma_arr.index(cmax)]

    # Trapezoidal AUC
    auc = 0.0
    for j in range(n_steps):
        auc += 0.5 * (c_plasma_arr[j] + c_plasma_arr[j + 1]) * dt_h

    fa = min(1.0, m_absorbed / dose_mg) if dose_mg > 0 else 0.0
    dissolution_limited = fa < 0.8

    notes = (
        f"Noyes-Whitney shrinking sphere + 1-cpt PK; "
        f"r0={particle_radius_um} um; Cs={cs_mg_mL} mg/mL; "
        f"peff={peff_cm_s:.2e} cm/s; "
        f"fa={fa:.3f}; dissolution_limited={dissolution_limited}"
    )

    return DissolutionAbsorptionResult(
        drug_name=drug_name,
        times_h=times_h,
        m_undissolved_mg=m_undiss_arr,
        c_lumen_mg_mL=c_lumen_arr,
        c_plasma_mg_L=c_plasma_arr,
        fa=fa,
        cmax=cmax,
        tmax_h=tmax_h,
        auc=auc,
        dissolution_limited=dissolution_limited,
        notes=notes,
    )


def particle_size_effect(
    drug_name: str,
    dose_mg: float,
    cs_mg_mL: float,
    radii_um: list[float],
    peff_cm_s: float = 1e-4,
    **kwargs,
) -> list[dict]:
    """Assess particle size effect on dissolution-limited absorption.

    Parameters
    ----------
    drug_name:
        Drug name.
    dose_mg:
        Dose in mg.
    cs_mg_mL:
        Saturation solubility in mg/mL.
    radii_um:
        List of particle radii in µm to evaluate.
    peff_cm_s:
        Effective permeability in cm/s.
    **kwargs:
        Additional keyword arguments forwarded to
        :func:`simulate_dissolution_absorption`.

    Returns
    -------
    list[dict]
        Each dict contains: radius_um, fa, cmax, auc, dissolution_limited.
        Sorted by radius_um ascending.
    """
    results: list[dict] = []
    for r in radii_um:
        res = simulate_dissolution_absorption(
            drug_name=drug_name,
            dose_mg=dose_mg,
            cs_mg_mL=cs_mg_mL,
            particle_radius_um=r,
            peff_cm_s=peff_cm_s,
            **kwargs,
        )
        results.append(
            {
                "radius_um": r,
                "fa": res.fa,
                "cmax": res.cmax,
                "auc": res.auc,
                "dissolution_limited": res.dissolution_limited,
            }
        )
    return sorted(results, key=lambda d: d["radius_um"])
