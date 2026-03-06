"""Pulmonary drug delivery PBPK model for inhaled aerosol/dry powder formulations.

Three-compartment model:
  1. Airway depot: receives inhaled dose x deposition fraction
  2. Lung tissue: absorbs from airway, transfers to plasma
  3. Plasma: systemic distribution and elimination
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz


@dataclass(frozen=True)
class PulmonaryPKResult:
    drug_name: str
    particle_size_um: float
    dose_mg: float
    route: str
    times_h: list[float]
    c_airway_mg_L: list[float]
    c_lung_tissue_mg_L: list[float]
    c_plasma_mg_L: list[float]
    cmax_airway: float
    cmax_lung_tissue: float
    cmax_plasma: float
    auc_lung_tissue: float
    auc_plasma: float
    lung_to_plasma_ratio: float
    pulmonary_bioavailability: float
    notes: str


def simulate_pulmonary_pk(
    drug_name: str,
    dose_mg: float,
    route: str = "inhaled",
    particle_size_um: float = 3.0,
    lung_deposition_fraction: float = 0.3,
    mucociliary_clearance_per_h: float = 0.5,
    k_lung_abs_per_h: float = 1.0,
    cl_sys_L_per_h: float = 1.0,
    vd_sys_L: float = 50.0,
    t_end_h: float = 12.0,
    dt_h: float = 0.01,
) -> PulmonaryPKResult:
    """Run a Forward-Euler pulmonary PBPK simulation."""

    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if particle_size_um <= 0:
        raise ValueError("particle_size_um must be > 0")
    if k_lung_abs_per_h <= 0:
        raise ValueError("k_lung_abs_per_h must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")

    # --- Particle-size effect on deposition ---
    if particle_size_um > 5.0:
        eff_deposition = lung_deposition_fraction * (3.0 / particle_size_um) ** 2
    elif particle_size_um < 1.0:
        eff_deposition = lung_deposition_fraction * 0.5
    else:
        eff_deposition = lung_deposition_fraction

    # --- Compartment volumes (nominal, for concentration calculation) ---
    v_airway = 0.1   # L  (airway lining fluid ~100 mL)
    v_lung = 1.0     # L  (lung tissue volume)
    v_plasma = vd_sys_L

    ke_sys = cl_sys_L_per_h / v_plasma

    n_steps = max(int(t_end_h / dt_h), 1)
    t = np.linspace(0.0, t_end_h, n_steps + 1)

    # --- State: amounts in mg ---
    a_airway = np.zeros(n_steps + 1)
    a_lung = np.zeros(n_steps + 1)
    a_plasma = np.zeros(n_steps + 1)

    # Initial: deposited dose enters airway
    a_airway[0] = dose_mg * eff_deposition

    total_absorbed_to_plasma = 0.0

    for i in range(n_steps):
        # Airway -> cleared (mucociliary) + absorbed to lung tissue
        loss_muco = mucociliary_clearance_per_h * a_airway[i] * dt_h
        transfer_to_lung = k_lung_abs_per_h * a_airway[i] * dt_h

        da_airway = -(loss_muco + transfer_to_lung)

        # Lung tissue -> plasma
        transfer_to_plasma = k_lung_abs_per_h * a_lung[i] * dt_h
        da_lung = transfer_to_lung - transfer_to_plasma

        # Plasma: systemic elimination
        elim = ke_sys * a_plasma[i] * dt_h
        da_plasma = transfer_to_plasma - elim

        total_absorbed_to_plasma += transfer_to_plasma

        a_airway[i + 1] = max(0.0, a_airway[i] + da_airway)
        a_lung[i + 1] = max(0.0, a_lung[i] + da_lung)
        a_plasma[i + 1] = max(0.0, a_plasma[i] + da_plasma)

    # --- Convert to concentrations ---
    c_airway = a_airway / v_airway
    c_lung = a_lung / v_lung
    c_plasma = a_plasma / v_plasma

    cmax_airway = float(np.max(c_airway))
    cmax_lung = float(np.max(c_lung))
    cmax_plasma = float(np.max(c_plasma))

    auc_lung = float(np_trapz(c_lung, t))
    auc_plasma = float(np_trapz(c_plasma, t))

    lung_to_plasma = cmax_lung / cmax_plasma if cmax_plasma > 0 else float("inf")
    pulm_bioavail = total_absorbed_to_plasma / dose_mg if dose_mg > 0 else 0.0

    notes = (
        f"Pulmonary PK: {drug_name}, {dose_mg}mg {route}. "
        f"Particle size={particle_size_um}um, deposition={eff_deposition:.3f}. "
        f"Cmax_plasma={cmax_plasma:.4f} mg/L, AUC_plasma={auc_plasma:.4f} mg*h/L. "
        f"Lung:plasma ratio={lung_to_plasma:.2f}."
    )

    return PulmonaryPKResult(
        drug_name=drug_name,
        particle_size_um=particle_size_um,
        dose_mg=dose_mg,
        route=route,
        times_h=t.tolist(),
        c_airway_mg_L=c_airway.tolist(),
        c_lung_tissue_mg_L=c_lung.tolist(),
        c_plasma_mg_L=c_plasma.tolist(),
        cmax_airway=cmax_airway,
        cmax_lung_tissue=cmax_lung,
        cmax_plasma=cmax_plasma,
        auc_lung_tissue=auc_lung,
        auc_plasma=auc_plasma,
        lung_to_plasma_ratio=lung_to_plasma,
        pulmonary_bioavailability=pulm_bioavail,
        notes=notes,
    )


def compare_particle_sizes(
    drug_name: str,
    dose_mg: float,
    particle_sizes_um: list[float],
    **kwargs,
) -> list[PulmonaryPKResult]:
    """Run simulations across several particle sizes and return the results."""
    return [
        simulate_pulmonary_pk(drug_name, dose_mg, particle_size_um=ps, **kwargs)
        for ps in particle_sizes_um
    ]


__all__ = [
    "PulmonaryPKResult",
    "simulate_pulmonary_pk",
    "compare_particle_sizes",
]
