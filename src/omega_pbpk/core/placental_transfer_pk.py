"""Placental transfer PK model (Phase 730).

Models drug transfer from maternal plasma to fetal plasma through the
placental barrier. Relevant for teratogenicity risk assessment during
pregnancy drug exposure.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PlacentalTransferResult:
    """Result of placental transfer PK simulation."""

    drug_name: str
    dose_mg: float
    times_h: list
    c_maternal_mg_L: list
    c_fetal_mg_L: list
    cmax_maternal: float
    cmax_fetal: float
    tmax_fetal_h: float
    auc_maternal: float
    auc_fetal: float
    fetal_to_maternal_auc_ratio: float
    fetal_to_maternal_cmax_ratio: float
    ps_actual_L_per_h: float
    teratogenicity_concern: str
    notes: str


def estimate_placental_ps(
    mw_da: float,
    logp: float,
    ps_coeff_L_per_h: float,
) -> float:
    """Estimate placental permeability-surface area product (PS).

    PS is reduced for high molecular weight compounds and for very hydrophilic
    or very lipophilic drugs. Moderate logP (around 2-3) is optimal for
    transcellular diffusion.

    Args:
        mw_da: Molecular weight in Daltons.
        logp: Octanol-water partition coefficient.
        ps_coeff_L_per_h: Base PS coefficient in L/h.

    Returns:
        Estimated PS in L/h (always > 0 for any non-zero ps_coeff).
    """
    mw_factor = max(0.1, 1.0 - (mw_da - 300.0) / 1000.0)
    logp_factor = max(0.1, min(1.0, (logp + 2.0) / 4.0))
    return ps_coeff_L_per_h * mw_factor * logp_factor


def simulate_placental_transfer(
    drug_name: str = "Drug",
    dose_mg: float = 100.0,
    mw_da: float = 300.0,
    logp: float = 2.0,
    cl_maternal_L_per_h: float = 5.0,
    cl_fetal_L_per_h: float = 0.5,
    vd_maternal_L: float = 50.0,
    vd_fetal_L: float = 5.0,
    ps_coeff_L_per_h: float = 0.5,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> PlacentalTransferResult:
    """Simulate maternal-fetal PK via placental transfer using forward Euler.

    Models IV bolus into maternal compartment with bidirectional placental
    diffusion into fetal compartment.

    Args:
        drug_name: Name of the drug.
        dose_mg: IV bolus dose into maternal compartment in mg.
        mw_da: Molecular weight in Daltons (affects placental PS).
        logp: Log octanol-water partition coefficient (affects placental PS).
        cl_maternal_L_per_h: Maternal clearance in L/h.
        cl_fetal_L_per_h: Fetal clearance in L/h.
        vd_maternal_L: Maternal volume of distribution in L.
        vd_fetal_L: Fetal volume of distribution in L.
        ps_coeff_L_per_h: Base placental PS coefficient in L/h.
        t_end_h: Simulation end time in hours.
        dt_h: Forward Euler time step in hours.

    Returns:
        PlacentalTransferResult with maternal/fetal PK profiles and risk assessment.

    Raises:
        ValueError: If dose_mg <= 0, vd_maternal_L <= 0, vd_fetal_L <= 0,
                    or cl_maternal_L_per_h <= 0.
    """
    if dose_mg <= 0.0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if vd_maternal_L <= 0.0:
        raise ValueError(f"vd_maternal_L must be > 0, got {vd_maternal_L}")
    if vd_fetal_L <= 0.0:
        raise ValueError(f"vd_fetal_L must be > 0, got {vd_fetal_L}")
    if cl_maternal_L_per_h <= 0.0:
        raise ValueError(f"cl_maternal_L_per_h must be > 0, got {cl_maternal_L_per_h}")

    # Estimate actual PS based on physicochemical properties
    ps_actual = estimate_placental_ps(mw_da, logp, ps_coeff_L_per_h)

    # Rate constants
    ke_mat = cl_maternal_L_per_h / vd_maternal_L
    ke_fet = cl_fetal_L_per_h / vd_fetal_L

    # Initial conditions: IV bolus into maternal compartment
    c_mat = dose_mg / vd_maternal_L  # mg/L
    c_fet = 0.0  # mg/L

    times_h: list = []
    c_maternal_list: list = []
    c_fetal_list: list = []

    t = 0.0
    n_steps = int(round(t_end_h / dt_h)) + 1

    cmax_fetal = 0.0
    tmax_fetal_h = 0.0

    for _ in range(n_steps):
        times_h.append(t)
        c_maternal_list.append(c_mat)
        c_fetal_list.append(c_fet)

        # Track fetal Cmax
        if c_fet > cmax_fetal:
            cmax_fetal = c_fet
            tmax_fetal_h = t

        # Net placental flux: from maternal to fetal (concentration-driven)
        # flux = PS * (C_mat - C_fet) in mg/h (PS in L/h, C in mg/L => mg/h)
        net_flux = ps_actual * (c_mat - c_fet)

        # ODEs
        dc_mat = -ke_mat * c_mat - net_flux / vd_maternal_L
        dc_fet = net_flux / vd_fetal_L - ke_fet * c_fet

        c_mat = max(0.0, c_mat + dc_mat * dt_h)
        c_fet = max(0.0, c_fet + dc_fet * dt_h)

        t += dt_h

    # Cmax maternal = initial concentration (IV bolus)
    cmax_maternal = c_maternal_list[0]

    # AUC via trapezoidal integration
    auc_maternal = 0.0
    auc_fetal = 0.0
    for i in range(1, len(times_h)):
        dt = times_h[i] - times_h[i - 1]
        auc_maternal += 0.5 * (c_maternal_list[i - 1] + c_maternal_list[i]) * dt
        auc_fetal += 0.5 * (c_fetal_list[i - 1] + c_fetal_list[i]) * dt

    # Ratios
    fetal_to_maternal_auc_ratio = auc_fetal / auc_maternal if auc_maternal > 0.0 else 0.0
    fetal_to_maternal_cmax_ratio = cmax_fetal / cmax_maternal if cmax_maternal > 0.0 else 0.0

    # Teratogenicity concern classification
    if fetal_to_maternal_auc_ratio < 0.1:
        teratogenicity_concern = "low"
    elif fetal_to_maternal_auc_ratio <= 0.5:
        teratogenicity_concern = "moderate"
    else:
        teratogenicity_concern = "high"

    notes = (
        f"Placental transfer PK for {drug_name}. "
        f"MW={mw_da:.0f} Da, logP={logp:.1f}, PS_actual={ps_actual:.3f} L/h. "
        f"Fetal/maternal AUC ratio={fetal_to_maternal_auc_ratio:.3f} "
        f"({teratogenicity_concern} teratogenicity concern). "
        f"Tmax_fetal={tmax_fetal_h:.1f} h."
    )

    return PlacentalTransferResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times_h,
        c_maternal_mg_L=c_maternal_list,
        c_fetal_mg_L=c_fetal_list,
        cmax_maternal=cmax_maternal,
        cmax_fetal=cmax_fetal,
        tmax_fetal_h=tmax_fetal_h,
        auc_maternal=auc_maternal,
        auc_fetal=auc_fetal,
        fetal_to_maternal_auc_ratio=fetal_to_maternal_auc_ratio,
        fetal_to_maternal_cmax_ratio=fetal_to_maternal_cmax_ratio,
        ps_actual_L_per_h=ps_actual,
        teratogenicity_concern=teratogenicity_concern,
        notes=notes,
    )
