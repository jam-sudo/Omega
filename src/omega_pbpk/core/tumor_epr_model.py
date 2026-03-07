"""Phase 370 — Tumor Vasculature Permeability Model (EPR Effect).

Models the Enhanced Permeability and Retention (EPR) effect in tumor vasculature
for nanoparticle or large molecule delivery using a 3-compartment ODE system.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class TumorEPRResult:
    """Result of tumor EPR effect simulation."""

    times_h: list
    c_systemic_mg_L: list
    c_tumor_mg_mL: list
    auc_systemic_mg_h_per_L: float
    auc_tumor_mg_h_per_mL: float
    cmax_tumor_mg_mL: float
    tmax_tumor_h: float
    epr_factor: float
    tumor_to_systemic_ratio: float
    total_tumor_dose_pct: float
    epr_class: str
    notes: str


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid function."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1.0 + exp_x)


def _trapezoidal_auc(x: list, y: list) -> float:
    """Manual trapezoidal AUC integration."""
    return sum(0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]) for i in range(1, len(x)))


def _compute_epr_factor(particle_size_nm: float) -> float:
    """Compute EPR factor based on particle size.

    Parameters
    ----------
    particle_size_nm:
        Effective particle size in nanometers.

    Returns
    -------
    float
        EPR factor in [0, 1].
    """
    if particle_size_nm < 8.0:
        # Below renal threshold — no EPR
        return 0.0
    elif particle_size_nm <= 200.0:
        # Optimal EPR window: sigmoid scaling
        return _sigmoid((particle_size_nm - 8.0) / 50.0)
    else:
        # Too large to extravasate
        return 0.0


def simulate_tumor_epr(
    drug_name: str,
    dose_mg: float,
    mw_kDa: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    k_epr_per_h: float,
    k_tumor_clearance_per_h: float,
    particle_size_nm: float | None = None,
    tumor_volume_mL: float = 1.0,
    normal_tissue_vascular_perm: float = 0.1,
    t_end_h: float = 24.0,
    dt_h: float = 0.01,
) -> TumorEPRResult:
    """Simulate EPR-mediated tumor drug accumulation using 3-compartment Forward Euler model.

    Parameters
    ----------
    drug_name:
        Name of the drug or nanoparticle formulation.
    dose_mg:
        IV bolus dose in mg.
    mw_kDa:
        Molecular weight in kDa (used if particle_size_nm is None or 0).
    cl_sys_L_per_h:
        Systemic clearance in L/h.
    vd_sys_L:
        Volume of distribution (systemic) in L.
    k_epr_per_h:
        EPR vascular permeation rate constant in h⁻¹.
    k_tumor_clearance_per_h:
        Tumor lymphatic/clearance rate constant in h⁻¹.
    particle_size_nm:
        Particle/molecule effective diameter in nm. If None or 0, estimated from mw_kDa.
    tumor_volume_mL:
        Tumor volume in mL.
    normal_tissue_vascular_perm:
        Normal tissue vascular permeability (0-1) relative to k_epr.
    t_end_h:
        Total simulation time in hours.
    dt_h:
        Forward Euler time step in hours.

    Returns
    -------
    TumorEPRResult
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    if tumor_volume_mL <= 0:
        raise ValueError("tumor_volume_mL must be > 0")
    if mw_kDa <= 0:
        raise ValueError("mw_kDa must be > 0")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be > 0")

    # --- Particle size and EPR factor ---
    if particle_size_nm is None or particle_size_nm <= 0:
        # Rough size estimation from molecular weight
        particle_size_nm = 0.3 * (mw_kDa**0.4) * 10.0

    epr_factor = _compute_epr_factor(particle_size_nm)

    # --- Rate constants ---
    ke_sys = cl_sys_L_per_h / vd_sys_L  # systemic elimination rate (h⁻¹)
    # Effective EPR rate incorporates EPR factor
    k_epr_effective = k_epr_per_h * epr_factor

    # Normal tissue leakage (informational, not tracked as separate compartment)
    # k_normal_leak = normal_tissue_vascular_perm * k_epr_per_h  # noqa: F841

    # --- Initial conditions ---
    # IV bolus: initial systemic concentration (mg/L)
    C_sys = dose_mg / vd_sys_L  # mg/L
    C_tumor = 0.0  # mg/mL

    n_steps = max(1, int(t_end_h / dt_h))
    dt_actual = t_end_h / n_steps

    times_h: list = [0.0]
    c_systemic_mg_L: list = [C_sys]
    c_tumor_mg_mL: list = [C_tumor]

    # Track Cmax_tumor
    cmax_tumor = C_tumor
    tmax_tumor = 0.0

    for i in range(1, n_steps + 1):
        t_now = i * dt_actual

        # Systemic compartment ODE:
        # dC_sys/dt = -ke_sys * C_sys - k_epr_effective * C_sys
        dC_sys_dt = -(ke_sys + k_epr_effective) * C_sys

        # Tumor interstitium ODE:
        # EPR flux: k_epr_effective * C_sys * vd_sys / tumor_volume (mg/mL/h)
        # The C_sys is in mg/L, vd_sys in L, tumor_volume in mL:
        # flux_mg_per_h = k_epr_effective * C_sys [mg/L] * vd_sys [L] = mg/h
        # concentration rate = flux / tumor_volume_mL [mg/mL/h]
        epr_flux_conc = k_epr_effective * C_sys * vd_sys_L / tumor_volume_mL
        dC_tumor_dt = epr_flux_conc - k_tumor_clearance_per_h * C_tumor

        C_sys_new = C_sys + dC_sys_dt * dt_actual
        C_tumor_new = C_tumor + dC_tumor_dt * dt_actual

        # Clamp to physical bounds
        C_sys = max(0.0, C_sys_new)
        C_tumor = max(0.0, C_tumor_new)

        times_h.append(t_now)
        c_systemic_mg_L.append(C_sys)
        c_tumor_mg_mL.append(C_tumor)

        if C_tumor > cmax_tumor:
            cmax_tumor = C_tumor
            tmax_tumor = t_now

    # --- AUC via trapezoidal integration ---
    auc_systemic = _trapezoidal_auc(times_h, c_systemic_mg_L)
    auc_tumor = _trapezoidal_auc(times_h, c_tumor_mg_mL)

    # Cmax systemic (IV bolus = initial concentration)
    cmax_systemic = c_systemic_mg_L[0]

    # tumor-to-systemic ratio based on Cmax
    if cmax_systemic > 1e-15:
        tumor_to_systemic_ratio = cmax_tumor / cmax_systemic
    else:
        tumor_to_systemic_ratio = 0.0

    # Total tumor dose as percentage of dose
    # Mass in tumor at each time = C_tumor [mg/mL] * tumor_volume_mL [mL] = mg
    # Total tumor exposure = AUC_tumor * tumor_volume_mL * k_tumor_clearance [mass cleared]
    # Simpler: integrate drug flux into tumor = k_epr_eff * C_sys * vd_sys dt
    # Approximate: total mass delivered = AUC_systemic * k_epr_effective * vd_sys_L [mg]
    total_mass_delivered_mg = auc_systemic * k_epr_effective * vd_sys_L
    total_tumor_dose_pct = (total_mass_delivered_mg / dose_mg * 100.0) if dose_mg > 0 else 0.0
    total_tumor_dose_pct = max(0.0, min(100.0, total_tumor_dose_pct))

    # EPR class
    if epr_factor == 0.0:
        epr_class = "absent"
    elif epr_factor <= 0.5:
        epr_class = "moderate"
    else:
        epr_class = "strong"

    # Notes
    size_str = f"{particle_size_nm:.1f} nm"
    notes = (
        f"drug={drug_name}; particle_size={size_str}; EPR_factor={epr_factor:.3f}; "
        f"epr_class={epr_class}; Cmax_tumor={cmax_tumor:.4f} mg/mL at {tmax_tumor:.2f}h; "
        f"AUC_sys={auc_systemic:.2f} mg*h/L; AUC_tumor={auc_tumor:.4f} mg*h/mL"
    )

    return TumorEPRResult(
        times_h=times_h,
        c_systemic_mg_L=c_systemic_mg_L,
        c_tumor_mg_mL=c_tumor_mg_mL,
        auc_systemic_mg_h_per_L=auc_systemic,
        auc_tumor_mg_h_per_mL=auc_tumor,
        cmax_tumor_mg_mL=cmax_tumor,
        tmax_tumor_h=tmax_tumor,
        epr_factor=epr_factor,
        tumor_to_systemic_ratio=tumor_to_systemic_ratio,
        total_tumor_dose_pct=total_tumor_dose_pct,
        epr_class=epr_class,
        notes=notes,
    )
