"""
Phase 636 — Drug Tumor Vascular Extravasation

Models drug extravasation from tumor vasculature into tumor interstitium,
including EPR effect and vascular permeability for different particle types.
"""

import math
from dataclasses import dataclass

_VALID_PARTICLE_TYPES = {"small_molecule", "nanoparticle", "antibody"}


@dataclass
class TumorExtravasationResult:
    drug_name: str
    dose_mg: float
    particle_type: str
    times_h: list
    c_plasma_mg_L: list
    c_tumor_vasculature_mg_L: list
    c_tumor_interstitium_mg_g: list
    cmax_plasma_mg_L: float
    cmax_tumor_mg_g: float
    auc_plasma_mg_h_per_L: float
    auc_tumor_mg_h_per_g: float
    epr_enhancement_factor: float
    tumor_accumulation_pct: float
    vascular_permeability_coeff: float
    notes: str


def simulate_tumor_extravasation(
    drug_name: str,
    dose_mg: float,
    particle_type: str,
    mw_Da: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    tumor_weight_g: float = 5.0,
    tumor_vascular_fraction: float = 0.1,
    t_end_h: float = 72.0,
    dt_h: float = 0.2,
) -> TumorExtravasationResult:
    """Simulate drug extravasation from tumor vasculature into tumor interstitium."""
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    if tumor_weight_g <= 0:
        raise ValueError("tumor_weight_g must be > 0")
    if particle_type not in _VALID_PARTICLE_TYPES:
        raise ValueError(f"particle_type must be one of {sorted(_VALID_PARTICLE_TYPES)}")

    # Vascular permeability (cm/s) by particle type
    if particle_type == "small_molecule":
        P_base = 1e-5
    elif particle_type == "antibody":
        P_base = 1e-7
    else:  # nanoparticle
        P_base = 5e-7

    # MW scaling: larger molecules have slightly lower permeability
    P = P_base * math.sqrt(500.0 / max(500.0, mw_Da))

    # Tumor blood flow
    Qv = 0.02  # L/h

    # Compartment volumes
    V_int_L = tumor_weight_g * 0.4 / 1000.0 * (1.0 - tumor_vascular_fraction)
    V_vasc_L = tumor_weight_g * tumor_vascular_fraction / 1000.0

    # Permeability-area product (L/h)
    PA = P * 3600.0 * tumor_weight_g * 100.0 / 1000.0

    # Transfer rates
    k_extrav = PA / V_vasc_L  # /h (vascular → interstitium)
    k_reabs = PA / V_int_L / 5.0  # /h (interstitium → vascular, slower due to elevated IFP)

    # Initial conditions (IV bolus)
    C_plasma = dose_mg / vd_sys_L
    C_vasc = C_plasma * 0.1
    C_int = 0.0  # mg/L interstitial concentration

    # Simulation
    n_steps = int(round(t_end_h / dt_h))

    times_h = [0.0]
    c_plasma_list = [C_plasma]
    c_vasc_list = [C_vasc]
    c_int_list = [C_int * V_int_L * 1000.0 / tumor_weight_g]

    for _ in range(n_steps):
        # Systemic PK — simplified (systemic clearance dominates)
        dC_plasma = -(cl_sys_L_per_h / vd_sys_L) * C_plasma

        # Tumor vascular compartment
        dC_vasc = (
            Qv / V_vasc_L * (C_plasma - C_vasc)
            - k_extrav * C_vasc
            + k_reabs * C_int * (V_int_L / V_vasc_L)
        )

        # Tumor interstitium
        dC_int = (
            k_extrav * C_vasc * (V_vasc_L / V_int_L)
            - k_reabs * C_int
            - 0.02 * C_int  # lymphatic drainage
        )

        C_plasma = max(0.0, C_plasma + dC_plasma * dt_h)
        C_vasc = max(0.0, C_vasc + dC_vasc * dt_h)
        C_int = max(0.0, C_int + dC_int * dt_h)

        times_h.append(times_h[-1] + dt_h)
        c_plasma_list.append(C_plasma)
        c_vasc_list.append(C_vasc)
        c_int_list.append(C_int * V_int_L * 1000.0 / tumor_weight_g)

    # Compute metrics
    cmax_plasma = max(c_plasma_list)
    cmax_tumor = max(c_int_list)

    auc_plasma = sum(
        0.5 * (c_plasma_list[i] + c_plasma_list[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )
    auc_tumor = sum(
        0.5 * (c_int_list[i] + c_int_list[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )

    # EPR enhancement factor: AUC_tumor / AUC_normal_tissue_proxy
    # Normal tissue proxy = 0.5 * AUC_plasma (in mg*h/g units)
    normal_tissue_auc = 0.5 * auc_plasma / 1000.0  # convert mg/L -> mg/g proxy
    if normal_tissue_auc > 0:
        epr_factor = auc_tumor / normal_tissue_auc
    else:
        epr_factor = 1.0
    epr_factor = max(0.1, min(100.0, epr_factor))

    # Tumor accumulation %: relative to total dose AUC
    total_dose_auc = dose_mg / cl_sys_L_per_h  # mg*h/L — systemic exposure
    if total_dose_auc > 0:
        tumor_accum_pct = auc_tumor / (total_dose_auc / 1000.0) * 100.0
    else:
        tumor_accum_pct = 0.0

    notes = (
        f"Tumor extravasation simulation for {drug_name} ({particle_type}, "
        f"MW={mw_Da} Da). "
        f"Permeability coeff P={P:.2e} cm/s. "
        f"EPR enhancement: {epr_factor:.2f}x. "
        f"Tumor accumulation: {tumor_accum_pct:.2f}% of dose AUC."
    )

    return TumorExtravasationResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        particle_type=particle_type,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_list,
        c_tumor_vasculature_mg_L=c_vasc_list,
        c_tumor_interstitium_mg_g=c_int_list,
        cmax_plasma_mg_L=cmax_plasma,
        cmax_tumor_mg_g=cmax_tumor,
        auc_plasma_mg_h_per_L=auc_plasma,
        auc_tumor_mg_h_per_g=auc_tumor,
        epr_enhancement_factor=epr_factor,
        tumor_accumulation_pct=tumor_accum_pct,
        vascular_permeability_coeff=P,
        notes=notes,
    )
