"""Phase 610 — Drug Distribution in Adipose Tissue.

Models drug distribution and accumulation in adipose tissue, particularly for
lipophilic compounds, using a 2-compartment model with a central and adipose
peripheral compartment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class AdiposeDrugDistributionResult:
    """Result of adipose tissue drug distribution simulation."""

    drug_name: str
    dose_mg: float
    bmi: float
    times_h: list
    c_plasma_mg_L: list
    c_adipose_mg_kg: list
    mass_adipose_mg: list
    cmax_plasma_mg_L: float
    tmax_plasma_h: float
    auc_plasma_mg_h_per_L: float
    t_half_effective_h: float
    adipose_accumulation_ratio: float
    redistribution_phase: bool
    notes: str


def simulate_adipose_distribution(
    drug_name: str,
    dose_mg: float,
    logP: float,
    cl_sys_L_per_h: float,
    vd_central_L: float,
    bmi: float = 25.0,
    age_years: float = 40.0,
    sex: str = "male",
    t_end_h: float = 72.0,
    dt_h: float = 0.1,
) -> AdiposeDrugDistributionResult:
    """Simulate drug distribution into adipose tissue.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Oral dose in milligrams.
    logP:
        Octanol-water partition coefficient (log units).
    cl_sys_L_per_h:
        Systemic clearance (L/h).
    vd_central_L:
        Central compartment volume of distribution (L).
    bmi:
        Body mass index (kg/m^2).
    age_years:
        Age in years.
    sex:
        "male" or "female".
    t_end_h:
        Simulation end time (h).
    dt_h:
        Forward Euler time step (h).

    Returns
    -------
    AdiposeDrugDistributionResult
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_central_L <= 0:
        raise ValueError("vd_central_L must be > 0")
    if bmi <= 0:
        raise ValueError("bmi must be > 0")
    if sex not in ("male", "female"):
        raise ValueError('sex must be "male" or "female"')

    # Fat mass calculation (simplified Deurenberg)
    if sex == "male":
        fat_mass_kg = bmi * 0.2 - 5.0 + age_years * 0.05
        fat_mass_kg = max(1.0, min(60.0, fat_mass_kg))
    else:
        fat_mass_kg = bmi * 0.25 - 2.0 + age_years * 0.07
        fat_mass_kg = max(2.0, min(70.0, fat_mass_kg))

    # Adipose volume (0.9 L/kg for fat tissue)
    vd_adipose_L = fat_mass_kg * 0.9

    # Adipose partition coefficient
    kp_adipose = max(0.01, logP * 2.0 + 1.0)

    # Transfer rates
    k12 = 0.1 * kp_adipose / max(1.0, vd_adipose_L / vd_central_L)  # central → adipose (/h)
    k21 = k12 / kp_adipose  # adipose → central (/h)

    # PK parameters
    f_oral = 0.85
    ka = 1.2  # /h
    ke = cl_sys_L_per_h / vd_central_L  # elimination from central (/h)

    # Initial conditions
    a_gut = dose_mg * f_oral
    c_central = 0.0
    c_adipose = 0.0

    n_steps = int(math.ceil(t_end_h / dt_h))

    times_h = []
    c_plasma_list = []
    c_adipose_mg_kg_list = []
    mass_adipose_list = []

    for step in range(n_steps + 1):
        t = step * dt_h

        mass_ad = c_adipose * vd_adipose_L
        c_ad_per_kg = mass_ad / fat_mass_kg

        times_h.append(t)
        c_plasma_list.append(max(0.0, c_central))
        c_adipose_mg_kg_list.append(max(0.0, c_ad_per_kg))
        mass_adipose_list.append(max(0.0, mass_ad))

        if step == n_steps:
            break

        # Forward Euler
        da_gut = -ka * a_gut
        dc_central = (
            ka * a_gut / vd_central_L
            - ke * c_central
            - k12 * c_central
            + k21 * c_adipose * (vd_adipose_L / vd_central_L)
        )
        dc_adipose = k12 * c_central * (vd_central_L / vd_adipose_L) - k21 * c_adipose

        a_gut = max(0.0, a_gut + da_gut * dt_h)
        c_central = max(0.0, c_central + dc_central * dt_h)
        c_adipose = max(0.0, c_adipose + dc_adipose * dt_h)

    # Derived metrics
    cmax_plasma = max(c_plasma_list)
    tmax_idx = c_plasma_list.index(cmax_plasma)
    tmax_plasma_h = times_h[tmax_idx]

    # AUC by trapezoidal rule
    auc = sum(
        0.5 * (c_plasma_list[i] + c_plasma_list[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )

    # Effective half-life: ln(2) / k_el_eff
    k_el_eff = cl_sys_L_per_h / (vd_central_L + vd_adipose_L / kp_adipose)
    t_half_effective_h = math.log(2.0) / k_el_eff if k_el_eff > 0 else float("inf")

    # Adipose accumulation ratio: peak c_adipose_mg_kg / c_plasma at same time
    peak_adipose = max(c_adipose_mg_kg_list)
    peak_adipose_idx = c_adipose_mg_kg_list.index(peak_adipose)
    c_plasma_at_adipose_peak = c_plasma_list[peak_adipose_idx]
    if c_plasma_at_adipose_peak > 0:
        adipose_accumulation_ratio = peak_adipose / c_plasma_at_adipose_peak
    else:
        adipose_accumulation_ratio = 0.0

    # Redistribution phase: effective t½ > 1.5 * 1-cpt t½
    t_half_1cpt = math.log(2.0) * vd_central_L / cl_sys_L_per_h
    redistribution_phase = t_half_effective_h > 1.5 * t_half_1cpt

    notes = (
        f"{drug_name}: dose={dose_mg} mg, logP={logP:.2f}, BMI={bmi:.1f}, "
        f"sex={sex}, fat_mass={fat_mass_kg:.1f} kg, "
        f"Kp_adipose={kp_adipose:.2f}, "
        f"Cmax={cmax_plasma:.3f} mg/L, AUC={auc:.2f} mg*h/L, "
        f"t½_eff={t_half_effective_h:.1f} h, "
        f"redistribution={'yes' if redistribution_phase else 'no'}."
    )

    return AdiposeDrugDistributionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        bmi=bmi,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_list,
        c_adipose_mg_kg=c_adipose_mg_kg_list,
        mass_adipose_mg=mass_adipose_list,
        cmax_plasma_mg_L=cmax_plasma,
        tmax_plasma_h=tmax_plasma_h,
        auc_plasma_mg_h_per_L=auc,
        t_half_effective_h=t_half_effective_h,
        adipose_accumulation_ratio=adipose_accumulation_ratio,
        redistribution_phase=redistribution_phase,
        notes=notes,
    )
