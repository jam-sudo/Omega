"""Drug placental transfer model.

Models drug transfer across the placenta from maternal to fetal circulation
using a simplified 2-compartment model (maternal plasma + fetal plasma) with
placental barrier characterisation.

Key mechanisms
--------------
- Placental permeability depends on logP and molecular weight
- Fetal compartment parameters scale with gestational age
- Placental surface area scales with gestational age
- Forward Euler ODE integration

References
----------
- Pacifici GM & Nottoli R, Clin Pharmacokinet. 1995;28(3):235-269
- Syme MR et al., Clin Pharmacokinet. 2004;43(8):487-514
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PlacentalTransferResult:
    """Result container for placental transfer simulation."""

    drug_name: str = ""
    dose_mg: float = 0.0
    times_h: list[float] = field(default_factory=list)
    c_maternal_mg_L: list[float] = field(default_factory=list)
    c_fetal_mg_L: list[float] = field(default_factory=list)
    fetal_to_maternal_ratio_auc: float = 0.0
    cmax_maternal_mg_L: float = 0.0
    cmax_fetal_mg_L: float = 0.0
    auc_maternal_mg_h_per_L: float = 0.0
    auc_fetal_mg_h_per_L: float = 0.0
    t50_fetal_h: float = 0.0
    placental_transfer_rate: float = 0.0
    fetal_exposure_category: str = ""
    mw_Da: float = 0.0
    logP: float = 0.0
    notes: str = ""


def _trapezoidal_auc(times: list[float], concs: list[float]) -> float:
    """Manual trapezoidal AUC."""
    return sum(
        0.5 * (concs[i] + concs[i - 1]) * (times[i] - times[i - 1]) for i in range(1, len(times))
    )


def simulate_placental_transfer(
    drug_name: str,
    dose_mg: float,
    logP: float,
    mw_Da: float,
    cl_maternal_L_per_h: float,
    vd_maternal_L: float,
    gestational_age_weeks: float = 28.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> PlacentalTransferResult:
    """Simulate drug transfer across the placenta.

    Parameters
    ----------
    drug_name : str
        Name of the drug compound.
    dose_mg : float
        Oral dose in mg (must be > 0).
    logP : float
        Logarithm of the octanol-water partition coefficient.
    mw_Da : float
        Molecular weight in Daltons.
    cl_maternal_L_per_h : float
        Maternal clearance in L/h (must be > 0).
    vd_maternal_L : float
        Maternal volume of distribution in L (must be > 0).
    gestational_age_weeks : float
        Gestational age in weeks (1-42, default 28).
    t_end_h : float
        Simulation end time in hours (default 24).
    dt_h : float
        Time step in hours (default 0.1).

    Returns
    -------
    PlacentalTransferResult
        Dataclass with simulation results.
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_maternal_L_per_h <= 0:
        raise ValueError("cl_maternal_L_per_h must be > 0")
    if vd_maternal_L <= 0:
        raise ValueError("vd_maternal_L must be > 0")
    if not 1 <= gestational_age_weeks <= 42:
        raise ValueError("gestational_age_weeks must be in [1, 42]")

    # Placental permeability
    p_placenta = 0.1 * 10 ** (logP * 0.4 - mw_Da / 600.0)
    p_placenta = max(0.001, min(2.0, p_placenta))

    # Fetal parameters scaled by gestational age
    ga_ratio = gestational_age_weeks / 40.0
    vd_fetal_L = 0.3 * ga_ratio**2
    cl_fetal_L_per_h = 0.02 * ga_ratio

    # Placental surface area factor
    sa_factor = max(0.1, min(1.0, ga_ratio**1.5))
    k_transfer = p_placenta * sa_factor

    # Oral absorption
    ka = 1.2  # /h
    a_gut = dose_mg * 0.85

    # Maternal elimination rate
    k_mat_el = cl_maternal_L_per_h / vd_maternal_L

    # Initial conditions
    c_maternal = 0.0
    c_fetal = 0.0

    # Storage
    times: list[float] = []
    c_mat_list: list[float] = []
    c_fet_list: list[float] = []

    t = 0.0
    n_steps = int(t_end_h / dt_h) + 1

    for _ in range(n_steps):
        times.append(t)
        c_mat_list.append(c_maternal)
        c_fet_list.append(c_fetal)

        # Forward Euler
        da_gut = -ka * a_gut
        dc_maternal = (
            ka * a_gut / vd_maternal_L
            - k_mat_el * c_maternal
            - k_transfer * c_maternal
            + k_transfer * c_fetal * (vd_fetal_L / vd_maternal_L)
        )
        dc_fetal = (
            k_transfer * c_maternal * (vd_maternal_L / vd_fetal_L)
            - k_transfer * c_fetal
            - (cl_fetal_L_per_h / vd_fetal_L) * c_fetal
        )

        a_gut += da_gut * dt_h
        c_maternal += dc_maternal * dt_h
        c_fetal += dc_fetal * dt_h

        # Ensure non-negative
        a_gut = max(0.0, a_gut)
        c_maternal = max(0.0, c_maternal)
        c_fetal = max(0.0, c_fetal)

        t += dt_h

    # Post-processing
    cmax_maternal = max(c_mat_list)
    cmax_fetal = max(c_fet_list)
    auc_maternal = _trapezoidal_auc(times, c_mat_list)
    auc_fetal = _trapezoidal_auc(times, c_fet_list)

    if auc_maternal > 0:
        fetal_to_maternal_ratio_auc = auc_fetal / auc_maternal
    else:
        fetal_to_maternal_ratio_auc = 0.0

    # t50_fetal: time for fetal concentration to reach 50% of maternal Cmax
    target_50 = 0.5 * cmax_maternal
    t50_fetal_h = 0.0
    for i, cf in enumerate(c_fet_list):
        if cf >= target_50:
            t50_fetal_h = times[i]
            break

    # Fetal exposure category
    if fetal_to_maternal_ratio_auc > 0.8:
        fetal_exposure_category = "high"
    elif fetal_to_maternal_ratio_auc > 0.4:
        fetal_exposure_category = "moderate"
    elif fetal_to_maternal_ratio_auc > 0.1:
        fetal_exposure_category = "low"
    else:
        fetal_exposure_category = "minimal"

    notes = (
        f"2-compartment placental model; GA={gestational_age_weeks}wk; "
        f"P_placenta={p_placenta:.4f}; k_transfer={k_transfer:.4f}/h"
    )

    return PlacentalTransferResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times,
        c_maternal_mg_L=c_mat_list,
        c_fetal_mg_L=c_fet_list,
        fetal_to_maternal_ratio_auc=fetal_to_maternal_ratio_auc,
        cmax_maternal_mg_L=cmax_maternal,
        cmax_fetal_mg_L=cmax_fetal,
        auc_maternal_mg_h_per_L=auc_maternal,
        auc_fetal_mg_h_per_L=auc_fetal,
        t50_fetal_h=t50_fetal_h,
        placental_transfer_rate=p_placenta,
        fetal_exposure_category=fetal_exposure_category,
        mw_Da=mw_Da,
        logP=logP,
        notes=notes,
    )
