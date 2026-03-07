"""Phase 756 — Drug Interaction with CYP Inducers.

Models time-dependent CYP induction effects on victim drug PK using an Emax
induction model combined with first-order enzyme turnover kinetics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class CYPInductionPKResult:
    """Result of CYP induction PK simulation."""

    drug_name: str
    inducer_name: str
    days: list
    cl_per_day: list
    auc_per_day: list
    cmax_per_day: list
    day_0_auc: float
    day_n_auc: float
    auc_ratio_final: float
    time_to_50pct_max_induction_days: float
    fold_induction: float
    notes: str


def _simulate_single_dose_pk(
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    ka_per_h: float,
    f_oral: float,
    t_end_h: float,
    dt_h: float,
) -> tuple[float, float]:
    """Simulate 1-compartment oral PK; return (AUC, Cmax) via forward Euler.

    Parameters
    ----------
    dose_mg:
        Oral dose in mg.
    cl_L_per_h:
        Clearance in L/h.
    vd_L:
        Volume of distribution in L.
    ka_per_h:
        Absorption rate constant (1/h).
    f_oral:
        Oral bioavailability fraction.
    t_end_h:
        Simulation end time in hours.
    dt_h:
        Integration time step in hours.

    Returns
    -------
    (auc, cmax) : tuple[float, float]
        Area under the curve (mg/L * h) and peak concentration (mg/L).
    """
    ke = cl_L_per_h / vd_L  # elimination rate constant (1/h)
    a_gut = dose_mg * f_oral  # amount in gut depot (mg)
    a_central = 0.0  # amount in central compartment (mg)

    auc = 0.0
    cmax = 0.0
    n_steps = max(1, int(round(t_end_h / dt_h)))
    dt = t_end_h / n_steps

    for _ in range(n_steps):
        conc = a_central / vd_L
        if conc > cmax:
            cmax = conc

        # trapezoidal AUC contribution
        da_gut = -ka_per_h * a_gut * dt
        da_central = (ka_per_h * a_gut - ke * a_central) * dt

        # update
        a_gut += da_gut
        a_central += da_central

        conc_new = a_central / vd_L
        auc += 0.5 * (conc + conc_new) * dt

    # final concentration check
    conc_final = a_central / vd_L
    if conc_final > cmax:
        cmax = conc_final

    return auc, cmax


def simulate_cyp_induction_pk(
    drug_name: str,
    inducer_name: str = "rifampin",
    dose_victim_mg: float = 100.0,
    cl_baseline_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    ka_per_h: float = 1.0,
    f_oral: float = 0.7,
    fold_induction: float = 5.0,
    t_half_enzyme_h: float = 70.0,
    n_days: int = 14,
    t_end_per_dose_h: float = 24.0,
    dt_h: float = 0.1,
) -> CYPInductionPKResult:
    """Simulate time course of CYP induction and its impact on victim drug PK.

    Parameters
    ----------
    drug_name:
        Name of victim drug.
    inducer_name:
        Name of the CYP inducer (e.g. "rifampin").
    dose_victim_mg:
        Single oral dose of victim drug (mg).
    cl_baseline_L_per_h:
        Baseline clearance before induction (L/h).
    vd_L:
        Volume of distribution of victim drug (L).
    ka_per_h:
        Absorption rate constant of victim drug (1/h).
    f_oral:
        Oral bioavailability of victim drug.
    fold_induction:
        Maximum fold increase in CL (must be > 1).
    t_half_enzyme_h:
        CYP enzyme half-life driving induction onset (h). Default 70 h for CYP3A4.
    n_days:
        Number of days of inducer co-administration.
    t_end_per_dose_h:
        Simulation window per dose (h).
    dt_h:
        Integration time step (h).

    Returns
    -------
    CYPInductionPKResult
    """
    if dose_victim_mg <= 0:
        raise ValueError(f"dose_victim_mg must be > 0, got {dose_victim_mg}")
    if cl_baseline_L_per_h <= 0:
        raise ValueError(f"cl_baseline_L_per_h must be > 0, got {cl_baseline_L_per_h}")
    if fold_induction <= 1:
        raise ValueError(f"fold_induction must be > 1, got {fold_induction}")
    if n_days <= 0:
        raise ValueError(f"n_days must be > 0, got {n_days}")

    k_deg = math.log(2.0) / t_half_enzyme_h  # enzyme degradation rate (1/h)

    # time to 50% max induction: (1 - exp(-k_deg * t_50 * 24)) = 0.5
    # => k_deg * t_50 * 24 = ln(2) => t_50 = ln(2) / (k_deg * 24)
    t_50_days = math.log(2.0) / (k_deg * 24.0)

    days_list: list[float] = []
    cl_list: list[float] = []
    auc_list: list[float] = []
    cmax_list: list[float] = []

    for d in range(n_days + 1):
        # CL at day d: enzyme induction follows 1-order onset
        induction_fraction = 1.0 - math.exp(-k_deg * d * 24.0)
        cl_d = cl_baseline_L_per_h * (1.0 + (fold_induction - 1.0) * induction_fraction)

        auc_d, cmax_d = _simulate_single_dose_pk(
            dose_mg=dose_victim_mg,
            cl_L_per_h=cl_d,
            vd_L=vd_L,
            ka_per_h=ka_per_h,
            f_oral=f_oral,
            t_end_h=t_end_per_dose_h,
            dt_h=dt_h,
        )

        days_list.append(float(d))
        cl_list.append(cl_d)
        auc_list.append(auc_d)
        cmax_list.append(cmax_d)

    day_0_auc = auc_list[0]
    day_n_auc = auc_list[-1]
    auc_ratio_final = day_n_auc / day_0_auc if day_0_auc > 0 else float("nan")

    notes = (
        f"Enzyme k_deg={k_deg:.4f}/h; t½_enzyme={t_half_enzyme_h:.1f}h; "
        f"fold_induction={fold_induction:.1f}; inducer={inducer_name}; "
        f"CL_max={cl_list[-1]:.2f} L/h at day {n_days}"
    )

    return CYPInductionPKResult(
        drug_name=drug_name,
        inducer_name=inducer_name,
        days=days_list,
        cl_per_day=cl_list,
        auc_per_day=auc_list,
        cmax_per_day=cmax_list,
        day_0_auc=day_0_auc,
        day_n_auc=day_n_auc,
        auc_ratio_final=auc_ratio_final,
        time_to_50pct_max_induction_days=t_50_days,
        fold_induction=fold_induction,
        notes=notes,
    )


def assess_induction_severity(auc_ratio: float) -> str:
    """Classify DDI severity based on AUC ratio after induction.

    Parameters
    ----------
    auc_ratio:
        Ratio of victim drug AUC with inducer vs. baseline (day_n / day_0).

    Returns
    -------
    str
        One of "none", "mild", "moderate", or "severe".
    """
    if auc_ratio > 0.8:
        return "none"
    elif auc_ratio >= 0.5:
        return "mild"
    elif auc_ratio >= 0.2:
        return "moderate"
    else:
        return "severe"
