"""Phase 669 — Drug Interaction with CYP Induction Lag.

Model the temporal dynamics of CYP enzyme induction, including the lag time
before induction manifests and the return to baseline after stopping the inducer.
Uses enzyme turnover model with Forward Euler ODE integration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class CYPInductionLagResult:
    """Result of CYP induction lag simulation."""

    drug_name_victim: str
    inducer_name: str
    enzyme_name: str
    times_h: list = field(default_factory=list)
    enzyme_activity: list = field(default_factory=list)
    c_victim_mg_L: list = field(default_factory=list)
    c_victim_no_induction_mg_L: list = field(default_factory=list)
    induction_ratio: list = field(default_factory=list)
    aucr: float = 0.0
    t_onset_50pct_h: float = 0.0
    t_offset_50pct_h: float = 0.0
    t_max_induction_h: float = 0.0
    max_fold_induction: float = 1.0
    victim_dose_mg: float = 0.0
    inducer_dose_mg: float = 0.0
    notes: str = ""


def simulate_cyp_induction_lag(
    victim_name: str,
    victim_dose_mg: float,
    victim_cl_L_per_h: float,
    victim_vd_L: float,
    inducer_name: str,
    inducer_dose_mg: float,
    enzyme_name: str,
    emax_induction: float,
    ec50_induction_mg_L: float,
    kdeg_per_h: float = 0.0077,
    t_pretreatment_h: float = 0.0,
    t_treatment_h: float = 168.0,
    t_washout_h: float = 72.0,
    dt_h: float = 1.0,
) -> CYPInductionLagResult:
    """Simulate CYP enzyme induction lag and its effect on victim drug PK.

    Parameters
    ----------
    victim_name : str
        Name of the victim drug.
    victim_dose_mg : float
        Dose of the victim drug (mg).
    victim_cl_L_per_h : float
        Clearance of the victim drug (L/h).
    victim_vd_L : float
        Volume of distribution of the victim drug (L).
    inducer_name : str
        Name of the inducer.
    inducer_dose_mg : float
        Dose of the inducer (mg).
    enzyme_name : str
        Name of the CYP enzyme.
    emax_induction : float
        Maximum fold induction above baseline.
    ec50_induction_mg_L : float
        EC50 for induction (mg/L).
    kdeg_per_h : float
        Enzyme degradation rate constant (1/h).
    t_pretreatment_h : float
        Pretreatment duration (h).
    t_treatment_h : float
        Treatment duration (h).
    t_washout_h : float
        Washout duration (h).
    dt_h : float
        Time step for Euler integration (h).

    Returns
    -------
    CYPInductionLagResult
    """
    # Validation
    if victim_cl_L_per_h <= 0:
        raise ValueError("victim_cl_L_per_h must be > 0")
    if victim_vd_L <= 0:
        raise ValueError("victim_vd_L must be > 0")
    if emax_induction < 0:
        raise ValueError("emax_induction must be >= 0")
    if ec50_induction_mg_L <= 0:
        raise ValueError("ec50_induction_mg_L must be > 0")
    if kdeg_per_h <= 0:
        raise ValueError("kdeg_per_h must be > 0")

    # Derived parameters
    ksyn = kdeg_per_h  # At baseline: synthesis = degradation
    c_inducer_ss = inducer_dose_mg / 10.0  # Simplified steady-state estimate

    # Total simulation time
    t_total = t_pretreatment_h + t_treatment_h + t_washout_h
    n_steps = max(1, int(math.ceil(t_total / dt_h)))

    # Victim drug PK parameters
    ka = 1.5  # absorption rate constant (1/h)
    f_oral = 0.85

    # State variables
    enzyme = 1.0  # normalized baseline
    a_gut = victim_dose_mg * f_oral
    c_victim = 0.0
    a_gut_noinduct = victim_dose_mg * f_oral
    c_victim_noinduct = 0.0

    # Storage
    times_h = [0.0]
    enzyme_activity = [1.0]
    c_victim_list = [0.0]
    c_victim_noinduct_list = [0.0]

    for i in range(1, n_steps + 1):
        t = i * dt_h

        # Determine inducer concentration based on phase
        if t <= t_pretreatment_h:
            c_inducer = 0.0
        elif t <= t_pretreatment_h + t_treatment_h:
            c_inducer = c_inducer_ss
        else:
            c_inducer = 0.0

        # Fold induction
        fold_induction = 1.0 + emax_induction * c_inducer / (ec50_induction_mg_L + c_inducer)

        # Enzyme ODE (Forward Euler)
        dE_dt = ksyn * fold_induction - kdeg_per_h * enzyme
        enzyme = enzyme + dE_dt * dt_h

        # Victim drug PK with induction
        k_el = victim_cl_L_per_h * enzyme / victim_vd_L
        dA_gut = -ka * a_gut * dt_h
        dC_victim = (ka * a_gut / victim_vd_L - k_el * c_victim) * dt_h
        a_gut = a_gut + dA_gut
        c_victim = max(0.0, c_victim + dC_victim)

        # Victim drug PK without induction (reference)
        k_el_ref = victim_cl_L_per_h / victim_vd_L
        dA_gut_noinduct = -ka * a_gut_noinduct * dt_h
        dC_victim_noinduct = (
            ka * a_gut_noinduct / victim_vd_L - k_el_ref * c_victim_noinduct
        ) * dt_h
        a_gut_noinduct = a_gut_noinduct + dA_gut_noinduct
        c_victim_noinduct = max(0.0, c_victim_noinduct + dC_victim_noinduct)

        times_h.append(t)
        enzyme_activity.append(enzyme)
        c_victim_list.append(c_victim)
        c_victim_noinduct_list.append(c_victim_noinduct)

    # Manual trapezoidal AUC
    auc_with = sum(
        0.5 * (c_victim_list[i] + c_victim_list[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )
    auc_without = sum(
        0.5
        * (c_victim_noinduct_list[i] + c_victim_noinduct_list[i - 1])
        * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )
    aucr = auc_with / auc_without if auc_without > 0 else 1.0

    # Induction ratio
    induction_ratio = [e / 1.0 for e in enzyme_activity]

    # Max fold induction and time
    max_enzyme = max(enzyme_activity)
    max_fold_induction = max_enzyme
    t_max_induction_h = times_h[enzyme_activity.index(max_enzyme)]

    # t_onset_50pct_h: time from start of treatment for enzyme to reach 50% of (max - baseline) + baseline
    threshold_onset = 1.0 + 0.5 * (max_enzyme - 1.0)
    t_onset_50pct_h = 0.0
    for i, t in enumerate(times_h):
        if t >= t_pretreatment_h and enzyme_activity[i] >= threshold_onset:
            t_onset_50pct_h = t - t_pretreatment_h
            break

    # t_offset_50pct_h: time from end of treatment for enzyme to return to 50% of (max - baseline) above baseline
    t_treatment_end = t_pretreatment_h + t_treatment_h
    t_offset_50pct_h = 0.0
    for i, t in enumerate(times_h):
        if t >= t_treatment_end and enzyme_activity[i] <= threshold_onset:
            t_offset_50pct_h = t - t_treatment_end
            break

    return CYPInductionLagResult(
        drug_name_victim=victim_name,
        inducer_name=inducer_name,
        enzyme_name=enzyme_name,
        times_h=times_h,
        enzyme_activity=enzyme_activity,
        c_victim_mg_L=c_victim_list,
        c_victim_no_induction_mg_L=c_victim_noinduct_list,
        induction_ratio=induction_ratio,
        aucr=aucr,
        t_onset_50pct_h=t_onset_50pct_h,
        t_offset_50pct_h=t_offset_50pct_h,
        t_max_induction_h=t_max_induction_h,
        max_fold_induction=max_fold_induction,
        victim_dose_mg=victim_dose_mg,
        inducer_dose_mg=inducer_dose_mg,
        notes=f"Enzyme turnover model for {enzyme_name} induction by {inducer_name}",
    )
