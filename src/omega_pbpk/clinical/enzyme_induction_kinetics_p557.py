"""Phase 557 - Hepatic Enzyme Induction Kinetics.

Models the time course of enzyme induction (CYP3A4/CYP2B6) during
repeated dosing using Forward Euler ODE integration.
"""

import math
from dataclasses import dataclass


@dataclass
class InductionKineticsResult:
    """Result of enzyme induction kinetics simulation."""

    drug_name: str
    times_days: list
    enzyme_activity_relative: list
    cl_relative: list
    auc_ratio_vs_victim: list
    t50_induction_days: float
    max_induction_fold: float
    induction_class: str
    days_to_steady_state: float
    notes: str


def simulate_induction_kinetics(
    drug_name: str,
    emax_fold: float,
    ec50_mg_L: float,
    plasma_conc_mg_L: float,
    kdeg_per_day: float = 0.5,
    t_end_days: float = 14.0,
    dt_days: float = 0.1,
) -> InductionKineticsResult:
    """Simulate enzyme induction kinetics using Forward Euler ODE.

    Parameters
    ----------
    drug_name : str
        Name of the inducer drug.
    emax_fold : float
        Maximum fold increase in enzyme activity (must be > 0).
    ec50_mg_L : float
        Plasma concentration for half-maximal induction (must be > 0).
    plasma_conc_mg_L : float
        Assumed steady-state plasma concentration of inducer (must be > 0).
    kdeg_per_day : float
        Enzyme degradation rate constant (1/day).
    t_end_days : float
        Duration of simulation in days.
    dt_days : float
        Time step for Forward Euler integration.

    Returns
    -------
    InductionKineticsResult
    """
    if emax_fold <= 0:
        raise ValueError("emax_fold must be > 0")
    if ec50_mg_L <= 0:
        raise ValueError("ec50_mg_L must be > 0")
    if plasma_conc_mg_L <= 0:
        raise ValueError("plasma_conc_mg_L must be > 0")

    # Steady-state enzyme activity
    e_ss = 1.0 + emax_fold * plasma_conc_mg_L / (ec50_mg_L + plasma_conc_mg_L)

    # Induction rate (constant since plasma_conc is constant)
    kin = kdeg_per_day * e_ss

    # Forward Euler integration
    times_days = []
    enzyme_activity_relative = []
    cl_relative = []
    auc_ratio_vs_victim = []

    e = 1.0
    t = 0.0
    n_steps = int(math.ceil(t_end_days / dt_days))

    t50_induction_days = float("inf")
    days_to_steady_state = float("inf")
    t50_found = False
    ss_found = False

    for i in range(n_steps + 1):
        t = i * dt_days
        times_days.append(t)
        enzyme_activity_relative.append(e)
        cl_relative.append(e)
        auc_ratio_vs_victim.append(1.0 / e)

        # Check t50 and steady-state thresholds
        if e_ss > 1.0:
            fraction = (e - 1.0) / (e_ss - 1.0)
            if not t50_found and fraction >= 0.5:
                t50_induction_days = t
                t50_found = True
            if not ss_found and fraction >= 0.9:
                days_to_steady_state = t
                ss_found = True

        # Forward Euler step
        de_dt = kin - kdeg_per_day * e
        e = e + de_dt * dt_days

    max_induction_fold = max(enzyme_activity_relative)

    if max_induction_fold < 2.0:
        induction_class = "weak"
    elif max_induction_fold <= 5.0:
        induction_class = "moderate"
    else:
        induction_class = "strong"

    notes = f"Forward Euler ODE, dt={dt_days} days, E_ss={e_ss:.3f}, kdeg={kdeg_per_day}/day"

    return InductionKineticsResult(
        drug_name=drug_name,
        times_days=times_days,
        enzyme_activity_relative=enzyme_activity_relative,
        cl_relative=cl_relative,
        auc_ratio_vs_victim=auc_ratio_vs_victim,
        t50_induction_days=t50_induction_days,
        max_induction_fold=max_induction_fold,
        induction_class=induction_class,
        days_to_steady_state=days_to_steady_state,
        notes=notes,
    )


def compare_inducers(inducers_list: list, plasma_conc_mg_L: float = 1.0) -> list:
    """Compare multiple inducers by their induction potency.

    Parameters
    ----------
    inducers_list : list of dict
        Each dict has keys: drug_name, emax_fold, ec50_mg_L.
    plasma_conc_mg_L : float
        Common plasma concentration for comparison.

    Returns
    -------
    list of InductionKineticsResult
        Sorted by max_induction_fold descending.
    """
    results = []
    for inducer in inducers_list:
        result = simulate_induction_kinetics(
            drug_name=inducer["drug_name"],
            emax_fold=inducer["emax_fold"],
            ec50_mg_L=inducer["ec50_mg_L"],
            plasma_conc_mg_L=plasma_conc_mg_L,
        )
        results.append(result)
    results.sort(key=lambda r: r.max_induction_fold, reverse=True)
    return results
