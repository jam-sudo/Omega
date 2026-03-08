"""Monoclonal antibody dosing simulation with target-mediated disposition.

Models free mAb concentration, receptor occupancy, and target-bound mAb
for IV and SC routes using a simplified TMDD quasi-steady-state approach.

References
----------
- Mager DE & Jusko WJ, J Pharmacokinet Pharmacodyn. 2001;28(6):507-32
- Gibiansky L et al, J Pharmacokinet Pharmacodyn. 2008;35(5):573-91
- Dua P et al, mAbs. 2015;7(6):1094-1100
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class MABDosingResult:
    """Result of a monoclonal antibody dosing simulation.

    Attributes
    ----------
    drug_name : Name of the mAb.
    dose_mg : Dose in milligrams.
    route : Dosing route ('iv' or 'sc').
    dosing_interval_weeks : Interval between doses in weeks.
    times_h : Simulation time points (hours).
    c_free_mab_ug_mL : Free mAb concentration (µg/mL).
    c_target_bound_ug_mL : Target-bound mAb concentration (µg/mL).
    c_free_target_nM : Free (unbound) target concentration (nM).
    receptor_occupancy : Fraction of target bound by mAb (0-1).
    cmax_ug_mL : Peak free mAb concentration (µg/mL).
    ctrough_ug_mL : Trough concentration before next dose (µg/mL).
    auc_ug_h_per_mL : Area under the concentration-time curve (µg·h/mL).
    target_coverage_pct : Percentage of time with receptor occupancy > 90%.
    accumulation_ratio : Cmax at steady state / Cmax after first dose.
    notes : Summary notes.
    """

    drug_name: str
    dose_mg: float
    route: str
    dosing_interval_weeks: float
    times_h: list
    c_free_mab_ug_mL: list
    c_target_bound_ug_mL: list
    c_free_target_nM: list
    receptor_occupancy: list
    cmax_ug_mL: float
    ctrough_ug_mL: float
    auc_ug_h_per_mL: float
    target_coverage_pct: float
    accumulation_ratio: float
    notes: str


def simulate_mab_dosing(
    drug_name: str,
    dose_mg: float,
    route: str,
    dosing_interval_weeks: float,
    kd_nM: float,
    target_conc_nM: float,
    cl_L_per_day: float,
    vd_central_L: float,
    mw_kDa: float = 150.0,
    n_doses: int = 4,
    sc_bioavailability: float = 0.75,
    sc_ka_per_h: float = 0.01,
) -> MABDosingResult:
    """Simulate mAb dosing with simplified TMDD.

    Parameters
    ----------
    drug_name : Name of the monoclonal antibody.
    dose_mg : Dose per administration (mg).
    route : 'iv' (intravenous bolus) or 'sc' (subcutaneous).
    dosing_interval_weeks : Interval between doses (weeks).
    kd_nM : Equilibrium dissociation constant for target binding (nM).
    target_conc_nM : Total target concentration (nM).
    cl_L_per_day : Systemic clearance (L/day).
    vd_central_L : Central volume of distribution (L).
    mw_kDa : Molecular weight of the mAb (kDa). Default 150.
    n_doses : Number of doses to simulate. Default 4.
    sc_bioavailability : Bioavailability fraction for SC route. Default 0.75.
    sc_ka_per_h : SC first-order absorption rate constant (/h). Default 0.01.

    Returns
    -------
    MABDosingResult
        Simulation results including PK profiles and PD metrics.

    Raises
    ------
    ValueError
        If any input parameter is invalid.
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive")
    if route not in ("iv", "sc"):
        raise ValueError("route must be 'iv' or 'sc'")
    if dosing_interval_weeks <= 0:
        raise ValueError("dosing_interval_weeks must be positive")
    if n_doses < 1:
        raise ValueError("n_doses must be >= 1")
    if kd_nM <= 0:
        raise ValueError("kd_nM must be positive")
    if target_conc_nM <= 0:
        raise ValueError("target_conc_nM must be positive")
    if cl_L_per_day <= 0:
        raise ValueError("cl_L_per_day must be positive")
    if vd_central_L <= 0:
        raise ValueError("vd_central_L must be positive")

    # --- Unit conversions ---
    vd_L_mL = vd_central_L * 1000.0  # mL
    tau_h = dosing_interval_weeks * 7.0 * 24.0  # h
    t_end_h = n_doses * tau_h + 168.0  # extra week tail

    # Kd in µg/mL (for Langmuir occupancy in concentration units)
    kd_ug_mL = kd_nM * mw_kDa / 1000.0

    # Elimination rate constant (/h)
    k_el = cl_L_per_day * 1000.0 / (24.0 * vd_L_mL)

    # --- Dose bolus in µg/mL ---
    dose_ug_mL_iv = dose_mg * 1000.0 / vd_L_mL  # IV bolus
    dose_mg_sc = dose_mg * sc_bioavailability  # SC depot dose (mg)

    # --- Build dosing times ---
    dosing_times_h = [i * tau_h for i in range(n_doses)]

    # --- Simulation ---
    dt_h = 0.5
    n_steps = int(math.ceil(t_end_h / dt_h)) + 1

    times_h: list = []
    c_free: list = []
    c_target_bound: list = []
    c_free_target: list = []
    ro_list: list = []

    C_free = 0.0  # µg/mL
    A_depot = 0.0  # mg (SC depot)

    dose_idx = 0

    for step in range(n_steps):
        t = step * dt_h

        # Apply doses at dosing times (within half a timestep)
        while dose_idx < len(dosing_times_h) and dosing_times_h[dose_idx] <= t + 1e-9:
            if route == "iv":
                C_free += dose_ug_mL_iv
            else:
                A_depot += dose_mg_sc
            dose_idx += 1

        # Record state
        times_h.append(t)
        c_free.append(C_free)

        # Receptor occupancy (Langmuir QSS)
        ro = C_free / (C_free + kd_ug_mL) if (C_free + kd_ug_mL) > 0.0 else 0.0
        ro_list.append(ro)

        # Target-bound mAb (µg/mL equivalent)
        tb = ro * target_conc_nM * mw_kDa / 1000.0
        c_target_bound.append(tb)

        # Free target (nM)
        ft = target_conc_nM * (1.0 - ro)
        c_free_target.append(ft)

        # ODE: dC_free/dt
        if route == "sc":
            input_rate = sc_ka_per_h * A_depot * 1000.0 / vd_L_mL  # µg/mL/h
        else:
            input_rate = 0.0

        dC = input_rate - k_el * C_free
        dA_depot = -sc_ka_per_h * A_depot if route == "sc" else 0.0

        C_free = max(0.0, C_free + dC * dt_h)
        A_depot = max(0.0, A_depot + dA_depot * dt_h)

        if t >= t_end_h:
            break

    # --- Derived PK/PD metrics ---
    cmax_ug_mL = max(c_free)

    # AUC via trapezoidal rule
    auc = sum(
        0.5 * (c_free[i] + c_free[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )

    # Trough: minimum in last full dosing interval
    if n_doses >= 2:
        last_dose_time = dosing_times_h[-1]
        trough_vals = [c_free[i] for i, t in enumerate(times_h) if t >= last_dose_time]
        ctrough = min(trough_vals) if trough_vals else c_free[-1]
    else:
        # Single dose: trough is last value before t_end
        ctrough = c_free[-1]

    # Target coverage: % time RO > 0.9
    n_over = sum(1 for r in ro_list if r > 0.9)
    target_coverage_pct = n_over / len(ro_list) * 100.0 if ro_list else 0.0

    # Accumulation ratio: Cmax last interval / Cmax first interval
    first_end = dosing_times_h[1] if n_doses >= 2 else tau_h
    first_vals = [c_free[i] for i, t in enumerate(times_h) if t < first_end]
    last_vals = [c_free[i] for i, t in enumerate(times_h) if t >= dosing_times_h[-1]]
    cmax_first = max(first_vals) if first_vals else 0.0
    cmax_last = max(last_vals) if last_vals else 0.0
    accumulation_ratio = (cmax_last / cmax_first) if cmax_first > 0.0 else 1.0

    notes = (
        f"Simulated {n_doses} doses of {dose_mg} mg {route.upper()} "
        f"every {dosing_interval_weeks:.1f} week(s). "
        f"kd={kd_nM} nM, target={target_conc_nM} nM, "
        f"CL={cl_L_per_day} L/day, Vd={vd_central_L} L."
    )

    return MABDosingResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route=route,
        dosing_interval_weeks=dosing_interval_weeks,
        times_h=times_h,
        c_free_mab_ug_mL=c_free,
        c_target_bound_ug_mL=c_target_bound,
        c_free_target_nM=c_free_target,
        receptor_occupancy=ro_list,
        cmax_ug_mL=cmax_ug_mL,
        ctrough_ug_mL=ctrough,
        auc_ug_h_per_mL=auc,
        target_coverage_pct=target_coverage_pct,
        accumulation_ratio=accumulation_ratio,
        notes=notes,
    )
