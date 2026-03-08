"""Phase 574 - Multi-Dose Steady State Accumulation.

Models accumulation of drug levels during multiple oral doses
to reach steady state using Forward Euler integration.
"""

import math
from dataclasses import dataclass


@dataclass
class MultidoseAccumulationResult:
    """Result of multi-dose accumulation simulation."""

    drug_name: str
    dose_mg: float
    dosing_interval_h: float
    n_doses: int
    times_h: list
    c_systemic_mg_L: list
    cmax_ss_mg_L: float
    cmin_ss_mg_L: float
    cavg_ss_mg_L: float
    accumulation_ratio: float
    fluctuation_pct: float
    doses_to_90pct_ss: int
    t_half_h: float
    notes: str


def simulate_multidose_accumulation(
    drug_name: str,
    dose_mg: float,
    ka_per_h: float,
    cl_L_per_h: float,
    vd_L: float,
    dosing_interval_h: float = 24.0,
    n_doses: int = 10,
    f_oral: float = 0.8,
    dt_h: float = 0.05,
) -> MultidoseAccumulationResult:
    """Simulate multi-dose oral PK accumulation to steady state.

    Args:
        drug_name: Name of the drug.
        dose_mg: Dose amount (mg).
        ka_per_h: Absorption rate constant (1/h).
        cl_L_per_h: Clearance (L/h).
        vd_L: Volume of distribution (L).
        dosing_interval_h: Dosing interval (h).
        n_doses: Number of doses.
        f_oral: Oral bioavailability fraction.
        dt_h: Time step (h).

    Returns:
        MultidoseAccumulationResult with full time course and steady-state metrics.
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if ka_per_h <= 0:
        raise ValueError("ka_per_h must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if n_doses <= 0:
        raise ValueError("n_doses must be > 0")

    ke = cl_L_per_h / vd_L
    t_half = math.log(2) / ke

    total_time = n_doses * dosing_interval_h
    n_steps = int(math.ceil(total_time / dt_h))

    a_gi = 0.0  # drug in GI compartment (mg)
    a_sys = 0.0  # drug in systemic compartment (mg)

    times_h = []
    c_systemic = []

    # Track dose times
    next_dose_index = 0

    for i in range(n_steps + 1):
        t = i * dt_h

        # Add dose at dosing times
        while next_dose_index < n_doses and t >= next_dose_index * dosing_interval_h - dt_h * 0.01:
            a_gi += dose_mg * f_oral
            next_dose_index += 1

        c = a_sys / vd_L
        times_h.append(t)
        c_systemic.append(c)

        if i < n_steps:
            da_gi = -ka_per_h * a_gi
            da_sys = ka_per_h * a_gi - ke * a_sys
            a_gi += da_gi * dt_h
            a_sys += da_sys * dt_h
            if a_gi < 0:
                a_gi = 0.0
            if a_sys < 0:
                a_sys = 0.0

    # Find Cmax for each dose cycle
    cmax_per_dose = []
    for d in range(n_doses):
        t_start = d * dosing_interval_h
        t_end = (d + 1) * dosing_interval_h
        cycle_max = 0.0
        for j in range(len(times_h)):
            if times_h[j] >= t_start - dt_h * 0.01 and times_h[j] <= t_end + dt_h * 0.01:
                if c_systemic[j] > cycle_max:
                    cycle_max = c_systemic[j]
        cmax_per_dose.append(cycle_max)

    cmax_dose1 = cmax_per_dose[0] if cmax_per_dose else 0.0

    # Steady state metrics from last dose cycle
    last_start = (n_doses - 1) * dosing_interval_h
    last_end = n_doses * dosing_interval_h

    last_cycle_times = []
    last_cycle_conc = []
    for j in range(len(times_h)):
        if times_h[j] >= last_start - dt_h * 0.01 and times_h[j] <= last_end + dt_h * 0.01:
            last_cycle_times.append(times_h[j])
            last_cycle_conc.append(c_systemic[j])

    cmax_ss = max(last_cycle_conc) if last_cycle_conc else 0.0
    cmin_ss = last_cycle_conc[-1] if last_cycle_conc else 0.0

    # AUC over last interval using trapezoidal rule
    auc_last = 0.0
    for j in range(1, len(last_cycle_times)):
        auc_last += (
            0.5
            * (last_cycle_conc[j] + last_cycle_conc[j - 1])
            * (last_cycle_times[j] - last_cycle_times[j - 1])
        )

    cavg_ss = auc_last / dosing_interval_h if dosing_interval_h > 0 else 0.0

    accumulation_ratio = cmax_ss / cmax_dose1 if cmax_dose1 > 0 else 1.0

    fluctuation_pct = (cmax_ss - cmin_ss) / cavg_ss * 100.0 if cavg_ss > 0 else 0.0

    # Doses to 90% steady state
    doses_to_90 = n_doses
    threshold = 0.9 * cmax_ss
    for d in range(len(cmax_per_dose)):
        if cmax_per_dose[d] >= threshold:
            doses_to_90 = d + 1  # 1-indexed
            break

    notes = (
        f"Multi-dose accumulation for {drug_name}: {dose_mg} mg every "
        f"{dosing_interval_h} h x {n_doses} doses. "
        f"ka={ka_per_h}/h, CL={cl_L_per_h} L/h, Vd={vd_L} L, F={f_oral}."
    )

    return MultidoseAccumulationResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        dosing_interval_h=dosing_interval_h,
        n_doses=n_doses,
        times_h=times_h,
        c_systemic_mg_L=c_systemic,
        cmax_ss_mg_L=cmax_ss,
        cmin_ss_mg_L=cmin_ss,
        cavg_ss_mg_L=cavg_ss,
        accumulation_ratio=accumulation_ratio,
        fluctuation_pct=fluctuation_pct,
        doses_to_90pct_ss=doses_to_90,
        t_half_h=t_half,
        notes=notes,
    )
