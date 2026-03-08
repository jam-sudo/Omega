"""Phase 625 — Drug Bile Secretion and Enterohepatic Cycling.

Models bile secretion of drugs and enterohepatic cycling (EHC): drug excreted
in bile, stored in gallbladder, released at meal times, and reabsorbed from gut.
"""

import math
from dataclasses import dataclass


@dataclass
class BileSecretionCyclingResult:
    """Result of bile secretion and enterohepatic cycling simulation."""

    drug_name: str
    dose_mg: float
    times_h: list
    c_plasma_mg_L: list
    c_bile_mg_mL: list
    c_duodenum_mg_mL: list
    cumulative_secreted_mg: list
    cumulative_reabsorbed_mg: list
    ehc_fraction: float
    n_cycling_events: int
    secondary_peak_times_h: list
    auc_mg_h_per_L: float
    t_half_effective_h: float
    notes: str


def _trapezoidal_auc(times: list, concs: list) -> float:
    """Manual trapezoidal AUC integration."""
    return sum(
        0.5 * (concs[i] + concs[i - 1]) * (times[i] - times[i - 1]) for i in range(1, len(times))
    )


def simulate_bile_secretion_cycling(
    drug_name: str,
    dose_mg: float,
    cl_biliary_L_per_h: float,
    cl_sys_L_per_h: float,
    vd_L: float,
    f_reabsorb: float = 0.6,
    bile_flow_mL_per_h: float = 30.0,
    meal_times_h: list[float] | None = None,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
) -> BileSecretionCyclingResult:
    """Simulate bile secretion and enterohepatic cycling.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Oral dose in mg.
    cl_biliary_L_per_h:
        Biliary clearance (L/h).
    cl_sys_L_per_h:
        Systemic (hepatic + renal) clearance (L/h).
    vd_L:
        Volume of distribution (L).
    f_reabsorb:
        Fraction of biliary drug reabsorbed from duodenum (0–1).
    bile_flow_mL_per_h:
        Bile flow rate into gallbladder (mL/h); stored for reference.
    meal_times_h:
        Times (h) at which gallbladder empties (meals). Default every 6 h.
    t_end_h:
        Simulation end time (h).
    dt_h:
        Integration step size (h).

    Returns
    -------
    BileSecretionCyclingResult
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_biliary_L_per_h < 0:
        raise ValueError("cl_biliary_L_per_h must be >= 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if not (0.0 <= f_reabsorb <= 1.0):
        raise ValueError("f_reabsorb must be in [0, 1]")

    if meal_times_h is None:
        meal_times_h = [0.0, 6.0, 12.0, 18.0, 24.0, 30.0, 36.0, 42.0]
    else:
        if len(meal_times_h) == 0:
            raise ValueError("meal_times_h must not be empty if provided")

    # Convert meal times to a set of rounded indices for fast lookup
    meal_set = set()
    for mt in meal_times_h:
        # Store as float for comparison
        meal_set.add(round(mt, 6))

    # --- Constants ---
    ka = 1.2  # /h, oral absorption rate constant
    f_oral = 0.85  # fraction absorbed from gut
    V_gb = 30.0  # mL, gallbladder volume
    V_duodenum = 50.0  # mL, effective duodenal volume
    # Duodenal absorption rate: t_half 0.5 h → ka_duodenum = f_reabsorb / 0.5
    ka_duodenum = f_reabsorb / 0.5  # /h

    # --- Initial conditions ---
    A_gut = dose_mg * f_oral  # mg
    A_gb = 0.0  # mg in gallbladder
    A_duodenum = 0.0  # mg in duodenum
    C_plasma = 0.0  # mg/L

    cum_secreted = 0.0
    cum_reabsorbed = 0.0

    # --- Output arrays ---
    n_steps = int(round(t_end_h / dt_h)) + 1
    times = []
    c_plasma_list = []
    c_bile_list = []
    c_duodenum_list = []
    cum_sec_list = []
    cum_rea_list = []

    # Track previous time for meal detection
    t = 0.0

    for step in range(n_steps):
        t = step * dt_h

        # Record state
        times.append(t)
        c_plasma_list.append(max(0.0, C_plasma))
        c_bile_list.append(A_gb / V_gb)
        c_duodenum_list.append(A_duodenum / V_duodenum)
        cum_sec_list.append(cum_secreted)
        cum_rea_list.append(cum_reabsorbed)

        if step == n_steps - 1:
            break

        # Check if next time step crosses a meal time
        t_next = (step + 1) * dt_h
        for mt in meal_set:
            if t < mt <= t_next or (t <= mt < t_next and step == 0 and mt == 0.0):
                # Gallbladder empties: bolus into duodenum
                A_duodenum += A_gb
                A_gb = 0.0

        # Handle t=0 meal (gallbladder empty at start, no effect at step 0)
        # Actually for step 0 meal time at t=0: we handle before the first step ODE
        if step == 0 and 0.0 in meal_set:
            # gallbladder already empty, no drug to release
            pass

        # --- ODE derivatives ---
        dA_gut_dt = -ka * A_gut

        biliary_rate = cl_biliary_L_per_h * C_plasma  # mg/h secreted into bile

        dC_plasma_dt = (
            ka * A_gut / vd_L
            + ka_duodenum * A_duodenum / vd_L
            - (cl_sys_L_per_h + cl_biliary_L_per_h) / vd_L * C_plasma
        )

        dA_gb_dt = biliary_rate  # mg/h

        dA_duodenum_dt = -ka_duodenum * A_duodenum  # mg/h (absorption from duodenum)

        # Cumulative trackers
        cum_secreted += biliary_rate * dt_h
        cum_reabsorbed += ka_duodenum * A_duodenum * dt_h

        # --- Forward Euler update ---
        A_gut = max(0.0, A_gut + dA_gut_dt * dt_h)
        C_plasma = max(0.0, C_plasma + dC_plasma_dt * dt_h)
        A_gb = max(0.0, A_gb + dA_gb_dt * dt_h)
        A_duodenum = max(0.0, A_duodenum + dA_duodenum_dt * dt_h)

    # --- Post-processing ---
    # AUC
    auc = _trapezoidal_auc(times, c_plasma_list)

    # Find tmax (first global maximum)
    cmax = max(c_plasma_list)
    tmax_idx = c_plasma_list.index(cmax)

    # Secondary peaks: local maxima after tmax_idx
    secondary_peak_times = []
    threshold = 0.05 * cmax if cmax > 0 else 0.0

    for i in range(tmax_idx + 1, len(c_plasma_list) - 1):
        c_prev = c_plasma_list[i - 1]
        c_curr = c_plasma_list[i]
        c_next = c_plasma_list[i + 1]
        if c_curr > c_prev and c_curr > c_next and c_curr > threshold:
            secondary_peak_times.append(times[i])

    n_cycling = len(secondary_peak_times)

    # EHC fraction
    ehc_fraction = min(1.0, cum_reabsorbed / dose_mg) if dose_mg > 0 else 0.0

    # Effective half-life (analytical)
    cl_effective = cl_sys_L_per_h + cl_biliary_L_per_h * (1.0 - f_reabsorb)
    if cl_effective > 0 and vd_L > 0:
        t_half_effective = math.log(2.0) * vd_L / cl_effective
    else:
        t_half_effective = float("inf")

    notes = (
        f"EHC simulation: {len(meal_times_h)} meal events, "
        f"biliary CL={cl_biliary_L_per_h:.2f} L/h, "
        f"f_reabsorb={f_reabsorb:.2f}"
    )

    return BileSecretionCyclingResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        c_bile_mg_mL=c_bile_list,
        c_duodenum_mg_mL=c_duodenum_list,
        cumulative_secreted_mg=cum_sec_list,
        cumulative_reabsorbed_mg=cum_rea_list,
        ehc_fraction=ehc_fraction,
        n_cycling_events=n_cycling,
        secondary_peak_times_h=secondary_peak_times,
        auc_mg_h_per_L=auc,
        t_half_effective_h=t_half_effective,
        notes=notes,
    )
