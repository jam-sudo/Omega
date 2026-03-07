"""Phase 354 — Multi-Dose Accumulation Calculator.

Calculates drug accumulation over multiple doses using Forward Euler 1-compartment
simulation with superposition-style bolus dosing events.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def _auc_trap(xs: list[float], ys: list[float]) -> float:
    return sum(0.5 * (ys[i] + ys[i - 1]) * (xs[i] - xs[i - 1]) for i in range(1, len(xs)))


@dataclass
class MultiDoseAccumulationResult:
    """Result of multi-dose accumulation simulation."""

    times_h: list
    c_plasma_mg_L: list
    dose_times_h: list
    auc_single_dose_mg_h_per_L: float
    auc_ss_mg_h_per_L: float
    accumulation_ratio: float
    cmax_single_mg_L: float
    cmax_ss_mg_L: float
    trough_single_mg_L: float
    trough_ss_mg_L: float
    t_half_h: float
    doses_to_steady_state: int
    notes: str


def simulate_multi_dose_accumulation(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    ka_per_h: float | None = None,
    f_oral: float = 1.0,
    tau_h: float = 12.0,
    n_doses: int = 10,
    t_end_h: float = 120.0,
    dt_h: float = 0.01,
) -> MultiDoseAccumulationResult:
    """Simulate multi-dose drug accumulation via Forward Euler.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    dose_mg:
        Dose in mg administered at each dosing event.
    cl_L_per_h:
        Total clearance (L/h).
    vd_L:
        Volume of distribution (L).
    ka_per_h:
        Oral absorption rate constant (per h). None = IV bolus.
    f_oral:
        Oral bioavailability fraction (0-1). Ignored for IV.
    tau_h:
        Dosing interval (h).
    n_doses:
        Number of doses to administer.
    t_end_h:
        Total simulation time (h).
    dt_h:
        Forward Euler time step (h).

    Returns
    -------
    MultiDoseAccumulationResult
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if tau_h <= 0:
        raise ValueError("tau_h must be > 0")
    if n_doses < 1:
        raise ValueError("n_doses must be >= 1")

    ke = cl_L_per_h / vd_L
    t_half = 0.693147 / ke

    is_oral = ka_per_h is not None and ka_per_h > 0.0
    if not is_oral:
        ka_per_h = 0.0  # unused

    # Build list of dose times
    dose_times = [k * tau_h for k in range(n_doses)]

    n_steps = max(int(round(t_end_h / dt_h)), 1)

    times_h: list[float] = []
    c_plasma: list[float] = []

    # State
    cp = 0.0  # plasma concentration (mg/L)
    ag = 0.0  # gut amount (mg) — only used for oral

    for i in range(n_steps + 1):
        t = round(i * dt_h, 10)

        times_h.append(t)
        c_plasma.append(cp)

        if i < n_steps:
            next_t = round((i + 1) * dt_h, 10)

            # Check if any dose falls within [t, next_t)
            for dose_t in dose_times:
                if t <= dose_t < next_t:
                    if is_oral:
                        ag += f_oral * dose_mg
                    else:
                        cp += dose_mg / vd_L

            if is_oral:
                dcp = (ka_per_h * ag / vd_L) - ke * cp
                dag = -ka_per_h * ag
                cp = max(cp + dcp * dt_h, 0.0)
                ag = max(ag + dag * dt_h, 0.0)
            else:
                dcp = -ke * cp
                cp = max(cp + dcp * dt_h, 0.0)

    # --- Single-dose AUC (0 to tau) ---
    tau_idx = 0
    for idx, t in enumerate(times_h):
        if t <= tau_h:
            tau_idx = idx
    auc_single = _auc_trap(times_h[: tau_idx + 1], c_plasma[: tau_idx + 1])
    cmax_single = max(c_plasma[: tau_idx + 1]) if tau_idx > 0 else c_plasma[0]
    trough_single = c_plasma[tau_idx]

    # --- Steady-state: last dosing interval ---
    last_dose_t = dose_times[-1]
    ss_start_idx = 0
    ss_end_idx = len(times_h) - 1
    for idx, t in enumerate(times_h):
        if t >= last_dose_t:
            ss_start_idx = idx
            break
    # End of last interval: last_dose_t + tau_h or t_end_h
    ss_end_t = min(last_dose_t + tau_h, t_end_h)
    for idx, t in enumerate(times_h):
        if t <= ss_end_t:
            ss_end_idx = idx

    ss_times = times_h[ss_start_idx : ss_end_idx + 1]
    ss_conc = c_plasma[ss_start_idx : ss_end_idx + 1]

    if len(ss_times) >= 2:
        auc_ss = _auc_trap(ss_times, ss_conc)
    else:
        auc_ss = auc_single

    cmax_ss = max(ss_conc) if ss_conc else cmax_single
    trough_ss = c_plasma[-1]

    accumulation_ratio = auc_ss / auc_single if auc_single > 0 else 1.0

    # doses to steady state: ceil(4.5 * t_half / tau)
    doses_to_ss = max(1, math.ceil(4.5 * t_half / tau_h))

    route = "oral" if is_oral else "IV bolus"
    notes = (
        f"Drug: {drug_name}; Route: {route}; ke: {ke:.4f}/h; "
        f"t½: {t_half:.2f}h; doses_to_ss: {doses_to_ss}"
    )

    return MultiDoseAccumulationResult(
        times_h=times_h,
        c_plasma_mg_L=c_plasma,
        dose_times_h=dose_times,
        auc_single_dose_mg_h_per_L=auc_single,
        auc_ss_mg_h_per_L=auc_ss,
        accumulation_ratio=accumulation_ratio,
        cmax_single_mg_L=cmax_single,
        cmax_ss_mg_L=cmax_ss,
        trough_single_mg_L=trough_single,
        trough_ss_mg_L=trough_ss,
        t_half_h=t_half,
        doses_to_steady_state=doses_to_ss,
        notes=notes,
    )
