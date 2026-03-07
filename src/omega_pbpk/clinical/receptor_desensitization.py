"""Receptor Desensitization / Tachyphylaxis Model (Phase 306).

Models receptor desensitization during repeated drug exposure via:
  - Beta-arrestin recruitment
  - Receptor internalization (endocytosis)
  - Receptor resensitization

Drug PK uses a 1-compartment oral model with multi-dose superposition.
Receptor state is tracked as R_active (fraction 0–1, starts at 1).

ODE system (Forward Euler):
  Absorption depot: dA/dt = -ka * A
  Drug:             dC/dt = ka * A / Vd - (CL/Vd) * C
  Receptor:         dR_active/dt = -k_desens * occupancy * R_active
                                  + k_resens * (1 - R_active)
  where occupancy = C / (EC50 + C)
  Effect = emax * occupancy * R_active
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DesensitizationResult:
    """Result from receptor desensitization simulation.

    Attributes:
        drug_name: Name of the drug.
        n_doses: Number of doses administered.
        interval_h: Dosing interval (h).
        k_desens_per_h: Receptor desensitization rate constant (1/h).
        k_resens_per_h: Receptor resensitization rate constant (1/h).
        times_h: Simulation time points (h).
        c_drug: Drug plasma concentration at each time point (mg/L).
        r_active: Active receptor fraction at each time point (0–1).
        effect: Pharmacodynamic effect at each time point.
        peak_effects: Effect at the peak concentration of each dose.
        tachyphylaxis_index: Percent reduction in effect from dose 1 to last dose.
        time_to_50pct_resensitization_h: Time after last dose for R to recover 50%
            of lost receptor activity (h).
        notes: Descriptive notes.
    """

    drug_name: str
    n_doses: int
    interval_h: float
    k_desens_per_h: float
    k_resens_per_h: float
    times_h: list[float]
    c_drug: list[float]
    r_active: list[float]
    effect: list[float]
    peak_effects: list[float]
    tachyphylaxis_index: float
    time_to_50pct_resensitization_h: float
    notes: str


def _validate_inputs(
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    ec50_mg_L: float,
    k_desens_per_h: float,
    k_resens_per_h: float,
    n_doses: int,
) -> None:
    """Validate simulation inputs, raising ValueError on invalid parameters."""
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if ec50_mg_L <= 0:
        raise ValueError(f"ec50_mg_L must be > 0, got {ec50_mg_L}")
    if k_desens_per_h <= 0:
        raise ValueError(f"k_desens_per_h must be > 0, got {k_desens_per_h}")
    if k_resens_per_h <= 0:
        raise ValueError(f"k_resens_per_h must be > 0, got {k_resens_per_h}")
    if n_doses < 1:
        raise ValueError(f"n_doses must be >= 1, got {n_doses}")


def tachyphylaxis_index(effects_at_peaks: list[float]) -> float:
    """Compute tachyphylaxis index from peak effects at each dose.

    TI = (first_peak_effect - last_peak_effect) / first_peak_effect * 100

    Returns 0 if there is only one dose or if first_peak_effect is zero.
    Range: 0 (no tachyphylaxis) to 100 (complete tachyphylaxis).

    Args:
        effects_at_peaks: List of peak effects, one per dose.

    Returns:
        Tachyphylaxis index as a percentage (0–100).
    """
    if len(effects_at_peaks) < 2:
        return 0.0
    first = effects_at_peaks[0]
    if first <= 0:
        return 0.0
    last = effects_at_peaks[-1]
    ti = (first - last) / first * 100.0
    return float(max(0.0, min(100.0, ti)))


def simulate_desensitization(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    ec50_mg_L: float,
    emax: float = 100.0,
    k_desens_per_h: float = 0.2,
    k_resens_per_h: float = 0.05,
    n_doses: int = 5,
    interval_h: float = 8.0,
    ka_per_h: float = 1.5,
    t_end_h: float | None = None,
    dt_h: float = 0.1,
) -> DesensitizationResult:
    """Simulate receptor desensitization during repeated drug dosing.

    Uses 1-compartment oral PK with Forward Euler integration.
    Multiple doses are applied by adding dose to the absorption depot at
    each scheduled dosing time during the simulation.

    Args:
        drug_name: Name of the drug.
        dose_mg: Oral dose per administration (mg).
        cl_L_per_h: Drug clearance (L/h).
        vd_L: Volume of distribution (L).
        ec50_mg_L: Drug concentration producing 50% of max effect (mg/L).
        emax: Maximum pharmacodynamic effect (default 100).
        k_desens_per_h: Receptor desensitization rate constant (1/h).
        k_resens_per_h: Receptor resensitization rate constant (1/h).
        n_doses: Number of doses to administer.
        interval_h: Dosing interval (h).
        ka_per_h: First-order oral absorption rate constant (1/h).
        t_end_h: Total simulation duration (h). Defaults to
            n_doses * interval_h + 3 * t_half.
        dt_h: Time step for Forward Euler integration (h).

    Returns:
        DesensitizationResult with full time-course and summary metrics.
    """
    _validate_inputs(dose_mg, cl_L_per_h, vd_L, ec50_mg_L, k_desens_per_h, k_resens_per_h, n_doses)

    ke = cl_L_per_h / vd_L  # elimination rate constant (1/h)
    t_half = math.log(2.0) / ke

    if t_end_h is None:
        t_end_h = n_doses * interval_h + 3.0 * t_half

    n_steps = max(1, round(t_end_h / dt_h))

    # Scheduled dose times (h)
    dose_times_h = [i * interval_h for i in range(n_doses)]

    # State variables
    a_depot = 0.0  # absorption depot (mg)
    c = 0.0  # plasma concentration (mg/L)
    r = 1.0  # active receptor fraction (0–1)

    times = np.zeros(n_steps + 1)
    arr_c = np.zeros(n_steps + 1)
    arr_r = np.zeros(n_steps + 1)
    arr_eff = np.zeros(n_steps + 1)

    arr_r[0] = r

    # Apply first dose at t=0
    dose_idx = 0
    if dose_times_h and abs(dose_times_h[0]) < 1e-9:
        a_depot += dose_mg
        dose_idx = 1

    def _effect(conc: float, r_frac: float) -> float:
        occ = conc / (ec50_mg_L + conc) if (ec50_mg_L + conc) > 0 else 0.0
        return emax * occ * r_frac

    arr_eff[0] = _effect(c, r)

    for i in range(n_steps):
        t_curr = i * dt_h
        t_next = (i + 1) * dt_h
        times[i + 1] = t_next

        # Apply any doses scheduled within (t_curr, t_next]
        while dose_idx < n_doses and dose_times_h[dose_idx] <= t_next + 1e-9:
            if dose_times_h[dose_idx] > t_curr + 1e-9:
                a_depot += dose_mg
            dose_idx += 1

        # Compute occupancy at current state for receptor ODE
        occupancy = c / (ec50_mg_L + c) if (ec50_mg_L + c) > 0 else 0.0

        # Forward Euler derivatives
        da = -ka_per_h * a_depot
        dc = ka_per_h * a_depot / vd_L - ke * c
        dr = -k_desens_per_h * occupancy * r + k_resens_per_h * (1.0 - r)

        a_depot = max(0.0, a_depot + da * dt_h)
        c = max(0.0, c + dc * dt_h)
        r = max(0.0, min(1.0, r + dr * dt_h))

        arr_c[i + 1] = c
        arr_r[i + 1] = r
        arr_eff[i + 1] = _effect(c, r)

    # Extract peak effects for each dose window
    peak_effects: list[float] = []
    for d in range(n_doses):
        t_start = dose_times_h[d]
        t_stop = dose_times_h[d + 1] if d < n_doses - 1 else t_start + 3.0 * t_half
        mask = (times >= t_start) & (times <= t_stop)
        peak_effects.append(float(np.max(arr_eff[mask])) if np.any(mask) else 0.0)

    ti = tachyphylaxis_index(peak_effects)

    # Time to 50% resensitization after last dose
    last_dose_t = dose_times_h[-1]
    last_dose_step = min(int(round(last_dose_t / dt_h)), n_steps)
    r_at_last = arr_r[last_dose_step]
    # Target: recover 50% of the receptor deficit
    r_target = r_at_last + 0.5 * (1.0 - r_at_last)

    time_to_50pct_h: float = float("inf")
    for i in range(last_dose_step, n_steps + 1):
        if arr_r[i] >= r_target:
            time_to_50pct_h = times[i] - last_dose_t
            break

    if not math.isfinite(time_to_50pct_h):
        # Analytical fallback: R(t) ≈ 1 - (1 - r0)*exp(-k_resens * t)
        deficit = 1.0 - r_at_last
        if deficit > 1e-9:
            frac = (r_target - r_at_last) / (1.0 - r_at_last)
            if 0.0 < frac < 1.0:
                time_to_50pct_h = -math.log(1.0 - frac) / k_resens_per_h
        else:
            time_to_50pct_h = 0.0

    notes = (
        f"k_desens={k_desens_per_h:.3f}/h, k_resens={k_resens_per_h:.3f}/h, "
        f"n_doses={n_doses}, TI={ti:.1f}%, "
        f"t50_resens={time_to_50pct_h:.1f}h"
    )

    return DesensitizationResult(
        drug_name=drug_name,
        n_doses=n_doses,
        interval_h=interval_h,
        k_desens_per_h=k_desens_per_h,
        k_resens_per_h=k_resens_per_h,
        times_h=times.tolist(),
        c_drug=arr_c.tolist(),
        r_active=arr_r.tolist(),
        effect=arr_eff.tolist(),
        peak_effects=peak_effects,
        tachyphylaxis_index=ti,
        time_to_50pct_resensitization_h=time_to_50pct_h,
        notes=notes,
    )
