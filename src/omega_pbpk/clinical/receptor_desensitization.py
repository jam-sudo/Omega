"""
Phase 233 — Receptor Desensitization Model

Models receptor desensitization and internalization after prolonged drug exposure.
Relevant for GPCRs, opioids, and beta-agonists.

3-state ODE model:
  R  = active receptors
  D  = desensitized receptors
  I  = internalized receptors

Drug effect is proportional to receptor availability (R/R0).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DesensitizationResult:
    """Result of a receptor desensitization simulation."""

    drug_name: str
    dose_mg: float
    route: str
    times_h: np.ndarray
    c_plasma_mg_L: np.ndarray
    r_active: np.ndarray
    r_desensitized: np.ndarray
    r_internalized: np.ndarray
    drug_effect: np.ndarray
    peak_effect: float
    effect_at_end: float
    r_active_min: float
    tolerance_developed: bool
    notes: list[str] = field(default_factory=list)


def simulate_desensitization(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    emax: float = 1.0,
    ec50_mg_L: float = 1.0,
    hill_n: float = 1.0,
    k_des_per_h: float = 0.5,
    k_recycle_per_h: float = 0.1,
    k_intern_per_h: float = 0.05,
    k_recycle_d_per_h: float = 0.02,
    r0: float = 100.0,
    ksyn_per_h: float = 2.0,
    kdeg_per_h: float = 0.02,
    route: str = "oral",
    ka_per_h: float = 1.0,
    t_end_h: float = 168.0,
    dt_h: float = 0.5,
) -> DesensitizationResult:
    """Simulate receptor desensitization and internalization over time.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Dose in milligrams (must be > 0).
    cl_L_per_h:
        Clearance in L/h (must be > 0).
    vd_L:
        Volume of distribution in L (must be > 0).
    emax:
        Maximum drug effect (must be > 0).
    ec50_mg_L:
        EC50 in mg/L (must be > 0).
    hill_n:
        Hill coefficient (must be > 0).
    k_des_per_h:
        Desensitization rate constant per h (must be >= 0).
    k_recycle_per_h:
        Recycling rate from internalized to active (must be >= 0).
    k_intern_per_h:
        Internalization rate of active receptors (must be >= 0).
    k_recycle_d_per_h:
        Recycling rate from desensitized state (must be >= 0).
    r0:
        Initial number of active receptors (must be > 0).
    ksyn_per_h:
        Receptor synthesis rate (receptors/h, must be >= 0).
    kdeg_per_h:
        Receptor degradation rate from internalized pool (must be >= 0).
    route:
        Administration route: "oral" or "iv".
    ka_per_h:
        Oral absorption rate constant (must be > 0 for oral route).
    t_end_h:
        Simulation end time in hours.
    dt_h:
        Integration time step in hours.

    Returns
    -------
    DesensitizationResult
        Simulation result containing time-course arrays and summary metrics.
    """
    # --- Input validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if emax <= 0:
        raise ValueError("emax must be > 0")
    if ec50_mg_L <= 0:
        raise ValueError("ec50_mg_L must be > 0")
    if hill_n <= 0:
        raise ValueError("hill_n must be > 0")
    if k_des_per_h < 0:
        raise ValueError("k_des_per_h must be >= 0")
    if k_recycle_per_h < 0:
        raise ValueError("k_recycle_per_h must be >= 0")
    if k_intern_per_h < 0:
        raise ValueError("k_intern_per_h must be >= 0")
    if k_recycle_d_per_h < 0:
        raise ValueError("k_recycle_d_per_h must be >= 0")
    if r0 <= 0:
        raise ValueError("r0 must be > 0")
    if ksyn_per_h < 0:
        raise ValueError("ksyn_per_h must be >= 0")
    if kdeg_per_h < 0:
        raise ValueError("kdeg_per_h must be >= 0")
    if route not in {"oral", "iv"}:
        raise ValueError("route must be 'oral' or 'iv'")
    if route == "oral" and ka_per_h <= 0:
        raise ValueError("ka_per_h must be > 0 for oral route")

    ke = cl_L_per_h / vd_L  # elimination rate constant

    n_steps = int(t_end_h / dt_h) + 1
    times = np.linspace(0.0, t_end_h, n_steps)

    c_plasma = np.zeros(n_steps)
    r_active = np.zeros(n_steps)
    r_desensitized = np.zeros(n_steps)
    r_internalized = np.zeros(n_steps)
    drug_effect = np.zeros(n_steps)

    # Initial conditions
    if route == "oral":
        gut_dose_mg = dose_mg  # drug in gut compartment (mg)
        c0_mg_L = 0.0
    else:
        gut_dose_mg = 0.0
        c0_mg_L = dose_mg / vd_L  # immediate IV bolus (mg/L)

    c_plasma[0] = c0_mg_L
    r_active[0] = r0
    r_desensitized[0] = 0.0
    r_internalized[0] = 0.0

    # Drug effect at t=0
    c_n = c0_mg_L**hill_n
    ec_n = ec50_mg_L**hill_n
    drug_effect[0] = emax * (r0 / r0) * c_n / (ec_n + c_n) if (c_n + ec_n) > 0 else 0.0

    # Forward Euler integration
    for i in range(1, n_steps):
        c = c_plasma[i - 1]
        R = r_active[i - 1]
        D = r_desensitized[i - 1]
        I_r = r_internalized[i - 1]

        # PK: 1-compartment with or without gut absorption
        if route == "oral":
            # gut → plasma
            gut_absorbed = ka_per_h * gut_dose_mg * dt_h
            gut_dose_mg = max(0.0, gut_dose_mg - ka_per_h * gut_dose_mg * dt_h)
            dc = (gut_absorbed / vd_L) - ke * c
        else:
            dc = -ke * c
            gut_absorbed = 0.0

        c_new = (
            max(0.0, c + dc * dt_h)
            if route == "iv"
            else max(0.0, c + (gut_absorbed / vd_L - ke * c) * dt_h)
        )
        # Recalculate properly for oral using explicit forward Euler on a single step:
        if route == "oral":
            # Reset gut tracking — use stored remaining gut drug
            pass  # already handled above

        c_plasma[i] = c_new

        # Receptor ODEs
        # dR/dt = k_recycle*I - k_des*C*R - k_intern*R + ksyn
        dR = k_recycle_per_h * I_r - k_des_per_h * c * R - k_intern_per_h * R + ksyn_per_h
        # dD/dt = k_des*C*R - k_recycle_d*D - k_intern_d*D
        dD = k_des_per_h * c * R - k_recycle_d_per_h * D - k_intern_per_h * D
        # dI/dt = k_intern*R + k_intern_d*D - k_recycle*I - kdeg*I
        dI = k_intern_per_h * R + k_intern_per_h * D - k_recycle_per_h * I_r - kdeg_per_h * I_r

        r_active[i] = max(0.0, R + dR * dt_h)
        r_desensitized[i] = max(0.0, D + dD * dt_h)
        r_internalized[i] = max(0.0, I_r + dI * dt_h)

        # Drug effect — proportional to receptor availability
        c_i = c_plasma[i]
        c_n_i = c_i**hill_n
        r_frac = r_active[i] / r0
        denom = ec50_mg_L**hill_n + c_n_i
        drug_effect[i] = emax * r_frac * c_n_i / denom if denom > 0 else 0.0

    peak_effect = float(np.max(drug_effect))
    effect_at_end = float(drug_effect[-1])
    r_active_min = float(np.min(r_active)) / r0  # fraction of initial

    notes: list[str] = []
    if k_des_per_h == 0.0:
        notes.append("k_des=0: no desensitization expected")
    if route == "oral":
        notes.append(f"Oral route, ka={ka_per_h:.2f}/h")

    tolerance_developed = effect_at_end < 0.5 * peak_effect if peak_effect > 0 else False

    return DesensitizationResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route=route,
        times_h=times,
        c_plasma_mg_L=c_plasma,
        r_active=r_active,
        r_desensitized=r_desensitized,
        r_internalized=r_internalized,
        drug_effect=drug_effect,
        peak_effect=peak_effect,
        effect_at_end=effect_at_end,
        r_active_min=r_active_min,
        tolerance_developed=tolerance_developed,
        notes=notes,
    )


def tolerance_index(result: DesensitizationResult) -> float:
    """Compute the tolerance index as percentage of effect lost from peak to end.

    Parameters
    ----------
    result:
        A :class:`DesensitizationResult` from :func:`simulate_desensitization`.

    Returns
    -------
    float
        Percentage tolerance: (peak_effect - effect_at_end) / peak_effect * 100.
        Returns 0.0 if peak_effect is zero.
    """
    if result.peak_effect <= 0:
        return 0.0
    ti = (result.peak_effect - result.effect_at_end) / result.peak_effect * 100.0
    return float(np.clip(ti, 0.0, 100.0))
