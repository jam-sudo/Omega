"""Pharmacodynamic tolerance and receptor desensitization model.

Models receptor desensitization (reduced sensitivity) due to repeated drug exposure.
Uses 1-compartment oral PK + Emax PD with time-varying receptor sensitivity.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ToleranceModelResult:
    """Result of a tolerance/desensitization simulation."""

    drug_name: str
    dose_mg: float
    times_h: list
    c_plasma_mg_L: list
    effect_pct: list
    sensitivity: list
    dose_1_peak_effect: float
    dose_n_peak_effect: float
    tolerance_ratio: float  # dose_n / dose_1; < 1 indicates tolerance
    min_sensitivity: float
    recovery_time_h: float
    notes: str


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------


def simulate_tolerance_pk_pd(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    ka_per_h: float = 1.0,
    f_oral: float = 0.8,
    emax: float = 100.0,
    ec50_mg_L: float = 1.0,
    k_desens_per_h: float = 0.1,
    k_recovery_per_h: float = 0.05,
    dosing_interval_h: float = 12.0,
    n_doses: int = 10,
    dt_h: float = 0.1,
) -> ToleranceModelResult:
    """Simulate multi-dose PK/PD with receptor desensitization tolerance.

    PK: 1-compartment oral (depot → central, forward Euler).
    PD: Emax model with time-varying receptor sensitivity S (0–1).

    Effect(t) = emax * S(t) * C(t) / (ec50 + C(t))
    dS/dt = k_recovery * (1 - S) - k_desens * Effect_norm * S
    """
    # --- validation ---
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
    if n_doses < 1:
        raise ValueError("n_doses must be >= 1")
    if k_desens_per_h <= 0:
        raise ValueError("k_desens_per_h must be > 0")
    if dosing_interval_h <= 0:
        raise ValueError("dosing_interval_h must be > 0")

    # derived rate constants
    ke = cl_L_per_h / vd_L  # elimination rate (/h)

    # total simulation time: n_doses intervals + extra recovery window
    recovery_window_h = 72.0
    t_total_h = n_doses * dosing_interval_h + recovery_window_h
    n_steps = int(round(t_total_h / dt_h))

    # dose amount in gut (mg)
    absorbed_dose_mg = f_oral * dose_mg

    # state variables
    gut_mg = 0.0  # amount in gut compartment
    central_mg = 0.0  # amount in central compartment
    S = 1.0  # receptor sensitivity

    # arrays
    times_h: list[float] = []
    c_plasma: list[float] = []
    effect_pct_arr: list[float] = []
    sensitivity_arr: list[float] = []

    # build set of dosing times (step indices)
    dosing_steps: set[int] = set()
    for d in range(n_doses):
        step_idx = int(round(d * dosing_interval_h / dt_h))
        dosing_steps.add(step_idx)

    last_dose_time_h = (n_doses - 1) * dosing_interval_h

    # --- forward Euler integration ---
    for step in range(n_steps + 1):
        t = step * dt_h

        # apply dose at dosing steps
        if step in dosing_steps:
            gut_mg += absorbed_dose_mg

        # concentrations
        C = central_mg / vd_L  # mg/L

        # effect (% of max)
        effect = emax * S * C / (ec50_mg_L + C)
        effect_norm = effect / emax  # 0 to 1

        # record
        times_h.append(t)
        c_plasma.append(C)
        effect_pct_arr.append(effect)
        sensitivity_arr.append(S)

        # derivatives (forward Euler)
        d_gut = -ka_per_h * gut_mg
        d_central = ka_per_h * gut_mg - ke * central_mg
        d_S = k_recovery_per_h * (1.0 - S) - k_desens_per_h * effect_norm * S

        # update
        gut_mg += d_gut * dt_h
        central_mg += d_central * dt_h
        S += d_S * dt_h

        # clamp S to [0, 1]
        if S < 0.0:
            S = 0.0
        if S > 1.0:
            S = 1.0

    # --- extract metrics ---
    # Dose 1: peak effect in first interval [0, tau)
    steps_per_dose = int(round(dosing_interval_h / dt_h))
    dose_1_indices = range(0, min(steps_per_dose, len(effect_pct_arr)))
    dose_1_peak = max(effect_pct_arr[i] for i in dose_1_indices) if dose_1_indices else 0.0

    # Dose N: peak effect in last dose interval [tau*(n-1), tau*n)
    last_dose_start_step = int(round(last_dose_time_h / dt_h))
    last_dose_end_step = last_dose_start_step + steps_per_dose
    last_dose_end_step = min(last_dose_end_step, len(effect_pct_arr))
    dose_n_indices = range(last_dose_start_step, last_dose_end_step)
    dose_n_peak = max(effect_pct_arr[i] for i in dose_n_indices) if dose_n_indices else 0.0

    # tolerance ratio
    if dose_1_peak > 0:
        tolerance_ratio = dose_n_peak / dose_1_peak
    else:
        tolerance_ratio = 1.0

    # minimum sensitivity
    min_sens = min(sensitivity_arr)

    # recovery time: scan from last dose time forward; find first t where S >= 0.9
    recovery_time_h = t_total_h  # default if never recovers
    last_dose_step = int(round(last_dose_time_h / dt_h))
    for idx in range(last_dose_step, len(sensitivity_arr)):
        if sensitivity_arr[idx] >= 0.9:
            recovery_time_h = times_h[idx] - last_dose_time_h
            break

    # notes
    severity_info = assess_tolerance_development(
        ToleranceModelResult(
            drug_name=drug_name,
            dose_mg=dose_mg,
            times_h=times_h,
            c_plasma_mg_L=c_plasma,
            effect_pct=effect_pct_arr,
            sensitivity=sensitivity_arr,
            dose_1_peak_effect=dose_1_peak,
            dose_n_peak_effect=dose_n_peak,
            tolerance_ratio=tolerance_ratio,
            min_sensitivity=min_sens,
            recovery_time_h=recovery_time_h,
            notes="",
        )
    )
    notes = (
        f"Tolerance severity: {severity_info['severity']}. "
        f"Dose-1 peak effect: {dose_1_peak:.1f}%, "
        f"Dose-{n_doses} peak effect: {dose_n_peak:.1f}%, "
        f"Tolerance ratio: {tolerance_ratio:.3f}, "
        f"Min receptor sensitivity: {min_sens:.3f}, "
        f"Recovery time to S=0.9: {recovery_time_h:.1f}h. "
        f"k_desens={k_desens_per_h}/h, k_recovery={k_recovery_per_h}/h."
    )

    return ToleranceModelResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times_h,
        c_plasma_mg_L=c_plasma,
        effect_pct=effect_pct_arr,
        sensitivity=sensitivity_arr,
        dose_1_peak_effect=dose_1_peak,
        dose_n_peak_effect=dose_n_peak,
        tolerance_ratio=tolerance_ratio,
        min_sensitivity=min_sens,
        recovery_time_h=recovery_time_h,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Assessment helper
# ---------------------------------------------------------------------------


def assess_tolerance_development(result: ToleranceModelResult) -> dict:
    """Assess tolerance severity based on tolerance_ratio.

    Returns:
        dict with keys:
            severity: "none" | "mild" | "moderate" | "severe"
            tolerance_ratio: float
            min_sensitivity: float
            description: str
    """
    ratio = result.tolerance_ratio

    if ratio > 0.9:
        severity = "none"
        description = "No clinically significant tolerance detected."
    elif ratio > 0.7:
        severity = "mild"
        description = "Mild tolerance: modest reduction in peak effect."
    elif ratio > 0.5:
        severity = "moderate"
        description = (
            "Moderate tolerance: notable reduction in peak effect, dose adjustment may be needed."
        )
    else:
        severity = "severe"
        description = (
            "Severe tolerance: substantial loss of efficacy, alternative therapy may be required."
        )

    return {
        "severity": severity,
        "tolerance_ratio": ratio,
        "min_sensitivity": result.min_sensitivity,
        "description": description,
    }
