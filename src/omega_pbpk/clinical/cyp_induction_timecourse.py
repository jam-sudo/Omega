"""CYP Enzyme Induction Time-Course Model — Phase 401.

Models the time-course of CYP enzyme induction after repeated dosing with an
inducer drug, including the washout kinetics.

ODE (Forward Euler):
    dE/dt = k_deg * (E_baseline * induction_factor - E)

where induction_factor = 1 + max_fold * conc / (conc + ec50) during dosing,
      induction_factor = 1.0 during washout (inducer removed)
      E_baseline = 1.0 (relative units)

References:
- Ripp et al. (2006) Drug Metab Dispos 34:1501–1510
- FDA Drug Interaction Guidance (2020)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "CYPInductionResult",
    "simulate_cyp_induction",
    "simulate_induction_washout",
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class CYPInductionResult:
    """Result from CYP induction time-course simulation."""

    drug_name: str
    cyp_enzyme: str
    times_h: list
    enzyme_activity_relative: list  # relative to baseline (1.0 = baseline)
    victim_cl_relative: list        # victim drug CL relative to baseline
    victim_auc_ratio: list          # victim AUC ratio (inverse of CL ratio)
    peak_induction_fold: float      # max enzyme activity
    time_to_peak_h: float
    time_to_50pct_induction_h: float
    washout_50pct_h: float          # time after washout for 50% return to baseline
    fda_induction_category: str     # "strong", "moderate", "weak", "none"
    auc_ratio_at_peak: float        # lowest victim AUC ratio = 1 / peak_fold
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fda_category(peak_fold: float) -> str:
    """Classify induction strength per FDA 2020 guidance."""
    if peak_fold > 5.0:
        return "strong"
    elif peak_fold > 2.0:
        return "moderate"
    elif peak_fold > 1.25:
        return "weak"
    else:
        return "none"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def simulate_cyp_induction(
    drug_name: str,
    cyp_enzyme: str,
    inducer_conc_uM: float,
    ec50_uM: float,
    max_fold_induction: float,
    k_deg_per_h: float = 0.02,
    t_end_h: float = 336.0,
    dt_h: float = 1.0,
    washout_start_h: float | None = None,
) -> CYPInductionResult:
    """Simulate CYP enzyme induction time-course.

    Parameters
    ----------
    drug_name:
        Name of the inducer drug.
    cyp_enzyme:
        CYP enzyme being induced (e.g., "CYP3A4").
    inducer_conc_uM:
        Unbound inducer concentration at enzyme site (uM). Use 0 for no induction.
    ec50_uM:
        EC50 for induction (uM). Must be > 0.
    max_fold_induction:
        Maximum fold induction (Emax, >= 1.0).
    k_deg_per_h:
        First-order enzyme degradation rate constant (1/h). Default 0.02 gives
        t1/2_enzyme ~ 34.7 h.
    t_end_h:
        Simulation end time (h). Default 336 h = 14 days.
    dt_h:
        Forward Euler time step (h). Default 1.0 h.
    washout_start_h:
        Time (h) when inducer is removed. None means no washout.

    Returns
    -------
    CYPInductionResult
    """
    # --- Validation ---
    if not cyp_enzyme:
        raise ValueError("cyp_enzyme must be a non-empty string")
    if inducer_conc_uM < 0:
        raise ValueError("inducer_conc_uM must be >= 0")
    if ec50_uM <= 0:
        raise ValueError("ec50_uM must be > 0")
    if max_fold_induction < 1.0:
        raise ValueError("max_fold_induction must be >= 1.0")
    if k_deg_per_h <= 0:
        raise ValueError("k_deg_per_h must be > 0")
    if washout_start_h is not None and washout_start_h < 0:
        raise ValueError("washout_start_h must be >= 0")

    # --- Induction factor at steady concentration ---
    # With inducer present: induction_factor = 1 + max_fold * C / (C + EC50)
    # At baseline (no inducer, or C=0): induction_factor = 1.0
    if inducer_conc_uM > 0:
        ss_induction_factor = 1.0 + max_fold_induction * inducer_conc_uM / (
            inducer_conc_uM + ec50_uM
        )
    else:
        ss_induction_factor = 1.0

    # --- Forward Euler integration ---
    n_steps = int(math.ceil(t_end_h / dt_h)) + 1
    times_h = []
    enzyme_activity = []

    E = 1.0  # baseline
    t = 0.0

    for _ in range(n_steps):
        times_h.append(t)
        enzyme_activity.append(E)

        # Determine induction factor at current time
        if washout_start_h is not None and t >= washout_start_h:
            induction_factor = 1.0  # inducer removed
        else:
            induction_factor = ss_induction_factor

        # dE/dt = k_deg * (E_baseline * induction_factor - E)
        dE = k_deg_per_h * (1.0 * induction_factor - E)
        E = E + dt_h * dE

        t = t + dt_h
        if t > t_end_h + 1e-9:
            break

    # Trim to t_end_h
    while len(times_h) > 1 and times_h[-1] > t_end_h + 1e-9:
        times_h.pop()
        enzyme_activity.pop()

    # --- Victim CL and AUC ratio ---
    victim_cl_relative = list(enzyme_activity)
    victim_auc_ratio = [1.0 / max(e, 1e-9) for e in enzyme_activity]

    # --- Peak induction fold ---
    peak_induction_fold = max(enzyme_activity)

    # --- Time to peak (first time enzyme reaches 90% of max) ---
    peak_threshold = 1.0 + 0.9 * (peak_induction_fold - 1.0)
    time_to_peak_h = t_end_h  # default fallback
    for _i, (t_val, e_val) in enumerate(zip(times_h, enzyme_activity)):
        if e_val >= peak_threshold:
            time_to_peak_h = t_val
            break

    # --- Time to 50% induction ---
    # 50% of (peak - 1) + 1 = (peak + 1) / 2
    half_induction_level = 1.0 + 0.5 * (peak_induction_fold - 1.0)
    time_to_50pct_induction_h = t_end_h  # default fallback
    for t_val, e_val in zip(times_h, enzyme_activity):
        if e_val >= half_induction_level:
            time_to_50pct_induction_h = t_val
            break

    # --- Washout: time after washout_start_h for E to return to (peak+1)/2 ---
    washout_50pct_h = 0.0
    if washout_start_h is not None:
        # Find washout start index
        washout_idx = None
        for i, t_val in enumerate(times_h):
            if t_val >= washout_start_h - 1e-9:
                washout_idx = i
                break

        if washout_idx is not None:
            # Peak enzyme value at washout start (may still be rising if washout early)
            washout_peak = max(enzyme_activity[: washout_idx + 1])
            washout_50pct_level = 1.0 + 0.5 * (washout_peak - 1.0)

            # Find when enzyme drops back to washout_50pct_level after washout
            found_washout = False
            for i in range(washout_idx, len(times_h)):
                if enzyme_activity[i] <= washout_50pct_level:
                    washout_50pct_h = times_h[i] - washout_start_h
                    found_washout = True
                    break
            if not found_washout:
                washout_50pct_h = t_end_h - washout_start_h  # not returned by end

    # --- FDA category ---
    fda_category = _fda_category(peak_induction_fold)
    auc_ratio_at_peak = 1.0 / max(peak_induction_fold, 1e-9)

    # --- Notes ---
    notes = (
        f"Enzyme t1/2 = {math.log(2) / k_deg_per_h:.1f} h; "
        f"SS induction factor = {ss_induction_factor:.2f}x; "
        f"FDA category: {fda_category}"
    )

    return CYPInductionResult(
        drug_name=drug_name,
        cyp_enzyme=cyp_enzyme,
        times_h=times_h,
        enzyme_activity_relative=enzyme_activity,
        victim_cl_relative=victim_cl_relative,
        victim_auc_ratio=victim_auc_ratio,
        peak_induction_fold=peak_induction_fold,
        time_to_peak_h=time_to_peak_h,
        time_to_50pct_induction_h=time_to_50pct_induction_h,
        washout_50pct_h=washout_50pct_h,
        fda_induction_category=fda_category,
        auc_ratio_at_peak=auc_ratio_at_peak,
        notes=notes,
    )


def simulate_induction_washout(
    drug_name: str,
    cyp_enzyme: str,
    inducer_conc_uM: float,
    ec50_uM: float,
    max_fold_induction: float,
    dosing_duration_h: float = 168.0,
    k_deg_per_h: float = 0.02,
) -> CYPInductionResult:
    """Convenience wrapper: simulate induction then 7-day washout.

    Parameters
    ----------
    drug_name:
        Name of the inducer drug.
    cyp_enzyme:
        CYP enzyme being induced.
    inducer_conc_uM:
        Unbound inducer concentration (uM).
    ec50_uM:
        EC50 for induction (uM).
    max_fold_induction:
        Maximum fold induction (>= 1.0).
    dosing_duration_h:
        Duration of dosing before washout (h). Default 168 h = 7 days.
    k_deg_per_h:
        Enzyme degradation rate (1/h). Default 0.02.

    Returns
    -------
    CYPInductionResult with washout_start_h = dosing_duration_h,
    t_end = dosing_duration_h + 168 h (7 days washout).
    """
    t_end = dosing_duration_h + 7.0 * 24.0  # 7 days washout
    return simulate_cyp_induction(
        drug_name=drug_name,
        cyp_enzyme=cyp_enzyme,
        inducer_conc_uM=inducer_conc_uM,
        ec50_uM=ec50_uM,
        max_fold_induction=max_fold_induction,
        k_deg_per_h=k_deg_per_h,
        t_end_h=t_end,
        dt_h=1.0,
        washout_start_h=dosing_duration_h,
    )
