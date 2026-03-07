"""Circadian dosing optimizer.

Optimizes dosing time based on circadian PK rhythms to maximize efficacy
or minimize toxicity using a 1-compartment oral steady-state PK model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz


@dataclass(frozen=True)
class CircadianDosingResult:
    """Result of circadian dosing optimization."""

    drug_name: str
    objective: str  # "max_efficacy", "min_toxicity", "balanced"
    optimal_dosing_hour: float  # Hour of day (0-24) for optimal dose
    optimal_auc_fraction: float  # AUC in therapeutic window as fraction of total AUC
    all_hours_auc_fraction: list  # AUC fraction for each hour 0-23
    trough_at_optimal_h: float  # Trough at optimal time
    peak_at_optimal_h: float  # Peak at optimal time
    rationale: str
    notes: str


def _ss_concentration(
    t_after_dose: float,
    dose_mg: float,
    vd_L: float,
    ka_per_h: float,
    ke_per_h: float,
    tau_h: float = 24.0,
) -> float:
    """Steady-state 1-compartment oral concentration at time t after dose.

    C_ss(t) = (F*dose/Vd * ka/(ka-ke)) *
              (exp(-ke*t)/(1-exp(-ke*tau)) - exp(-ka*t)/(1-exp(-ka*tau)))

    Assumes F=1 (bioavailability folded into dose_mg).
    """
    if abs(ka_per_h - ke_per_h) < 1e-6:
        # Avoid division by zero: perturb slightly
        ka_per_h = ka_per_h + 1e-5

    factor = (dose_mg / vd_L) * ka_per_h / (ka_per_h - ke_per_h)
    exp_ke = math.exp(-ke_per_h * t_after_dose) / (1.0 - math.exp(-ke_per_h * tau_h))
    exp_ka = math.exp(-ka_per_h * t_after_dose) / (1.0 - math.exp(-ka_per_h * tau_h))
    return factor * (exp_ke - exp_ka)


def _compute_ss_profile(
    dosing_hour: float,
    dose_mg: float,
    vd_L: float,
    ka_per_h: float,
    ke_per_h: float,
    n_points: int = 2400,
    tau_h: float = 24.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute steady-state concentration profile over one 24h cycle.

    Returns absolute time array (0 to 24h) and corresponding concentrations.
    The dose is given at dosing_hour within the 24h day cycle.
    """
    t_abs = np.linspace(0.0, tau_h, n_points + 1)
    t_after_dose = (t_abs - dosing_hour) % tau_h
    conc = np.array(
        [_ss_concentration(t, dose_mg, vd_L, ka_per_h, ke_per_h, tau_h) for t in t_after_dose]
    )
    return t_abs, conc


def _window_auc_fraction(
    t_abs: np.ndarray,
    conc: np.ndarray,
    window_center_h: float,
    window_half_width_h: float,
    tau_h: float = 24.0,
) -> float:
    """Compute AUC fraction for concentrations falling within a time window.

    Window is defined on the absolute 24h day clock.
    Handles wrap-around at midnight.
    """
    total_auc = float(np_trapz(conc, t_abs))
    if total_auc <= 0.0:
        return 0.0

    low = window_center_h - window_half_width_h
    high = window_center_h + window_half_width_h

    # Mask for times in window (handle wrap-around)
    if low < 0:
        mask = (t_abs <= high) | (t_abs >= low + tau_h)
    elif high > tau_h:
        mask = (t_abs >= low) | (t_abs <= high - tau_h)
    else:
        mask = (t_abs >= low) & (t_abs <= high)

    if not np.any(mask):
        return 0.0

    # Build a masked concentration array (zero outside window)
    conc_masked = np.where(mask, conc, 0.0)
    window_auc = float(np_trapz(conc_masked, t_abs))

    return max(0.0, window_auc / total_auc)


def optimize_dosing_time(
    drug_name: str,
    cl_L_per_h: float,
    vd_L: float,
    dose_mg: float,
    ka_per_h: float = 1.5,
    efficacy_peak_h: float = 10.0,
    toxicity_peak_h: float = 2.0,
    therapeutic_low: float | None = None,
    therapeutic_high: float | None = None,
    objective: str = "balanced",
    n_cycles: int = 3,
) -> CircadianDosingResult:
    """Optimize dosing time based on circadian PK rhythms.

    Evaluates all 24 candidate dosing hours (0-23) using steady-state
    1-compartment oral PK, selecting the hour that best meets the objective.

    Parameters
    ----------
    drug_name:
        Identifier string for the drug.
    cl_L_per_h:
        Clearance in L/h (must be > 0).
    vd_L:
        Volume of distribution in L (must be > 0).
    dose_mg:
        Dose in mg (must be > 0).
    ka_per_h:
        Absorption rate constant in 1/h (must be > 0). Default 1.5.
    efficacy_peak_h:
        Hour of day (0-24) when efficacy is maximized. Default 10.0.
    toxicity_peak_h:
        Hour of day (0-24) when toxicity is maximized. Default 2.0.
    therapeutic_low:
        Lower bound of therapeutic window (mg/L). Optional.
    therapeutic_high:
        Upper bound of therapeutic window (mg/L). Optional.
    objective:
        Optimization objective: "max_efficacy", "min_toxicity", or "balanced".
    n_cycles:
        Number of dosing cycles simulated (informational; steady-state used).

    Returns
    -------
    CircadianDosingResult
        Frozen dataclass with optimization results.
    """
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if ka_per_h <= 0:
        raise ValueError(f"ka_per_h must be > 0, got {ka_per_h}")
    valid_objectives = {"max_efficacy", "min_toxicity", "balanced"}
    if objective not in valid_objectives:
        raise ValueError(f"objective must be one of {valid_objectives}, got '{objective}'")

    ke_per_h = cl_L_per_h / vd_L
    tau_h = 24.0

    efficacy_half_width = 4.0  # ±4h window
    toxicity_half_width = 2.0  # ±2h window

    # Evaluate each candidate dosing hour 0-23
    all_efficacy_fractions: list[float] = []
    all_toxicity_fractions: list[float] = []
    all_scores: list[float] = []
    all_auc_fractions: list[float] = []

    for h in range(24):
        t_abs, conc = _compute_ss_profile(
            dosing_hour=float(h),
            dose_mg=dose_mg,
            vd_L=vd_L,
            ka_per_h=ka_per_h,
            ke_per_h=ke_per_h,
        )

        eff_frac = _window_auc_fraction(t_abs, conc, efficacy_peak_h, efficacy_half_width, tau_h)
        tox_frac = _window_auc_fraction(t_abs, conc, toxicity_peak_h, toxicity_half_width, tau_h)

        all_efficacy_fractions.append(eff_frac)
        all_toxicity_fractions.append(tox_frac)
        all_auc_fractions.append(eff_frac)

        if objective == "max_efficacy":
            score = eff_frac
        elif objective == "min_toxicity":
            score = -tox_frac
        else:  # balanced
            score = eff_frac - 0.5 * tox_frac

        all_scores.append(score)

    optimal_idx = int(np.argmax(all_scores))
    optimal_hour = float(optimal_idx)
    optimal_auc_fraction = all_auc_fractions[optimal_idx]

    # Compute trough and peak for the optimal dosing hour
    t_abs_opt, conc_opt = _compute_ss_profile(
        dosing_hour=optimal_hour,
        dose_mg=dose_mg,
        vd_L=vd_L,
        ka_per_h=ka_per_h,
        ke_per_h=ke_per_h,
    )
    trough = float(np.min(conc_opt))
    peak = float(np.max(conc_opt))

    # Build rationale string
    if objective == "max_efficacy":
        rationale = (
            f"Dosing at hour {optimal_hour:.0f} maximizes drug exposure during the "
            f"efficacy window (centered at {efficacy_peak_h:.0f}h ± {efficacy_half_width:.0f}h). "
            f"Efficacy AUC fraction: {optimal_auc_fraction:.3f}."
        )
    elif objective == "min_toxicity":
        tox_at_opt = all_toxicity_fractions[optimal_idx]
        rationale = (
            f"Dosing at hour {optimal_hour:.0f} minimizes drug exposure during the "
            f"toxicity window (centered at {toxicity_peak_h:.0f}h ± {toxicity_half_width:.0f}h). "
            f"Toxicity AUC fraction: {tox_at_opt:.3f}."
        )
    else:
        tox_at_opt = all_toxicity_fractions[optimal_idx]
        rationale = (
            f"Dosing at hour {optimal_hour:.0f} provides balanced optimization: "
            f"efficacy fraction={optimal_auc_fraction:.3f}, "
            f"toxicity fraction={tox_at_opt:.3f} (score={all_scores[optimal_idx]:.3f})."
        )

    notes = (
        f"{drug_name}: CL={cl_L_per_h:.2f} L/h, Vd={vd_L:.1f} L, "
        f"ke={ke_per_h:.3f} 1/h, ka={ka_per_h:.3f} 1/h, "
        f"dose={dose_mg:.1f} mg, n_cycles={n_cycles}, objective={objective}"
    )

    return CircadianDosingResult(
        drug_name=drug_name,
        objective=objective,
        optimal_dosing_hour=optimal_hour,
        optimal_auc_fraction=optimal_auc_fraction,
        all_hours_auc_fraction=all_auc_fractions,
        trough_at_optimal_h=trough,
        peak_at_optimal_h=peak,
        rationale=rationale,
        notes=notes,
    )


def compare_dosing_schedules(
    drug_name: str,
    cl_L_per_h: float,
    vd_L: float,
    dose_mg: float,
    hours: list | None = None,
    **kwargs,
) -> list[CircadianDosingResult]:
    """Evaluate and compare specific dosing hours.

    Evaluates each specified dosing hour using optimize_dosing_time (fixing
    the candidate to that specific hour), then returns results sorted by
    optimal_auc_fraction descending.

    Parameters
    ----------
    drug_name:
        Identifier string for the drug.
    cl_L_per_h:
        Clearance in L/h (must be > 0).
    vd_L:
        Volume of distribution in L (must be > 0).
    dose_mg:
        Dose in mg (must be > 0).
    hours:
        List of dosing hours to compare. Default [0, 6, 12, 18].
    **kwargs:
        Additional keyword arguments passed to optimize_dosing_time.

    Returns
    -------
    list[CircadianDosingResult]
        Results sorted by optimal_auc_fraction descending.
    """
    if hours is None:
        hours = [0, 6, 12, 18]

    # Extract parameters needed for direct profile evaluation
    ka_per_h = kwargs.get("ka_per_h", 1.5)
    efficacy_peak_h = kwargs.get("efficacy_peak_h", 10.0)
    toxicity_peak_h = kwargs.get("toxicity_peak_h", 2.0)
    objective = kwargs.get("objective", "balanced")
    n_cycles = kwargs.get("n_cycles", 3)

    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if ka_per_h <= 0:
        raise ValueError(f"ka_per_h must be > 0, got {ka_per_h}")
    valid_objectives = {"max_efficacy", "min_toxicity", "balanced"}
    if objective not in valid_objectives:
        raise ValueError(f"objective must be one of {valid_objectives}, got '{objective}'")

    ke_per_h = cl_L_per_h / vd_L
    tau_h = 24.0
    efficacy_half_width = 4.0
    toxicity_half_width = 2.0

    results: list[CircadianDosingResult] = []

    for h in hours:
        h_float = float(h)
        t_abs, conc = _compute_ss_profile(
            dosing_hour=h_float,
            dose_mg=dose_mg,
            vd_L=vd_L,
            ka_per_h=ka_per_h,
            ke_per_h=ke_per_h,
        )

        eff_frac = _window_auc_fraction(t_abs, conc, efficacy_peak_h, efficacy_half_width, tau_h)
        tox_frac = _window_auc_fraction(t_abs, conc, toxicity_peak_h, toxicity_half_width, tau_h)
        trough = float(np.min(conc))
        peak = float(np.max(conc))

        if objective == "max_efficacy":
            score = eff_frac
        elif objective == "min_toxicity":
            score = -tox_frac
        else:
            score = eff_frac - 0.5 * tox_frac

        if objective == "max_efficacy":
            rationale = f"Dosing at hour {h_float:.0f}: efficacy AUC fraction = {eff_frac:.3f}."
        elif objective == "min_toxicity":
            rationale = f"Dosing at hour {h_float:.0f}: toxicity AUC fraction = {tox_frac:.3f}."
        else:
            rationale = (
                f"Dosing at hour {h_float:.0f}: efficacy={eff_frac:.3f}, "
                f"toxicity={tox_frac:.3f}, score={score:.3f}."
            )

        notes = (
            f"{drug_name}: CL={cl_L_per_h:.2f} L/h, Vd={vd_L:.1f} L, "
            f"ke={ke_per_h:.3f} 1/h, ka={ka_per_h:.3f} 1/h, "
            f"dose={dose_mg:.1f} mg, n_cycles={n_cycles}, objective={objective}"
        )

        # Build a dummy all_hours_auc_fraction with the single evaluated hour
        all_hours_auc_fraction = [eff_frac if i == int(h) % 24 else 0.0 for i in range(24)]

        results.append(
            CircadianDosingResult(
                drug_name=drug_name,
                objective=objective,
                optimal_dosing_hour=h_float,
                optimal_auc_fraction=eff_frac,
                all_hours_auc_fraction=all_hours_auc_fraction,
                trough_at_optimal_h=trough,
                peak_at_optimal_h=peak,
                rationale=rationale,
                notes=notes,
            )
        )

    # Sort by optimal_auc_fraction descending
    results.sort(key=lambda r: r.optimal_auc_fraction, reverse=True)
    return results
