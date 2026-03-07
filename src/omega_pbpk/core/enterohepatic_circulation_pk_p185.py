"""Enterohepatic circulation (EHC) PK model — Phase 185.

Models bile secretion, gallbladder storage, intestinal reabsorption,
and meal-triggered recycling producing multiple plasma peaks.

4-Compartment forward-Euler ODE:
  C_plasma  (mg/L)  – systemic drug
  A_bile    (mg)    – bile duct / gallbladder pool
  A_gut     (mg)    – gut lumen awaiting reabsorption
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = [
    "EHCPKResult",
    "simulate_ehc_pk",
    "compare_ehc_levels",
]

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class EHCPKResult:
    """Result of an enterohepatic circulation PK simulation."""

    drug_name: str
    dose_mg: float
    f_reabs: float
    times_h: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    cmax_mg_L: float = 0.0
    tmax_h: float = 0.0
    auc_mg_h_per_L: float = 0.0
    secondary_peaks_count: int = 0
    t_half_effective_h: float = 0.0
    notes: str = ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _trapz(xs: list, ys: list) -> float:
    """Manual trapezoidal integration."""
    total = 0.0
    for i in range(1, len(xs)):
        total += 0.5 * (ys[i] + ys[i - 1]) * (xs[i] - xs[i - 1])
    return total


def _count_secondary_peaks(times: list, concs: list, cmax: float) -> int:
    """Count peaks > 10% of Cmax that occur after Tmax."""
    if not concs or cmax <= 0.0:
        return 0

    # find tmax index
    peak_idx = concs.index(max(concs))
    threshold = 0.10 * cmax
    count = 0
    # scan from peak_idx+1 onward for local maxima above threshold
    n = len(concs)
    for i in range(peak_idx + 2, n - 1):
        if concs[i] > concs[i - 1] and concs[i] > concs[i + 1]:
            if concs[i] > threshold:
                count += 1
    return count


def _effective_half_life(times: list, concs: list, cmax: float) -> float:
    """Estimate effective half-life from terminal slope after Cmax."""
    if cmax <= 0.0 or len(concs) < 4:
        return 0.0

    peak_idx = concs.index(max(concs))
    # use last 30% of time points after Cmax for terminal slope
    post = [(times[i], concs[i]) for i in range(peak_idx, len(concs)) if concs[i] > 0]
    if len(post) < 3:
        return 0.0

    start = len(post) * 7 // 10
    segment = post[start:]
    if len(segment) < 2:
        segment = post[-2:]

    # log-linear regression
    log_c = [math.log(max(c, 1e-12)) for _, c in segment]
    t_vals = [t for t, _ in segment]
    n = len(t_vals)
    if n < 2:
        return 0.0
    mean_t = sum(t_vals) / n
    mean_lc = sum(log_c) / n
    num = sum((t_vals[i] - mean_t) * (log_c[i] - mean_lc) for i in range(n))
    den = sum((t_vals[i] - mean_t) ** 2 for i in range(n))
    if den == 0.0 or num >= 0.0:
        return 0.0
    slope = num / den  # negative
    return math.log(2.0) / (-slope)


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------


def simulate_ehc_pk(
    drug_name: str,
    dose_mg: float,
    k_bile_per_h: float,
    f_reabs: float,
    meal_times_h: list,
    cl_L_per_h: float,
    vd_L: float,
    t_end_h: float = 48.0,
    dt_h: float = 0.01,
    iv: bool = False,
) -> EHCPKResult:
    """Simulate enterohepatic circulation pharmacokinetics.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    dose_mg:
        Single IV or oral bolus dose (mg).  For oral a first-pass factor of
        0.8 is applied to Vd scaling (simple approximation).
    k_bile_per_h:
        First-order biliary secretion rate constant (h⁻¹).
    f_reabs:
        Fraction of drug exiting bile that is reabsorbed (0–1).
    meal_times_h:
        Times (h) at which a meal triggers gallbladder emptying.
    cl_L_per_h:
        Systemic clearance (L/h).
    vd_L:
        Volume of distribution (L).
    t_end_h:
        Simulation end time (h).
    dt_h:
        Forward-Euler step size (h).
    iv:
        If True, full dose enters plasma directly; otherwise oral bolus.

    Returns
    -------
    EHCPKResult
    """
    # --- validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if not (0.0 <= f_reabs <= 1.0):
        raise ValueError("f_reabs must be in [0, 1]")

    ke = cl_L_per_h / vd_L  # elimination rate (h⁻¹)
    k_reabs = f_reabs * 1.0  # reabsorption rate = f_reabs × 1 h⁻¹ (normalised)
    k_feces = (1.0 - f_reabs) * 1.0  # fecal loss rate
    gallbladder_base = 0.0
    GALLBLADDER_EMPTYING_RATE = 2.0  # h⁻¹ during meal window (1 h)

    # Sort meal times for efficient look-up
    sorted_meals = sorted(meal_times_h)

    # Initial conditions
    if iv:
        c_plasma = dose_mg / vd_L
    else:
        c_plasma = dose_mg / vd_L  # simplified oral bolus (absorbed instantly)
    a_bile = 0.0
    a_gut = 0.0

    times: list = []
    concs: list = []

    t = 0.0
    steps = int(round(t_end_h / dt_h))
    for _ in range(steps + 1):
        times.append(t)
        concs.append(max(c_plasma, 0.0))

        # Determine gallbladder emptying rate at time t
        k_gb = gallbladder_base
        for tm in sorted_meals:
            if tm <= t < tm + 1.0:
                k_gb = GALLBLADDER_EMPTYING_RATE
                break

        # ODEs (forward Euler)
        bile_out = k_gb * a_bile
        dC = -ke * c_plasma - k_bile_per_h * c_plasma + k_reabs * a_gut / vd_L
        dA_bile = k_bile_per_h * c_plasma * vd_L - bile_out
        dA_gut = bile_out - k_reabs * a_gut - k_feces * a_gut

        c_plasma = max(c_plasma + dC * dt_h, 0.0)
        a_bile = max(a_bile + dA_bile * dt_h, 0.0)
        a_gut = max(a_gut + dA_gut * dt_h, 0.0)

        t = round(t + dt_h, 10)

    cmax = max(concs)
    tmax = times[concs.index(cmax)]
    auc = _trapz(times, concs)
    secondary = _count_secondary_peaks(times, concs, cmax)
    t_half = _effective_half_life(times, concs, cmax)

    notes_parts = []
    if meal_times_h:
        notes_parts.append(f"Meals at {sorted_meals} h")
    if f_reabs == 0.0:
        notes_parts.append("No EHC (f_reabs=0)")
    notes = "; ".join(notes_parts)

    return EHCPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        f_reabs=f_reabs,
        times_h=times,
        c_plasma_mg_L=concs,
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc_mg_h_per_L=auc,
        secondary_peaks_count=secondary,
        t_half_effective_h=t_half,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------------


def compare_ehc_levels(
    drug_name: str,
    dose_mg: float,
    f_reabs_values: list,
    k_bile_per_h: float = 0.05,
    meal_times_h: list | None = None,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    t_end_h: float = 48.0,
    dt_h: float = 0.01,
) -> list:
    """Simulate EHC at multiple f_reabs levels and return sorted by AUC descending.

    Parameters
    ----------
    f_reabs_values:
        List of reabsorption fractions to evaluate.
    All other parameters are passed to :func:`simulate_ehc_pk`.

    Returns
    -------
    list[EHCPKResult] sorted by auc_mg_h_per_L descending.
    """
    if meal_times_h is None:
        meal_times_h = [4.0, 12.0, 24.0]

    results = []
    for f in f_reabs_values:
        res = simulate_ehc_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            k_bile_per_h=k_bile_per_h,
            f_reabs=f,
            meal_times_h=meal_times_h,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            t_end_h=t_end_h,
            dt_h=dt_h,
        )
        results.append(res)

    results.sort(key=lambda r: r.auc_mg_h_per_L, reverse=True)
    return results
