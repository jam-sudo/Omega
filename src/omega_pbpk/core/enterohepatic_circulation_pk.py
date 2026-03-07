"""Enterohepatic circulation (EHC) pharmacokinetics model.

Models the recycling of drug via bile excretion, gallbladder storage,
and intestinal re-absorption triggered by meals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = ["EHCResult", "simulate_ehc_pk", "compare_ehc_vs_no_ehc"]


@dataclass
class EHCResult:
    """Result of enterohepatic circulation PK simulation."""

    drug_name: str
    dose_mg: float
    f_biliary: float
    f_reabsorbed: float

    times_h: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    a_bile_mg: list = field(default_factory=list)
    a_intestine_mg: list = field(default_factory=list)

    cmax_mg_L: float = 0.0
    tmax_h: float = 0.0
    auc_mg_h_per_L: float = 0.0
    t_half_apparent_h: float = 0.0
    n_ehc_peaks: int = 0
    fe_biliary_pct: float = 0.0
    notes: str = ""


def _gallbladder_gate(t: float, meal_interval_h: float, meal_duration_h: float) -> float:
    """Return 1.0 during meal times, 0.0 otherwise."""
    phase = t % meal_interval_h
    return 1.0 if phase < meal_duration_h else 0.0


def _trapezoidal_auc(times: list, values: list) -> float:
    """Compute AUC via trapezoidal rule."""
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        auc += 0.5 * (values[i] + values[i - 1]) * dt
    return auc


def _count_secondary_peaks(c_plasma: list) -> int:
    """Count local maxima after the first (primary) peak."""
    if len(c_plasma) < 3:
        return 0

    # Find the first peak index
    first_peak_idx = 0
    for i in range(1, len(c_plasma) - 1):
        if c_plasma[i] >= c_plasma[i - 1] and c_plasma[i] >= c_plasma[i + 1]:
            first_peak_idx = i
            break

    # Count secondary peaks after the first peak
    n_peaks = 0
    for i in range(first_peak_idx + 2, len(c_plasma) - 1):
        if c_plasma[i] > c_plasma[i - 1] and c_plasma[i] > c_plasma[i + 1]:
            # Must be a meaningful peak (>5% of cmax)
            cmax = max(c_plasma)
            if cmax > 0 and c_plasma[i] / cmax > 0.05:
                n_peaks += 1

    return n_peaks


def simulate_ehc_pk(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    f_biliary: float = 0.3,
    f_reabsorbed: float = 0.8,
    meal_interval_h: float = 6.0,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
) -> EHCResult:
    """Simulate enterohepatic circulation pharmacokinetics.

    Args:
        drug_name: Name of the drug.
        dose_mg: IV bolus dose in mg.
        cl_L_per_h: Total clearance in L/h.
        vd_L: Volume of distribution in L.
        f_biliary: Fraction of dose routed through biliary excretion (0 to <1).
        f_reabsorbed: Fraction of bile drug reabsorbed from intestine (0 to 1).
        meal_interval_h: Hours between meals (gallbladder emptying).
        t_end_h: Simulation end time in hours.
        dt_h: Forward Euler time step in hours.

    Returns:
        EHCResult with plasma concentration and compartment amounts over time.
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if not (0.0 <= f_biliary < 1.0):
        raise ValueError(f"f_biliary must be in [0, 1), got {f_biliary}")
    if not (0.0 <= f_reabsorbed <= 1.0):
        raise ValueError(f"f_reabsorbed must be in [0, 1], got {f_reabsorbed}")

    # Rate constants
    # f_biliary fraction of total CL is diverted to biliary excretion (not additive)
    k_liver = cl_L_per_h * f_biliary / vd_L  # biliary diversion rate (1/h) from plasma
    k_direct = cl_L_per_h * (1.0 - f_biliary) / vd_L  # direct elimination rate (1/h)
    k_bile = 0.5  # biliary secretion rate (1/h)
    k_empty = 2.0  # gallbladder emptying rate during meals (1/h)
    meal_duration_h = 1.0
    k_intestine = 1.0  # intestinal re-absorption rate (1/h)

    # Initial conditions (IV bolus)
    c_plasma = dose_mg / vd_L  # mg/L
    a_liver = 0.0  # mg
    a_bile = 0.0  # mg
    a_intestine = 0.0  # mg

    times = []
    c_plasma_list = []
    a_bile_list = []
    a_intestine_list = []

    n_steps = int(t_end_h / dt_h) + 1
    t = 0.0

    for _ in range(n_steps):
        times.append(t)
        c_plasma_list.append(c_plasma)
        a_bile_list.append(a_bile)
        a_intestine_list.append(a_intestine)

        gate = _gallbladder_gate(t, meal_interval_h, meal_duration_h)

        # ODE right-hand sides
        # Reabsorption brings f_reabsorbed fraction back to plasma
        # k_intestine removes all drug from intestine; f_reabsorbed fraction returns to plasma
        intestine_outflow = k_intestine * a_intestine  # mg/h total
        reabsorption_rate = intestine_outflow * f_reabsorbed / vd_L  # mg/L/h into plasma

        dc_plasma = -k_direct * c_plasma - k_liver * c_plasma + reabsorption_rate
        da_liver = k_liver * c_plasma * vd_L - k_bile * a_liver
        da_bile = k_bile * a_liver - k_empty * a_bile * gate
        # Loss from intestine includes both reabsorbed and fecal (total)
        da_intestine = k_empty * a_bile * gate - k_intestine * a_intestine

        # Forward Euler update
        c_plasma = max(0.0, c_plasma + dc_plasma * dt_h)
        a_liver = max(0.0, a_liver + da_liver * dt_h)
        a_bile = max(0.0, a_bile + da_bile * dt_h)
        a_intestine = max(0.0, a_intestine + da_intestine * dt_h)

        t += dt_h

    # Compute summary metrics
    cmax = max(c_plasma_list)
    tmax = times[c_plasma_list.index(cmax)]
    auc = _trapezoidal_auc(times, c_plasma_list)

    # Apparent half-life: use terminal phase (last 25% of time)
    terminal_start = int(0.75 * len(times))
    t_half = 0.0
    c_terminal = c_plasma_list[terminal_start:]
    t_terminal = times[terminal_start:]
    if len(c_terminal) >= 2 and c_terminal[0] > 0 and c_terminal[-1] > 0:
        ratio = c_terminal[-1] / c_terminal[0]
        if 0 < ratio < 1:
            elapsed = t_terminal[-1] - t_terminal[0]
            # C(t) = C0 * exp(-lambda * t)  =>  lambda = -ln(ratio)/elapsed
            lam = -math.log(ratio) / elapsed
            if lam > 0:
                t_half = math.log(2) / lam

    n_peaks = _count_secondary_peaks(c_plasma_list)
    fe_biliary_pct = f_biliary * (1.0 - f_reabsorbed) * 100.0

    notes = (
        f"EHC simulation: f_biliary={f_biliary:.2f}, f_reabsorbed={f_reabsorbed:.2f}. "
        f"Meals every {meal_interval_h}h. {n_peaks} secondary plasma peak(s) detected."
    )

    return EHCResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        f_biliary=f_biliary,
        f_reabsorbed=f_reabsorbed,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        a_bile_mg=a_bile_list,
        a_intestine_mg=a_intestine_list,
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc_mg_h_per_L=auc,
        t_half_apparent_h=t_half,
        n_ehc_peaks=n_peaks,
        fe_biliary_pct=fe_biliary_pct,
        notes=notes,
    )


def compare_ehc_vs_no_ehc(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    f_biliary: float = 0.3,
) -> dict:
    """Compare EHC simulation vs no-EHC (f_biliary=0).

    Args:
        drug_name: Name of the drug.
        dose_mg: IV bolus dose in mg.
        cl_L_per_h: Total clearance in L/h.
        vd_L: Volume of distribution in L.
        f_biliary: Fraction routed through biliary excretion.

    Returns:
        Dict with keys: ehc_result, no_ehc_result, auc_ratio, notes.
    """
    ehc_result = simulate_ehc_pk(
        drug_name=drug_name,
        dose_mg=dose_mg,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
        f_biliary=f_biliary,
    )
    no_ehc_result = simulate_ehc_pk(
        drug_name=drug_name,
        dose_mg=dose_mg,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
        f_biliary=0.0,
    )

    auc_ehc = ehc_result.auc_mg_h_per_L
    auc_no_ehc = no_ehc_result.auc_mg_h_per_L
    auc_ratio = auc_ehc / auc_no_ehc if auc_no_ehc > 0 else 1.0

    notes = (
        f"EHC (f_biliary={f_biliary:.2f}) AUC={auc_ehc:.2f} vs "
        f"no-EHC AUC={auc_no_ehc:.2f}. AUC ratio={auc_ratio:.3f}."
    )

    return {
        "ehc_result": ehc_result,
        "no_ehc_result": no_ehc_result,
        "auc_ratio": auc_ratio,
        "notes": notes,
    }
