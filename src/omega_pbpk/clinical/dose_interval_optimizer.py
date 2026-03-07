"""Dosing interval optimizer — minimize peak-trough fluctuation while maintaining therapeutic."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class DoseIntervalResult:
    """Result of dosing interval optimization."""

    drug_name: str
    dose_mg: float
    target_trough_mg_L: float
    target_peak_mg_L: float
    intervals_tested: list[float] = field(default_factory=list)
    troughs: list[float] = field(default_factory=list)
    peaks: list[float] = field(default_factory=list)
    fluctuations: list[float] = field(default_factory=list)
    optimal_interval_h: float = 0.0
    optimal_trough: float = 0.0
    optimal_peak: float = 0.0
    optimal_fluctuation: float = 0.0
    meets_criteria: bool = False
    notes: str = ""


def compute_accumulation_ratio(
    cl_L_per_h: float,
    vd_L: float,
    interval_h: float,
) -> float:
    """Compute steady-state accumulation ratio for a 1-compartment model.

    R = 1 / (1 - exp(-ke * interval_h))

    Parameters
    ----------
    cl_L_per_h : float
        Clearance (L/h).
    vd_L : float
        Volume of distribution (L).
    interval_h : float
        Dosing interval (h).

    Returns
    -------
    float
        Accumulation ratio (>= 1).
    """
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if interval_h <= 0:
        raise ValueError("interval_h must be > 0")

    ke = cl_L_per_h / vd_L
    exp_val = math.exp(-ke * interval_h)
    # Avoid division by zero for very short intervals
    if exp_val >= 1.0 - 1e-12:
        return 1e9  # effectively infinite accumulation
    return 1.0 / (1.0 - exp_val)


def _simulate_multidose_pk(
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    interval_h: float,
    n_doses: int,
    ka_per_h: float,
    f_oral: float,
    dt_h: float = 0.05,
) -> tuple[list[float], list[float]]:
    """Simulate multi-dose 1-compartment oral PK via forward Euler.

    Returns (times, concentrations).
    """
    ke = cl_L_per_h / vd_L
    t_end = interval_h * n_doses
    n_steps = int(round(t_end / dt_h)) + 1

    times: list[float] = []
    concs: list[float] = []

    c_central = 0.0  # plasma concentration (mg/L)
    a_gut = 0.0  # absorption compartment (mg)

    t = 0.0
    dose_absorbed = dose_mg * f_oral

    for i in range(n_steps):
        # Check if a new dose is due at this time point
        # Dose is given at t = 0, interval_h, 2*interval_h, ...
        for d in range(n_doses):
            dose_time = d * interval_h
            # Add dose at the start of each interval (only once per step)
            if abs(t - dose_time) < dt_h * 0.5 and i > 0 or (i == 0 and d == 0):
                # Only add dose at exact dose time; avoid double-dosing
                pass

        times.append(t)
        concs.append(c_central)

        # Dose administration: at t = n * interval_h (integer multiples)
        # Check if current step is a dose time
        # Use integer index approach: dose at step index = round(d * interval_h / dt_h)
        for d in range(n_doses):
            dose_idx = round(d * interval_h / dt_h)
            if i == dose_idx:
                if ka_per_h > 0:
                    a_gut += dose_absorbed
                else:
                    # IV bolus equivalent: instant absorption
                    c_central += dose_absorbed / vd_L

        # Forward Euler
        if ka_per_h > 0:
            da_gut = -ka_per_h * a_gut * dt_h
            dc = (ka_per_h * a_gut / vd_L - ke * c_central) * dt_h
            a_gut = max(0.0, a_gut + da_gut)
        else:
            dc = -ke * c_central * dt_h

        c_central = max(0.0, c_central + dc)
        t = round(t + dt_h, 10)

    return times, concs


def optimize_dose_interval(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    target_trough_mg_L: float,
    target_peak_mg_L: float,
    intervals_to_test: list[float] | None = None,
    n_doses: int = 10,
    ka_per_h: float = 1.0,
    f_oral: float = 1.0,
    dt_h: float = 0.05,
) -> DoseIntervalResult:
    """Optimize dosing interval to meet therapeutic trough/peak targets.

    For each candidate interval, simulates multi-dose PK and computes
    steady-state trough, peak, and fluctuation index.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Dose administered per interval (mg).
    cl_L_per_h : float
        Clearance (L/h).
    vd_L : float
        Volume of distribution (L).
    target_trough_mg_L : float
        Minimum acceptable trough concentration (mg/L).
    target_peak_mg_L : float
        Maximum acceptable peak concentration (mg/L).
    intervals_to_test : list[float]
        Candidate dosing intervals (h). Default: [6, 8, 12, 24].
    n_doses : int
        Number of doses to simulate. Default: 10.
    ka_per_h : float
        First-order absorption rate constant (1/h). Default: 1.0.
    f_oral : float
        Oral bioavailability fraction. Default: 1.0.
    dt_h : float
        ODE integration step (h). Default: 0.05.

    Returns
    -------
    DoseIntervalResult
    """
    if not drug_name:
        raise ValueError("drug_name must not be empty")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if target_trough_mg_L < 0:
        raise ValueError("target_trough_mg_L must be >= 0")
    if target_peak_mg_L <= 0:
        raise ValueError("target_peak_mg_L must be > 0")
    if target_trough_mg_L >= target_peak_mg_L:
        raise ValueError("target_trough_mg_L must be < target_peak_mg_L")
    if n_doses < 1:
        raise ValueError("n_doses must be >= 1")
    if ka_per_h < 0:
        raise ValueError("ka_per_h must be >= 0")
    if not (0.0 < f_oral <= 1.0):
        raise ValueError("f_oral must be in (0, 1]")
    if dt_h <= 0:
        raise ValueError("dt_h must be > 0")

    if intervals_to_test is None:
        intervals_to_test = [6.0, 8.0, 12.0, 24.0]

    if len(intervals_to_test) == 0:
        raise ValueError("intervals_to_test must contain at least one interval")

    for iv in intervals_to_test:
        if iv <= 0:
            raise ValueError("All intervals in intervals_to_test must be > 0")

    troughs: list[float] = []
    peaks: list[float] = []
    fluctuations: list[float] = []

    for interval_h in intervals_to_test:
        times, concs = _simulate_multidose_pk(
            dose_mg=dose_mg,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            interval_h=interval_h,
            n_doses=n_doses,
            ka_per_h=ka_per_h,
            f_oral=f_oral,
            dt_h=dt_h,
        )

        # Use the last dosing interval for steady-state metrics
        last_dose_time = (n_doses - 1) * interval_h
        next_dose_time = n_doses * interval_h

        # Extract concentrations in the last interval
        last_interval_concs = [
            c for t, c in zip(times, concs, strict=False) if last_dose_time <= t <= next_dose_time
        ]

        if len(last_interval_concs) < 2:
            last_interval_concs = concs[-max(2, len(concs) // n_doses) :]

        trough = min(last_interval_concs)
        peak = max(last_interval_concs)

        # Average concentration in last interval
        cavg = sum(last_interval_concs) / len(last_interval_concs) if last_interval_concs else 1e-9
        if cavg < 1e-12:
            cavg = 1e-12

        fluctuation = (peak - trough) / cavg * 100.0

        troughs.append(trough)
        peaks.append(peak)
        fluctuations.append(fluctuation)

    # Select best interval: prefer ones meeting both trough and peak criteria
    # Among those, select smallest fluctuation
    meets_trough = [t >= target_trough_mg_L for t in troughs]
    meets_peak = [p <= target_peak_mg_L for p in peaks]
    meets_both = [mt and mp for mt, mp in zip(meets_trough, meets_peak, strict=False)]

    best_idx: int
    meets_criteria = any(meets_both)

    if meets_criteria:
        # Among intervals that meet both criteria, pick lowest fluctuation
        valid_indices = [i for i, m in enumerate(meets_both) if m]
        best_idx = min(valid_indices, key=lambda i: fluctuations[i])
    else:
        # None meet full criteria; pick smallest fluctuation overall
        best_idx = min(range(len(fluctuations)), key=lambda i: fluctuations[i])

    optimal_interval = intervals_to_test[best_idx]
    optimal_trough = troughs[best_idx]
    optimal_peak = peaks[best_idx]
    optimal_fluctuation = fluctuations[best_idx]

    notes_parts = [f"Tested {len(intervals_to_test)} intervals."]
    if meets_criteria:
        notes_parts.append(
            f"Optimal interval {optimal_interval} h meets both trough and peak targets."
        )
    else:
        notes_parts.append(
            f"No interval meets both targets. "
            f"Selected interval {optimal_interval} h with lowest fluctuation."
        )
    notes = " ".join(notes_parts)

    return DoseIntervalResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        target_trough_mg_L=target_trough_mg_L,
        target_peak_mg_L=target_peak_mg_L,
        intervals_tested=list(intervals_to_test),
        troughs=troughs,
        peaks=peaks,
        fluctuations=fluctuations,
        optimal_interval_h=optimal_interval,
        optimal_trough=optimal_trough,
        optimal_peak=optimal_peak,
        optimal_fluctuation=optimal_fluctuation,
        meets_criteria=meets_criteria,
        notes=notes,
    )
