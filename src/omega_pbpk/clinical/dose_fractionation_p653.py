"""Dose fractionation optimizer for repeated dosing schedules.

Compares QD, BID, TID, and QID regimens to maximize time within the
therapeutic window while minimizing peak toxicity.  Uses a 1-compartment
oral PK model with Forward Euler integration over 3 days (72 h) and
evaluates steady-state metrics from the last 24 h.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DoseFractionationResult:
    """Steady-state metrics for a single fractionation schedule."""

    drug_name: str
    total_daily_dose_mg: float
    n_doses_per_day: int
    dose_interval_h: float
    cmax_ss_mg_L: float
    cmin_ss_mg_L: float
    cavg_ss_mg_L: float
    fluctuation_pct: float
    time_above_mec_h_per_day: float
    time_above_mtc_h_per_day: float
    time_in_window_h_per_day: float
    peak_trough_ratio: float
    recommended: bool
    score: float
    notes: str


def simulate_dose_fractionation(
    drug_name: str,
    total_daily_dose_mg: float,
    n_doses_per_day: int,
    cl_L_per_h: float,
    vd_L: float,
    ka_per_h: float = 1.0,
    mec_mg_L: float = 0.1,
    mtc_mg_L: float = 2.0,
    t_end_h: float = 72.0,
    dt_h: float = 0.1,
) -> DoseFractionationResult:
    """Simulate a fractionated dosing schedule and return SS metrics."""

    # --- validation ---
    if total_daily_dose_mg <= 0:
        raise ValueError("total_daily_dose_mg must be > 0")
    if n_doses_per_day not in (1, 2, 3, 4):
        raise ValueError("n_doses_per_day must be 1, 2, 3, or 4")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if mec_mg_L <= 0:
        raise ValueError("mec_mg_L must be > 0")
    if mtc_mg_L <= mec_mg_L:
        raise ValueError("mtc_mg_L must be > mec_mg_L")

    dose_per_admin = total_daily_dose_mg / n_doses_per_day
    dose_interval = 24.0 / n_doses_per_day
    k_el = cl_L_per_h / vd_L
    bioavailability = 0.85

    n_steps = int(t_end_h / dt_h) + 1

    a_gut = 0.0
    c_plasma = 0.0

    # Build set of dose times
    dose_times: set[int] = set()
    t = 0.0
    while t < t_end_h - 1e-9:
        step_idx = int(round(t / dt_h))
        dose_times.add(step_idx)
        t += dose_interval

    # Storage for last 24 h
    ss_start_step = int(round((t_end_h - 24.0) / dt_h))
    c_last24: list[float] = []

    for i in range(n_steps):
        if i in dose_times:
            a_gut += dose_per_admin * bioavailability

        # Record before advancing (state at time i*dt_h)
        if i >= ss_start_step:
            c_last24.append(c_plasma)

        # Forward Euler
        da_gut = -ka_per_h * a_gut
        dc_plasma = ka_per_h * a_gut / vd_L - k_el * c_plasma
        a_gut += da_gut * dt_h
        c_plasma += dc_plasma * dt_h

    # --- steady-state metrics from last 24 h ---
    cmax_ss = max(c_last24)
    cmin_ss = min(c_last24)

    # trapezoidal AUC for last 24 h
    auc_24 = sum(0.5 * (c_last24[i] + c_last24[i - 1]) * dt_h for i in range(1, len(c_last24)))
    cavg_ss = auc_24 / 24.0 if auc_24 > 0 else 0.0

    fluctuation_pct = (cmax_ss - cmin_ss) / cavg_ss * 100.0 if cavg_ss > 0 else 0.0
    peak_trough_ratio = cmax_ss / cmin_ss if cmin_ss > 0 else float("inf")

    time_above_mec = sum(dt_h for c in c_last24 if c >= mec_mg_L)
    time_above_mtc = sum(dt_h for c in c_last24 if c >= mtc_mg_L)
    time_in_window = sum(dt_h for c in c_last24 if mec_mg_L <= c < mtc_mg_L)

    score = time_in_window - time_above_mtc * 2.0

    recommended = fluctuation_pct < 100 and time_in_window > 12

    notes = (
        f"{n_doses_per_day}x/day, interval={dose_interval:.1f}h, dose={dose_per_admin:.2f}mg each"
    )

    return DoseFractionationResult(
        drug_name=drug_name,
        total_daily_dose_mg=total_daily_dose_mg,
        n_doses_per_day=n_doses_per_day,
        dose_interval_h=dose_interval,
        cmax_ss_mg_L=cmax_ss,
        cmin_ss_mg_L=cmin_ss,
        cavg_ss_mg_L=cavg_ss,
        fluctuation_pct=fluctuation_pct,
        time_above_mec_h_per_day=time_above_mec,
        time_above_mtc_h_per_day=time_above_mtc,
        time_in_window_h_per_day=time_in_window,
        peak_trough_ratio=peak_trough_ratio,
        recommended=recommended,
        score=score,
        notes=notes,
    )


def compare_fractionation(
    drug_name: str,
    total_daily_dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    ka_per_h: float = 1.0,
    mec_mg_L: float = 0.1,
    mtc_mg_L: float = 2.0,
) -> list[DoseFractionationResult]:
    """Compare QD/BID/TID/QID and mark the best as recommended."""

    results: list[DoseFractionationResult] = []
    for n in (1, 2, 3, 4):
        r = simulate_dose_fractionation(
            drug_name=drug_name,
            total_daily_dose_mg=total_daily_dose_mg,
            n_doses_per_day=n,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            ka_per_h=ka_per_h,
            mec_mg_L=mec_mg_L,
            mtc_mg_L=mtc_mg_L,
        )
        results.append(r)

    best_score = max(r.score for r in results)

    # Rebuild with recommended flag
    out: list[DoseFractionationResult] = []
    for r in results:
        out.append(
            DoseFractionationResult(
                drug_name=r.drug_name,
                total_daily_dose_mg=r.total_daily_dose_mg,
                n_doses_per_day=r.n_doses_per_day,
                dose_interval_h=r.dose_interval_h,
                cmax_ss_mg_L=r.cmax_ss_mg_L,
                cmin_ss_mg_L=r.cmin_ss_mg_L,
                cavg_ss_mg_L=r.cavg_ss_mg_L,
                fluctuation_pct=r.fluctuation_pct,
                time_above_mec_h_per_day=r.time_above_mec_h_per_day,
                time_above_mtc_h_per_day=r.time_above_mtc_h_per_day,
                time_in_window_h_per_day=r.time_in_window_h_per_day,
                peak_trough_ratio=r.peak_trough_ratio,
                recommended=(r.score == best_score),
                score=r.score,
                notes=r.notes,
            )
        )

    return out
