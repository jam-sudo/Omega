"""Phase 674 — Dose Frequency Optimizer.

Optimize dosing frequency for a drug to achieve target therapeutic outcomes.
Focuses on QD vs BID vs TID vs QID based on PK/PD criteria.

This module complements regimen_optimizer.py (Phase 26), which uses
scipy minimize_scalar for total daily dose/regimen; here we use Forward
Euler simulation and analytical accumulation for frequency selection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FrequencyOptimizationResult:
    drug_name: str
    frequency_per_day: int
    dosing_interval_h: float
    dose_per_administration_mg: float
    daily_dose_mg: float
    css_max_mg_L: float
    css_min_mg_L: float
    css_avg_mg_L: float
    fluctuation_pct: float
    time_in_therapeutic_pct: float
    accumulation_ratio: float
    patient_convenience_score: float
    composite_score: float
    recommended: bool
    notes: str


_CONVENIENCE = {1: 100.0, 2: 75.0, 3: 50.0, 4: 25.0}


def _simulate_ss_1cpt_oral(
    dose_mg: float,
    tau_h: float,
    t_half_h: float,
    vd_L: float,
    f_oral: float,
    ka_per_h: float,
    dt: float = 0.05,
) -> tuple[float, float, float, float]:
    """Forward Euler simulation of 1-cpt oral until steady state.

    Simulate n_intervals dosing intervals and extract last interval Cmax/Cmin/Cavg.
    Returns (css_max, css_min, css_avg, accumulation_ratio).
    """
    ke = math.log(2.0) / t_half_h
    # Run until steady state: use 10 intervals or 5 half-lives worth of doses,
    # whichever gives more intervals; minimum 8.
    n_intervals = max(8, int(5.0 * t_half_h / tau_h) + 2)

    A_gut = 0.0
    C = 0.0  # plasma concentration mg/L

    # Track single-dose Cmax for accumulation ratio
    single_dose_Cmax: float | None = None

    # Lists to store the last interval for analysis
    last_C: list[float] = []
    last_t: list[float] = []

    t = 0.0

    steps_per_interval = int(tau_h / dt) + 1

    for interval in range(n_intervals):
        # Administer dose
        A_gut += f_oral * dose_mg

        t_local = 0.0
        local_C: list[float] = []
        local_t: list[float] = []

        steps = steps_per_interval
        for _ in range(steps):
            local_C.append(C)
            local_t.append(t)

            # Forward Euler
            dA_gut = -ka_per_h * A_gut
            dC = (ka_per_h * A_gut) / vd_L - ke * C

            A_gut += dA_gut * dt
            C += dC * dt
            t += dt
            t_local += dt

            if t_local >= tau_h:
                break

        # Single dose Cmax
        if single_dose_Cmax is None:
            single_dose_Cmax = max(local_C)

        if interval == n_intervals - 1:
            last_C = local_C
            last_t = local_t

    css_max = max(last_C) if last_C else C
    css_min = min(last_C) if last_C else C

    # Trapezoidal AUC over last interval
    auc = 0.0
    for i in range(len(last_C) - 1):
        auc += (last_C[i] + last_C[i + 1]) / 2.0 * (last_t[i + 1] - last_t[i])
    t_span = last_t[-1] - last_t[0] if len(last_t) > 1 else tau_h
    css_avg = auc / t_span if t_span > 0.0 else (css_max + css_min) / 2.0

    acc_ratio = css_max / max(1e-9, single_dose_Cmax or 1e-9)
    acc_ratio = max(1.0, acc_ratio)

    return css_max, css_min, css_avg, acc_ratio


def _time_in_therapeutic(
    dose_mg: float,
    tau_h: float,
    t_half_h: float,
    vd_L: float,
    f_oral: float,
    ka_per_h: float,
    mec: float,
    mtc: float,
    dt: float = 0.05,
) -> float:
    """Compute % of last steady-state interval spent between MEC and MTC."""
    ke = math.log(2.0) / t_half_h
    n_intervals = max(8, int(5.0 * t_half_h / tau_h) + 2)

    A_gut = 0.0
    C = 0.0
    t = 0.0

    steps_per_interval = int(tau_h / dt) + 1
    last_C: list[float] = []

    for interval in range(n_intervals):
        A_gut += f_oral * dose_mg
        t_local = 0.0
        local_C: list[float] = []

        steps = steps_per_interval
        for _ in range(steps):
            local_C.append(C)

            dA_gut = -ka_per_h * A_gut
            dC = (ka_per_h * A_gut) / vd_L - ke * C
            A_gut += dA_gut * dt
            C += dC * dt
            t += dt
            t_local += dt

            if t_local >= tau_h:
                break

        if interval == n_intervals - 1:
            last_C = local_C

    if not last_C:
        return 0.0

    total_time = 0.0
    in_window_time = 0.0
    for _i, c_val in enumerate(last_C):
        step = dt  # uniform steps
        total_time += step
        if mec <= c_val <= mtc:
            in_window_time += step

    return min(100.0, 100.0 * in_window_time / max(1e-9, total_time))


def optimize_dosing_frequency(
    drug_name: str,
    total_daily_dose_mg: float,
    t_half_h: float,
    mec_mg_L: float = 0.5,
    mtc_mg_L: float = 5.0,
    vd_L: float = 50.0,
    f_oral: float = 0.8,
    ka_per_h: float = 1.0,
    frequencies_per_day: list[int] | None = None,
) -> list[FrequencyOptimizationResult]:
    """Return list of results for each frequency, sorted by composite_score descending."""
    # Validation
    if total_daily_dose_mg <= 0.0:
        raise ValueError("total_daily_dose_mg must be positive")
    if t_half_h <= 0.0:
        raise ValueError("t_half_h must be positive")
    if mec_mg_L <= 0.0 or mtc_mg_L <= 0.0:
        raise ValueError("mec_mg_L and mtc_mg_L must be positive")
    if mec_mg_L >= mtc_mg_L:
        raise ValueError("mec_mg_L must be less than mtc_mg_L")
    if vd_L <= 0.0:
        raise ValueError("vd_L must be positive")
    if not (0.0 < f_oral <= 1.0):
        raise ValueError("f_oral must be in (0, 1]")

    if frequencies_per_day is None:
        frequencies_per_day = [1, 2, 3, 4]

    results: list[FrequencyOptimizationResult] = []

    for freq in frequencies_per_day:
        tau_h = 24.0 / freq
        dose_per_admin = total_daily_dose_mg / freq
        convenience = _CONVENIENCE.get(freq, max(0.0, 100.0 - (freq - 1) * 25.0))

        css_max, css_min, css_avg, acc_ratio = _simulate_ss_1cpt_oral(
            dose_mg=dose_per_admin,
            tau_h=tau_h,
            t_half_h=t_half_h,
            vd_L=vd_L,
            f_oral=f_oral,
            ka_per_h=ka_per_h,
        )

        # Fluctuation
        if css_avg > 1e-9:
            fluctuation = max(0.0, (css_max - css_min) / css_avg * 100.0)
        else:
            fluctuation = 0.0

        # TIR
        tir = _time_in_therapeutic(
            dose_mg=dose_per_admin,
            tau_h=tau_h,
            t_half_h=t_half_h,
            vd_L=vd_L,
            f_oral=f_oral,
            ka_per_h=ka_per_h,
            mec=mec_mg_L,
            mtc=mtc_mg_L,
        )

        # Composite score: 0.5*TIR + 0.3*convenience + 0.2*(100 - fluctuation/2)
        fluct_term = max(0.0, 100.0 - fluctuation / 2.0)
        composite = 0.5 * tir + 0.3 * convenience + 0.2 * fluct_term
        composite = min(100.0, max(0.0, composite))

        notes_parts = [f"Freq={freq}x/day, tau={tau_h:.1f}h."]
        if css_max > mtc_mg_L:
            notes_parts.append("Cmax exceeds MTC.")
        if css_min < mec_mg_L:
            notes_parts.append("Trough below MEC.")
        notes = " ".join(notes_parts)

        results.append(
            FrequencyOptimizationResult(
                drug_name=drug_name,
                frequency_per_day=freq,
                dosing_interval_h=tau_h,
                dose_per_administration_mg=dose_per_admin,
                daily_dose_mg=total_daily_dose_mg,
                css_max_mg_L=css_max,
                css_min_mg_L=css_min,
                css_avg_mg_L=css_avg,
                fluctuation_pct=fluctuation,
                time_in_therapeutic_pct=tir,
                accumulation_ratio=acc_ratio,
                patient_convenience_score=convenience,
                composite_score=composite,
                recommended=False,
                notes=notes,
            )
        )

    # Sort descending by composite_score
    results.sort(key=lambda r: r.composite_score, reverse=True)

    # Mark best as recommended
    if results:
        best = results[0]
        results[0] = FrequencyOptimizationResult(
            drug_name=best.drug_name,
            frequency_per_day=best.frequency_per_day,
            dosing_interval_h=best.dosing_interval_h,
            dose_per_administration_mg=best.dose_per_administration_mg,
            daily_dose_mg=best.daily_dose_mg,
            css_max_mg_L=best.css_max_mg_L,
            css_min_mg_L=best.css_min_mg_L,
            css_avg_mg_L=best.css_avg_mg_L,
            fluctuation_pct=best.fluctuation_pct,
            time_in_therapeutic_pct=best.time_in_therapeutic_pct,
            accumulation_ratio=best.accumulation_ratio,
            patient_convenience_score=best.patient_convenience_score,
            composite_score=best.composite_score,
            recommended=True,
            notes=best.notes,
        )

    return results


def quick_frequency_recommendation(
    drug_name: str,
    t_half_h: float,
    mec_mg_L: float,
    mtc_mg_L: float,
) -> str:
    """Return simple frequency recommendation based on t_half and therapeutic window width."""
    # Narrow therapeutic window check
    mtw = (mtc_mg_L - mec_mg_L) / max(1e-9, mec_mg_L)  # relative window width

    if t_half_h > 16.0:
        freq = "QD"
    elif t_half_h >= 8.0:
        freq = "BID"
    elif t_half_h >= 4.0:
        freq = "TID"
    else:
        freq = "QID"

    # Narrow TW (MTW < 2): increase frequency by one step
    if mtw < 2.0:
        if freq == "QD":
            freq = "BID"
        elif freq == "BID":
            freq = "TID"
        elif freq == "TID":
            freq = "QID"
        # QID stays QID

    return freq
