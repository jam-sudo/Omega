"""
Phase 927 — Chronotherapy Optimization

Optimal dosing time selection based on circadian PK variation.
Minimizes toxicity or maximizes efficacy by timing the dose to match circadian rhythms.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["ChronotherapyResult", "optimize_dosing_time", "compare_chronotherapy_vs_fixed"]


@dataclass
class ChronotherapyResult:
    drug_name: str
    optimal_dosing_hour: float
    optimal_auc_above_eff: float
    optimal_time_in_tox: float
    worst_dosing_hour: float
    dosing_hour_results: list
    amplitude_cl: float
    notes: str


def _cl_at_time(t_clock: float, cl_mean: float, amplitude_cl: float, peak_cl_h: float) -> float:
    """Compute circadian CL at a given clock hour."""
    return cl_mean * (1.0 + amplitude_cl * math.cos(2.0 * math.pi * (t_clock - peak_cl_h) / 24.0))


def _simulate_single_dose(
    dose_mg: float,
    route: str,
    dosing_hour: float,
    cl_mean: float,
    vd_L: float,
    ka: float,
    amplitude_cl: float,
    peak_cl_h: float,
    t_end_h: float,
    dt_h: float,
) -> tuple:
    """
    Simulate 1-cpt PK with circadian CL variation for a single dose starting at dosing_hour.
    Returns (times, concentrations).
    """
    times = []
    concentrations = []

    if route == "oral":
        A = dose_mg  # absorption depot (mg)
        C = 0.0  # plasma concentration (mg/L)
        t = 0.0
        while t <= t_end_h + 1e-9:
            t_clock = (dosing_hour + t) % 24.0
            cl_t = _cl_at_time(t_clock, cl_mean, amplitude_cl, peak_cl_h)
            ke_t = cl_t / vd_L
            times.append(t)
            concentrations.append(C)
            # Forward Euler
            dA = -ka * A
            dC = ka * A / vd_L - ke_t * C
            A = max(0.0, A + dA * dt_h)
            C = max(0.0, C + dC * dt_h)
            t = round(t + dt_h, 10)
    else:  # iv bolus
        C = dose_mg / vd_L
        t = 0.0
        while t <= t_end_h + 1e-9:
            t_clock = (dosing_hour + t) % 24.0
            cl_t = _cl_at_time(t_clock, cl_mean, amplitude_cl, peak_cl_h)
            ke_t = cl_t / vd_L
            times.append(t)
            concentrations.append(C)
            dC = -ke_t * C
            C = max(0.0, C + dC * dt_h)
            t = round(t + dt_h, 10)

    return times, concentrations


def _trapezoidal_auc_above_threshold(times: list, concentrations: list, threshold: float) -> float:
    """Compute AUC above a concentration threshold using trapezoidal rule."""
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        c0 = max(0.0, concentrations[i - 1] - threshold)
        c1 = max(0.0, concentrations[i] - threshold)
        auc += 0.5 * (c0 + c1) * dt
    return auc


def _time_above_threshold(times: list, concentrations: list, threshold: float) -> float:
    """Compute time (hours) spent above a concentration threshold."""
    time_above = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        c0 = concentrations[i - 1]
        c1 = concentrations[i]
        # Both above
        if c0 >= threshold and c1 >= threshold:
            time_above += dt
        # Crossing upward: c0 < threshold, c1 >= threshold
        elif c0 < threshold <= c1:
            frac = (c1 - threshold) / (c1 - c0) if c1 != c0 else 0.0
            time_above += frac * dt
        # Crossing downward: c0 >= threshold, c1 < threshold
        elif c0 >= threshold > c1:
            frac = (c0 - threshold) / (c0 - c1) if c0 != c1 else 0.0
            time_above += frac * dt
    return time_above


def _find_cmax(concentrations: list) -> float:
    return max(concentrations) if concentrations else 0.0


def optimize_dosing_time(
    drug_name: str,
    dose_mg: float,
    route: str = "oral",
    dosing_hours: list[float] | None = None,
    cl_mean_L_per_h: float = 10.0,
    vd_L: float = 50.0,
    ka_per_h: float = 1.0,
    amplitude_cl: float = 0.3,
    peak_cl_h: float = 14.0,
    c_eff_threshold_mg_L: float | None = None,
    c_tox_threshold_mg_L: float | None = None,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> ChronotherapyResult:
    """
    Find the optimal dosing hour to maximize the therapeutic ratio
    (AUC above efficacy threshold) / (time above toxicity threshold + 0.1).
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_mean_L_per_h <= 0:
        raise ValueError("cl_mean_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if not (0.0 <= amplitude_cl <= 1.0):
        raise ValueError("amplitude_cl must be in [0, 1]")
    if route not in {"oral", "iv"}:
        raise ValueError(f"route must be 'oral' or 'iv', got '{route}'")

    if dosing_hours is None:
        dosing_hours = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0]

    # First pass: find global Cmax to set default thresholds
    all_concs = []
    for h in dosing_hours:
        _, concs = _simulate_single_dose(
            dose_mg,
            route,
            h,
            cl_mean_L_per_h,
            vd_L,
            ka_per_h,
            amplitude_cl,
            peak_cl_h,
            t_end_h,
            dt_h,
        )
        all_concs.extend(concs)
    global_cmax = max(all_concs) if all_concs else 1.0

    if c_eff_threshold_mg_L is None:
        c_eff_threshold_mg_L = global_cmax * 0.3
    if c_tox_threshold_mg_L is None:
        c_tox_threshold_mg_L = global_cmax * 0.8

    # Second pass: compute metrics per dosing hour
    hour_results = []
    for h in dosing_hours:
        times, concs = _simulate_single_dose(
            dose_mg,
            route,
            h,
            cl_mean_L_per_h,
            vd_L,
            ka_per_h,
            amplitude_cl,
            peak_cl_h,
            t_end_h,
            dt_h,
        )
        auc_eff = _trapezoidal_auc_above_threshold(times, concs, c_eff_threshold_mg_L)
        t_tox = _time_above_threshold(times, concs, c_tox_threshold_mg_L)
        ratio = auc_eff / (t_tox + 0.1)
        hour_results.append(
            {
                "hour": h,
                "auc_above_eff": auc_eff,
                "time_in_tox": t_tox,
                "therapeutic_ratio": ratio,
            }
        )

    # Find optimal and worst
    best = max(hour_results, key=lambda x: x["therapeutic_ratio"])
    worst = min(hour_results, key=lambda x: x["therapeutic_ratio"])

    notes = (
        f"Circadian CL variation amplitude={amplitude_cl:.2f}, peak at {peak_cl_h:.1f}h. "
        f"Efficacy threshold={c_eff_threshold_mg_L:.3f} mg/L, "
        f"Toxicity threshold={c_tox_threshold_mg_L:.3f} mg/L. "
        f"Optimal hour={best['hour']:.1f}h with therapeutic_ratio={best['therapeutic_ratio']:.3f}."
    )

    return ChronotherapyResult(
        drug_name=drug_name,
        optimal_dosing_hour=float(best["hour"]),
        optimal_auc_above_eff=float(best["auc_above_eff"]),
        optimal_time_in_tox=float(best["time_in_tox"]),
        worst_dosing_hour=float(worst["hour"]),
        dosing_hour_results=hour_results,
        amplitude_cl=amplitude_cl,
        notes=notes,
    )


def compare_chronotherapy_vs_fixed(
    drug_name: str,
    dose_mg: float,
    route: str = "oral",
    dosing_hours: list[float] | None = None,
    cl_mean_L_per_h: float = 10.0,
    vd_L: float = 50.0,
    ka_per_h: float = 1.0,
    amplitude_cl: float = 0.3,
    peak_cl_h: float = 14.0,
    c_eff_threshold_mg_L: float | None = None,
    c_tox_threshold_mg_L: float | None = None,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> dict:
    """
    Compare optimal dosing time vs fixed 8am dosing.

    Returns dict with keys: "optimal", "fixed_8am", "improvement_pct".
    """
    optimal = optimize_dosing_time(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route=route,
        dosing_hours=dosing_hours,
        cl_mean_L_per_h=cl_mean_L_per_h,
        vd_L=vd_L,
        ka_per_h=ka_per_h,
        amplitude_cl=amplitude_cl,
        peak_cl_h=peak_cl_h,
        c_eff_threshold_mg_L=c_eff_threshold_mg_L,
        c_tox_threshold_mg_L=c_tox_threshold_mg_L,
        t_end_h=t_end_h,
        dt_h=dt_h,
    )

    # Extract 8am result from dosing_hour_results (if available)
    fixed_8am_data = None
    for r in optimal.dosing_hour_results:
        if r["hour"] == 8.0:
            fixed_8am_data = r
            break

    # If 8am wasn't in the list, simulate it separately using the same thresholds
    if fixed_8am_data is None:
        # Re-derive thresholds from the optimal result's notes is complex;
        # Instead re-run simulate for 8am
        times_8, concs_8 = _simulate_single_dose(
            dose_mg,
            route,
            8.0,
            cl_mean_L_per_h,
            vd_L,
            ka_per_h,
            amplitude_cl,
            peak_cl_h,
            t_end_h,
            dt_h,
        )
        # Use the same thresholds as optimal (derive from global cmax)
        all_concs = []
        hrs = (
            dosing_hours
            if dosing_hours is not None
            else [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0]
        )
        for h in hrs:
            _, concs = _simulate_single_dose(
                dose_mg,
                route,
                h,
                cl_mean_L_per_h,
                vd_L,
                ka_per_h,
                amplitude_cl,
                peak_cl_h,
                t_end_h,
                dt_h,
            )
            all_concs.extend(concs)
        global_cmax = max(all_concs) if all_concs else 1.0
        c_eff = c_eff_threshold_mg_L if c_eff_threshold_mg_L is not None else global_cmax * 0.3
        c_tox = c_tox_threshold_mg_L if c_tox_threshold_mg_L is not None else global_cmax * 0.8
        auc_eff_8 = _trapezoidal_auc_above_threshold(times_8, concs_8, c_eff)
        t_tox_8 = _time_above_threshold(times_8, concs_8, c_tox)
        ratio_8 = auc_eff_8 / (t_tox_8 + 0.1)
        fixed_8am_data = {
            "hour": 8.0,
            "auc_above_eff": auc_eff_8,
            "time_in_tox": t_tox_8,
            "therapeutic_ratio": ratio_8,
        }

    optimal_ratio = max(r["therapeutic_ratio"] for r in optimal.dosing_hour_results)
    fixed_ratio = fixed_8am_data["therapeutic_ratio"]
    if fixed_ratio > 0:
        improvement_pct = (optimal_ratio - fixed_ratio) / fixed_ratio * 100.0
    else:
        improvement_pct = 0.0 if optimal_ratio == 0.0 else float("inf")
    improvement_pct = max(0.0, improvement_pct)  # optimal is always >= fixed by definition

    return {
        "optimal": optimal,
        "fixed_8am": fixed_8am_data,
        "improvement_pct": improvement_pct,
    }
