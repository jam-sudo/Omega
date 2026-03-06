"""Drug accumulation ratio, loading dose, and time-to-steady-state analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AccumulationResult:
    drug_name: str
    dose_mg: float
    dosing_interval_h: float
    n_doses_to_ss: int
    accumulation_ratio_r: float
    peak_trough_ratio: float
    css_max_mg_L: float
    css_min_mg_L: float
    css_avg_mg_L: float
    loading_dose_mg: float
    time_to_ss_h: float
    fluctuation_pct: float
    doses_profile: list[dict]
    warnings: list[str]


def accumulation_ratio(t_half_h: float, tau_h: float) -> float:
    """Accumulation ratio R = 1 / (1 - exp(-0.693 * tau / t_half))."""
    return 1.0 / (1.0 - math.exp(-0.693 * tau_h / t_half_h))


def loading_dose(maintenance_dose_mg: float, t_half_h: float, tau_h: float) -> float:
    """Loading dose = maintenance_dose / (1 - exp(-ke * tau))."""
    ke = 0.693 / t_half_h
    return maintenance_dose_mg / (1.0 - math.exp(-ke * tau_h))


def simulate_accumulation(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    ka_per_h: float = 1.0,
    route: str = "oral",
    dosing_interval_h: float = 12.0,
    n_doses: int = 10,
    dt_h: float = 0.05,
    f: float = 1.0,
    therapeutic_low_mg_L: float | None = None,
    therapeutic_high_mg_L: float | None = None,
) -> AccumulationResult:
    """Simulate multi-dose accumulation using forward Euler 1-compartment model."""
    ke = cl_L_per_h / vd_L
    t_half_h = 0.693 / ke
    R = accumulation_ratio(t_half_h, dosing_interval_h)
    LD = loading_dose(dose_mg, t_half_h, dosing_interval_h)

    n_doses_to_ss = math.ceil(3.32 * t_half_h / dosing_interval_h)
    if n_doses_to_ss < 1:
        n_doses_to_ss = 1

    total_time = n_doses * dosing_interval_h
    n_steps = int(total_time / dt_h) + 1
    t = np.linspace(0.0, total_time, n_steps)

    gut = np.zeros(n_steps)
    central = np.zeros(n_steps)

    dose_times = [i * dosing_interval_h for i in range(n_doses)]

    is_oral = route.lower() == "oral"

    for i in range(1, n_steps):
        dt_actual = t[i] - t[i - 1]

        # Check if a dose should be administered at this step
        for _d_idx, d_time in enumerate(dose_times):
            if t[i - 1] <= d_time < t[i]:
                if is_oral:
                    gut[i - 1] += dose_mg * f
                else:
                    central[i - 1] += dose_mg * f / vd_L

        if is_oral:
            d_gut = -ka_per_h * gut[i - 1] * dt_actual
            d_central = (ka_per_h * gut[i - 1] / vd_L - ke * central[i - 1]) * dt_actual
            gut[i] = gut[i - 1] + d_gut
        else:
            d_central = -ke * central[i - 1] * dt_actual

        central[i] = central[i - 1] + d_central

    # Extract per-dose Cmax and Ctrough
    doses_profile: list[dict] = []
    for d_idx in range(n_doses):
        interval_start = d_idx * dosing_interval_h
        interval_end = (d_idx + 1) * dosing_interval_h
        mask = (t >= interval_start) & (t < interval_end)
        if d_idx == n_doses - 1:
            mask = (t >= interval_start) & (t <= interval_end)
        interval_conc = central[mask]
        cmax = float(np.max(interval_conc)) if len(interval_conc) > 0 else 0.0
        ctrough = float(interval_conc[-1]) if len(interval_conc) > 0 else 0.0
        doses_profile.append({"dose_n": d_idx + 1, "cmax": cmax, "ctrough": ctrough})

    css_max = doses_profile[-1]["cmax"]
    css_min = doses_profile[-1]["ctrough"]
    css_avg = dose_mg * f / (cl_L_per_h * dosing_interval_h)

    peak_trough_ratio = css_max / css_min if css_min > 0 else float("inf")
    fluctuation = (css_max - css_min) / css_avg * 100.0 if css_avg > 0 else 0.0
    time_to_ss = n_doses_to_ss * dosing_interval_h

    warnings: list[str] = []
    if fluctuation > 100.0:
        warnings.append("High fluctuation (>100%): consider extended-release")
    if peak_trough_ratio > 4.0:
        warnings.append("Wide peak-trough ratio: monitor for toxicity/subtherapeutic windows")

    return AccumulationResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        dosing_interval_h=dosing_interval_h,
        n_doses_to_ss=n_doses_to_ss,
        accumulation_ratio_r=R,
        peak_trough_ratio=peak_trough_ratio,
        css_max_mg_L=css_max,
        css_min_mg_L=css_min,
        css_avg_mg_L=css_avg,
        loading_dose_mg=LD,
        time_to_ss_h=time_to_ss,
        fluctuation_pct=fluctuation,
        doses_profile=doses_profile,
        warnings=warnings,
    )


def dosing_regimen_comparison(
    drug_name: str,
    daily_dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    ka_per_h: float = 1.0,
) -> list[dict]:
    """Compare once-daily, BID, TID, QID regimens."""
    regimens = [
        ("once-daily", 24.0, 1),
        ("BID", 12.0, 2),
        ("TID", 8.0, 3),
        ("QID", 6.0, 4),
    ]
    results: list[dict] = []
    for name, tau_h, n_per_day in regimens:
        dose_mg = daily_dose_mg / n_per_day
        r = simulate_accumulation(
            drug_name=drug_name,
            dose_mg=dose_mg,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            ka_per_h=ka_per_h,
            dosing_interval_h=tau_h,
            n_doses=20,
        )
        results.append(
            {
                "regimen": name,
                "tau_h": tau_h,
                "dose_mg": dose_mg,
                "css_avg": r.css_avg_mg_L,
                "fluctuation_pct": r.fluctuation_pct,
                "peak_trough_ratio": r.peak_trough_ratio,
            }
        )
    return results


__all__ = [
    "AccumulationResult",
    "accumulation_ratio",
    "simulate_accumulation",
    "dosing_regimen_comparison",
]
