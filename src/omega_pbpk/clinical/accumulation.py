"""Drug accumulation index and multiple-dose PK at steady state."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz


@dataclass(frozen=True)
class AccumulationResult:
    drug_name: str
    dose_mg: float
    dosing_interval_h: float
    n_doses: int
    route: str
    times_h: list[float]
    cp_profile: list[float]  # full multi-dose concentration profile
    cmax_ss: float  # steady-state Cmax
    cmin_ss: float  # steady-state Cmin (trough)
    cavg_ss: float  # average concentration at SS
    auc_ss: float  # AUC over one dosing interval at SS
    accumulation_index: float  # R = Cmax_ss / Cmax_1 (or AUC_ss/AUC_1)
    fluctuation_pct: float  # (Cmax-Cmin)/Cavg * 100
    t_ss_90pct_h: float  # time to reach 90% of SS Cmax
    n_doses_to_ss: int  # doses needed to reach 90% SS
    thalf_effective_h: float  # effective t½ = 0.693/ke
    analytical_r: float  # Analytical R = 1/(1-exp(-ke*tau))
    # Phase 74 aliases
    accumulation_ratio_r: float = 1.0
    css_max_mg_L: float = 0.0
    css_min_mg_L: float = 0.0
    loading_dose_mg: float = 0.0
    doses_profile: list[float] | None = None
    time_to_ss_h: float = 0.0


def simulate_multiple_dose(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    dosing_interval_h: float = 24.0,
    n_doses: int = 10,
    route: str = "iv",
    ka_per_h: float = 1.5,
    f: float = 1.0,
    dt_h: float = 0.05,
) -> AccumulationResult:
    """Simulate multiple-dose PK and compute accumulation metrics.

    Parameters
    ----------
    route : str
        'iv' for IV bolus or 'oral' for first-order oral absorption.
    f : float
        Oral bioavailability (0-1). Only used for oral route.
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if dosing_interval_h <= 0:
        raise ValueError("dosing_interval_h must be > 0")
    if n_doses < 1:
        raise ValueError("n_doses must be >= 1")

    ke = cl_L_per_h / vd_L
    thalf = math.log(2) / ke
    t_end = n_doses * dosing_interval_h
    n_pts = max(int(t_end / dt_h), 1)
    t = np.linspace(0.0, t_end, n_pts + 1)

    cp = np.zeros(n_pts + 1)

    # Superposition of single-dose profiles
    # Single-dose at t=0: IV bolus C(t) = D/Vd * exp(-ke*t)
    # Oral: C(t) = F*D*ka/(Vd*(ka-ke)) * (exp(-ke*t) - exp(-ka*t))
    dose_times = [i * dosing_interval_h for i in range(n_doses)]

    if route == "iv":
        c0 = dose_mg / vd_L
        for dose_t in dose_times:
            for i, ti in enumerate(t):
                dt_since = ti - dose_t
                if dt_since >= 0:
                    cp[i] += c0 * math.exp(-ke * dt_since)
    else:  # oral
        effective_dose = dose_mg * f
        if abs(ka_per_h - ke) < 1e-9:
            ka_per_h = ka_per_h * 1.001  # avoid singularity
        coef = effective_dose * ka_per_h / (vd_L * (ka_per_h - ke))
        for dose_t in dose_times:
            for i, ti in enumerate(t):
                dt_since = ti - dose_t
                if dt_since >= 0:
                    cp[i] += coef * (math.exp(-ke * dt_since) - math.exp(-ka_per_h * dt_since))

    cp = np.maximum(cp, 0.0)

    # Single-dose Cmax for accumulation index
    if route == "iv":
        cmax_1 = dose_mg / vd_L
        auc_1 = dose_mg / cl_L_per_h
    else:
        # Tmax_1 = ln(ka/ke)/(ka-ke)
        if ka_per_h > ke:
            tmax_1 = math.log(ka_per_h / ke) / (ka_per_h - ke)
            effective_dose_f = dose_mg * f
            coef = effective_dose_f * ka_per_h / (vd_L * (ka_per_h - ke))
            cmax_1 = coef * (math.exp(-ke * tmax_1) - math.exp(-ka_per_h * tmax_1))
        else:
            cmax_1 = float(np.max(cp[: int(dosing_interval_h / dt_h) + 1]))
        auc_1 = dose_mg * f / cl_L_per_h

    # Last dosing interval for SS metrics
    last_dose_idx = int((n_doses - 1) * dosing_interval_h / dt_h)
    end_idx = min(int(n_doses * dosing_interval_h / dt_h), len(cp) - 1)
    cp_last = cp[last_dose_idx : end_idx + 1]
    t_last = t[last_dose_idx : end_idx + 1]

    cmax_ss = float(np.max(cp_last)) if len(cp_last) > 0 else float(np.max(cp))
    cmin_ss = float(np.min(cp_last)) if len(cp_last) > 0 else float(np.min(cp))
    auc_ss = float(np_trapz(cp_last, t_last)) if len(cp_last) > 1 else auc_1
    cavg_ss = auc_ss / dosing_interval_h if dosing_interval_h > 0 else cmax_ss

    # Accumulation index
    acc_index = cmax_ss / cmax_1 if cmax_1 > 0 else 1.0
    analytical_r = 1.0 / (1.0 - math.exp(-ke * dosing_interval_h))

    fluctuation = 100.0 * (cmax_ss - cmin_ss) / max(cavg_ss, 1e-9)

    # Time to 90% SS
    target = 0.9 * cmax_ss
    t_ss = t[-1]
    n_to_ss = n_doses
    dose_cmax = []
    for i in range(n_doses):
        start = int(i * dosing_interval_h / dt_h)
        end = min(int((i + 1) * dosing_interval_h / dt_h), len(cp))
        if end > start:
            dose_cmax.append((i + 1, float(np.max(cp[start:end])), float(t[start])))

    for dose_num, cm, t_start in dose_cmax:
        if cm >= target:
            t_ss = t_start
            n_to_ss = dose_num
            break

    loading = dose_mg * analytical_r

    # Per-dose Cmax list
    doses_profile_list = [cm for _, cm, _ in dose_cmax]

    return AccumulationResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        dosing_interval_h=dosing_interval_h,
        n_doses=n_doses,
        route=route,
        times_h=t.tolist(),
        cp_profile=cp.tolist(),
        cmax_ss=cmax_ss,
        cmin_ss=cmin_ss,
        cavg_ss=cavg_ss,
        auc_ss=auc_ss,
        accumulation_index=acc_index,
        fluctuation_pct=fluctuation,
        t_ss_90pct_h=float(t_ss),
        n_doses_to_ss=n_to_ss,
        thalf_effective_h=thalf,
        analytical_r=analytical_r,
        accumulation_ratio_r=analytical_r,
        css_max_mg_L=cmax_ss,
        css_min_mg_L=cmin_ss,
        loading_dose_mg=loading,
        doses_profile=doses_profile_list,
        time_to_ss_h=float(t_ss),
    )


def accumulation_index_analytical(ke: float, tau: float) -> float:
    """Analytical accumulation index R = 1/(1-exp(-ke*tau)) for IV bolus."""
    if ke <= 0 or tau <= 0:
        raise ValueError("ke and tau must be > 0")
    return 1.0 / (1.0 - math.exp(-ke * tau))


def accumulation_ratio(t_half_h: float, tau_h: float) -> float:
    """Compute accumulation ratio R from half-life and dosing interval."""
    if t_half_h <= 0 or tau_h <= 0:
        raise ValueError("t_half_h and tau_h must be > 0")
    ke = math.log(2) / t_half_h
    return 1.0 / (1.0 - math.exp(-ke * tau_h))


def simulate_accumulation(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    dosing_interval_h: float = 24.0,
    n_doses: int = 10,
    route: str = "iv",
    **kwargs,
) -> AccumulationResult:
    """Alias for simulate_multiple_dose (Phase 74 API)."""
    return simulate_multiple_dose(
        drug_name=drug_name,
        dose_mg=dose_mg,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
        dosing_interval_h=dosing_interval_h,
        n_doses=n_doses,
        route=route,
        **kwargs,
    )


def dosing_regimen_comparison(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
) -> list[dict]:
    """Compare QID, TID, BID, QD regimens for the same daily dose."""
    regimens = [
        ("QID", 6.0, 4),
        ("TID", 8.0, 3),
        ("BID", 12.0, 2),
        ("QD", 24.0, 1),
    ]
    results = []
    daily_dose = dose_mg
    for label, tau, n_per_day in regimens:
        per_dose = daily_dose / n_per_day
        r = simulate_multiple_dose(
            drug_name=drug_name,
            dose_mg=per_dose,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            dosing_interval_h=tau,
            n_doses=max(20, int(10 * (math.log(2) / (cl_L_per_h / vd_L)) / tau) + 1),
        )
        results.append(
            {
                "regimen": label,
                "tau_h": tau,
                "dose_mg": per_dose,
                "css_avg": r.cavg_ss,
                "fluctuation_pct": r.fluctuation_pct,
                "peak_trough_ratio": r.cmax_ss / max(r.cmin_ss, 1e-12),
            }
        )
    return results


__all__ = [
    "AccumulationResult",
    "simulate_multiple_dose",
    "accumulation_index_analytical",
    "accumulation_ratio",
    "simulate_accumulation",
    "dosing_regimen_comparison",
]
