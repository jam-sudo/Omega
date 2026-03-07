"""Prodrug PK/PD model: prodrug → active drug conversion and pharmacological effect."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz

__all__ = [
    "ProdrugPKResult",
    "simulate_prodrug_pk",
    "compare_prodrug_strategies",
]


@dataclass(frozen=True)
class ProdrugPKResult:
    """Results from a prodrug PK/PD simulation."""

    prodrug_name: str
    active_name: str
    prodrug_dose_mg: float
    times_h: list[float]
    c_prodrug: list[float]  # mg/L
    c_active: list[float]  # mg/L
    effect_pct: list[float]  # % of Emax
    cmax_prodrug: float
    cmax_active: float
    tmax_active_h: float
    auc_prodrug: float  # mg*h/L
    auc_active: float  # mg*h/L
    peak_effect_pct: float
    time_peak_effect_h: float
    notes: str


def _validate_inputs(
    prodrug_dose_mg: float,
    ka_per_h: float,
    f_oral: float,
    cl_prodrug_L_per_h: float,
    vd_prodrug_L: float,
    k_conversion_per_h: float,
    cl_active_L_per_h: float,
    vd_active_L: float,
    ec50_active_mg_L: float,
) -> None:
    if prodrug_dose_mg <= 0:
        raise ValueError("prodrug_dose_mg must be > 0")
    if ka_per_h <= 0:
        raise ValueError("ka_per_h must be > 0")
    if not (0 < f_oral <= 1.0):
        raise ValueError("f_oral must be in (0, 1]")
    if cl_prodrug_L_per_h <= 0:
        raise ValueError("cl_prodrug_L_per_h must be > 0")
    if vd_prodrug_L <= 0:
        raise ValueError("vd_prodrug_L must be > 0")
    if k_conversion_per_h <= 0:
        raise ValueError("k_conversion_per_h must be > 0")
    if cl_active_L_per_h <= 0:
        raise ValueError("cl_active_L_per_h must be > 0")
    if vd_active_L <= 0:
        raise ValueError("vd_active_L must be > 0")
    if ec50_active_mg_L <= 0:
        raise ValueError("ec50_active_mg_L must be > 0")


def simulate_prodrug_pk(
    prodrug_name: str,
    active_name: str,
    prodrug_dose_mg: float,
    ka_per_h: float = 1.5,
    f_oral: float = 0.5,
    cl_prodrug_L_per_h: float = 20.0,
    vd_prodrug_L: float = 10.0,
    k_conversion_per_h: float = 2.0,
    cl_active_L_per_h: float = 5.0,
    vd_active_L: float = 50.0,
    ec50_active_mg_L: float = 0.1,
    emax: float = 100.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> ProdrugPKResult:
    """Simulate prodrug PK/PD using a 3-compartment Forward Euler model.

    Compartments: gut depot -> prodrug plasma -> active drug plasma.

    ODEs
    ----
    dA_gut/dt  = -ka * A_gut
    dC_pro/dt  = ka * f_oral * A_gut / vd_pro - (cl_pro/vd_pro + k_conv) * C_pro
    dC_act/dt  = k_conv * C_pro * (vd_pro/vd_act) - (cl_act/vd_act) * C_act

    Effect uses Hill equation with n=1: Emax * C_act / (EC50 + C_act).
    """
    _validate_inputs(
        prodrug_dose_mg,
        ka_per_h,
        f_oral,
        cl_prodrug_L_per_h,
        vd_prodrug_L,
        k_conversion_per_h,
        cl_active_L_per_h,
        vd_active_L,
        ec50_active_mg_L,
    )

    n_steps = int(t_end_h / dt_h) + 1
    times = np.linspace(0.0, t_end_h, n_steps)

    # State: A_gut (mg), C_pro (mg/L), C_act (mg/L)
    a_gut = prodrug_dose_mg
    c_pro = 0.0
    c_act = 0.0

    c_pro_arr = np.zeros(n_steps)
    c_act_arr = np.zeros(n_steps)

    for i in range(n_steps):
        c_pro_arr[i] = c_pro
        c_act_arr[i] = c_act

        # Derivatives
        da_gut = -ka_per_h * a_gut
        dc_pro = (
            ka_per_h * f_oral * a_gut / vd_prodrug_L
            - (cl_prodrug_L_per_h / vd_prodrug_L + k_conversion_per_h) * c_pro
        )
        dc_act = (
            k_conversion_per_h * c_pro * (vd_prodrug_L / vd_active_L)
            - (cl_active_L_per_h / vd_active_L) * c_act
        )

        # Forward Euler
        a_gut = max(0.0, a_gut + da_gut * dt_h)
        c_pro = max(0.0, c_pro + dc_pro * dt_h)
        c_act = max(0.0, c_act + dc_act * dt_h)

    # Effect: Emax model (Hill n=1)
    effect_arr = emax * c_act_arr / (ec50_active_mg_L + c_act_arr)

    # Summary metrics
    cmax_prodrug = float(np.max(c_pro_arr))
    cmax_active = float(np.max(c_act_arr))
    idx_tmax_active = int(np.argmax(c_act_arr))
    tmax_active_h = float(times[idx_tmax_active])

    auc_prodrug = float(np_trapz(c_pro_arr, times))
    auc_active = float(np_trapz(c_act_arr, times))

    peak_effect_pct = float(np.max(effect_arr))
    idx_peak_effect = int(np.argmax(effect_arr))
    time_peak_effect_h = float(times[idx_peak_effect])

    notes = (
        f"Prodrug {prodrug_name} converted to {active_name} at k_conv={k_conversion_per_h}/h. "
        f"Peak active Cmax={cmax_active:.3f} mg/L at Tmax={tmax_active_h:.1f} h. "
        f"Peak effect={peak_effect_pct:.1f}% Emax at {time_peak_effect_h:.1f} h."
    )

    return ProdrugPKResult(
        prodrug_name=prodrug_name,
        active_name=active_name,
        prodrug_dose_mg=prodrug_dose_mg,
        times_h=times.tolist(),
        c_prodrug=c_pro_arr.tolist(),
        c_active=c_act_arr.tolist(),
        effect_pct=effect_arr.tolist(),
        cmax_prodrug=cmax_prodrug,
        cmax_active=cmax_active,
        tmax_active_h=tmax_active_h,
        auc_prodrug=auc_prodrug,
        auc_active=auc_active,
        peak_effect_pct=peak_effect_pct,
        time_peak_effect_h=time_peak_effect_h,
        notes=notes,
    )


def compare_prodrug_strategies(
    prodrug_name: str,
    active_name: str,
    dose_mg: float,
    k_conv_fast: float = 5.0,
    k_conv_slow: float = 0.5,
) -> dict:
    """Compare fast vs slow conversion strategies.

    Returns a dict with keys 'fast' and 'slow', each containing a ProdrugPKResult,
    plus a 'comparison' dict summarising differences in active drug Cmax, AUC, and
    sustained effect (time with effect > 50% of peak).
    """
    if k_conv_fast <= 0:
        raise ValueError("k_conv_fast must be > 0")
    if k_conv_slow <= 0:
        raise ValueError("k_conv_slow must be > 0")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")

    result_fast = simulate_prodrug_pk(
        prodrug_name=prodrug_name,
        active_name=active_name,
        prodrug_dose_mg=dose_mg,
        k_conversion_per_h=k_conv_fast,
    )
    result_slow = simulate_prodrug_pk(
        prodrug_name=prodrug_name,
        active_name=active_name,
        prodrug_dose_mg=dose_mg,
        k_conversion_per_h=k_conv_slow,
    )

    # Sustained effect: fraction of time with effect > 50% of peak
    def _sustained_h(result: ProdrugPKResult) -> float:
        times_arr = np.array(result.times_h)
        effect_arr = np.array(result.effect_pct)
        threshold = 0.5 * result.peak_effect_pct
        mask = effect_arr >= threshold
        if not mask.any():
            return 0.0
        dt = float(times_arr[1] - times_arr[0]) if len(times_arr) > 1 else 0.0
        return float(np.sum(mask) * dt)

    comparison = {
        "cmax_active_fast": result_fast.cmax_active,
        "cmax_active_slow": result_slow.cmax_active,
        "auc_active_fast": result_fast.auc_active,
        "auc_active_slow": result_slow.auc_active,
        "tmax_active_fast_h": result_fast.tmax_active_h,
        "tmax_active_slow_h": result_slow.tmax_active_h,
        "peak_effect_fast_pct": result_fast.peak_effect_pct,
        "peak_effect_slow_pct": result_slow.peak_effect_pct,
        "sustained_effect_fast_h": _sustained_h(result_fast),
        "sustained_effect_slow_h": _sustained_h(result_slow),
        "preferred_strategy": (
            "slow_conversion"
            if result_slow.auc_active >= result_fast.auc_active
            else "fast_conversion"
        ),
        "rationale": (
            "Slow conversion typically yields more sustained exposure; "
            "fast conversion gives higher early Cmax."
        ),
    }

    return {
        "fast": result_fast,
        "slow": result_slow,
        "comparison": comparison,
    }
