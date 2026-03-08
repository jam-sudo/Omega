"""
Phase 400 — Drug Accumulation Index Calculator

Analytical 1-compartment multiple-dose accumulation calculations:
- Accumulation index (Rac)
- Trough and Cmax accumulation factors
- Steady-state concentrations
- Time to steady state
- Dosing regimen comparison

Pure Python stdlib only (math, dataclasses).
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class AccumulationIndexResult:
    """Results from accumulation index calculation."""

    drug_name: str
    dose_mg: float
    dosing_interval_h: float
    t_half_h: float
    ke_per_h: float
    accumulation_index: float  # Rac = 1 / (1 - exp(-ke * tau))
    trough_accumulation_factor: float  # Ctrough_ss / Ctrough_1
    cmax_accumulation_factor: float  # Cmax_ss / Cmax_1
    css_avg_mg_L: float  # steady-state average concentration
    css_max_mg_L: float  # steady-state peak concentration (simplified)
    css_min_mg_L: float  # steady-state trough concentration
    fluctuation_pct: float  # (Cmax - Cmin) / Cavg * 100
    doses_to_90pct_ss: int  # doses needed to reach 90% SS
    time_to_90pct_ss_h: float
    time_to_95pct_ss_h: float
    notes: str


def calculate_accumulation(
    drug_name: str,
    dose_mg: float,
    t_half_h: float,
    dosing_interval_h: float,
    vd_L: float,
    ka_per_h: float = 1.5,
    f_oral: float = 1.0,
) -> AccumulationIndexResult:
    """
    Calculate drug accumulation index and steady-state PK parameters.

    Uses an analytical 1-compartment multiple-dose model.

    Parameters
    ----------
    drug_name : str
    dose_mg : float
        Dose per administration (mg).
    t_half_h : float
        Elimination half-life (h).
    dosing_interval_h : float
        Dosing interval tau (h).
    vd_L : float
        Volume of distribution (L).
    ka_per_h : float
        Absorption rate constant (1/h). Default 1.5.
    f_oral : float
        Oral bioavailability (0 < f <= 1). Default 1.0.

    Returns
    -------
    AccumulationIndexResult
    """
    # --- validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive")
    if t_half_h <= 0:
        raise ValueError("t_half_h must be positive")
    if dosing_interval_h <= 0:
        raise ValueError("dosing_interval_h must be positive")
    if vd_L <= 0:
        raise ValueError("vd_L must be positive")
    if not (0 < f_oral <= 1.0):
        raise ValueError("f_oral must be in (0, 1]")

    ke = 0.693147180559945 / t_half_h  # elimination rate constant (1/h)
    tau = dosing_interval_h

    # --- accumulation index ---
    exp_ke_tau = math.exp(-ke * tau)
    # guard against exp_ke_tau approaching 1 (very long half-life vs. interval)
    denominator = 1.0 - exp_ke_tau
    if denominator < 1.0e-12:
        denominator = 1.0e-12
    Rac = 1.0 / denominator

    # For 1-cpt oral multiple dose, Cmax and Ctrough SS / 1st dose ratio ≈ Rac
    # (exact only for IV; oral approximation valid when ka >> ke)
    trough_acc = Rac
    cmax_acc = Rac

    # --- steady-state average concentration ---
    # Css_avg = F * Dose / (CL * tau) = F * Dose / (Vd * ke * tau)
    css_avg = f_oral * dose_mg / (vd_L * ke * tau)

    # --- simplified peak and trough (Widmark approximation) ---
    # For oral 1-cpt SS, peak ~ css_avg * (1 + ke*tau/2),
    # trough ~ css_avg * (1 - ke*tau/2)
    half_swing = ke * tau / 2.0
    css_max = css_avg * (1.0 + half_swing)
    css_min = max(0.0, css_avg * (1.0 - half_swing))

    # --- fluctuation ---
    if css_avg > 0:
        fluctuation_pct = (css_max - css_min) / css_avg * 100.0
    else:
        fluctuation_pct = 0.0

    # --- time to steady state ---
    # 90% SS: t = -ln(0.1) / ke = ln(10) / ke = 3.32 * t_half
    time_to_90pct = 3.32 * t_half_h
    # 95% SS: t = -ln(0.05) / ke ≈ 4.32 * t_half
    time_to_95pct = 4.32 * t_half_h

    doses_to_90pct = max(1, math.ceil(time_to_90pct / tau))

    # --- notes ---
    notes_parts = []
    if Rac >= 5.0:
        notes_parts.append(f"High accumulation (Rac={Rac:.1f}); consider dose adjustment.")
    elif Rac >= 2.0:
        notes_parts.append(f"Moderate accumulation (Rac={Rac:.1f}).")
    else:
        notes_parts.append(f"Low accumulation (Rac={Rac:.1f}).")

    if fluctuation_pct > 100:
        notes_parts.append("High fluctuation; consider controlled-release formulation.")

    notes = " ".join(notes_parts)

    return AccumulationIndexResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        dosing_interval_h=dosing_interval_h,
        t_half_h=t_half_h,
        ke_per_h=ke,
        accumulation_index=Rac,
        trough_accumulation_factor=trough_acc,
        cmax_accumulation_factor=cmax_acc,
        css_avg_mg_L=css_avg,
        css_max_mg_L=css_max,
        css_min_mg_L=css_min,
        fluctuation_pct=fluctuation_pct,
        doses_to_90pct_ss=doses_to_90pct,
        time_to_90pct_ss_h=time_to_90pct,
        time_to_95pct_ss_h=time_to_95pct,
        notes=notes,
    )


def compare_dosing_regimens(
    drug_name: str,
    dose_mg: float,
    t_half_h: float,
    vd_L: float,
    f_oral: float = 1.0,
    regimens: list[float] | None = None,
) -> list[AccumulationIndexResult]:
    """
    Compare accumulation across multiple dosing intervals.

    Parameters
    ----------
    drug_name : str
    dose_mg : float
    t_half_h : float
    vd_L : float
    f_oral : float
    regimens : list of float, optional
        Dosing intervals (h). Default [6, 8, 12, 24, 48].

    Returns
    -------
    list of AccumulationIndexResult sorted by accumulation_index descending
    """
    if regimens is None:
        regimens = [6.0, 8.0, 12.0, 24.0, 48.0]

    results = []
    for tau in regimens:
        r = calculate_accumulation(
            drug_name=drug_name,
            dose_mg=dose_mg,
            t_half_h=t_half_h,
            dosing_interval_h=tau,
            vd_L=vd_L,
            f_oral=f_oral,
        )
        results.append(r)

    results.sort(key=lambda x: x.accumulation_index, reverse=True)
    return results
