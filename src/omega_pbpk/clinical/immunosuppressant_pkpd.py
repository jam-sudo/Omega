"""Immunosuppressant (calcineurin inhibitor) PK/PD model — Phase 254.

Models cyclosporine A (CsA) and tacrolimus (Tac) 1-compartment PK
with IL-2 inhibition pharmacodynamics for transplant immunosuppression.

References
----------
- Serkova N et al., Clin Pharmacokinet. 2004;43(2):83-105
- Staatz CE & Tett SE, Clin Pharmacokinet. 2004;43(10):623-53
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class ImmunosuppressantPKResult:
    """Result from immunosuppressant PK/PD simulation."""

    drug_name: str
    dose_mg: float
    interval_h: float
    n_doses: int
    cl_L_per_h: float
    vd_L: float

    times_h: list[float] = field(default_factory=list)
    c_whole_blood_ng_mL: list[float] = field(default_factory=list)  # whole-blood conc
    il2_inhibition_pct: list[float] = field(default_factory=list)   # % IL-2 inhibition

    c_trough_ng_mL: float = 0.0    # Pre-dose trough at steady state
    c_peak_ng_mL: float = 0.0     # Peak whole-blood concentration
    auc_tau_ng_h_mL: float = 0.0  # AUC over last dosing interval
    in_therapeutic_range: bool = False
    therapeutic_range_ng_mL: tuple[float, float] = (0.0, 0.0)
    avg_il2_inhibition_pct: float = 0.0
    notes: str = ""


def simulate_immunosuppressant_pk(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    ka_per_h: float = 1.5,
    f_oral: float = 0.3,
    interval_h: float = 12.0,
    n_doses: int = 10,
    ec50_ng_mL: float = 200.0,
    emax_il2: float = 100.0,
    therapeutic_range: tuple[float, float] = (100.0, 300.0),
    dt_h: float = 0.1,
) -> ImmunosuppressantPKResult:
    """Simulate multiple-dose PK/PD of a calcineurin inhibitor.

    Parameters
    ----------
    drug_name : Drug name (e.g., 'cyclosporine', 'tacrolimus').
    dose_mg : Dose per administration (mg).
    cl_L_per_h : Systemic clearance (L/h). Must be > 0.
    vd_L : Volume of distribution (L). Must be > 0.
    ka_per_h : Oral absorption rate constant (h^-1). Must be > 0.
    f_oral : Oral bioavailability fraction (0, 1]. Must be in (0, 1].
    interval_h : Dosing interval (h). Must be > 0.
    n_doses : Number of doses to simulate. Must be >= 1.
    ec50_ng_mL : Concentration producing 50% IL-2 inhibition (ng/mL).
    emax_il2 : Maximum IL-2 inhibition (%). Default 100.
    therapeutic_range : (lower, upper) trough target in ng/mL.
    dt_h : Integration time step (h).

    Returns
    -------
    ImmunosuppressantPKResult
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if ka_per_h <= 0:
        raise ValueError("ka_per_h must be > 0")
    if not (0 < f_oral <= 1):
        raise ValueError("f_oral must be in (0, 1]")
    if interval_h <= 0:
        raise ValueError("interval_h must be > 0")
    if n_doses < 1:
        raise ValueError("n_doses must be >= 1")

    ke = cl_L_per_h / vd_L

    # Convert dose to ng (whole blood units: 1 mg = 1e6 ng; Vd in L)
    # Whole blood conc [ng/mL] = amount [ng] / (Vd [L] * 1000 [mL/L])
    dose_ng = dose_mg * 1e6  # mg -> ng
    bioavail_dose_ng = dose_ng * f_oral

    t_end_h = n_doses * interval_h
    n_steps = int(t_end_h / dt_h) + 1
    times = [i * dt_h for i in range(n_steps)]

    c_blood = [0.0] * n_steps  # ng/mL
    a_gut = [0.0] * n_steps    # ng in gut depot

    # Administer first dose at t=0
    a_gut[0] = bioavail_dose_ng
    dose_times = set()
    for d in range(n_doses):
        dose_times.add(round(d * interval_h / dt_h))

    for i in range(1, n_steps):
        cb = c_blood[i - 1]
        ag = a_gut[i - 1]

        absorption = ka_per_h * ag
        elimination = ke * cb * vd_L * 1000.0  # ng/h (Vd * 1000 mL/L * conc_ng/mL)

        da_gut = -absorption * dt_h
        dc_blood = (absorption - elimination) / (vd_L * 1000.0) * dt_h

        a_gut[i] = max(0.0, ag + da_gut)
        c_blood[i] = max(0.0, cb + dc_blood)

        # Administer dose at start of each interval
        if i in dose_times and i > 0:
            a_gut[i] += bioavail_dose_ng

    # IL-2 inhibition via Emax model
    il2_inhib = [emax_il2 * c / (ec50_ng_mL + c) if c > 0 else 0.0 for c in c_blood]

    # Trough = concentration at end of last interval (just before last dose)
    last_dose_idx = round((n_doses - 1) * interval_h / dt_h)
    c_trough = c_blood[last_dose_idx] if last_dose_idx < n_steps else c_blood[-1]
    c_peak = max(c_blood[last_dose_idx:]) if last_dose_idx < n_steps else max(c_blood)

    # AUC over last dosing interval
    last_start = last_dose_idx
    last_end = min(n_steps, last_dose_idx + int(interval_h / dt_h) + 1)
    auc_tau = 0.0
    for i in range(last_start + 1, last_end):
        auc_tau += 0.5 * (c_blood[i - 1] + c_blood[i]) * dt_h

    in_range = therapeutic_range[0] <= c_trough <= therapeutic_range[1]
    avg_il2 = sum(il2_inhib[last_start:last_end]) / max(1, last_end - last_start)

    notes = (
        f"{drug_name}: {dose_mg} mg q{interval_h:.0f}h x{n_doses} doses. "
        f"Trough={c_trough:.1f} ng/mL "
        f"({'in' if in_range else 'outside'} target "
        f"{therapeutic_range[0]}-{therapeutic_range[1]} ng/mL). "
        f"Avg IL-2 inhibition={avg_il2:.1f}%."
    )

    return ImmunosuppressantPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        interval_h=interval_h,
        n_doses=n_doses,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
        times_h=times,
        c_whole_blood_ng_mL=c_blood,
        il2_inhibition_pct=il2_inhib,
        c_trough_ng_mL=c_trough,
        c_peak_ng_mL=c_peak,
        auc_tau_ng_h_mL=auc_tau,
        in_therapeutic_range=in_range,
        therapeutic_range_ng_mL=therapeutic_range,
        avg_il2_inhibition_pct=avg_il2,
        notes=notes,
    )


def recommend_dose(
    drug_name: str,
    observed_trough_ng_mL: float,
    current_dose_mg: float,
    target_trough_ng_mL: float,
    cl_L_per_h: float,
    vd_L: float,
) -> dict:
    """Recommend a dose adjustment to achieve target trough.

    Uses linear proportionality assumption: trough ∝ dose.

    Parameters
    ----------
    drug_name : Drug name.
    observed_trough_ng_mL : Current observed trough (ng/mL). Must be > 0.
    current_dose_mg : Current dose (mg). Must be > 0.
    target_trough_ng_mL : Target trough (ng/mL). Must be > 0.
    cl_L_per_h : Clearance (L/h). Used to compute t_half.
    vd_L : Volume of distribution (L).

    Returns
    -------
    dict with keys:
        recommended_dose_mg (float): New recommended dose.
        dose_change_pct (float): Percent change from current.
        t_half_h (float): Elimination half-life.
        time_to_new_ss_h (float): ~3.3 * t_half to reach 90% of new SS.
    """
    if observed_trough_ng_mL <= 0:
        raise ValueError("observed_trough_ng_mL must be > 0")
    if current_dose_mg <= 0:
        raise ValueError("current_dose_mg must be > 0")
    if target_trough_ng_mL <= 0:
        raise ValueError("target_trough_ng_mL must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")

    ratio = target_trough_ng_mL / observed_trough_ng_mL
    new_dose = current_dose_mg * ratio
    change_pct = (ratio - 1.0) * 100.0
    ke = cl_L_per_h / vd_L
    t_half = math.log(2) / ke
    time_to_ss = 3.3 * t_half  # ~90% of new SS

    return {
        "drug_name": drug_name,
        "recommended_dose_mg": new_dose,
        "dose_change_pct": change_pct,
        "t_half_h": t_half,
        "time_to_new_ss_h": time_to_ss,
    }


__all__ = [
    "ImmunosuppressantPKResult",
    "simulate_immunosuppressant_pk",
    "recommend_dose",
]
