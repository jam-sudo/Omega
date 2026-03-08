"""Drug accumulation index for multiple dosing regimens.

Calculates the drug accumulation index (AI) which quantifies how much
drug builds up at steady state compared to the first dose.  Supports
comparison of different dosing intervals for the same total daily dose.

References
----------
- Rowland M, Tozer TN. Clinical Pharmacokinetics, 4th ed. (accumulation theory)
- Gibaldi M, Perrier D. Pharmacokinetics, 2nd ed. (multiple dosing)
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class AccumulationIndexResult:
    """Result of drug accumulation index calculation."""

    drug_name: str
    dose_mg: float
    dosing_interval_h: float
    t_half_h: float
    accumulation_index: float
    cmax_ratio: float
    cmin_ratio: float
    trough_steady_state_mg_L: float
    peak_steady_state_mg_L: float
    auc_ratio_ss_to_single: float
    time_to_90pct_steady_state_h: float
    n_doses_to_steady_state: int
    fluctuation_index: float
    notes: str


def calculate_accumulation_index(
    drug_name: str,
    dose_mg: float,
    dosing_interval_h: float,
    cl_L_per_h: float,
    vd_L: float,
    ka_per_h: float = 1.0,
    f_oral: float = 0.85,
) -> AccumulationIndexResult:
    """Calculate drug accumulation index for multiple dosing.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Dose per administration (mg).
    dosing_interval_h : float
        Dosing interval tau (hours).
    cl_L_per_h : float
        Total clearance (L/h).
    vd_L : float
        Volume of distribution (L).
    ka_per_h : float
        Absorption rate constant (1/h).
    f_oral : float
        Oral bioavailability fraction.

    Returns
    -------
    AccumulationIndexResult
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive")
    if dosing_interval_h <= 0:
        raise ValueError("dosing_interval_h must be positive")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be positive")
    if vd_L <= 0:
        raise ValueError("vd_L must be positive")
    if ka_per_h <= 0:
        raise ValueError("ka_per_h must be positive")

    # Derived PK parameters
    k_el = cl_L_per_h / vd_L
    t_half = 0.693 / k_el
    tau = dosing_interval_h

    # Accumulation index
    exp_kel_tau = math.exp(-k_el * tau)
    ai = 1.0 / (1.0 - exp_kel_tau)

    # Trough at steady state (1-compartment)
    cmin_ss = (dose_mg * f_oral / vd_L) * exp_kel_tau / (1.0 - exp_kel_tau)

    # First-dose trough
    cmin_first = (dose_mg * f_oral / vd_L) * math.exp(-k_el * tau)

    # Peak calculations - handle ka ~ k_el
    ka = ka_per_h
    if abs(ka - k_el) < 1e-6:
        # Limiting case: tmax = 1/k_el
        tmax = 1.0 / k_el
        cmax_first = (dose_mg * f_oral / vd_L) * math.exp(-1.0)
        cmax_ss = cmax_first * ai
    else:
        tmax = math.log(ka / k_el) / (ka - k_el)
        if tmax < 0:
            tmax = 0.5  # fallback
        # Single-dose Cmax (oral 1-cpt)
        cmax_first = (
            (dose_mg * f_oral / vd_L)
            * (ka / (ka - k_el))
            * (math.exp(-k_el * tmax) - math.exp(-ka * tmax))
        )
        # Steady-state Cmax (oral 1-cpt, multiple dosing)
        exp_ka_tau = math.exp(-ka * tau)
        cmax_ss = (
            (dose_mg * f_oral / vd_L)
            * (ka / (ka - k_el))
            * (
                math.exp(-k_el * tmax) / (1.0 - exp_kel_tau)
                - math.exp(-ka * tmax) / (1.0 - exp_ka_tau)
            )
        )
        # Ensure positive
        cmax_ss = max(cmax_ss, cmin_ss)

    cmax_first = max(cmax_first, 1e-15)
    cmax_ratio = cmax_ss / cmax_first if cmax_first > 0 else ai
    cmin_ratio = cmin_ss / cmin_first if cmin_first > 0 else ai

    # AUC ratio at steady state vs single dose = AI
    auc_ratio = ai

    # Time to 90% steady state
    time_to_90 = -math.log(0.1) / k_el  # = 2.303 / k_el = 2.303 * t_half / 0.693

    # Number of doses to steady state
    n_doses = math.ceil(time_to_90 / tau)
    n_doses = max(n_doses, 1)

    # Fluctuation index
    if cmin_ss > 0:
        fluct = (cmax_ss - cmin_ss) / cmin_ss
        fluct = max(0.0, min(fluct, 100.0))
    else:
        fluct = 100.0  # clamped max

    notes_parts = []
    if ai > 3.0:
        notes_parts.append("Significant accumulation expected")
    if fluct > 10.0:
        notes_parts.append("High peak-trough fluctuation")
    if t_half > 24.0:
        notes_parts.append("Long half-life drug")

    return AccumulationIndexResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        dosing_interval_h=tau,
        t_half_h=t_half,
        accumulation_index=ai,
        cmax_ratio=cmax_ratio,
        cmin_ratio=cmin_ratio,
        trough_steady_state_mg_L=cmin_ss,
        peak_steady_state_mg_L=cmax_ss,
        auc_ratio_ss_to_single=auc_ratio,
        time_to_90pct_steady_state_h=time_to_90,
        n_doses_to_steady_state=n_doses,
        fluctuation_index=fluct,
        notes="; ".join(notes_parts),
    )


def compare_regimens(
    drug_name: str,
    cl_L_per_h: float,
    vd_L: float,
    daily_dose_mg: float,
    intervals: list[int] | None = None,
) -> list[AccumulationIndexResult]:
    """Compare dosing regimens for the same total daily dose.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    cl_L_per_h : float
        Total clearance (L/h).
    vd_L : float
        Volume of distribution (L).
    daily_dose_mg : float
        Total daily dose (mg).
    intervals : list[int] or None
        Dosing intervals to compare (hours). Defaults to [6, 8, 12, 24].

    Returns
    -------
    list[AccumulationIndexResult]
        Results sorted by fluctuation_index ascending (smoother first).
    """
    if intervals is None:
        intervals = [6, 8, 12, 24]

    results = []
    for tau in intervals:
        n_doses_per_day = 24.0 / tau
        dose_per_admin = daily_dose_mg / n_doses_per_day
        result = calculate_accumulation_index(
            drug_name=drug_name,
            dose_mg=dose_per_admin,
            dosing_interval_h=float(tau),
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
        )
        results.append(result)

    results.sort(key=lambda r: r.fluctuation_index)
    return results
