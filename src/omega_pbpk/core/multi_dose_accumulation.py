"""Phase 612 — Multi-Dose Accumulation Kinetics.

Analytical and simulated accumulation factor, steady-state prediction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class AccumulationResult:
    drug_name: str
    dose_mg: float
    interval_h: float
    n_doses: int
    t_half_h: float
    accumulation_factor: float  # R = 1/(1 - exp(-ke*tau)) for 1-cpt IV
    ss_cmax_mg_L: float  # steady-state Cmax
    ss_cmin_mg_L: float  # steady-state Cmin (trough)
    ss_cavg_mg_L: float  # average steady-state concentration
    ss_auc_per_interval: float  # AUC over one dosing interval at SS
    n_doses_to_90pct_ss: int  # doses to reach 90% of SS
    n_doses_to_95pct_ss: int  # doses to reach 95% of SS
    fluctuation_pct: float  # (Cmax-Cmin)/Cavg * 100 at SS
    time_to_ss_h: float  # time to reach 90% SS
    notes: str


def analytical_ss_1cpt_iv(
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    interval_h: float,
) -> dict:
    """Analytical steady-state for 1-compartment IV bolus.

    Cmax_ss = (dose/vd) / (1 - exp(-ke*tau))
    Cmin_ss = Cmax_ss * exp(-ke*tau)
    AUC_ss  = dose / CL  (identical to single-dose AUC for linear PK)
    """
    ke = cl_L_per_h / vd_L
    tau = interval_h
    exp_term = math.exp(-ke * tau)
    cmax_ss = (dose_mg / vd_L) / (1.0 - exp_term)
    cmin_ss = cmax_ss * exp_term
    auc_ss = dose_mg / cl_L_per_h
    acc_factor = cmax_ss / (dose_mg / vd_L)
    return {
        "cmax_ss": cmax_ss,
        "cmin_ss": cmin_ss,
        "auc_ss": auc_ss,
        "accumulation_factor": acc_factor,
    }


def doses_to_steady_state(ke_per_h: float, tau_h: float, threshold_pct: float = 90.0) -> int:
    """Number of doses to reach threshold% of steady-state.

    Uses: C_n / C_ss = 1 - exp(-n * ke * tau)
    Solves for n: n = ceil(-log(1 - threshold/100) / (ke * tau))
    """
    fraction = threshold_pct / 100.0
    n = math.ceil(-math.log(1.0 - fraction) / (ke_per_h * tau_h))
    return int(n)


def simulate_accumulation(
    drug_name: str,
    dose_mg: float,
    interval_h: float,
    cl_L_per_h: float,
    vd_L: float,
    route: str = "oral",
    ka_per_h: float = 1.0,
    f_oral: float = 1.0,
    n_doses: int = 10,
    t_end_h: float | None = None,
    dt_h: float = 0.1,
) -> AccumulationResult:
    """Simulate multi-dose PK accumulation using forward Euler.

    For IV: one-compartment with first-order elimination.
    For oral: one-compartment with first-order absorption and elimination.
    """
    # ---- Validation ----
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be positive, got {dose_mg}")
    if interval_h <= 0:
        raise ValueError(f"interval_h must be positive, got {interval_h}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be positive, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be positive, got {vd_L}")
    if n_doses < 1:
        raise ValueError(f"n_doses must be >= 1, got {n_doses}")
    if route not in {"oral", "iv"}:
        raise ValueError(f"route must be 'oral' or 'iv', got {route!r}")
    if dt_h <= 0:
        raise ValueError(f"dt_h must be positive, got {dt_h}")

    ke = cl_L_per_h / vd_L
    t_half_h = math.log(2.0) / ke

    if t_end_h is None:
        t_end_h = n_doses * interval_h

    # ---- Forward Euler simulation ----
    # State: C (concentration in central compartment, mg/L)
    #        Agut (drug in GI tract, mg) — oral only
    C = 0.0
    Agut = 0.0

    times: list[float] = []
    concs: list[float] = []

    t = 0.0
    dose_count = 0

    # Schedule doses at t=0, tau, 2*tau, ...
    next_dose_t = 0.0

    n_steps = int(round(t_end_h / dt_h)) + 1
    for step in range(n_steps):
        t = step * dt_h

        # Administer dose if we've reached a dose time
        if dose_count < n_doses and t >= next_dose_t - 1e-9:
            if route == "iv":
                C += dose_mg / vd_L
            else:  # oral
                Agut += dose_mg * f_oral
            dose_count += 1
            next_dose_t = dose_count * interval_h

        times.append(t)
        concs.append(C)

        # Forward Euler step
        if route == "iv":
            dC_dt = -ke * C
            C += dC_dt * dt_h
        else:
            dC_dt = (ka_per_h * Agut / vd_L) - ke * C
            dAgut_dt = -ka_per_h * Agut
            C += dC_dt * dt_h
            Agut += dAgut_dt * dt_h

        # Clamp to avoid tiny negatives
        C = max(0.0, C)
        Agut = max(0.0, Agut)

    # ---- Extract steady-state from last interval ----
    last_dose_start = (n_doses - 1) * interval_h
    last_dose_end = last_dose_start + interval_h

    ss_times: list[float] = []
    ss_concs: list[float] = []
    for t_i, c_i in zip(times, concs):
        if last_dose_start - 1e-9 <= t_i <= last_dose_end + 1e-9:
            ss_times.append(t_i)
            ss_concs.append(c_i)

    if not ss_concs:
        # Fallback: use all data
        ss_concs = concs

    ss_cmax = max(ss_concs)
    ss_cmin = min(ss_concs)

    # AUC of last interval via trapezoidal rule
    auc_ss = 0.0
    for i in range(1, len(ss_times)):
        auc_ss += 0.5 * (ss_concs[i - 1] + ss_concs[i]) * (ss_times[i] - ss_times[i - 1])

    interval_duration = ss_times[-1] - ss_times[0] if len(ss_times) > 1 else interval_h
    ss_cavg = auc_ss / interval_duration if interval_duration > 0 else ss_cmax

    # ---- Accumulation factor ----
    # Reference: peak of first dose
    first_dose_end = interval_h
    first_times: list[float] = []
    first_concs: list[float] = []
    for t_i, c_i in zip(times, concs):
        if t_i <= first_dose_end + 1e-9:
            first_times.append(t_i)
            first_concs.append(c_i)

    first_cmax = max(first_concs) if first_concs else dose_mg / vd_L
    acc_factor = ss_cmax / first_cmax if first_cmax > 0 else 1.0

    # ---- Doses to steady-state ----
    n_to_90 = doses_to_steady_state(ke, interval_h, 90.0)
    n_to_95 = doses_to_steady_state(ke, interval_h, 95.0)

    # ---- Fluctuation ----
    fluctuation_pct = (ss_cmax - ss_cmin) / ss_cavg * 100.0 if ss_cavg > 0 else 0.0

    # ---- Time to SS ----
    time_to_ss_h = n_to_90 * interval_h

    notes = (
        f"Route: {route}; ke={ke:.3f} /h; t½={t_half_h:.1f} h; "
        f"Doses to 90% SS: {n_to_90}; Accumulation factor: {acc_factor:.2f}."
    )

    return AccumulationResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        interval_h=interval_h,
        n_doses=n_doses,
        t_half_h=t_half_h,
        accumulation_factor=acc_factor,
        ss_cmax_mg_L=ss_cmax,
        ss_cmin_mg_L=ss_cmin,
        ss_cavg_mg_L=ss_cavg,
        ss_auc_per_interval=auc_ss,
        n_doses_to_90pct_ss=n_to_90,
        n_doses_to_95pct_ss=n_to_95,
        fluctuation_pct=fluctuation_pct,
        time_to_ss_h=time_to_ss_h,
        notes=notes,
    )


__all__ = [
    "AccumulationResult",
    "simulate_accumulation",
    "analytical_ss_1cpt_iv",
    "doses_to_steady_state",
]
