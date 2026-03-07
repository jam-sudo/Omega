"""Indirect Response PD Models (Dayneka/Jusko 1993).

Implements the 4 classical indirect response (IDR) models for pharmacodynamic
response modeling. Drug concentration follows a simple 1-compartment IV PK model.

Models:
    I:   dR/dt = kin * (1 - Imax*C/(IC50+C)) - kout*R  (inhibit production)
    II:  dR/dt = kin - kout * (1 - Imax*C/(IC50+C)) * R  (inhibit degradation)
    III: dR/dt = kin * (1 + Emax*C/(EC50+C)) - kout*R  (stimulate production)
    IV:  dR/dt = kin - kout * (1 / (1 + Emax*C/(EC50+C))) * R  (stimulate degradation)

Reference: Dayneka NL, Garg V, Jusko WJ. J Pharmacokinet Biopharm. 1993;21(4):457-78.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz

_VALID_MODELS = (1, 2, 3, 4)


@dataclass(frozen=True)
class IndirectResponseResult:
    """Result of an indirect response PD simulation."""

    model_num: int
    dose_mg: float
    kin: float
    kout: float
    ec50_ic50_mg_L: float
    emax_imax: float
    times_h: list[float]
    c_drug_mg_L: list[float]
    response: list[float]
    baseline_r0: float
    peak_response: float
    tmax_response_h: float
    time_to_return_pct: float  # -1 if never returned within 10% of baseline
    auc_response: float
    notes: str


def _drug_conc(dose_mg: float, vd_L: float, ke: float, t: float) -> float:
    """1-compartment IV concentration at time t."""
    return (dose_mg / vd_L) * math.exp(-ke * t)


def _idr_deriv(
    model_num: int,
    r: float,
    c: float,
    kin: float,
    kout: float,
    ec50_ic50: float,
    emax_imax: float,
) -> float:
    """Compute dR/dt for the given IDR model."""
    if model_num == 1:
        # Inhibit production
        inhibition = emax_imax * c / (ec50_ic50 + c) if (ec50_ic50 + c) > 0 else 0.0
        return kin * (1.0 - inhibition) - kout * r
    elif model_num == 2:
        # Inhibit degradation
        inhibition = emax_imax * c / (ec50_ic50 + c) if (ec50_ic50 + c) > 0 else 0.0
        return kin - kout * (1.0 - inhibition) * r
    elif model_num == 3:
        # Stimulate production
        stimulation = emax_imax * c / (ec50_ic50 + c) if (ec50_ic50 + c) > 0 else 0.0
        return kin * (1.0 + stimulation) - kout * r
    else:
        # Model IV: stimulate degradation (inhibit via inverse term)
        stimulation = emax_imax * c / (ec50_ic50 + c) if (ec50_ic50 + c) > 0 else 0.0
        return kin - kout * (1.0 / (1.0 + stimulation)) * r


def simulate_indirect_response(
    model_num: int,
    dose_mg: float,
    vd_L: float,
    cl_L_per_h: float,
    kin: float,
    kout: float,
    ec50_ic50_mg_L: float,
    emax_imax: float = 1.0,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
) -> IndirectResponseResult:
    """Simulate an indirect response PD model.

    Parameters
    ----------
    model_num : int
        IDR model number: 1 (inhibit production), 2 (inhibit degradation),
        3 (stimulate production), 4 (stimulate degradation).
    dose_mg : float
        IV dose in mg.
    vd_L : float
        Volume of distribution (L).
    cl_L_per_h : float
        Clearance (L/h).
    kin : float
        Zero-order rate constant for response production (response units/h).
    kout : float
        First-order rate constant for response degradation (1/h).
    ec50_ic50_mg_L : float
        EC50 or IC50 concentration (mg/L).
    emax_imax : float
        Maximum effect (0-1 for inhibitory models; >0 for stimulatory models).
    t_end_h : float
        Simulation end time (h).
    dt_h : float
        Integration time step (h).

    Returns
    -------
    IndirectResponseResult
        Simulation results including time course of drug and response.
    """
    # --- Validation ---
    if model_num not in _VALID_MODELS:
        raise ValueError(f"model_num must be one of {_VALID_MODELS}, got {model_num}")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if kin <= 0:
        raise ValueError("kin must be > 0")
    if kout <= 0:
        raise ValueError("kout must be > 0")
    if ec50_ic50_mg_L <= 0:
        raise ValueError("ec50_ic50_mg_L must be > 0")
    if model_num in (1, 2):
        if not (0.0 < emax_imax <= 1.0):
            raise ValueError(
                f"emax_imax must be in (0, 1] for inhibitory models (1, 2), got {emax_imax}"
            )
    else:
        if emax_imax <= 0:
            raise ValueError(
                f"emax_imax must be > 0 for stimulatory models (3, 4), got {emax_imax}"
            )

    ke = cl_L_per_h / vd_L
    baseline_r0 = kin / kout

    n_steps = max(int(t_end_h / dt_h), 1)
    times = np.linspace(0.0, t_end_h, n_steps + 1)

    c_arr = np.zeros(n_steps + 1)
    r_arr = np.zeros(n_steps + 1)
    r_arr[0] = baseline_r0
    c_arr[0] = _drug_conc(dose_mg, vd_L, ke, 0.0)

    for i in range(n_steps):
        t_now = times[i]
        c_now = _drug_conc(dose_mg, vd_L, ke, t_now)
        r_now = r_arr[i]

        dr_dt = _idr_deriv(model_num, r_now, c_now, kin, kout, ec50_ic50_mg_L, emax_imax)
        r_next = r_now + dr_dt * dt_h
        # Prevent negative response
        r_arr[i + 1] = max(r_next, 0.0)
        c_arr[i + 1] = _drug_conc(dose_mg, vd_L, ke, times[i + 1])

    # --- Derived metrics ---
    peak_response = float(np.max(r_arr))
    tmax_idx = int(np.argmax(r_arr))
    tmax_response_h = float(times[tmax_idx])

    # Time to return within 10% of baseline (after peak)
    tolerance = 0.10 * baseline_r0
    time_to_return = -1.0
    if tmax_idx < n_steps:
        for i in range(tmax_idx, n_steps + 1):
            if abs(r_arr[i] - baseline_r0) <= tolerance:
                time_to_return = float(times[i])
                break

    auc_response = float(np_trapz(r_arr, times))

    # Build notes
    model_labels = {
        1: "inhibit production",
        2: "inhibit degradation",
        3: "stimulate production",
        4: "stimulate degradation",
    }
    notes = (
        f"IDR Model {model_num} ({model_labels[model_num]}); "
        f"ke={ke:.4f}/h; baseline R0={baseline_r0:.4f}; "
        f"peak response={peak_response:.4f} at t={tmax_response_h:.2f}h"
    )

    return IndirectResponseResult(
        model_num=model_num,
        dose_mg=dose_mg,
        kin=kin,
        kout=kout,
        ec50_ic50_mg_L=ec50_ic50_mg_L,
        emax_imax=emax_imax,
        times_h=times.tolist(),
        c_drug_mg_L=c_arr.tolist(),
        response=r_arr.tolist(),
        baseline_r0=baseline_r0,
        peak_response=peak_response,
        tmax_response_h=tmax_response_h,
        time_to_return_pct=time_to_return,
        auc_response=auc_response,
        notes=notes,
    )


def compare_idr_models(
    dose_mg: float,
    vd_L: float,
    cl_L_per_h: float,
    kin: float,
    kout: float,
    ec50_mg_L: float,
    emax_imax: float = 1.0,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
) -> dict[str, IndirectResponseResult]:
    """Simulate all four IDR models and return results keyed by "I", "II", "III", "IV".

    Parameters
    ----------
    dose_mg : float
        IV dose in mg.
    vd_L : float
        Volume of distribution (L).
    cl_L_per_h : float
        Clearance (L/h).
    kin : float
        Zero-order production rate constant.
    kout : float
        First-order degradation rate constant.
    ec50_mg_L : float
        EC50/IC50 used for all 4 models.
    emax_imax : float
        Maximum effect parameter. Applied to all models. For inhibitory models (I, II)
        clamped to (0, 1]. For stimulatory models (III, IV) must be > 0.
    t_end_h : float
        Simulation end time (h).
    dt_h : float
        Integration time step (h).

    Returns
    -------
    dict[str, IndirectResponseResult]
        Keys: "I", "II", "III", "IV"
    """
    keys = ["I", "II", "III", "IV"]
    model_nums = [1, 2, 3, 4]

    results: dict[str, IndirectResponseResult] = {}
    for key, num in zip(keys, model_nums, strict=True):
        # Clamp emax_imax appropriately per model type
        eff = min(emax_imax, 1.0) if num in (1, 2) else max(emax_imax, 1e-9)
        results[key] = simulate_indirect_response(
            model_num=num,
            dose_mg=dose_mg,
            vd_L=vd_L,
            cl_L_per_h=cl_L_per_h,
            kin=kin,
            kout=kout,
            ec50_ic50_mg_L=ec50_mg_L,
            emax_imax=eff,
            t_end_h=t_end_h,
            dt_h=dt_h,
        )
    return results


__all__ = [
    "IndirectResponseResult",
    "simulate_indirect_response",
    "compare_idr_models",
]
