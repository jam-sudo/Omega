"""Cardiac output-dependent PBPK (Phase 202).

Distribution changes in heart failure, exercise, and sepsis.
4-compartment model (plasma, liver, kidney, muscle) with forward Euler integration.
Cardiac output shifts effective CL and Vd, modulating tissue concentrations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz  # noqa: E402

__all__ = ["CardiacOutputPKResult", "simulate_cardiac_output_pk", "compare_cardiac_states"]


# ---------------------------------------------------------------------------
# Cardiac state parameters
# ---------------------------------------------------------------------------

CARDIAC_STATES: dict[str, dict[str, float]] = {
    "normal":        {"co_L_per_min": 5.0,  "cl_factor": 1.0, "vd_factor": 1.0},
    "heart_failure": {"co_L_per_min": 2.5,  "cl_factor": 0.6, "vd_factor": 0.8},
    "exercise":      {"co_L_per_min": 15.0, "cl_factor": 1.3, "vd_factor": 1.1},
    "sepsis":        {"co_L_per_min": 8.0,  "cl_factor": 0.7, "vd_factor": 1.3},
}

# Tissue-to-plasma partition coefficients (fixed heuristics)
_KP_LIVER = 3.0
_KP_KIDNEY = 2.0
_KP_MUSCLE = 0.5

_NORMAL_CO = 5.0  # L/min, reference cardiac output


@dataclass(frozen=True)
class CardiacOutputPKResult:
    """PK simulation outcome under a given cardiac output state."""

    drug_name: str
    cardiac_state: str              # "normal", "heart_failure", "exercise", "sepsis"
    cardiac_output_L_per_min: float
    dose_mg: float
    route: str
    times_h: list[float]
    c_plasma_mg_L: list[float]
    c_liver_mg_L: list[float]
    c_kidney_mg_L: list[float]
    c_muscle_mg_L: list[float]
    cmax_plasma: float
    auc_plasma: float               # mg·h/L (trapezoid)
    cl_effective_L_per_h: float     # effective CL given cardiac state
    redistribution_index: float     # |cl_factor-1| + |vd_factor-1|
    notes: str


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def simulate_cardiac_output_pk(
    drug_name: str,
    dose_mg: float,
    cl_nominal_L_per_h: float,
    vd_L: float,
    cardiac_state: str = "normal",
    route: str = "iv",
    ka_per_h: float = 1.0,
    f_oral: float = 1.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> CardiacOutputPKResult:
    """Simulate plasma and tissue PK under a specified cardiac output state.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Administered dose (mg).
    cl_nominal_L_per_h : float
        Nominal (healthy) clearance (L/h).
    vd_L : float
        Nominal (healthy) volume of distribution (L).
    cardiac_state : str
        One of "normal", "heart_failure", "exercise", "sepsis".
    route : str
        "iv" or "oral".
    ka_per_h : float
        Absorption rate constant (h⁻¹); only used for oral route.
    f_oral : float
        Oral bioavailability fraction (0–1).
    t_end_h : float
        Simulation end time (h).
    dt_h : float
        Integration step size (h).

    Returns
    -------
    CardiacOutputPKResult
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be positive, got {dose_mg}")
    if cl_nominal_L_per_h <= 0:
        raise ValueError(f"cl_nominal_L_per_h must be positive, got {cl_nominal_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be positive, got {vd_L}")
    if cardiac_state not in CARDIAC_STATES:
        valid = list(CARDIAC_STATES.keys())
        raise ValueError(f"cardiac_state must be one of {valid}, got '{cardiac_state}'")

    params = CARDIAC_STATES[cardiac_state]
    co = params["co_L_per_min"]
    cl_eff = cl_nominal_L_per_h * params["cl_factor"]
    vd_eff = vd_L * params["vd_factor"]
    ke = cl_eff / vd_eff

    # Redistribution index
    redistribution_index = abs(params["cl_factor"] - 1.0) + abs(params["vd_factor"] - 1.0)

    # Build time array
    n_steps = max(int(t_end_h / dt_h) + 1, 2)
    times = np.linspace(0.0, t_end_h, n_steps)

    # State: [depot (oral), central_amount]
    c_plasma_arr = np.zeros(n_steps)

    if route == "iv":
        # Instantaneous bolus: initial concentration = dose/Vd
        amount_central = dose_mg
        depot = 0.0
    else:
        # Oral: dose enters depot
        depot = dose_mg * f_oral
        amount_central = 0.0

    # Forward Euler integration
    for i, _t in enumerate(times):
        c_plasma_arr[i] = amount_central / vd_eff

        if i == n_steps - 1:
            break

        # Absorption (oral only)
        if route == "oral" and depot > 0.0:
            absorbed = ka_per_h * depot * dt_h
            depot -= absorbed
            amount_central += absorbed

        # Elimination
        amount_central -= ke * amount_central * dt_h
        amount_central = max(amount_central, 0.0)

    # Tissue concentrations derived from plasma + cardiac output scaling
    co_ratio = co / _NORMAL_CO          # relative to normal (5 L/min)
    co_ratio_inv = _NORMAL_CO / co      # inverse for muscle

    c_liver_arr = c_plasma_arr * _KP_LIVER * co_ratio
    c_kidney_arr = c_plasma_arr * _KP_KIDNEY * co_ratio
    c_muscle_arr = c_plasma_arr * _KP_MUSCLE * co_ratio_inv

    # PK metrics
    cmax_plasma = float(np.max(c_plasma_arr))
    auc_plasma = float(np_trapz(c_plasma_arr, times))

    notes_parts: list[str] = []
    if cardiac_state == "heart_failure":
        notes_parts.append("Reduced CL and Vd in heart failure → higher exposure")
    elif cardiac_state == "exercise":
        notes_parts.append("Elevated CO → faster distribution, higher CL")
    elif cardiac_state == "sepsis":
        notes_parts.append("Sepsis: redistributed Vd, reduced CL")
    notes = "; ".join(notes_parts) if notes_parts else f"Cardiac state: {cardiac_state}"

    return CardiacOutputPKResult(
        drug_name=drug_name,
        cardiac_state=cardiac_state,
        cardiac_output_L_per_min=float(co),
        dose_mg=dose_mg,
        route=route,
        times_h=times.tolist(),
        c_plasma_mg_L=c_plasma_arr.tolist(),
        c_liver_mg_L=c_liver_arr.tolist(),
        c_kidney_mg_L=c_kidney_arr.tolist(),
        c_muscle_mg_L=c_muscle_arr.tolist(),
        cmax_plasma=cmax_plasma,
        auc_plasma=auc_plasma,
        cl_effective_L_per_h=float(cl_eff),
        redistribution_index=float(redistribution_index),
        notes=notes,
    )


def compare_cardiac_states(
    drug_name: str,
    dose_mg: float,
    cl_nominal_L_per_h: float,
    vd_L: float,
    **kwargs,
) -> list[CardiacOutputPKResult]:
    """Simulate all four cardiac states and return sorted by AUC descending.

    Parameters
    ----------
    drug_name, dose_mg, cl_nominal_L_per_h, vd_L :
        Passed directly to :func:`simulate_cardiac_output_pk`.
    **kwargs :
        Additional keyword arguments forwarded to each simulation call
        (e.g., route, ka_per_h, f_oral, t_end_h, dt_h).

    Returns
    -------
    list[CardiacOutputPKResult]
        All four cardiac states; order determined by AUC (highest first).
    """
    results: list[CardiacOutputPKResult] = []
    for state in CARDIAC_STATES:
        res = simulate_cardiac_output_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            cl_nominal_L_per_h=cl_nominal_L_per_h,
            vd_L=vd_L,
            cardiac_state=state,
            **kwargs,
        )
        results.append(res)
    results.sort(key=lambda r: r.auc_plasma, reverse=True)
    return results
