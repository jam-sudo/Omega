"""
Phase 398 — Oral Controlled Release Pharmacokinetics

Models PK of oral controlled-release formulations:
OROS, matrix tablet, enteric-coated, extended-release, immediate-release.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class OralCRResult:
    """Result of an oral controlled-release PK simulation (plain dataclass — has list fields)."""
    drug_name: str
    formulation: str
    dose_mg: float
    times_h: list
    c_gi_depot_mg: list
    c_plasma_mg_L: list
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    t_half_apparent_h: float
    relative_bioavailability: float
    peak_trough_ratio: float
    absorption_duration_h: float
    notes: str


_FORMULATION_PARAMS: dict = {
    "immediate_release": {"k_release_mod": 1.0,  "t_lag_h": 0.0, "f_bioav_mod": 1.0},
    "matrix":            {"k_release_mod": 0.15, "t_lag_h": 0.5, "f_bioav_mod": 0.95},
    "oros":              {"k_release_mod": 0.08, "t_lag_h": 1.0, "f_bioav_mod": 0.90},
    "enteric":           {"k_release_mod": 0.5,  "t_lag_h": 2.0, "f_bioav_mod": 0.98},
    "extended":          {"k_release_mod": 0.12, "t_lag_h": 0.0, "f_bioav_mod": 0.92},
}

_VALID_FORMULATIONS = set(_FORMULATION_PARAMS.keys())


def _trapezoidal_auc(x: list, y: list) -> float:
    return sum(0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]) for i in range(1, len(x)))


def simulate_oral_cr_pk(
    drug_name: str,
    dose_mg: float,
    formulation: str,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    ka_absorption_per_h: float = 1.5,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> OralCRResult:
    """
    Simulate oral controlled-release PK using a 3-compartment Forward Euler model.

    Compartments:
      M_gi   — drug in GI formulation (mg)
      M_rel  — drug released and dissolved, awaiting absorption (mg)
      C      — systemic plasma concentration (mg/L)

    Parameters
    ----------
    drug_name : str
    dose_mg : float
    formulation : str
        One of "immediate_release", "matrix", "oros", "enteric", "extended".
    cl_sys_L_per_h : float
        Systemic clearance (L/h).
    vd_sys_L : float
        Volume of distribution (L).
    ka_absorption_per_h : float
        First-order absorption rate constant (1/h).
    t_end_h : float
        Simulation end time (h).
    dt_h : float
        Forward-Euler time step (h).

    Returns
    -------
    OralCRResult
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    if ka_absorption_per_h <= 0:
        raise ValueError("ka_absorption_per_h must be > 0")
    if formulation not in _VALID_FORMULATIONS:
        raise ValueError(
            f"formulation must be one of {sorted(_VALID_FORMULATIONS)}, got '{formulation}'"
        )

    params = _FORMULATION_PARAMS[formulation]
    k_release_mod: float = params["k_release_mod"]
    t_lag_h: float = params["t_lag_h"]
    f_bioav_mod: float = params["f_bioav_mod"]

    effective_k_release = k_release_mod * 0.5  # baseline * modifier
    ka = ka_absorption_per_h
    k_elim = cl_sys_L_per_h / vd_sys_L

    # Initial conditions
    M_gi = dose_mg
    M_rel = 0.0
    C = 0.0

    # Storage
    times_h: list = []
    c_gi_list: list = []
    c_plasma_list: list = []

    n_steps = int(round(t_end_h / dt_h))
    t = 0.0
    for _ in range(n_steps + 1):
        times_h.append(t)
        c_gi_list.append(M_gi)
        c_plasma_list.append(C)

        if t < t_lag_h:
            # No release during lag phase
            dM_gi = 0.0
            dM_rel = 0.0
        else:
            dM_gi = -effective_k_release * M_gi
            dM_rel = effective_k_release * M_gi - ka * M_rel

        dC = ka * M_rel * f_bioav_mod / vd_sys_L - k_elim * C

        M_gi += dM_gi * dt_h
        M_rel += dM_rel * dt_h
        C += dC * dt_h

        # Clamp negatives
        if M_gi < 0.0:
            M_gi = 0.0
        if M_rel < 0.0:
            M_rel = 0.0
        if C < 0.0:
            C = 0.0

        t += dt_h

    auc = _trapezoidal_auc(times_h, c_plasma_list)

    cmax = max(c_plasma_list)
    tmax = times_h[c_plasma_list.index(cmax)]

    # Apparent half-life: dominated by slower of k_release or k_elim
    # Use combined: t½ = ln2 / min(effective_k_release, k_elim)
    rate_dominant = min(effective_k_release, k_elim) if effective_k_release > 0 else k_elim
    t_half_apparent = math.log(2.0) / rate_dominant if rate_dominant > 0 else float("inf")

    # Trough at 24h (last time point)
    c_trough = c_plasma_list[-1]
    if c_trough > 0:
        peak_trough_ratio = cmax / c_trough
    else:
        peak_trough_ratio = float("inf")

    # Time for 90% release from formulation: M_gi(t) = dose * exp(-k_rel*(t-t_lag))
    # => 0.1 * dose => t = t_lag + ln(10)/k_rel
    if effective_k_release > 0:
        absorption_duration_h = t_lag_h + math.log(10.0) / effective_k_release
    else:
        absorption_duration_h = float("inf")

    notes = (
        f"formulation={formulation}, k_release={effective_k_release:.4f}/h, "
        f"t_lag={t_lag_h}h, f_bioav_mod={f_bioav_mod}"
    )

    return OralCRResult(
        drug_name=drug_name,
        formulation=formulation,
        dose_mg=dose_mg,
        times_h=times_h,
        c_gi_depot_mg=c_gi_list,
        c_plasma_mg_L=c_plasma_list,
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc_mg_h_per_L=auc,
        t_half_apparent_h=t_half_apparent,
        relative_bioavailability=f_bioav_mod,
        peak_trough_ratio=peak_trough_ratio,
        absorption_duration_h=absorption_duration_h,
        notes=notes,
    )


def compare_formulations_cr(
    drug_name: str,
    dose_mg: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
) -> list:
    """
    Compare all 5 formulations.

    Returns a list of OralCRResult sorted by peak_trough_ratio ascending
    (most controlled — lowest ratio — first).
    """
    results = []
    for formulation in _VALID_FORMULATIONS:
        res = simulate_oral_cr_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            formulation=formulation,
            cl_sys_L_per_h=cl_sys_L_per_h,
            vd_sys_L=vd_sys_L,
        )
        results.append(res)

    results.sort(key=lambda r: (r.peak_trough_ratio if r.peak_trough_ratio != float("inf") else 1e18))
    return results
