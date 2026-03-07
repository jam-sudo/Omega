"""Cardiac output-dependent PBPK (Phase 202 + Phase 435).

Phase 202: Simple cardiac-state PK scaling model (simulate_cardiac_output_pk,
compare_cardiac_states) — 4-compartment with algebraic tissue concentrations.

Phase 435: Full perfusion-limited 4-tissue PBPK with explicit ODE states
(simulate_co_pbpk, cardiac_output_sensitivity).  Cardiac output fraction
scales all organ blood flows; hepatic CL uses well-stirred model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz  # noqa: E402

__all__ = [
    # Phase 202
    "CardiacOutputPKResult",
    "simulate_cardiac_output_pk",
    "compare_cardiac_states",
    "CARDIAC_STATES",
    # Phase 435
    "COPBPKResult",
    "simulate_co_pbpk",
    "cardiac_output_sensitivity",
]


# ===========================================================================
# Phase 202 — Cardiac-state scaling model
# ===========================================================================

# Cardiac state parameters
CARDIAC_STATES: dict[str, dict[str, float]] = {
    "normal": {"co_L_per_min": 5.0, "cl_factor": 1.0, "vd_factor": 1.0},
    "heart_failure": {"co_L_per_min": 2.5, "cl_factor": 0.6, "vd_factor": 0.8},
    "exercise": {"co_L_per_min": 15.0, "cl_factor": 1.3, "vd_factor": 1.1},
    "sepsis": {"co_L_per_min": 8.0, "cl_factor": 0.7, "vd_factor": 1.3},
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
    cardiac_state: str  # "normal", "heart_failure", "exercise", "sepsis"
    cardiac_output_L_per_min: float
    dose_mg: float
    route: str
    times_h: list[float]
    c_plasma_mg_L: list[float]
    c_liver_mg_L: list[float]
    c_kidney_mg_L: list[float]
    c_muscle_mg_L: list[float]
    cmax_plasma: float
    auc_plasma: float  # mg·h/L (trapezoid)
    cl_effective_L_per_h: float  # effective CL given cardiac state
    redistribution_index: float  # |cl_factor-1| + |vd_factor-1|
    notes: str


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
    co_ratio = co / _NORMAL_CO  # relative to normal (5 L/min)
    co_ratio_inv = _NORMAL_CO / co  # inverse for muscle

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


# ===========================================================================
# Phase 435 — Full perfusion-limited 4-tissue PBPK with ODE states
# ===========================================================================

# Fixed physiological organ volumes (L)
_ORGAN_VOLUMES: dict[str, float] = {
    "liver": 1.8,
    "kidney": 0.3,
    "muscle": 28.0,
    "fat": 14.0,
}

# Default normal organ blood flows (L/h) — reference healthy adult
_DEFAULT_ORGAN_FLOWS: dict[str, float] = {
    "liver": 54.0,  # hepatic artery + portal; ~0.9 L/min
    "kidney": 72.0,  # ~1.2 L/min
    "muscle": 15.0,  # resting skeletal muscle
    "fat": 5.0,  # adipose
}

# Default tissue-to-plasma partition coefficients
_DEFAULT_ORGAN_KP: dict[str, float] = {
    "liver": 3.0,
    "kidney": 5.0,
    "muscle": 1.0,
    "fat": 0.5,
}

_ORGANS = ["liver", "kidney", "muscle", "fat"]


@dataclass
class COPBPKResult:
    """Result of a cardiac-output-dependent PBPK simulation (Phase 435)."""

    drug_name: str
    dose_mg: float
    cardiac_output_fraction: float  # 1.0 = normal CO
    cl_hep_effective_L_per_h: float  # well-stirred effective hepatic CL

    times_h: list[float]
    c_plasma_mg_L: list[float]
    tissue_concs: dict[str, list[float]]  # organ → concentration time course

    auc_plasma: float  # mg·h/L (trapezoidal)
    t_half_h: float  # terminal half-life (h)
    notes: str


def simulate_co_pbpk(
    drug_name: str,
    dose_mg: float,
    cl_hep_L_per_h: float,
    cardiac_output_fraction: float,
    organ_flows: dict[str, float] | None = None,
    organ_kp: dict[str, float] | None = None,
    vd_L: float = 50.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> COPBPKResult:
    """Simulate plasma and tissue PK with altered cardiac output.

    Parameters
    ----------
    drug_name:
        Identifier for the compound.
    dose_mg:
        IV bolus dose (mg).
    cl_hep_L_per_h:
        Intrinsic (nominal) hepatic clearance at normal CO (L/h).
    cardiac_output_fraction:
        Fraction of normal cardiac output; 1.0 = healthy, 0.5 = heart failure.
        Must be in (0, 5].
    organ_flows:
        Normal (CO_fraction=1) blood flows (L/h) for each organ.
        Keys: ``"liver"``, ``"kidney"``, ``"muscle"``, ``"fat"``.
        Defaults to physiological reference values.
    organ_kp:
        Tissue-to-plasma partition coefficients for each organ.
        Defaults to ``{"liver": 3, "kidney": 5, "muscle": 1, "fat": 0.5}``.
    vd_L:
        Apparent volume of distribution (L); used to set initial plasma conc.
    t_end_h:
        Simulation duration (h).
    dt_h:
        Forward Euler time step (h).

    Returns
    -------
    COPBPKResult
    """
    # Input validation
    if not drug_name:
        raise ValueError("drug_name must be a non-empty string")
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be positive, got {dose_mg}")
    if cl_hep_L_per_h <= 0:
        raise ValueError(f"cl_hep_L_per_h must be positive, got {cl_hep_L_per_h}")
    if not (0 < cardiac_output_fraction <= 5.0):
        raise ValueError(
            f"cardiac_output_fraction must be in (0, 5], got {cardiac_output_fraction}"
        )
    if vd_L <= 0:
        raise ValueError(f"vd_L must be positive, got {vd_L}")
    if t_end_h <= 0:
        raise ValueError(f"t_end_h must be positive, got {t_end_h}")
    if dt_h <= 0:
        raise ValueError(f"dt_h must be positive, got {dt_h}")

    # Merge user-supplied dicts with defaults
    q_normal = dict(_DEFAULT_ORGAN_FLOWS)
    if organ_flows is not None:
        for k, v in organ_flows.items():
            if k not in _ORGANS:
                raise ValueError(f"Unknown organ in organ_flows: '{k}'. Must be in {_ORGANS}")
            if v <= 0:
                raise ValueError(f"organ_flows['{k}'] must be positive, got {v}")
            q_normal[k] = float(v)

    kp = dict(_DEFAULT_ORGAN_KP)
    if organ_kp is not None:
        for k, v in organ_kp.items():
            if k not in _ORGANS:
                raise ValueError(f"Unknown organ in organ_kp: '{k}'. Must be in {_ORGANS}")
            if v <= 0:
                raise ValueError(f"organ_kp['{k}'] must be positive, got {v}")
            kp[k] = float(v)

    # Scale flows by cardiac output fraction
    q_scaled: dict[str, float] = {org: q_normal[org] * cardiac_output_fraction for org in _ORGANS}

    # Well-stirred hepatic CL: CL_eff = Q_h * CL_int / (Q_h + CL_int)
    q_h = q_scaled["liver"]
    cl_hep_eff = q_h * cl_hep_L_per_h / (q_h + cl_hep_L_per_h)

    # Total systemic CL (hepatic only in this model)
    cl_total = cl_hep_eff

    # Terminal half-life from Vd and CL
    if cl_total > 0:
        t_half = math.log(2.0) * vd_L / cl_total
    else:
        t_half = float("inf")

    # Forward Euler integration — flow-limited (instantaneous equilibrium) PBPK
    # In flow-limited PBPK, tissues equilibrate instantaneously with plasma:
    #   C_tissue = Kp_tissue * C_plasma
    # The effective volume of distribution accounts for all tissue compartments:
    #   Vd_eff = V_plasma + sum(Kp_org * V_org)
    # Plasma elimination ODE:
    #   dC_plasma/dt = -CL_eff * C_plasma / Vd_eff
    # Cardiac output affects CL_eff via the well-stirred model (already computed).
    # To capture CO effects on tissue redistribution, we model time-varying
    # Kp_eff using an exponential approach:
    #   C_tissue(t) = Kp_org * C_plasma(t) * (1 - exp(-Q_org / V_org * t))
    #                 + C_tissue(0) * exp(-Q_org / V_org * t)
    # This tracks the wash-in/wash-out of each tissue as a function of organ flow.

    # Effective Vd for plasma ODE (steady-state distribution)
    v_plasma = vd_L - sum(kp[org] * _ORGAN_VOLUMES[org] for org in _ORGANS)
    v_plasma = max(v_plasma, 3.0)  # clamp to minimum physiological plasma volume

    # Effective elimination rate constant for plasma
    ke = cl_total / vd_L  # elimination from plasma using given Vd

    n_steps = max(int(round(t_end_h / dt_h)), 2)
    times = np.linspace(0.0, t_end_h, n_steps + 1)
    n_pts = len(times)

    # Initial conditions: IV bolus → drug distributed at t=0 according to Kp
    c_plasma_arr = np.zeros(n_pts)
    # Each tissue tracked as C_tissue (mg/L); initial equilibrium with plasma
    c_tissue_arr: dict[str, np.ndarray] = {org: np.zeros(n_pts) for org in _ORGANS}

    c0_plasma = dose_mg / vd_L
    c_plasma_arr[0] = c0_plasma
    for org in _ORGANS:
        c_tissue_arr[org][0] = kp[org] * c0_plasma

    # Tissue wash-in rate constants (h⁻¹): how fast each tissue equilibrates
    k_tissue: dict[str, float] = {org: q_scaled[org] / _ORGAN_VOLUMES[org] for org in _ORGANS}

    for i in range(n_pts - 1):
        cp = c_plasma_arr[i]

        # Plasma ODE: single-compartment elimination
        dcp_dt = -ke * cp
        new_cp = cp + dcp_dt * dt_h
        c_plasma_arr[i + 1] = max(0.0, new_cp)

        # Tissue ODE: first-order approach to equilibrium with plasma
        # dC_tissue/dt = k_tissue * (Kp * C_plasma - C_tissue)
        for org in _ORGANS:
            ct = c_tissue_arr[org][i]
            eq_conc = kp[org] * cp
            dct_dt = k_tissue[org] * (eq_conc - ct)
            new_ct = ct + dct_dt * dt_h
            c_tissue_arr[org][i + 1] = max(0.0, new_ct)

    # Build tissue concentration time courses
    tissue_concs: dict[str, list[float]] = {}
    for org in _ORGANS:
        tissue_concs[org] = c_tissue_arr[org].tolist()

    # AUC by trapezoidal rule
    auc_plasma = float(np_trapz(c_plasma_arr, times))

    notes_parts = [
        f"CO fraction: {cardiac_output_fraction:.2f}; "
        f"Q_liver_scaled: {q_h:.1f} L/h; "
        f"CL_hep_eff: {cl_hep_eff:.3f} L/h"
    ]
    if cardiac_output_fraction < 0.7:
        notes_parts.append("Reduced CO → lower hepatic CL → increased exposure")
    elif cardiac_output_fraction > 1.3:
        notes_parts.append("Elevated CO → higher hepatic CL → reduced exposure")

    return COPBPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        cardiac_output_fraction=cardiac_output_fraction,
        cl_hep_effective_L_per_h=float(cl_hep_eff),
        times_h=times.tolist(),
        c_plasma_mg_L=c_plasma_arr.tolist(),
        tissue_concs=tissue_concs,
        auc_plasma=auc_plasma,
        t_half_h=float(t_half),
        notes="; ".join(notes_parts),
    )


def cardiac_output_sensitivity(
    drug_name: str,
    dose_mg: float,
    cl_hep_L_per_h: float,
    vd_L: float,
    co_fractions: list[float],
    organ_flows: dict[str, float] | None = None,
    organ_kp: dict[str, float] | None = None,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> list[dict]:
    """Evaluate AUC, t½, and effective CL across a range of CO fractions.

    Parameters
    ----------
    drug_name:
        Compound identifier.
    dose_mg:
        IV bolus dose (mg).
    cl_hep_L_per_h:
        Nominal hepatic clearance at normal CO (L/h).
    vd_L:
        Volume of distribution (L).
    co_fractions:
        List of cardiac output fractions to evaluate; each must be in (0, 5].
    organ_flows:
        Normal organ flows; forwarded to :func:`simulate_co_pbpk`.
    organ_kp:
        Organ Kp values; forwarded to :func:`simulate_co_pbpk`.
    t_end_h:
        Simulation duration (h).
    dt_h:
        Forward Euler time step (h).

    Returns
    -------
    list[dict]
        One dict per CO fraction with keys:
        ``co_fraction``, ``auc_plasma``, ``t_half_h``, ``cl_hep_effective_L_per_h``.
    """
    if not co_fractions:
        raise ValueError("co_fractions must be a non-empty list")
    for f in co_fractions:
        if not (0 < f <= 5.0):
            raise ValueError(f"Each co_fraction must be in (0, 5], got {f}")

    results: list[dict] = []
    for frac in co_fractions:
        res = simulate_co_pbpk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            cl_hep_L_per_h=cl_hep_L_per_h,
            cardiac_output_fraction=frac,
            organ_flows=organ_flows,
            organ_kp=organ_kp,
            vd_L=vd_L,
            t_end_h=t_end_h,
            dt_h=dt_h,
        )
        results.append(
            {
                "co_fraction": frac,
                "auc_plasma": res.auc_plasma,
                "t_half_h": res.t_half_h,
                "cl_hep_effective_L_per_h": res.cl_hep_effective_L_per_h,
            }
        )
    return results
