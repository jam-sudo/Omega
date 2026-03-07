"""Subcutaneous (SC) drug absorption pharmacokinetics.

Models drug absorption after SC injection through two parallel pathways:
  1. Direct capillary absorption (predominant for small molecules)
  2. Lymphatic absorption (important for large molecules/biologics)

Reference: Supersaxo et al. (1990), Kagan et al. (2007).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz

__all__ = [
    "SCPKResult",
    "simulate_sc_pk",
    "sc_bioavailability",
    "compare_sc_iv",
]


@dataclass(frozen=True)
class SCPKResult:
    """Result of a subcutaneous PK simulation.

    Attributes
    ----------
    drug_name : str
        Drug name.
    dose_mg : float
        Administered SC dose (mg).
    mw_kDa : float
        Drug molecular weight (kDa).
    f_lymph : float
        Fraction of dose absorbed via lymphatic route (0-1).
    times_h : list[float]
        Time points (h).
    c_plasma : list[float]
        Plasma concentration at each time point (ng/mL or consistent units).
    cmax : float
        Maximum plasma concentration.
    tmax_h : float
        Time to maximum plasma concentration (h).
    auc : float
        AUC from 0 to t_end using trapezoidal rule.
    f_via_lymph : float
        Fraction of absorbed drug delivered via lymphatic route.
    f_via_capillary : float
        Fraction of absorbed drug delivered via capillary route.
    notes : str
        Simulation notes.
    """

    drug_name: str
    dose_mg: float
    mw_kDa: float
    f_lymph: float
    times_h: list[float]
    c_plasma: list[float]
    cmax: float
    tmax_h: float
    auc: float
    f_via_lymph: float
    f_via_capillary: float
    notes: str


def _auto_f_lymph(mw_kDa: float) -> float:
    """Auto-calculate lymphatic absorption fraction based on molecular weight.

    Sigmoid relationship: f_lymph = 1 / (1 + exp(-(MW_kDa - 3) / 1))
    - Small molecules (<1 kDa): ~0% lymphatic
    - Transition zone (~3 kDa)
    - Large molecules (>10 kDa): ~50% lymphatic

    Parameters
    ----------
    mw_kDa : float
        Molecular weight in kDa.

    Returns
    -------
    float
        Fraction absorbed via lymphatic route (0 to ~0.5).
    """
    f = 1.0 / (1.0 + math.exp(-(mw_kDa - 3.0) / 1.0))
    # Scale to [0, 0.5]: f_lymph = sigmoid * 0.5
    return f * 0.5


def simulate_sc_pk(
    drug_name: str,
    dose_mg: float,
    mw_kDa: float = 0.5,
    ka_capillary_per_h: float = 0.5,
    ka_lymph_per_h: float = 0.02,
    f_lymph: float | None = None,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
) -> SCPKResult:
    """Simulate subcutaneous drug absorption pharmacokinetics.

    SC depot splits into two parallel absorption pathways:
      - Capillary: A_cap[0] = dose * (1 - f_lymph), ka_cap per h
      - Lymph:     A_lymph[0] = dose * f_lymph, ka_lymph per h

    Both pathways deliver drug to central plasma compartment.

    ODE system (Forward Euler):
        dA_cap/dt   = -ka_cap * A_cap
        dA_lymph/dt = -ka_lymph * A_lymph
        dC/dt       = ka_cap*A_cap/Vd + ka_lymph*A_lymph/Vd - (CL/Vd)*C

    Parameters
    ----------
    drug_name : str
        Drug name.
    dose_mg : float
        SC dose (mg, must be > 0).
    mw_kDa : float
        Molecular weight (kDa, must be > 0). Used to auto-calculate f_lymph.
    ka_capillary_per_h : float
        Capillary absorption rate constant (1/h, must be > 0).
    ka_lymph_per_h : float
        Lymphatic absorption rate constant (1/h, must be > 0).
    f_lymph : float | None
        Fraction absorbed via lymphatics (0-1). If None, auto-calculated
        from MW using sigmoid: f_lymph = sigmoid(mw_kDa, center=3, scale=1) * 0.5
    cl_L_per_h : float
        Total body clearance (L/h, must be > 0).
    vd_L : float
        Volume of distribution (L, must be > 0).
    t_end_h : float
        Simulation end time (h).
    dt_h : float
        Time step for Forward Euler integration (h).

    Returns
    -------
    SCPKResult
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if mw_kDa <= 0:
        raise ValueError("mw_kDa must be > 0")
    if ka_capillary_per_h <= 0:
        raise ValueError("ka_capillary_per_h must be > 0")
    if ka_lymph_per_h <= 0:
        raise ValueError("ka_lymph_per_h must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if f_lymph is not None and not (0.0 <= f_lymph <= 1.0):
        raise ValueError("f_lymph must be in [0, 1]")

    # Auto-calculate f_lymph if not provided
    if f_lymph is None:
        f_lymph_val = _auto_f_lymph(mw_kDa)
    else:
        f_lymph_val = float(f_lymph)

    f_cap = 1.0 - f_lymph_val

    # Initial conditions
    A_cap = dose_mg * f_cap    # mg in capillary depot
    A_lymph = dose_mg * f_lymph_val  # mg in lymphatic depot
    C = 0.0                    # plasma concentration (mg/L)

    ke = cl_L_per_h / vd_L  # elimination rate constant (1/h)

    n_steps = max(int(round(t_end_h / dt_h)), 1)
    times = np.linspace(0.0, t_end_h, n_steps + 1)
    c_arr = np.empty(n_steps + 1)
    c_arr[0] = C

    # Track absorbed amounts for route fractions
    absorbed_cap = 0.0
    absorbed_lymph = 0.0

    # Forward Euler integration
    for i in range(n_steps):
        dA_cap = -ka_capillary_per_h * A_cap
        dA_lymph = -ka_lymph_per_h * A_lymph
        dC = (
            ka_capillary_per_h * A_cap / vd_L
            + ka_lymph_per_h * A_lymph / vd_L
            - ke * C
        )

        # Absorbed in this step
        absorbed_cap += -dA_cap * dt_h
        absorbed_lymph += -dA_lymph * dt_h

        A_cap = max(0.0, A_cap + dA_cap * dt_h)
        A_lymph = max(0.0, A_lymph + dA_lymph * dt_h)
        C = max(0.0, C + dC * dt_h)
        c_arr[i + 1] = C

    # Compute PK metrics
    cmax = float(np.max(c_arr))
    tmax_h = float(times[int(np.argmax(c_arr))])
    auc = float(np_trapz(c_arr, times))

    total_absorbed = absorbed_cap + absorbed_lymph
    if total_absorbed > 0:
        f_via_cap = absorbed_cap / total_absorbed
        f_via_lymph_final = absorbed_lymph / total_absorbed
    else:
        f_via_cap = f_cap
        f_via_lymph_final = f_lymph_val

    notes = (
        f"Forward Euler SC PK. MW={mw_kDa} kDa, f_lymph={f_lymph_val:.3f}. "
        f"ka_cap={ka_capillary_per_h}/h, ka_lymph={ka_lymph_per_h}/h. "
        f"CL={cl_L_per_h} L/h, Vd={vd_L} L, ke={ke:.4f}/h. dt={dt_h} h."
    )

    return SCPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        mw_kDa=mw_kDa,
        f_lymph=f_lymph_val,
        times_h=times.tolist(),
        c_plasma=c_arr.tolist(),
        cmax=cmax,
        tmax_h=tmax_h,
        auc=auc,
        f_via_lymph=f_via_lymph_final,
        f_via_capillary=f_via_cap,
        notes=notes,
    )


def sc_bioavailability(
    mw_kDa: float,
    logP: float,
    solubility_mg_mL: float,
) -> float:
    """Estimate SC bioavailability of a drug.

    SC bioavailability is generally high for small hydrophilic drugs,
    and decreases for:
      - Lipophilic drugs (local depot formation at injection site)
      - Large molecules (proteolytic degradation in SC tissue)

    Formula:
        f_sc = 0.9 * (1 - 0.3 * max(0, logP - 2) / 5) * min(1.0, 1 / (1 + 0.01*mw_kDa))
    Clamped to [0.1, 0.99].

    Parameters
    ----------
    mw_kDa : float
        Molecular weight (kDa).
    logP : float
        Lipophilicity.
    solubility_mg_mL : float
        Aqueous solubility (mg/mL). Not directly used in formula but validated.

    Returns
    -------
    float
        Estimated SC bioavailability fraction (0.1 to 0.99).
    """
    if mw_kDa <= 0:
        raise ValueError("mw_kDa must be > 0")
    if solubility_mg_mL < 0:
        raise ValueError("solubility_mg_mL must be >= 0")

    # Lipophilicity penalty: applies when logP > 2
    lipo_factor = 1.0 - 0.3 * max(0.0, logP - 2.0) / 5.0

    # MW penalty: large molecules have lower SC bioavailability due to
    # proteolytic degradation
    mw_factor = min(1.0, 1.0 / (1.0 + 0.01 * mw_kDa))

    f_sc = 0.9 * lipo_factor * mw_factor

    # Clamp to [0.1, 0.99]
    return max(0.1, min(0.99, f_sc))


def compare_sc_iv(
    drug_name: str,
    dose_mg: float,
    mw_kDa: float,
    cl_L_per_h: float,
    vd_L: float,
) -> dict:
    """Compare SC vs IV pharmacokinetics for the same drug.

    For IV: instantaneous bolus, analytical 1-cpt solution.
    For SC: simulate_sc_pk with default parameters.

    Parameters
    ----------
    drug_name : str
        Drug name.
    dose_mg : float
        Dose (mg, same for both routes).
    mw_kDa : float
        Molecular weight (kDa).
    cl_L_per_h : float
        Total body clearance (L/h).
    vd_L : float
        Volume of distribution (L).

    Returns
    -------
    dict
        Keys: ``auc_sc``, ``auc_iv``, ``auc_ratio``, ``cmax_sc``, ``cmax_iv``,
        ``cmax_ratio``, ``tmax_sc_h``, ``f_lymph``.
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if mw_kDa <= 0:
        raise ValueError("mw_kDa must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")

    t_end_h = max(48.0, 5.0 * vd_L / cl_L_per_h)

    sc_result = simulate_sc_pk(
        drug_name=drug_name,
        dose_mg=dose_mg,
        mw_kDa=mw_kDa,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
        t_end_h=t_end_h,
    )

    # IV: C(t) = (dose/Vd) * exp(-ke*t), AUC = dose/CL
    ke = cl_L_per_h / vd_L
    auc_iv = dose_mg / cl_L_per_h
    cmax_iv = dose_mg / vd_L  # at t=0

    auc_ratio = sc_result.auc / auc_iv if auc_iv > 0 else float("nan")
    cmax_ratio = sc_result.cmax / cmax_iv if cmax_iv > 0 else float("nan")

    return {
        "drug_name": drug_name,
        "auc_sc": sc_result.auc,
        "auc_iv": auc_iv,
        "auc_ratio": auc_ratio,
        "cmax_sc": sc_result.cmax,
        "cmax_iv": cmax_iv,
        "cmax_ratio": cmax_ratio,
        "tmax_sc_h": sc_result.tmax_h,
        "f_lymph": sc_result.f_lymph,
        "ke_per_h": ke,
    }
