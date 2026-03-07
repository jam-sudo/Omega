"""Advanced lymphatic absorption model with TRL (triglyceride-rich lipoprotein) association.

Extends core/lymphatic.py with chylomicron/TRL-mediated drug transport:
  - Lipophilic drugs (logP > 5) associate with TRL in intestinal lymph
  - TRL association bypasses portal first-pass and follows slow lipolysis clearance
  - Three absorption pathways: portal, lymphatic free, TRL-bound

References:
  - Trevaskis et al. 2008 (lymphatic drug absorption)
  - Porter et al. 2007 (lipid-based formulations and TRL)
  - Charman & Stella 1986 (intestinal lymphatic drug transport)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz


@dataclass(frozen=True)
class LymphaticAdvancedResult:
    """Result of advanced lymphatic absorption simulation with TRL association."""

    drug_name: str
    dose_mg: float
    logP: float
    fat_content_g: float
    trl_fraction: float  # fraction of lymphatic dose carried by TRL
    lymph_fraction: float  # fraction of total dose via lymphatics (portal bypass)
    times_h: list[float]
    c_plasma: list[float]  # total plasma concentration (mg/L)
    cmax: float  # peak plasma concentration (mg/L)
    tmax_h: float  # time of peak (h)
    auc: float  # AUC (mg*h/L)
    f_absorbed_portal: float  # fraction of dose absorbed via portal
    f_absorbed_lymph: float  # fraction of dose absorbed via lymph free
    f_absorbed_trl: float  # fraction of dose absorbed via TRL
    notes: str


def _sigmoid(x: float) -> float:
    """Standard logistic sigmoid function."""
    return 1.0 / (1.0 + math.exp(-x))


def _lymph_fraction_from_logp(logP: float) -> float:
    """Estimate lymphatic fraction from logP.

    Uses sigmoid with inflection at logP=3, steepness 1/1.5,
    scaled to maximum 0.8 (literature upper bound for lymphatic absorption).

    From core/lymphatic.py logic:
        f_lymph = sigmoid((logP - 3) / 1.5) * 0.8
    """
    f = _sigmoid((logP - 3.0) / 1.5) * 0.8
    return float(np.clip(f, 0.0, 0.8))


def trl_association_fraction(
    logP: float,
    fat_content_g: float = 20.0,
    drug_dose_mg: float = 100.0,
) -> float:
    """Estimate fraction of lymphatic drug dose carried by TRL (chylomicrons).

    TRL association is driven by:
      - Drug lipophilicity (logP): higher logP → more TRL association
      - Fat content of meal: more fat → more chylomicron formation
      - Drug dose is implicit (assumes sink conditions in TRL)

    Formula:
        f_trl = sigmoid(logP - 4) * min(1.0, fat_g/50) * min(1.0, logP/7)
        clamped to [0, 0.95]

    Parameters
    ----------
    logP : float
        Drug lipophilicity.
    fat_content_g : float
        Fat content of co-administered meal (g). 0 = fasted.
    drug_dose_mg : float
        Drug dose (mg), retained for API signature compatibility.

    Returns
    -------
    float
        Fraction of lymphatic drug dose in TRL [0, 0.95].
    """
    if fat_content_g < 0:
        raise ValueError("fat_content_g must be >= 0")

    lipophilicity_factor = _sigmoid(logP - 4.0)
    fat_factor = min(1.0, fat_content_g / 50.0)
    logp_saturation = min(1.0, logP / 7.0) if logP > 0 else 0.0

    f_trl = lipophilicity_factor * fat_factor * logp_saturation
    return float(np.clip(f_trl, 0.0, 0.95))


def simulate_lymphatic_advanced_pk(
    drug_name: str,
    dose_mg: float,
    logP: float,
    cl_L_per_h: float,
    vd_L: float,
    fat_content_g: float = 20.0,
    ka_portal_per_h: float = 1.5,
    ka_lymph_per_h: float = 0.15,
    trl_clearance_per_h: float = 0.05,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> LymphaticAdvancedResult:
    """Simulate oral absorption with portal, lymphatic free, and TRL pathways.

    Three absorption compartments deliver drug to a single central plasma compartment:
      1. Portal gut depot  → fast absorption (ka_portal), undergoes hepatic first-pass
      2. Lymph free depot  → slow absorption (ka_lymph), bypasses liver
      3. TRL depot         → very slow, governed by lipolysis (trl_clearance_per_h)

    ODE system (Forward Euler):
        dA_portal/dt   = -ka_portal * A_portal
        dA_lymph/dt    = -ka_lymph * A_lymph
        dA_trl/dt      = -trl_clearance * A_trl
        dA_central/dt  = ka_portal*A_portal + ka_lymph*A_lymph
                         + trl_clearance*A_trl - (CL/Vd)*A_central

    Note: Portal fraction undergoes hepatic first-pass (well-stirred model).
    Lymph and TRL fractions bypass hepatic first-pass.

    Parameters
    ----------
    drug_name : str
        Drug name label.
    dose_mg : float
        Oral dose (mg).
    logP : float
        Drug lipophilicity.
    cl_L_per_h : float
        Systemic clearance (L/h).
    vd_L : float
        Volume of distribution (L).
    fat_content_g : float
        Fat content of co-administered meal (g).
    ka_portal_per_h : float
        Portal absorption rate constant (1/h).
    ka_lymph_per_h : float
        Free lymphatic absorption rate constant (1/h).
    trl_clearance_per_h : float
        TRL lipolysis/clearance rate (1/h), governs TRL drug release.
    t_end_h : float
        Simulation end time (h).
    dt_h : float
        Forward Euler time step (h).

    Returns
    -------
    LymphaticAdvancedResult
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if fat_content_g < 0:
        raise ValueError("fat_content_g must be >= 0")
    if ka_portal_per_h <= 0:
        raise ValueError("ka_portal_per_h must be > 0")
    if ka_lymph_per_h <= 0:
        raise ValueError("ka_lymph_per_h must be > 0")
    if trl_clearance_per_h <= 0:
        raise ValueError("trl_clearance_per_h must be > 0")

    # Compute absorption fractions
    f_lymph = _lymph_fraction_from_logp(logP)
    f_portal_total = 1.0 - f_lymph  # fraction absorbed via portal route

    f_trl = trl_association_fraction(logP, fat_content_g, dose_mg)

    # Lymphatic sub-fractions
    f_lymph_free = f_lymph * (1.0 - f_trl)
    f_lymph_trl = f_lymph * f_trl

    # Hepatic first-pass for portal fraction (well-stirred model)
    q_liver_L_per_h = 90.0
    e_h = min(cl_L_per_h / q_liver_L_per_h, 0.999)
    fh = 1.0 - e_h  # survival fraction through liver

    # Dose allocation to absorption compartments (mg)
    a_portal = dose_mg * f_portal_total
    a_lymph_free = dose_mg * f_lymph_free
    a_trl = dose_mg * f_lymph_trl
    a_central = 0.0

    ke = cl_L_per_h / vd_L  # elimination rate constant (1/h)

    n_steps = max(int(t_end_h / dt_h), 1)
    times = np.linspace(0.0, t_end_h, n_steps + 1)
    c_plasma_arr = np.zeros(n_steps + 1)

    # Track absorbed amounts for bioavailability fractions
    absorbed_portal = 0.0
    absorbed_lymph_free = 0.0
    absorbed_trl = 0.0

    for i in range(n_steps):
        # Absorption fluxes (mg/h)
        flux_portal = ka_portal_per_h * a_portal * fh  # first-pass applied at input
        flux_lymph_free = ka_lymph_per_h * a_lymph_free
        flux_trl = trl_clearance_per_h * a_trl

        # Forward Euler ODE
        d_portal = -ka_portal_per_h * a_portal
        d_lymph_free = -ka_lymph_per_h * a_lymph_free
        d_trl = -trl_clearance_per_h * a_trl
        d_central = flux_portal + flux_lymph_free + flux_trl - ke * a_central

        a_portal = max(a_portal + d_portal * dt_h, 0.0)
        a_lymph_free = max(a_lymph_free + d_lymph_free * dt_h, 0.0)
        a_trl = max(a_trl + d_trl * dt_h, 0.0)
        a_central = max(a_central + d_central * dt_h, 0.0)

        # Accumulate absorbed amounts for bioavailability
        absorbed_portal += flux_portal * dt_h
        absorbed_lymph_free += flux_lymph_free * dt_h
        absorbed_trl += flux_trl * dt_h

        c_plasma_arr[i + 1] = a_central / vd_L

    auc = float(np_trapz(c_plasma_arr, times))
    cmax = float(np.max(c_plasma_arr))
    tmax_h = float(times[int(np.argmax(c_plasma_arr))])

    # Bioavailability fractions (as fraction of dose)
    f_abs_portal = absorbed_portal / dose_mg
    f_abs_lymph = absorbed_lymph_free / dose_mg
    f_abs_trl = absorbed_trl / dose_mg

    notes = (
        f"Lymphatic advanced PK ({drug_name}): logP={logP:.2f}, "
        f"f_lymph={f_lymph:.3f}, f_trl={f_trl:.3f}, "
        f"fat={fat_content_g:.1f} g, "
        f"Cmax={cmax:.4g} mg/L, AUC={auc:.4g} mg*h/L"
    )

    return LymphaticAdvancedResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        logP=logP,
        fat_content_g=fat_content_g,
        trl_fraction=f_trl,
        lymph_fraction=f_lymph,
        times_h=times.tolist(),
        c_plasma=c_plasma_arr.tolist(),
        cmax=cmax,
        tmax_h=tmax_h,
        auc=auc,
        f_absorbed_portal=f_abs_portal,
        f_absorbed_lymph=f_abs_lymph,
        f_absorbed_trl=f_abs_trl,
        notes=notes,
    )


def compare_fat_meals(
    drug_name: str,
    dose_mg: float,
    logP: float,
    cl_L_per_h: float,
    vd_L: float,
) -> dict[str, LymphaticAdvancedResult]:
    """Compare lymphatic absorption under fasted, low-fat, and high-fat conditions.

    Runs simulate_lymphatic_advanced_pk for:
      - fat_content_g = 0   (fasted)
      - fat_content_g = 10  (low-fat meal)
      - fat_content_g = 40  (high-fat meal)

    Parameters
    ----------
    drug_name : str
        Drug name label.
    dose_mg : float
        Oral dose (mg).
    logP : float
        Drug lipophilicity.
    cl_L_per_h : float
        Systemic clearance (L/h).
    vd_L : float
        Volume of distribution (L).

    Returns
    -------
    dict with keys "fasted", "low_fat", "high_fat"
    """
    results: dict[str, LymphaticAdvancedResult] = {}

    for label, fat_g in [
        ("fasted", 0.0),
        ("low_fat", 10.0),
        ("high_fat", 40.0),
    ]:
        results[label] = simulate_lymphatic_advanced_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            logP=logP,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            fat_content_g=fat_g,
        )

    return results


__all__ = [
    "LymphaticAdvancedResult",
    "trl_association_fraction",
    "simulate_lymphatic_advanced_pk",
    "compare_fat_meals",
]
