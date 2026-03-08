"""Phase 517 — Mucosal Drug Absorption Model.

Models drug absorption through mucosal membranes with mucus layer
diffusion and tight junction effects using a 2-compartment approach.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class MucosalAbsorptionResult:
    """Result of mucosal absorption simulation."""

    drug_name: str
    dose_mg: float
    mw_Da: float
    logP: float
    f_mucus: float
    ka_apparent_per_h: float
    e_hepatic: float
    f_hepatic: float
    times_h: list = field(default_factory=list)
    c_systemic_mg_L: list = field(default_factory=list)
    cmax_mg_L: float = 0.0
    tmax_h: float = 0.0
    auc_mg_h_per_L: float = 0.0
    bioavailability_pct: float = 0.0
    t_half_h: float = 0.0
    notes: str = ""


def _compute_f_mucus(mw_Da: float, logP: float) -> float:
    """Compute mucus penetration fraction.

    f_mucus = exp(-MW / 500) * (1 + logP) / (1 + abs(logP))
    Clamped to [0.05, 1.0].
    Hydrophilic drugs penetrate mucus better.
    """
    size_factor = math.exp(-mw_Da / 500.0)
    # logP factor: for logP >= 0, gives (1+logP)/(1+logP) = 1.0
    # for logP < 0, gives (1+logP)/(1-logP) which decreases with more negative logP
    lp_factor = (1.0 + logP) / (1.0 + abs(logP))
    f_mucus = size_factor * lp_factor
    return max(0.05, min(1.0, f_mucus))


def simulate_mucosal_absorption(
    drug_name: str,
    dose_mg: float,
    mw_Da: float,
    logP: float,
    ka_per_h: float = 1.0,
    cl_hepatic_L_per_h: float = 20.0,
    vd_sys_L: float = 50.0,
    f_abs: float = 0.9,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> MucosalAbsorptionResult:
    """Simulate mucosal drug absorption.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Oral dose in mg.
    mw_Da:
        Molecular weight in Da.
    logP:
        Octanol-water partition coefficient.
    ka_per_h:
        Intrinsic absorption rate constant (1/h).
    cl_hepatic_L_per_h:
        Hepatic clearance (L/h).
    vd_sys_L:
        Systemic volume of distribution (L).
    f_abs:
        Gut availability (fraction absorbed past GI wall).
    t_end_h:
        Simulation end time (h).
    dt_h:
        Time step (h).

    Returns
    -------
    MucosalAbsorptionResult
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if mw_Da <= 0:
        raise ValueError("mw_Da must be > 0")
    if cl_hepatic_L_per_h <= 0:
        raise ValueError("cl_hepatic_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")

    # Mucus penetration factor
    f_mucus = _compute_f_mucus(mw_Da, logP)

    # Apparent absorption rate constant accounting for mucus
    ka_app = ka_per_h * f_mucus

    # Portal compartment parameters
    Q_portal = 72.0  # L/h
    Vd_portal = 3.0  # L
    k_portal = Q_portal / Vd_portal  # 1/h — portal clearance rate

    # Hepatic extraction ratio and availability
    E_h = cl_hepatic_L_per_h / (cl_hepatic_L_per_h + Q_portal)
    f_hepatic = 1.0 - E_h

    # Systemic elimination rate constant
    k_el = cl_hepatic_L_per_h / vd_sys_L

    # Initial conditions
    A_lumen = dose_mg  # mg in GI lumen
    A_portal = 0.0  # mg in portal blood
    C_sys = 0.0  # mg/L systemic concentration

    times: list[float] = []
    c_sys_list: list[float] = []

    t = 0.0
    n_steps = int(round(t_end_h / dt_h))

    for _ in range(n_steps + 1):
        times.append(t)
        c_sys_list.append(C_sys)

        # Forward Euler derivatives
        dA_lumen = -ka_app * A_lumen
        dA_portal = ka_app * f_abs * A_lumen - k_portal * A_portal
        dC_sys = (1.0 - E_h) * k_portal * A_portal / vd_sys_L - k_el * C_sys

        A_lumen += dA_lumen * dt_h
        A_portal += dA_portal * dt_h
        C_sys += dC_sys * dt_h

        # Clamp negatives from numerical noise
        if A_lumen < 0.0:
            A_lumen = 0.0
        if A_portal < 0.0:
            A_portal = 0.0
        if C_sys < 0.0:
            C_sys = 0.0

        t += dt_h

    # PK metrics
    cmax = max(c_sys_list)
    tmax = times[c_sys_list.index(cmax)]

    auc = sum(
        0.5 * (c_sys_list[i] + c_sys_list[i - 1]) * (times[i] - times[i - 1])
        for i in range(1, len(times))
    )

    # Approximate half-life from terminal elimination
    t_half = math.log(2.0) / k_el if k_el > 0 else float("inf")

    bioavailability_pct = f_mucus * f_abs * f_hepatic * 100.0

    notes = (
        f"f_mucus={f_mucus:.3f}, ka_app={ka_app:.3f}/h, E_h={E_h:.3f}, F={bioavailability_pct:.1f}%"
    )

    return MucosalAbsorptionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        mw_Da=mw_Da,
        logP=logP,
        f_mucus=f_mucus,
        ka_apparent_per_h=ka_app,
        e_hepatic=E_h,
        f_hepatic=f_hepatic,
        times_h=times,
        c_systemic_mg_L=c_sys_list,
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc_mg_h_per_L=auc,
        bioavailability_pct=bioavailability_pct,
        t_half_h=t_half,
        notes=notes,
    )


def compare_mw_effect(
    drug_name: str,
    dose_mg: float,
    logP: float,
    mw_values_Da: list,
) -> list:
    """Compare mucosal absorption across different molecular weights.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Oral dose in mg.
    logP:
        Octanol-water partition coefficient.
    mw_values_Da:
        List of molecular weights (Da) to compare.

    Returns
    -------
    list of MucosalAbsorptionResult, sorted by bioavailability_pct descending.
    """
    results = [
        simulate_mucosal_absorption(
            drug_name=drug_name,
            dose_mg=dose_mg,
            mw_Da=mw,
            logP=logP,
        )
        for mw in mw_values_Da
    ]
    results.sort(key=lambda r: r.bioavailability_pct, reverse=True)
    return results
