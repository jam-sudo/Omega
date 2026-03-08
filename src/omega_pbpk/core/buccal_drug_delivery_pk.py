"""Phase 545 — Buccal Drug Delivery PK model.

Models drug absorption via the buccal mucosa (between cheek and gum),
accounting for buccal permeability, swallowing loss, and systemic
distribution using Forward Euler ODE integration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class BuccalDrugDeliveryResult:
    """Result of buccal drug delivery PK simulation."""

    drug_name: str
    dose_mg: float
    logP: float
    residence_time_h: float

    ka_buccal_per_h: float
    f_buccal_absorbed: float

    times_h: list = field(default_factory=list)
    c_systemic_mg_L: list = field(default_factory=list)

    cmax_mg_L: float = 0.0
    tmax_h: float = 0.0
    auc_mg_h_per_L: float = 0.0

    buccal_vs_oral_tmax_factor: float = 0.0
    bioavailability_pct: float = 0.0

    notes: str = ""


def _compute_ka_buccal(logP: float) -> float:
    """Compute buccal absorption rate constant (per h) from logP.

    Linear mapping reflecting increase in mucosal permeability with
    lipophilicity, clamped to physiological range [0.01, 2.0] /h.
    """
    ka = 0.3 * logP + 0.5
    return max(0.01, min(2.0, ka))


def simulate_buccal_delivery(
    drug_name: str,
    dose_mg: float,
    logP: float,
    cl_L_per_h: float,
    vd_sys_L: float,
    residence_time_h: float = 0.5,
    t_end_h: float = 12.0,
    dt_h: float = 0.05,
) -> BuccalDrugDeliveryResult:
    """Simulate buccal drug delivery PK using Forward Euler.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Dose placed in buccal cavity (mg).
    logP:
        Octanol-water partition coefficient.
    cl_L_per_h:
        Systemic clearance (L/h).
    vd_sys_L:
        Systemic volume of distribution (L).
    residence_time_h:
        Time drug remains in buccal cavity before swallowed (h).
    t_end_h:
        Simulation end time (h).
    dt_h:
        Forward Euler time step (h).

    Returns
    -------
    BuccalDrugDeliveryResult
    """
    if dose_mg <= 0.0:
        raise ValueError("dose_mg must be positive")
    if cl_L_per_h <= 0.0:
        raise ValueError("cl_L_per_h must be positive")
    if vd_sys_L <= 0.0:
        raise ValueError("vd_sys_L must be positive")
    if residence_time_h <= 0.0:
        raise ValueError("residence_time_h must be positive")

    ka_buccal = _compute_ka_buccal(logP)
    k_swallow = 0.3  # /h
    ka_gi = 0.5  # /h
    f_oral = 0.5  # first-pass fraction for swallowed portion
    k_el = cl_L_per_h / vd_sys_L

    # Fraction effectively absorbed buccally (analytical estimate for notes)
    f_buccal = (
        ka_buccal
        / (ka_buccal + k_swallow)
        * (1.0 - math.exp(-(ka_buccal + k_swallow) * residence_time_h))
    )

    # State: A_buccal (mg), A_stomach (mg), C_sys (mg/L)
    A_buccal = dose_mg
    A_stomach = 0.0
    C_sys = 0.0

    times_h = []
    c_systemic = []

    t = 0.0
    n_steps = int(round(t_end_h / dt_h))

    for _ in range(n_steps + 1):
        times_h.append(t)
        c_systemic.append(C_sys)

        # Derivatives
        dA_buccal = -(ka_buccal + k_swallow) * A_buccal
        dA_stomach = k_swallow * A_buccal - ka_gi * A_stomach
        dC_sys = (
            ka_buccal * A_buccal / vd_sys_L + ka_gi * A_stomach * f_oral / vd_sys_L - k_el * C_sys
        )

        # Euler update
        A_buccal += dA_buccal * dt_h
        A_stomach += dA_stomach * dt_h
        C_sys += dC_sys * dt_h

        # Clamp negatives
        A_buccal = max(0.0, A_buccal)
        A_stomach = max(0.0, A_stomach)
        C_sys = max(0.0, C_sys)

        t += dt_h

    # PK metrics
    cmax = max(c_systemic)
    tmax = times_h[c_systemic.index(cmax)]

    auc = sum(
        0.5 * (c_systemic[i] + c_systemic[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )

    # Theoretical AUC for 100% bioavailability: dose/(CL)
    auc_theoretical = dose_mg / cl_L_per_h
    bioav_pct = min(100.0, (auc / auc_theoretical) * 100.0) if auc_theoretical > 0 else 0.0

    typical_oral_tmax = 1.5  # h
    buccal_vs_oral_factor = tmax / typical_oral_tmax if typical_oral_tmax > 0 else 0.0

    notes_parts = []
    if logP < 0:
        notes_parts.append("logP<0: very low buccal permeability; ka clamped to minimum")
    if logP > 3:
        notes_parts.append("logP>3: good lipophilicity for buccal absorption")
    if f_buccal < 0.2:
        notes_parts.append("low buccal absorption fraction; most drug swallowed")

    return BuccalDrugDeliveryResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        logP=logP,
        residence_time_h=residence_time_h,
        ka_buccal_per_h=ka_buccal,
        f_buccal_absorbed=f_buccal,
        times_h=times_h,
        c_systemic_mg_L=c_systemic,
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc_mg_h_per_L=auc,
        buccal_vs_oral_tmax_factor=buccal_vs_oral_factor,
        bioavailability_pct=bioav_pct,
        notes="; ".join(notes_parts),
    )


def compare_logp_buccal(
    drug_name: str,
    dose_mg: float,
    logp_values: list,
    cl_L_per_h: float = 10.0,
    vd_sys_L: float = 50.0,
) -> list:
    """Compare buccal PK across different logP values.

    Parameters
    ----------
    drug_name:
        Base name for the drug (logP value appended).
    dose_mg:
        Dose (mg).
    logp_values:
        List of logP values to compare.
    cl_L_per_h:
        Systemic clearance (L/h).
    vd_sys_L:
        Systemic volume of distribution (L).

    Returns
    -------
    list of BuccalDrugDeliveryResult sorted by AUC descending.
    """
    results = []
    for lp in logp_values:
        res = simulate_buccal_delivery(
            drug_name=f"{drug_name}_logP{lp}",
            dose_mg=dose_mg,
            logP=lp,
            cl_L_per_h=cl_L_per_h,
            vd_sys_L=vd_sys_L,
        )
        results.append(res)

    results.sort(key=lambda r: r.auc_mg_h_per_L, reverse=True)
    return results
