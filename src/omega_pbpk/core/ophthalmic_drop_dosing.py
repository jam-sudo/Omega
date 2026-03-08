"""Phase 531: Ophthalmic Drop Dosing Model.

PK model for topical ophthalmic eye drop dosing.
Compartments: tear film, aqueous humor, systemic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class OphthalmicDropResult:
    """Result of ophthalmic drop dosing simulation."""

    drug_name: str
    dose_mg: float
    logP: float
    n_drops: int
    k_corneal_per_h: float
    k_drain_per_h: float
    times_h: list
    a_tear_mg: list
    c_aqueous_mg_L: list
    c_systemic_mg_L: list
    cmax_aqueous_mg_L: float
    tmax_aqueous_h: float
    auc_aqueous_mg_h_per_L: float
    cmax_systemic_mg_L: float
    systemic_fraction_pct: float
    corneal_bioavailability_pct: float
    notes: str


def _compute_k_corneal(logP: float) -> float:
    """Compute corneal absorption rate constant per hour.

    Simplified model:
    - logP > 0: k_corneal = 0.1 * (1 + logP) / 2
    - logP <= 0: k_corneal = 0.01
    Clamped to [0.01, 5.0].
    """
    if logP > 0:
        k = 0.1 * (1.0 + logP) / 2.0
    else:
        k = 0.01
    # Clamp
    if k < 0.01:
        k = 0.01
    if k > 5.0:
        k = 5.0
    return k


def _trapezoidal_auc(x: list, y: list) -> float:
    """Manual trapezoidal AUC."""
    return sum(0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]) for i in range(1, len(x)))


def simulate_ophthalmic_drop(
    drug_name: str,
    dose_mg: float,
    logP: float,
    n_drops: int = 1,
    cl_sys_L_per_h: float = 10.0,
    vd_sys_L: float = 50.0,
    t_end_h: float = 8.0,
    dt_h: float = 0.01,
) -> OphthalmicDropResult:
    """Simulate topical ophthalmic eye drop pharmacokinetics.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Dose per drop in mg.
    logP:
        Lipophilicity (log octanol-water partition coefficient).
    n_drops:
        Number of drops administered simultaneously.
    cl_sys_L_per_h:
        Systemic clearance (L/h).
    vd_sys_L:
        Systemic volume of distribution (L).
    t_end_h:
        Simulation end time (h).
    dt_h:
        Forward Euler time step (h).

    Returns
    -------
    OphthalmicDropResult
    """
    # Validate inputs
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if n_drops < 1:
        raise ValueError("n_drops must be >= 1")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")

    # Total dose
    total_dose_mg = dose_mg * n_drops

    # Rate constants
    k_drain = 30.0  # /h (0.5/min drainage)
    k_corneal = _compute_k_corneal(logP)  # /h
    k_aq_out = 0.1  # /h (aqueous humor turnover)
    f_nl = 0.6  # fraction of drained dose absorbed systemically
    k_el = cl_sys_L_per_h / vd_sys_L  # /h

    # Aqueous humor volume
    V_aq = 0.00025  # L (250 µL)

    # Initial conditions
    A_tear = total_dose_mg  # mg in tear film
    A_aq = 0.0  # mg in aqueous humor
    C_sys = 0.0  # mg/L systemic concentration

    # Storage
    n_steps = int(math.ceil(t_end_h / dt_h)) + 1
    times_h: list = []
    a_tear_mg: list = []
    c_aqueous_mg_L: list = []
    c_systemic_mg_L: list = []

    t = 0.0
    times_h.append(t)
    a_tear_mg.append(A_tear)
    c_aqueous_mg_L.append(A_aq / V_aq)
    c_systemic_mg_L.append(C_sys)

    systemic_absorbed_mg = 0.0

    for _ in range(n_steps - 1):
        # Derivatives
        dA_tear = -(k_drain + k_corneal) * A_tear
        dA_aq = k_corneal * A_tear - k_aq_out * A_aq
        # Systemic: nasolacrimal drainage -> systemic
        dC_sys = (k_drain * f_nl * A_tear) / vd_sys_L - k_el * C_sys

        # Track systemic absorption flux (for fraction calculation)
        systemic_absorbed_mg += k_drain * f_nl * A_tear * dt_h

        # Forward Euler update
        A_tear += dA_tear * dt_h
        A_aq += dA_aq * dt_h
        C_sys += dC_sys * dt_h

        # Clamp negatives
        if A_tear < 0.0:
            A_tear = 0.0
        if A_aq < 0.0:
            A_aq = 0.0
        if C_sys < 0.0:
            C_sys = 0.0

        t += dt_h
        times_h.append(t)
        a_tear_mg.append(A_tear)
        c_aqueous_mg_L.append(A_aq / V_aq)
        c_systemic_mg_L.append(C_sys)

    # Derived metrics
    cmax_aqueous = max(c_aqueous_mg_L)
    tmax_aqueous = times_h[c_aqueous_mg_L.index(cmax_aqueous)]
    auc_aqueous = _trapezoidal_auc(times_h, c_aqueous_mg_L)

    cmax_systemic = max(c_systemic_mg_L)
    systemic_fraction_pct = (systemic_absorbed_mg / total_dose_mg) * 100.0

    # Corneal bioavailability: fraction that reached aqueous (via corneal path)
    # Estimate from AUC_aq * k_aq_out * V_aq relative to dose
    auc_amount_aq = _trapezoidal_auc(times_h, a_tear_mg)
    corneal_amount = k_corneal * auc_amount_aq  # total mg absorbed cornally
    corneal_bioavailability_pct = (corneal_amount / total_dose_mg) * 100.0
    if corneal_bioavailability_pct > 100.0:
        corneal_bioavailability_pct = 100.0

    notes = (
        f"k_corneal={k_corneal:.3f}/h, k_drain={k_drain:.1f}/h, "
        f"f_nl={f_nl:.1f}; systemic exposure via nasolacrimal route"
    )

    return OphthalmicDropResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        logP=logP,
        n_drops=n_drops,
        k_corneal_per_h=k_corneal,
        k_drain_per_h=k_drain,
        times_h=times_h,
        a_tear_mg=a_tear_mg,
        c_aqueous_mg_L=c_aqueous_mg_L,
        c_systemic_mg_L=c_systemic_mg_L,
        cmax_aqueous_mg_L=cmax_aqueous,
        tmax_aqueous_h=tmax_aqueous,
        auc_aqueous_mg_h_per_L=auc_aqueous,
        cmax_systemic_mg_L=cmax_systemic,
        systemic_fraction_pct=systemic_fraction_pct,
        corneal_bioavailability_pct=corneal_bioavailability_pct,
        notes=notes,
    )


def compare_logp_ophthalmic(drug_name: str, dose_mg: float, logp_values: list) -> list:
    """Compare ophthalmic PK across different logP values.

    Parameters
    ----------
    drug_name:
        Drug name.
    dose_mg:
        Dose per drop in mg.
    logp_values:
        List of logP values to simulate.

    Returns
    -------
    List of OphthalmicDropResult sorted by auc_aqueous_mg_h_per_L descending.
    """
    results = [simulate_ophthalmic_drop(drug_name, dose_mg, logP) for logP in logp_values]
    results.sort(key=lambda r: r.auc_aqueous_mg_h_per_L, reverse=True)
    return results
