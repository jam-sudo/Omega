"""Intranasal drug delivery pharmacokinetics.

Models intranasal drug absorption via a 3-compartment forward-Euler ODE:
  - Nasal cavity: drug deposited, absorbed via olfactory (CNS) and respiratory
    (systemic) epithelia, cleared by mucociliary clearance (MCC).
  - Systemic plasma: receives drug from respiratory epithelium.
  - Brain/CNS (optional): receives drug directly from olfactory epithelium,
    bypassing the blood-brain barrier.

This route is relevant for CNS drugs, vaccines, and rapid systemic delivery.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from omega_pbpk._compat import np_trapz


@dataclass(frozen=True)
class IntranasalPKResult:
    """Result of an intranasal PK simulation."""

    drug_name: str
    dose_mg: float
    times_h: list[float]
    m_nasal_mg: list[float]
    c_systemic_mg_L: list[float]
    c_cns_mg_L: list[float]
    cmax_systemic: float
    tmax_systemic_h: float
    auc_systemic: float
    cmax_cns: float
    auc_cns: float
    f_absorbed_systemic: float
    f_absorbed_cns: float
    nose_to_brain_ratio: float
    notes: list[str] = field(default_factory=list)


def simulate_intranasal_pk(
    drug_name: str,
    dose_mg: float,
    ka_systemic_per_h: float = 2.0,
    ka_cns_per_h: float = 0.5,
    k_mcc_per_h: float = 1.0,
    cl_systemic_L_per_h: float = 10.0,
    vd_systemic_L: float = 50.0,
    v_cns_L: float = 1.5,
    cl_cns_per_h: float = 0.1,
    t_end_h: float = 12.0,
    dt_h: float = 0.05,
) -> IntranasalPKResult:
    """Simulate intranasal drug pharmacokinetics.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Intranasal dose in mg.
    ka_systemic_per_h:
        First-order absorption rate from nasal respiratory epithelium to
        systemic plasma (h^-1).
    ka_cns_per_h:
        First-order absorption rate from nasal olfactory epithelium directly
        to CNS (h^-1). Set to 0 to disable nose-to-brain pathway.
    k_mcc_per_h:
        Mucociliary clearance rate from nasal cavity (h^-1). Drug cleared by
        MCC is removed from the system (swallowed / lost).
    cl_systemic_L_per_h:
        Systemic clearance (L/h).
    vd_systemic_L:
        Volume of distribution for systemic compartment (L).
    v_cns_L:
        Volume of CNS compartment (L).
    cl_cns_per_h:
        CNS elimination clearance (L/h).
    t_end_h:
        Simulation end time (h).
    dt_h:
        Integration step size (h).

    Returns
    -------
    IntranasalPKResult
    """
    # --- Input validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_systemic_L_per_h <= 0:
        raise ValueError("cl_systemic_L_per_h must be > 0")
    if vd_systemic_L <= 0:
        raise ValueError("vd_systemic_L must be > 0")
    if ka_systemic_per_h <= 0:
        raise ValueError("ka_systemic_per_h must be > 0")
    if ka_cns_per_h < 0:
        raise ValueError("ka_cns_per_h must be >= 0")
    if k_mcc_per_h <= 0:
        raise ValueError("k_mcc_per_h must be > 0")
    if v_cns_L <= 0:
        raise ValueError("v_cns_L must be > 0")
    if cl_cns_per_h < 0:
        raise ValueError("cl_cns_per_h must be >= 0")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be > 0")
    if dt_h <= 0:
        raise ValueError("dt_h must be > 0")

    n_steps = int(math.ceil(t_end_h / dt_h))
    times = np.linspace(0.0, t_end_h, n_steps + 1)

    # State variables
    m_nasal = np.zeros(n_steps + 1)  # mg in nasal cavity
    c_sys = np.zeros(n_steps + 1)  # mg/L systemic plasma
    c_cns = np.zeros(n_steps + 1)  # mg/L CNS compartment

    # Initial conditions
    m_nasal[0] = dose_mg

    # Rate constants
    k_el_sys = cl_systemic_L_per_h / vd_systemic_L  # systemic elimination (h^-1)
    k_el_cns = cl_cns_per_h / v_cns_L  # CNS elimination (h^-1)
    k_out_nasal = ka_systemic_per_h + ka_cns_per_h + k_mcc_per_h

    for i in range(n_steps):
        dt = times[i + 1] - times[i]

        dm_nasal = -k_out_nasal * m_nasal[i]
        dc_sys = (ka_systemic_per_h * m_nasal[i]) / vd_systemic_L - k_el_sys * c_sys[i]
        dc_cns = (ka_cns_per_h * m_nasal[i]) / v_cns_L - k_el_cns * c_cns[i]

        m_nasal[i + 1] = max(m_nasal[i] + dt * dm_nasal, 0.0)
        c_sys[i + 1] = max(c_sys[i] + dt * dc_sys, 0.0)
        c_cns[i + 1] = max(c_cns[i] + dt * dc_cns, 0.0)

    # --- Derived PK metrics ---
    auc_systemic = float(np_trapz(c_sys, times))
    auc_cns = float(np_trapz(c_cns, times))

    cmax_systemic = float(np.max(c_sys))
    tmax_idx = int(np.argmax(c_sys))
    tmax_systemic_h = float(times[tmax_idx])

    cmax_cns = float(np.max(c_cns))

    # Fraction absorbed: recovered from AUC × CL / dose
    # f_systemic = (AUC_sys * CL_sys) / dose
    f_absorbed_systemic = float(np.clip(auc_systemic * cl_systemic_L_per_h / dose_mg, 0.0, 1.0))

    # f_cns: drug delivered to CNS relative to dose
    # Analytically: ka_cns / k_out_nasal * (1 - exp(-k_out_nasal * t_end))
    # Use trapezoid integral over rate of input to CNS / dose
    total_cns_input_mg = float(np_trapz(ka_cns_per_h * m_nasal, times))
    f_absorbed_cns = float(np.clip(total_cns_input_mg / dose_mg, 0.0, 1.0))

    nose_to_brain_ratio = auc_cns / auc_systemic if auc_systemic > 0.0 else 0.0

    notes: list[str] = []
    if k_mcc_per_h > ka_systemic_per_h + ka_cns_per_h:
        notes.append("MCC rate exceeds absorption rates; significant drug lost via clearance")
    if ka_cns_per_h == 0.0:
        notes.append("Nose-to-brain pathway disabled (ka_cns_per_h=0)")

    return IntranasalPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times.tolist(),
        m_nasal_mg=m_nasal.tolist(),
        c_systemic_mg_L=c_sys.tolist(),
        c_cns_mg_L=c_cns.tolist(),
        cmax_systemic=cmax_systemic,
        tmax_systemic_h=tmax_systemic_h,
        auc_systemic=auc_systemic,
        cmax_cns=cmax_cns,
        auc_cns=auc_cns,
        f_absorbed_systemic=f_absorbed_systemic,
        f_absorbed_cns=f_absorbed_cns,
        nose_to_brain_ratio=nose_to_brain_ratio,
        notes=notes,
    )


def compare_delivery_routes(
    drug_name: str,
    dose_mg: float,
    cl_systemic_L_per_h: float,
    vd_systemic_L: float,
    ka_iv_equivalent: float | None = None,
    **kwargs,
) -> dict:
    """Compare intranasal vs IV (or oral) delivery for the same drug.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Dose in mg (same for both routes).
    cl_systemic_L_per_h:
        Systemic clearance (L/h).
    vd_systemic_L:
        Volume of distribution (L).
    ka_iv_equivalent:
        If None, compares against IV (instantaneous bolus).
        If a float, treats it as an oral ka and compares vs oral.
    **kwargs:
        Additional keyword arguments forwarded to ``simulate_intranasal_pk``.

    Returns
    -------
    dict with keys:
        - intranasal: IntranasalPKResult
        - reference_label: str ("iv" or "oral")
        - reference_auc: float
        - reference_cmax: float
        - bioavailability_ratio: float (intranasal AUC / reference AUC)
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_systemic_L_per_h <= 0:
        raise ValueError("cl_systemic_L_per_h must be > 0")
    if vd_systemic_L <= 0:
        raise ValueError("vd_systemic_L must be > 0")

    intranasal_result = simulate_intranasal_pk(
        drug_name=drug_name,
        dose_mg=dose_mg,
        cl_systemic_L_per_h=cl_systemic_L_per_h,
        vd_systemic_L=vd_systemic_L,
        **kwargs,
    )

    t_end_h = kwargs.get("t_end_h", 12.0)
    dt_h = kwargs.get("dt_h", 0.05)
    n_steps = int(math.ceil(t_end_h / dt_h))
    times = np.linspace(0.0, t_end_h, n_steps + 1)

    k_el = cl_systemic_L_per_h / vd_systemic_L

    if ka_iv_equivalent is None:
        # IV bolus: C(t) = (dose / Vd) * exp(-kel * t)
        c_ref = (dose_mg / vd_systemic_L) * np.exp(-k_el * times)
        reference_label = "iv"
    else:
        # Oral 1-compartment: C(t) = dose * ka / (Vd * (ka - kel)) * (exp(-kel*t) - exp(-ka*t))
        ka = float(ka_iv_equivalent)
        if abs(ka - k_el) < 1e-9:
            ka += 1e-6
        c_ref = (
            dose_mg
            * ka
            / (vd_systemic_L * (ka - k_el))
            * (np.exp(-k_el * times) - np.exp(-ka * times))
        )
        c_ref = np.maximum(c_ref, 0.0)
        reference_label = "oral"

    reference_auc = float(np_trapz(c_ref, times))
    reference_cmax = float(np.max(c_ref))

    bioavailability_ratio = (
        intranasal_result.auc_systemic / reference_auc if reference_auc > 0.0 else 0.0
    )

    return {
        "intranasal": intranasal_result,
        "reference_label": reference_label,
        "reference_auc": reference_auc,
        "reference_cmax": reference_cmax,
        "bioavailability_ratio": bioavailability_ratio,
    }


__all__ = [
    "IntranasalPKResult",
    "simulate_intranasal_pk",
    "compare_delivery_routes",
]
