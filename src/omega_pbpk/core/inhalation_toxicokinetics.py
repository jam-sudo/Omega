"""Inhalation Toxicokinetics — Phase 1025.

Models systemic absorption of inhaled toxic agents (solvents, gases, fine particles)
using a 2-compartment (blood + tissue) ODE with Fickian lung uptake.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def _trapz(times: list, values: list) -> float:
    """Manual trapezoidal integration."""
    if len(times) < 2:
        return 0.0
    area = 0.0
    for i in range(len(times) - 1):
        dt = times[i + 1] - times[i]
        area += 0.5 * (values[i] + values[i + 1]) * dt
    return area


@dataclass
class InhalationToxResult:
    compound_name: str
    exposure_ppm: float
    exposure_duration_h: float
    times_h: list
    c_alveolar_mg_L: list  # alveolar air concentration (mg/L air)
    c_blood_mg_L: list  # blood concentration
    c_tissue_mg_L: list  # deep tissue (fat/liver) concentration
    auc_blood: float  # mg/L · h
    auc_tissue: float
    cmax_blood: float
    tmax_blood_h: float
    clearance_t50_h: float  # time from exposure end to 50% of peak blood conc
    notes: str


def simulate_inhalation_toxicokinetics(
    compound_name: str,
    exposure_ppm: float,
    mw_Da: float,
    exposure_duration_h: float = 8.0,
    blood_air_partition: float = 10.0,
    tissue_blood_partition: float = 3.0,
    alveolar_ventilation_L_h: float = 300.0,
    cardiac_output_L_h: float = 300.0,
    vd_blood_L: float = 5.0,
    vd_tissue_L: float = 40.0,
    cl_metabolic_L_per_h: float = 2.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> InhalationToxResult:
    """Simulate inhalation toxicokinetics using a 2-compartment ODE model.

    Parameters
    ----------
    compound_name : str
        Name of the inhaled compound.
    exposure_ppm : float
        Air concentration in parts per million (ppm).
    mw_Da : float
        Molecular weight in Da.
    exposure_duration_h : float
        Duration of inhalation exposure (hours).
    blood_air_partition : float
        Blood:air partition coefficient.
    tissue_blood_partition : float
        Tissue:blood partition coefficient.
    alveolar_ventilation_L_h : float
        Alveolar ventilation rate (L/h).
    cardiac_output_L_h : float
        Cardiac output (L/h).
    vd_blood_L : float
        Volume of distribution — blood compartment (L).
    vd_tissue_L : float
        Volume of distribution — tissue compartment (L).
    cl_metabolic_L_per_h : float
        Metabolic clearance (L/h).
    t_end_h : float
        Total simulation time (hours).
    dt_h : float
        Forward Euler time step (hours).

    Returns
    -------
    InhalationToxResult
    """
    # Validation
    if exposure_ppm <= 0:
        raise ValueError("exposure_ppm must be > 0")
    if mw_Da <= 0:
        raise ValueError("mw_Da must be > 0")
    if exposure_duration_h <= 0:
        raise ValueError("exposure_duration_h must be > 0")
    if alveolar_ventilation_L_h <= 0:
        raise ValueError("alveolar_ventilation_L_h must be > 0")
    if t_end_h <= exposure_duration_h:
        raise ValueError("t_end_h must be > exposure_duration_h")

    # Convert ppm to mg/L air (ideal gas at 25°C: molar volume = 24.45 L/mol)
    c_air_mg_L = exposure_ppm * mw_Da / 24450.0

    # Rate constants
    k12 = cardiac_output_L_h / vd_blood_L
    k21 = cardiac_output_L_h / (vd_tissue_L * tissue_blood_partition)
    ke_blood = cl_metabolic_L_per_h / vd_blood_L

    # Forward Euler simulation
    n_steps = int(math.ceil(t_end_h / dt_h)) + 1
    times_h: list = []
    c_blood_list: list = []
    c_tissue_list: list = []
    c_alveolar_list: list = []

    c_blood = 0.0
    c_tissue = 0.0

    for step in range(n_steps):
        t = step * dt_h
        if t > t_end_h:
            break

        times_h.append(t)
        c_blood_list.append(c_blood)
        c_tissue_list.append(c_tissue)

        # Alveolar concentration: present only during exposure
        if t <= exposure_duration_h:
            c_alveolar_list.append(c_air_mg_L)
        else:
            c_alveolar_list.append(0.0)

        # Absorption rate into blood (Fickian uptake)
        if t <= exposure_duration_h:
            absorption_rate = alveolar_ventilation_L_h * (
                c_air_mg_L - c_blood / blood_air_partition
            )
        else:
            absorption_rate = 0.0

        # ODE derivatives
        dc_blood = (
            absorption_rate / vd_blood_L - k12 * c_blood + k21 * c_tissue - ke_blood * c_blood
        )
        dc_tissue = k12 * c_blood * (vd_blood_L / vd_tissue_L) - k21 * c_tissue

        # Forward Euler update
        c_blood = max(0.0, c_blood + dc_blood * dt_h)
        c_tissue = max(0.0, c_tissue + dc_tissue * dt_h)

    # Derived metrics
    auc_blood = _trapz(times_h, c_blood_list)
    auc_tissue = _trapz(times_h, c_tissue_list)

    cmax_blood = max(c_blood_list)
    tmax_blood_h = times_h[c_blood_list.index(cmax_blood)]

    # clearance_t50_h: time from exposure_end until c_blood drops to 0.5 * cmax_blood
    half_peak = 0.5 * cmax_blood
    clearance_t50_h = t_end_h - exposure_duration_h  # default: full post-exposure window

    # Only search post-exposure portion for decline through half_peak
    in_decline = False
    for idx in range(len(times_h)):
        t_i = times_h[idx]
        if t_i <= tmax_blood_h:
            continue
        if not in_decline and t_i > exposure_duration_h:
            in_decline = True
        if in_decline and c_blood_list[idx] <= half_peak:
            clearance_t50_h = t_i - exposure_duration_h
            break

    notes = (
        f"Inhalation TK: {compound_name}, {exposure_ppm} ppm, "
        f"MW={mw_Da} Da, exposure {exposure_duration_h} h. "
        f"c_air={c_air_mg_L:.4f} mg/L. "
        f"Blood:air partition={blood_air_partition}, "
        f"tissue:blood partition={tissue_blood_partition}."
    )

    return InhalationToxResult(
        compound_name=compound_name,
        exposure_ppm=exposure_ppm,
        exposure_duration_h=exposure_duration_h,
        times_h=times_h,
        c_alveolar_mg_L=c_alveolar_list,
        c_blood_mg_L=c_blood_list,
        c_tissue_mg_L=c_tissue_list,
        auc_blood=auc_blood,
        auc_tissue=auc_tissue,
        cmax_blood=cmax_blood,
        tmax_blood_h=tmax_blood_h,
        clearance_t50_h=clearance_t50_h,
        notes=notes,
    )
