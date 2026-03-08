"""Phase 464 — Solid Lipid Nanoparticle PK (SLN PK).

3-compartment PBPK model for solid lipid nanoparticles:
  Plasma (SLN intact) -> MPS organs (liver/spleen) -> Free drug (systemic)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

_VALID_COATINGS = {"PEG", "bare"}


@dataclass
class SLNPKResult:
    """Result of SLN PK simulation."""

    drug_name: str
    dose_mg: float
    particle_size_nm: float
    surface_coating: str
    logP: float
    times_h: list = field(default_factory=list)
    c_sln_plasma_mg_L: list = field(default_factory=list)
    c_free_drug_mg_L: list = field(default_factory=list)
    cmax_free_drug_mg_L: float = 0.0
    tmax_free_drug_h: float = 0.0
    auc_free_drug_mg_h_per_L: float = 0.0
    auc_sln_mg_h_per_L: float = 0.0
    mps_uptake_fraction: float = 0.0
    circulation_time_h: float = 0.0
    notes: str = ""


def _calc_k_release(logP: float) -> float:
    """Calculate drug release rate from SLN.

    Higher logP -> slower release (drug retained in lipid matrix).
    """
    raw = 0.1 - 0.01 * logP
    return max(0.01, min(0.5, raw))


def _calc_k_mps_effective(surface_coating: str, particle_size_nm: float) -> float:
    """Calculate effective MPS clearance rate."""
    if surface_coating == "bare":
        k_mps_base = 2.0
    else:  # PEG
        k_mps_base = 0.3

    # Size effect: larger particles cleared faster
    size_factor = 1.0 + (particle_size_nm - 100.0) / 500.0
    return k_mps_base * max(0.1, size_factor)


def _trapezoidal_auc(times: list, concs: list) -> float:
    """Manual trapezoidal AUC calculation."""
    return sum(
        0.5 * (concs[i] + concs[i - 1]) * (times[i] - times[i - 1]) for i in range(1, len(times))
    )


def simulate_sln_pk(
    drug_name: str,
    dose_mg: float,
    particle_size_nm: float = 200.0,
    surface_coating: str = "PEG",
    logP: float = 3.0,
    cl_sys_L_per_h: float = 5.0,
    vd_sys_L: float = 50.0,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
) -> SLNPKResult:
    """Simulate SLN pharmacokinetics (3-compartment model).

    Compartments:
      1. Plasma SLN (intact nanoparticles in circulation)
      2. MPS compartment (liver/spleen uptake - sink)
      3. Free drug in systemic circulation (released from SLN)

    Args:
        drug_name: Name of the drug/formulation.
        dose_mg: IV dose of SLN in mg.
        particle_size_nm: Particle size in nanometers.
        surface_coating: Surface coating type ("PEG" or "bare").
        logP: Octanol-water partition coefficient.
        cl_sys_L_per_h: Systemic clearance for free drug in L/h.
        vd_sys_L: Volume of distribution for free drug in L.
        t_end_h: Simulation end time in hours.
        dt_h: Time step in hours.

    Returns:
        SLNPKResult with SLN and free drug concentration-time profiles.
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if particle_size_nm <= 0:
        raise ValueError("particle_size_nm must be > 0")
    if surface_coating not in _VALID_COATINGS:
        raise ValueError(f"surface_coating must be one of {_VALID_COATINGS}")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")

    # Kinetic parameters
    k_mps = _calc_k_mps_effective(surface_coating, particle_size_nm)
    k_release = _calc_k_release(logP)
    k_el_free = cl_sys_L_per_h / vd_sys_L  # elimination of free drug

    # Volume for SLN in plasma (assume similar volume as free drug)
    vd_sln_L = vd_sys_L

    # Initial conditions: IV bolus SLN
    # SLN concentration in plasma
    c_sln = dose_mg / vd_sln_L  # mg/L
    c_free = 0.0  # mg/L (no free drug initially)

    times: list = []
    c_sln_list: list = []
    c_free_list: list = []

    n_steps = int(round(t_end_h / dt_h)) + 1

    for step in range(n_steps):
        t = step * dt_h
        times.append(t)
        c_sln_list.append(c_sln)
        c_free_list.append(c_free)

        if step < n_steps - 1:
            # dC_sln/dt = -(k_mps + k_release) * C_sln
            dc_sln = -(k_mps + k_release) * c_sln

            # Release from SLN contributes to free drug
            # dC_free/dt = k_release * C_sln * Vd_sln / Vd_sys - k_el_free * C_free
            dc_free = k_release * c_sln * vd_sln_L / vd_sys_L - k_el_free * c_free

            c_sln = max(0.0, c_sln + dc_sln * dt_h)
            c_free = max(0.0, c_free + dc_free * dt_h)

    # Derived PK metrics
    cmax_free = max(c_free_list)
    tmax_free = 0.0
    if cmax_free > 0:
        tmax_free = times[c_free_list.index(cmax_free)]

    auc_free = _trapezoidal_auc(times, c_free_list)
    auc_sln = _trapezoidal_auc(times, c_sln_list)

    # MPS uptake fraction: fraction of SLN cleared by MPS vs released
    # MPS fraction ≈ k_mps / (k_mps + k_release)
    total_rate = k_mps + k_release
    mps_fraction = k_mps / total_rate if total_rate > 0 else 0.0

    # Circulation time: t½ of SLN in plasma
    # SLN decays as exp(-(k_mps + k_release)*t), so t½ = ln(2)/(k_mps + k_release)
    circulation_time = math.log(2) / total_rate if total_rate > 0 else float("inf")

    notes = (
        f"k_mps={k_mps:.3f}/h, k_release={k_release:.3f}/h, "
        f"k_el_free={k_el_free:.3f}/h, "
        f"MPS_fraction={mps_fraction:.2%}"
    )

    return SLNPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        particle_size_nm=particle_size_nm,
        surface_coating=surface_coating,
        logP=logP,
        times_h=times,
        c_sln_plasma_mg_L=c_sln_list,
        c_free_drug_mg_L=c_free_list,
        cmax_free_drug_mg_L=cmax_free,
        tmax_free_drug_h=tmax_free,
        auc_free_drug_mg_h_per_L=auc_free,
        auc_sln_mg_h_per_L=auc_sln,
        mps_uptake_fraction=mps_fraction,
        circulation_time_h=circulation_time,
        notes=notes,
    )


def compare_surface_coatings(
    drug_name: str,
    dose_mg: float,
    particle_size_nm: float = 200.0,
    logP: float = 3.0,
    cl_sys_L_per_h: float = 5.0,
    vd_sys_L: float = 50.0,
) -> list:
    """Compare bare vs PEG-coated SLN formulations.

    Returns list [bare_result, PEG_result] sorted by circulation_time_h ascending.
    """
    bare_result = simulate_sln_pk(
        drug_name=drug_name,
        dose_mg=dose_mg,
        particle_size_nm=particle_size_nm,
        surface_coating="bare",
        logP=logP,
        cl_sys_L_per_h=cl_sys_L_per_h,
        vd_sys_L=vd_sys_L,
    )
    peg_result = simulate_sln_pk(
        drug_name=drug_name,
        dose_mg=dose_mg,
        particle_size_nm=particle_size_nm,
        surface_coating="PEG",
        logP=logP,
        cl_sys_L_per_h=cl_sys_L_per_h,
        vd_sys_L=vd_sys_L,
    )
    results = [bare_result, peg_result]
    results.sort(key=lambda r: r.circulation_time_h)
    return results
