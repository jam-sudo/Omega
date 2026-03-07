"""
Phase 739 — Amorphous Solid Dispersion PK (core/asd_pk.py)

Models supersaturation → crystallization → precipitation → absorption dynamics
using a 4-state forward-Euler ODE system.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class ASDPKResult:
    """Result of ASD PK simulation."""

    drug_name: str
    dose_mg: float
    times_h: list[float]
    c_plasma_mg_L: list[float]
    c_super_mg_L: list[float]
    cmax_plasma: float
    tmax_plasma_h: float
    auc_plasma: float
    cmax_super: float
    fraction_precipitated: float
    fraction_absorbed: float
    auc_enhancement_vs_sat: float
    notes: str


def _validate_inputs(
    dose_mg: float,
    c_sat_mg_L: float,
    supersaturation_ratio: float,
) -> None:
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if c_sat_mg_L <= 0:
        raise ValueError(f"c_sat_mg_L must be > 0, got {c_sat_mg_L}")
    if supersaturation_ratio < 1.0:
        raise ValueError(f"supersaturation_ratio must be >= 1.0, got {supersaturation_ratio}")


def _trapezoidal_auc(times: list[float], values: list[float]) -> float:
    """Compute AUC using trapezoidal rule."""
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        auc += 0.5 * (values[i] + values[i - 1]) * dt
    return auc


def simulate_asd_pk(
    drug_name: str = "Drug",
    dose_mg: float = 100.0,
    c_sat_mg_L: float = 1.0,
    supersaturation_ratio: float = 5.0,
    k_diss_per_h: float = 2.0,
    k_abs_per_h: float = 1.5,
    k_precip_per_h: float = 0.5,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    v_gi_L: float = 0.25,
    t_end_h: float = 12.0,
    dt_h: float = 0.02,
) -> ASDPKResult:
    """Simulate amorphous solid dispersion PK.

    States:
        A_am   : amorphous pool (mg)
        C_super: supersaturated GI solution (mg/L)
        A_cryst: crystalline precipitate (mg)
        C_plasma: systemic plasma (mg/L)
    """
    _validate_inputs(dose_mg, c_sat_mg_L, supersaturation_ratio)

    ke = cl_L_per_h / vd_L

    # Initial conditions
    c_super_init = min(dose_mg / v_gi_L, supersaturation_ratio * c_sat_mg_L)
    a_am = dose_mg - c_super_init * v_gi_L
    if a_am < 0.0:
        a_am = 0.0

    c_super = c_super_init
    a_cryst = max(0.0, dose_mg - c_super_init * v_gi_L - a_am)
    c_plasma = 0.0

    times: list[float] = []
    c_plasma_list: list[float] = []
    c_super_list: list[float] = []

    n_steps = int(round(t_end_h / dt_h))
    t = 0.0

    for _ in range(n_steps + 1):
        times.append(t)
        c_plasma_list.append(c_plasma)
        c_super_list.append(c_super)

        if _ == n_steps:
            break

        # ODEs
        excess = max(0.0, c_super - c_sat_mg_L)

        da_am_dt = -k_diss_per_h * a_am
        dc_super_dt = (
            k_diss_per_h * a_am / v_gi_L
            - k_abs_per_h * c_super
            - k_precip_per_h * excess * c_super / c_sat_mg_L
        )
        da_cryst_dt = k_precip_per_h * excess * v_gi_L
        dc_plasma_dt = k_abs_per_h * c_super * v_gi_L / vd_L - ke * c_plasma

        # Forward Euler
        a_am = max(0.0, a_am + da_am_dt * dt_h)
        c_super = max(0.0, c_super + dc_super_dt * dt_h)
        a_cryst = max(0.0, a_cryst + da_cryst_dt * dt_h)
        c_plasma = max(0.0, c_plasma + dc_plasma_dt * dt_h)

        t += dt_h

    # Metrics
    cmax_plasma = max(c_plasma_list)
    tmax_idx = c_plasma_list.index(cmax_plasma)
    tmax_plasma_h = times[tmax_idx]
    auc_plasma = _trapezoidal_auc(times, c_plasma_list)
    cmax_super = max(c_super_list)
    fraction_precipitated = min(1.0, a_cryst / dose_mg)

    # fraction_absorbed estimated from AUC × CL / dose
    fraction_absorbed = min(1.0, auc_plasma * cl_L_per_h / dose_mg)

    # AUC enhancement vs crystalline reference
    # Reference: only C_sat dissolves, simple 1-cpt
    reference_auc = (
        c_sat_mg_L * v_gi_L / vd_L * (1.0 / ke) * (1.0 - math.exp(-ke * t_end_h))
        if ke > 0.0
        else float("inf")
    )
    auc_enhancement_vs_sat = auc_plasma / reference_auc if reference_auc > 0.0 else float("inf")

    notes = (
        f"ASD simulation: dose={dose_mg}mg, SR={supersaturation_ratio}x, "
        f"k_precip={k_precip_per_h}/h. "
        f"Precipitation fraction={fraction_precipitated:.2f}, "
        f"AUC enhancement={auc_enhancement_vs_sat:.2f}x vs crystalline."
    )

    return ASDPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        c_super_mg_L=c_super_list,
        cmax_plasma=cmax_plasma,
        tmax_plasma_h=tmax_plasma_h,
        auc_plasma=auc_plasma,
        cmax_super=cmax_super,
        fraction_precipitated=fraction_precipitated,
        fraction_absorbed=fraction_absorbed,
        auc_enhancement_vs_sat=auc_enhancement_vs_sat,
        notes=notes,
    )


def compare_asd_vs_crystalline(
    drug_name: str = "Drug",
    dose_mg: float = 100.0,
    c_sat_mg_L: float = 1.0,
    k_precip_asd: float = 0.5,
    k_precip_cryst: float = 5.0,
    supersaturation_ratio: float = 5.0,
    k_diss_per_h: float = 2.0,
    k_abs_per_h: float = 1.5,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    v_gi_L: float = 0.25,
    t_end_h: float = 12.0,
    dt_h: float = 0.02,
) -> dict:
    """Compare ASD formulation vs crystalline (high k_precip) formulation.

    Returns a dict with results for both and key comparative metrics.
    """
    asd_result = simulate_asd_pk(
        drug_name=drug_name,
        dose_mg=dose_mg,
        c_sat_mg_L=c_sat_mg_L,
        supersaturation_ratio=supersaturation_ratio,
        k_diss_per_h=k_diss_per_h,
        k_abs_per_h=k_abs_per_h,
        k_precip_per_h=k_precip_asd,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
        v_gi_L=v_gi_L,
        t_end_h=t_end_h,
        dt_h=dt_h,
    )
    cryst_result = simulate_asd_pk(
        drug_name=drug_name,
        dose_mg=dose_mg,
        c_sat_mg_L=c_sat_mg_L,
        supersaturation_ratio=supersaturation_ratio,
        k_diss_per_h=k_diss_per_h,
        k_abs_per_h=k_abs_per_h,
        k_precip_per_h=k_precip_cryst,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
        v_gi_L=v_gi_L,
        t_end_h=t_end_h,
        dt_h=dt_h,
    )

    auc_ratio = (
        asd_result.auc_plasma / cryst_result.auc_plasma
        if cryst_result.auc_plasma > 0.0
        else float("inf")
    )
    cmax_ratio = (
        asd_result.cmax_plasma / cryst_result.cmax_plasma
        if cryst_result.cmax_plasma > 0.0
        else float("inf")
    )

    return {
        "asd": asd_result,
        "crystalline": cryst_result,
        "auc_ratio_asd_vs_cryst": auc_ratio,
        "cmax_ratio_asd_vs_cryst": cmax_ratio,
        "asd_fraction_precipitated": asd_result.fraction_precipitated,
        "cryst_fraction_precipitated": cryst_result.fraction_precipitated,
        "summary": (f"ASD vs crystalline: AUC ratio={auc_ratio:.2f}, Cmax ratio={cmax_ratio:.2f}"),
    }
