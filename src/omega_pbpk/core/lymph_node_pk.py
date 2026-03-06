"""Lymph node drug accumulation — Phase 163.

Two-compartment model (plasma + lymph) for predicting drug accumulation
in lymph nodes after IV bolus administration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz


@dataclass(frozen=True)
class LymphNodePKResult:
    """Result of lymph node PK simulation."""

    drug_name: str
    dose_mg: float
    times_h: list[float]
    c_plasma_mg_L: list[float]
    c_lymph_mg_L: list[float]
    cmax_plasma: float
    cmax_lymph: float
    auc_plasma: float
    auc_lymph: float
    lymph_to_plasma_ratio: float
    lymph_enrichment_factor: float
    t_half_lymph_h: float
    notes: list[str]


def simulate_lymph_node_pk(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_systemic_L: float,
    k_lymph_in_per_h: float = 0.05,
    k_lymph_out_per_h: float = 0.02,
    lymph_volume_L: float = 2.0,
    f_lymph_uptake: float = 0.1,
    t_end_h: float = 48.0,
    dt_h: float = 0.05,
) -> LymphNodePKResult:
    """Simulate two-compartment plasma + lymph node PK after IV bolus.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        IV bolus dose (mg). Must be > 0.
    cl_L_per_h : float
        Systemic clearance (L/h). Must be > 0.
    vd_systemic_L : float
        Systemic volume of distribution (L). Must be > 0.
    k_lymph_in_per_h : float
        Transfer rate constant from plasma to lymph (1/h).
    k_lymph_out_per_h : float
        Transfer rate constant from lymph to plasma (1/h).
    lymph_volume_L : float
        Total lymph compartment volume (L). Must be > 0.
    f_lymph_uptake : float
        Fraction of dose entering lymphatic route directly (0-1).
    t_end_h : float
        Simulation end time (h).
    dt_h : float
        Time step (h).

    Returns
    -------
    LymphNodePKResult
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_systemic_L <= 0:
        raise ValueError("vd_systemic_L must be > 0")
    if f_lymph_uptake < 0 or f_lymph_uptake > 1:
        raise ValueError("f_lymph_uptake must be between 0 and 1")
    if lymph_volume_L <= 0:
        raise ValueError("lymph_volume_L must be > 0")

    ke = cl_L_per_h / vd_systemic_L

    # Initial conditions: IV bolus split
    a_plasma = dose_mg * (1.0 - f_lymph_uptake)
    a_lymph = dose_mg * f_lymph_uptake

    n = max(int(t_end_h / dt_h), 1)
    t = np.linspace(0.0, t_end_h, n + 1)
    c_plasma = np.zeros(n + 1)
    c_lymph = np.zeros(n + 1)

    c_plasma[0] = a_plasma / vd_systemic_L
    c_lymph[0] = a_lymph / lymph_volume_L

    for i in range(n):
        da_plasma = (
            -ke * a_plasma
            - k_lymph_in_per_h * a_plasma
            + k_lymph_out_per_h * a_lymph
        ) * dt_h
        da_lymph = (k_lymph_in_per_h * a_plasma - k_lymph_out_per_h * a_lymph) * dt_h

        a_plasma = max(a_plasma + da_plasma, 0.0)
        a_lymph = max(a_lymph + da_lymph, 0.0)

        c_plasma[i + 1] = a_plasma / vd_systemic_L
        c_lymph[i + 1] = a_lymph / lymph_volume_L

    # PK metrics
    cmax_plasma = float(np.max(c_plasma))
    cmax_lymph = float(np.max(c_lymph))
    auc_plasma = float(np_trapz(c_plasma, t))
    auc_lymph = float(np_trapz(c_lymph, t))

    lymph_to_plasma_ratio = auc_lymph / auc_plasma if auc_plasma > 0 else 0.0
    enrichment_factor = cmax_lymph / cmax_plasma if cmax_plasma > 0 else 0.0

    # Estimate t_half_lymph
    idx_cmax_lymph = int(np.argmax(c_lymph))
    half_cmax = cmax_lymph / 2.0
    t_half_lymph_h = math.log(2) / k_lymph_out_per_h  # fallback
    if idx_cmax_lymph < n:
        terminal = c_lymph[idx_cmax_lymph:]
        below = np.where(terminal <= half_cmax)[0]
        if len(below) > 0:
            idx_half = idx_cmax_lymph + below[0]
            t_half_lymph_h = float(t[idx_half] - t[idx_cmax_lymph])
            if t_half_lymph_h <= 0:
                t_half_lymph_h = math.log(2) / k_lymph_out_per_h

    # Notes
    notes: list[str] = []
    if lymph_to_plasma_ratio > 5:
        notes.append("High lymph node accumulation observed")
    if enrichment_factor > 10:
        notes.append("Significant lymph enrichment — favorable for lymph-targeted therapy")
    if t_half_lymph_h > 24:
        notes.append("Prolonged lymph node retention (>24h)")

    return LymphNodePKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=t.tolist(),
        c_plasma_mg_L=c_plasma.tolist(),
        c_lymph_mg_L=c_lymph.tolist(),
        cmax_plasma=cmax_plasma,
        cmax_lymph=cmax_lymph,
        auc_plasma=auc_plasma,
        auc_lymph=auc_lymph,
        lymph_to_plasma_ratio=lymph_to_plasma_ratio,
        lymph_enrichment_factor=enrichment_factor,
        t_half_lymph_h=t_half_lymph_h,
        notes=notes,
    )


def compare_formulation_lymph(
    drug_name: str,
    dose_mg: float,
    formulations: list[dict],
    cl_L_per_h: float = 1.0,
    vd_systemic_L: float = 10.0,
    **kwargs,
) -> list[LymphNodePKResult]:
    """Compare multiple formulations for lymph node drug accumulation.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        IV bolus dose (mg).
    formulations : list[dict]
        Each dict must have 'name' and may override simulation parameters.
    cl_L_per_h : float
        Default systemic clearance.
    vd_systemic_L : float
        Default systemic Vd.
    **kwargs
        Additional defaults passed to simulate_lymph_node_pk.

    Returns
    -------
    list[LymphNodePKResult]
    """
    if not formulations:
        raise ValueError("formulations must be a non-empty list")

    results: list[LymphNodePKResult] = []
    for form in formulations:
        sim_kwargs = dict(kwargs)
        for key in (
            "k_lymph_in_per_h",
            "k_lymph_out_per_h",
            "lymph_volume_L",
            "f_lymph_uptake",
            "t_end_h",
            "dt_h",
        ):
            if key in form:
                sim_kwargs[key] = form[key]

        result = simulate_lymph_node_pk(
            drug_name=form.get("name", drug_name),
            dose_mg=dose_mg,
            cl_L_per_h=cl_L_per_h,
            vd_systemic_L=vd_systemic_L,
            **sim_kwargs,
        )
        results.append(result)

    return results


__all__ = [
    "LymphNodePKResult",
    "simulate_lymph_node_pk",
    "compare_formulation_lymph",
]
