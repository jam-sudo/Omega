"""Mechanistic renal PK: glomerular filtration + active secretion + passive reabsorption."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz


@dataclass(frozen=True)
class KidneyPKResult:
    cl_renal_L_per_h: float
    cl_filtration_L_per_h: float
    cl_secretion_mean_L_per_h: float
    reabsorption_fraction: float
    fe_urine: float  # math.nan if dose_mg <= 0
    urine_rate_mg_h: list[float]
    urine_cumulative_mg: float
    urine_conc_mg_L: list[float]
    gfr_mL_per_min: float
    secretion_vmax_mg_h: float
    secretion_km_mg_L: float


def predict_renal_pk(
    plasma_times: list[float],
    plasma_conc: list[float],
    fup: float,
    gfr_mL_per_min: float = 120.0,
    logP: float = 0.0,
    secretion_vmax_mg_h: float = 0.0,
    secretion_km_mg_L: float = 1.0,
    urine_flow_mL_h: float = 60.0,
    dose_mg: float = 0.0,
) -> KidneyPKResult:
    """Predict renal PK from plasma concentration-time profile.

    Parameters
    ----------
    plasma_times : list[float]
        Time points (h).
    plasma_conc : list[float]
        Plasma concentrations at each time point (mg/L).
    fup : float
        Fraction unbound in plasma (0, 1].
    gfr_mL_per_min : float
        Glomerular filtration rate (mL/min), default 120.
    logP : float
        Lipophilicity for reabsorption estimate.
    secretion_vmax_mg_h : float
        Vmax for active tubular secretion (mg/h).
    secretion_km_mg_L : float
        Km for active tubular secretion (mg/L).
    urine_flow_mL_h : float
        Urine flow rate (mL/h), must be > 0.
    dose_mg : float
        Administered dose (mg); used only for fe calculation.

    Returns
    -------
    KidneyPKResult
    """
    # --- Input validation ---
    if urine_flow_mL_h <= 0:
        raise ValueError("urine_flow_mL_h must be > 0")
    if secretion_vmax_mg_h < 0:
        raise ValueError("secretion_vmax_mg_h must be >= 0")

    fup = float(np.clip(fup, 1e-6, 1.0))

    t = np.asarray(plasma_times, dtype=np.float64)
    cp = np.clip(np.asarray(plasma_conc, dtype=np.float64), 0.0, None)

    # --- Filtration clearance ---
    cl_filtration = gfr_mL_per_min * 0.06 * fup  # L/h

    # --- Active secretion clearance (time-varying) ---
    if secretion_vmax_mg_h > 0:
        cls_t = secretion_vmax_mg_h / (secretion_km_mg_L + cp)
    else:
        cls_t = np.zeros_like(cp)
    cls_mean = float(np.mean(cls_t))

    # --- Reabsorption fraction (logP-based sigmoid, clamped 0–0.95) ---
    fr = 1.0 / (1.0 + math.exp(-(logP - 1.0) * 2.0))
    fr = max(0.0, min(fr, 0.95))

    # --- Urine excretion rate (mg/h) ---
    dau_dt = (cl_filtration + cls_t) * cp * (1.0 - fr)

    # --- Cumulative urine amount (mg) ---
    urine_cumulative = float(np_trapz(dau_dt, t))

    # --- Urine concentration (mg/L) ---
    urine_flow_L_h = urine_flow_mL_h / 1000.0
    cu = dau_dt / urine_flow_L_h

    # --- Effective renal clearance ---
    cl_renal = (cl_filtration + cls_mean) * (1.0 - fr)

    # --- Fraction excreted in urine ---
    fe = urine_cumulative / dose_mg if dose_mg > 0 else math.nan

    return KidneyPKResult(
        cl_renal_L_per_h=cl_renal,
        cl_filtration_L_per_h=cl_filtration,
        cl_secretion_mean_L_per_h=cls_mean,
        reabsorption_fraction=fr,
        fe_urine=fe,
        urine_rate_mg_h=dau_dt.tolist(),
        urine_cumulative_mg=urine_cumulative,
        urine_conc_mg_L=cu.tolist(),
        gfr_mL_per_min=gfr_mL_per_min,
        secretion_vmax_mg_h=secretion_vmax_mg_h,
        secretion_km_mg_L=secretion_km_mg_L,
    )


__all__ = ["KidneyPKResult", "predict_renal_pk"]
