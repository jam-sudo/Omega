"""Buccal and sublingual drug delivery pharmacokinetics (Phase 298).

Models drug absorption through oral mucosal surfaces using a forward-Euler
ODE based on Papp (apparent permeability), surface area, and saliva dynamics.

Key features:
  - Sublingual: very fast absorption, small surface area (1.2 cm²)
  - Buccal: moderate absorption, large surface area (50 cm²)
  - Both routes bypass hepatic first-pass metabolism
  - Saliva dilutes drug and washes it away via saliva flow

ODE system:
  dc_saliva/dt = -(Papp * area / saliva_vol) * c_saliva
                 - (saliva_flow / saliva_vol) * c_saliva
  dC_plasma/dt = (Papp * area * c_saliva) / (Vd * 1000)  [unit conversion mL→L]
                 - (CL / Vd) * C_plasma

where c_saliva is in mg/mL and C_plasma is in mg/L.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz

# Surface areas (cm²) by route
_SURFACE_AREA_CM2 = {
    "sublingual": 1.2,
    "buccal": 50.0,
}

# Unit conversion: cm/s → cm/h
_CM_PER_S_TO_CM_PER_H = 3600.0

# mL to L conversion factor for absorption flux
_ML_TO_L = 1e-3


@dataclass(frozen=True)
class BuccalPKResult:
    """Result of a buccal/sublingual PK simulation."""

    drug_name: str
    dose_mg: float
    route: str
    times_h: list[float]
    c_plasma: list[float]
    c_saliva_profile: list[float]
    cmax: float
    tmax_h: float
    auc: float
    f_absorbed: float  # fraction absorbed vs swallowed/diluted away
    notes: str


def simulate_buccal_pk(
    drug_name: str,
    dose_mg: float,
    route: str = "sublingual",
    papp_cm_s: float = 5e-6,
    saliva_volume_mL: float = 1.0,
    saliva_flow_mL_per_h: float = 0.6,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    t_end_h: float = 4.0,
    dt_h: float = 0.01,
) -> BuccalPKResult:
    """Simulate buccal or sublingual drug delivery pharmacokinetics.

    Parameters
    ----------
    drug_name:
        Name or identifier of the drug.
    dose_mg:
        Total dose placed in the oral cavity (mg).
    route:
        "sublingual" (under tongue, area 1.2 cm²) or
        "buccal" (cheek, area 50 cm²).
    papp_cm_s:
        Apparent permeability across oral mucosa (cm/s). Typical range:
        1e-7 (poorly permeable) to 1e-4 (highly permeable).
    saliva_volume_mL:
        Volume of saliva in which drug is dissolved (mL). Default 1.0 mL.
    saliva_flow_mL_per_h:
        Saliva flow rate that washes drug away (mL/h). Default 0.6 mL/h.
        Set to 0 for no saliva washout.
    cl_L_per_h:
        Systemic clearance (L/h).
    vd_L:
        Volume of distribution (L).
    t_end_h:
        Simulation end time (h).
    dt_h:
        Forward Euler integration step size (h).

    Returns
    -------
    BuccalPKResult
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if route not in ("sublingual", "buccal"):
        raise ValueError(f"route must be 'sublingual' or 'buccal', got {route!r}")
    if papp_cm_s <= 0:
        raise ValueError(f"papp_cm_s must be > 0, got {papp_cm_s}")
    if saliva_volume_mL <= 0:
        raise ValueError(f"saliva_volume_mL must be > 0, got {saliva_volume_mL}")
    if saliva_flow_mL_per_h < 0:
        raise ValueError(f"saliva_flow_mL_per_h must be >= 0, got {saliva_flow_mL_per_h}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if t_end_h <= 0:
        raise ValueError(f"t_end_h must be > 0, got {t_end_h}")
    if dt_h <= 0:
        raise ValueError(f"dt_h must be > 0, got {dt_h}")

    # Route-specific surface area
    area_cm2 = _SURFACE_AREA_CM2[route]

    # Convert Papp from cm/s to cm/h
    papp_cm_h = papp_cm_s * _CM_PER_S_TO_CM_PER_H

    # Permeability-area product (cm³/h = mL/h)
    pa_mL_per_h = papp_cm_h * area_cm2

    # Rate constant for saliva-mediated drug loss (absorption + washout)
    k_abs_h = pa_mL_per_h / saliva_volume_mL  # 1/h, absorption from saliva
    k_wash_h = saliva_flow_mL_per_h / saliva_volume_mL  # 1/h, saliva washout
    k_total_saliva_h = k_abs_h + k_wash_h  # total saliva loss rate

    # Systemic elimination rate
    ke = cl_L_per_h / vd_L  # 1/h

    n_steps = max(int(math.ceil(t_end_h / dt_h)), 1)
    times = np.linspace(0.0, t_end_h, n_steps + 1)

    c_saliva = np.zeros(n_steps + 1)  # mg/mL
    c_plasma = np.zeros(n_steps + 1)  # mg/L

    # Initial saliva concentration
    c_saliva[0] = dose_mg / saliva_volume_mL

    for i in range(n_steps):
        # dc_saliva/dt = -(k_abs + k_wash) * c_saliva
        dc_sal = -k_total_saliva_h * c_saliva[i]

        # dC_plasma/dt = (Papp * area_cm2 * c_saliva [mg/mL] * 1000 mL/L) / Vd_L
        #              - ke * C_plasma
        # Absorption flux: pa_mL_per_h * c_saliva [mg/mL] = mg/h absorbed
        # Convert to mg/L/h in plasma: divide by Vd_L
        absorbed_rate_mg_per_h = pa_mL_per_h * c_saliva[i]  # mg/h
        dc_pl = absorbed_rate_mg_per_h / vd_L - ke * c_plasma[i]

        c_saliva[i + 1] = max(c_saliva[i] + dc_sal * dt_h, 0.0)
        c_plasma[i + 1] = max(c_plasma[i] + dc_pl * dt_h, 0.0)

    # --- PK metrics ---
    auc = float(np_trapz(c_plasma, times))
    cmax = float(np.max(c_plasma))
    tmax_idx = int(np.argmax(c_plasma))
    tmax_h = float(times[tmax_idx])

    # f_absorbed: fraction of drug removed from saliva via absorption (vs washout)
    # = k_abs / (k_abs + k_wash)
    if k_total_saliva_h > 0:
        f_absorbed = k_abs_h / k_total_saliva_h
    else:
        f_absorbed = 0.0

    # Build notes
    note_parts: list[str] = []
    note_parts.append(
        f"Route: {route}; surface area = {area_cm2:.1f} cm²; "
        f"Papp = {papp_cm_s:.2e} cm/s"
    )
    if f_absorbed < 0.5:
        note_parts.append(
            "Less than 50% of saliva drug absorbed (saliva washout dominant); "
            "consider higher Papp or slower saliva flow"
        )
    if route == "sublingual":
        note_parts.append(
            "Sublingual route: very fast onset expected due to rich vascularity"
        )
    else:
        note_parts.append(
            "Buccal route: larger area compensates for lower Papp vs sublingual"
        )
    notes = "; ".join(note_parts)

    return BuccalPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route=route,
        times_h=times.tolist(),
        c_plasma=c_plasma.tolist(),
        c_saliva_profile=c_saliva.tolist(),
        cmax=cmax,
        tmax_h=tmax_h,
        auc=auc,
        f_absorbed=f_absorbed,
        notes=notes,
    )


def compare_buccal_routes(
    drug_name: str,
    dose_mg: float,
    papp_cm_s: float,
    cl_L_per_h: float,
    vd_L: float,
    saliva_volume_mL: float = 1.0,
    saliva_flow_mL_per_h: float = 0.6,
    t_end_h: float = 4.0,
    dt_h: float = 0.01,
) -> dict:
    """Compare sublingual vs buccal route for the same drug and dose.

    Parameters
    ----------
    drug_name:
        Drug identifier.
    dose_mg:
        Dose in mg (same for both routes).
    papp_cm_s:
        Apparent permeability (cm/s); same value used for both routes.
    cl_L_per_h:
        Systemic clearance (L/h).
    vd_L:
        Volume of distribution (L).
    saliva_volume_mL:
        Saliva volume (mL).
    saliva_flow_mL_per_h:
        Saliva flow rate (mL/h).
    t_end_h:
        Simulation duration (h).
    dt_h:
        Integration step size (h).

    Returns
    -------
    dict with keys:
        sublingual : BuccalPKResult
        buccal : BuccalPKResult
        sublingual_cmax : float
        buccal_cmax : float
        sublingual_tmax_h : float
        buccal_tmax_h : float
        sublingual_auc : float
        buccal_auc : float
        sublingual_f_absorbed : float
        buccal_f_absorbed : float
        buccal_vs_sublingual_auc_ratio : float
    """
    sl_result = simulate_buccal_pk(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route="sublingual",
        papp_cm_s=papp_cm_s,
        saliva_volume_mL=saliva_volume_mL,
        saliva_flow_mL_per_h=saliva_flow_mL_per_h,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
        t_end_h=t_end_h,
        dt_h=dt_h,
    )

    bu_result = simulate_buccal_pk(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route="buccal",
        papp_cm_s=papp_cm_s,
        saliva_volume_mL=saliva_volume_mL,
        saliva_flow_mL_per_h=saliva_flow_mL_per_h,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
        t_end_h=t_end_h,
        dt_h=dt_h,
    )

    auc_ratio = bu_result.auc / sl_result.auc if sl_result.auc > 0 else float("inf")

    return {
        "sublingual": sl_result,
        "buccal": bu_result,
        "sublingual_cmax": sl_result.cmax,
        "buccal_cmax": bu_result.cmax,
        "sublingual_tmax_h": sl_result.tmax_h,
        "buccal_tmax_h": bu_result.tmax_h,
        "sublingual_auc": sl_result.auc,
        "buccal_auc": bu_result.auc,
        "sublingual_f_absorbed": sl_result.f_absorbed,
        "buccal_f_absorbed": bu_result.f_absorbed,
        "buccal_vs_sublingual_auc_ratio": auc_ratio,
    }


__all__ = [
    "BuccalPKResult",
    "simulate_buccal_pk",
    "compare_buccal_routes",
]
