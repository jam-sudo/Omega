"""Phase 226 — Wound PK Model.

3-compartment forward-Euler ODE model for drug distribution into wound
tissue following topical application.

Compartments:
  C_surface  (mg/mL)  — drug at wound surface / application site
  C_wound    (mg/g)   — drug in wound tissue / exudate
  C_systemic (mg/L)   — systemic circulation (absorbed from wound)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_V_SURFACE_ML: float = 2.0  # wound exudate volume (mL)
_V_WOUND_G: float = 5.0  # wound tissue equivalent (g)

_K_EVAP: float = 0.05  # /h   evaporation / washout from surface
_K_WOUND_ELIM: float = 0.02  # /h   local metabolism / drainage
_K_WS: float = 0.01  # /h   wound → systemic absorption

_PENETRATION_FACTORS = {
    "superficial": 0.5,
    "partial_thickness": 1.0,
    "full_thickness": 2.0,
}

_DEFAULT_MIC_MG_G: float = 1.0  # default MIC for mic_coverage_h calculation
_DT_H: float = 0.01  # forward Euler time step (h)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class WoundPKResult:
    """Result from wound PK simulation."""

    drug_name: str
    dose_mg: float
    wound_depth: str
    times_h: list[float]
    c_surface_mg_mL: list[float]
    c_wound_mg_g: list[float]
    c_systemic_mg_L: list[float]
    cmax_wound_mg_g: float
    tmax_wound_h: float
    auc_wound_mg_h_per_g: float
    cmax_systemic_mg_L: float
    mic_coverage_h: float
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _validate_inputs(
    dose_mg: float,
    wound_depth: str,
    cl_L_per_h: float,
) -> None:
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0; got {dose_mg}")
    if wound_depth not in _PENETRATION_FACTORS:
        raise ValueError(
            f"wound_depth must be one of {list(_PENETRATION_FACTORS.keys())}; got {wound_depth!r}"
        )
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0; got {cl_L_per_h}")


def _trapz_auc(times: list[float], conc: list[float]) -> float:
    """Manual trapezoidal AUC integration."""
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        auc += 0.5 * (conc[i] + conc[i - 1]) * dt
    return auc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def simulate_wound_pk(
    drug_name: str,
    dose_mg: float,
    wound_depth: str,
    formulation_viscosity_cp: float = 1.0,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    t_end_h: float = 24.0,
    dt_h: float = _DT_H,
    mic_mg_g: float = _DEFAULT_MIC_MG_G,
) -> WoundPKResult:
    """Simulate wound PK following topical drug application.

    Parameters
    ----------
    drug_name:
        Drug identifier.
    dose_mg:
        Topical dose applied in mg.
    wound_depth:
        One of "superficial", "partial_thickness", "full_thickness".
    formulation_viscosity_cp:
        Formulation viscosity in centipoise; higher → slower penetration.
        Penetration rate divided by sqrt(viscosity / 1.0).
    cl_L_per_h:
        Systemic clearance in L/h (must be > 0).
    vd_L:
        Volume of distribution in L for systemic compartment.
    t_end_h:
        Simulation end time in hours.
    dt_h:
        Forward Euler time step in hours.
    mic_mg_g:
        MIC threshold for mic_coverage_h calculation (mg/g).

    Returns
    -------
    WoundPKResult
    """
    _validate_inputs(dose_mg, wound_depth, cl_L_per_h)

    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0; got {vd_L}")
    if formulation_viscosity_cp <= 0:
        raise ValueError(f"formulation_viscosity_cp must be > 0; got {formulation_viscosity_cp}")

    # Penetration rate constant (h^-1)
    pen_factor = _PENETRATION_FACTORS[wound_depth]
    visc_factor = math.sqrt(max(formulation_viscosity_cp, 1e-9))
    k_pen = 0.1 * pen_factor / visc_factor

    # Systemic elimination rate constant
    ke = cl_L_per_h / vd_L  # /h

    # Initial conditions
    c_surf = dose_mg / _V_SURFACE_ML  # mg/mL
    c_wound = 0.0  # mg/g
    c_sys = 0.0  # mg/L

    times: list[float] = []
    c_surf_arr: list[float] = []
    c_wound_arr: list[float] = []
    c_sys_arr: list[float] = []

    t = 0.0
    n_steps = max(1, int(round(t_end_h / dt_h)))

    for _ in range(n_steps + 1):
        times.append(t)
        c_surf_arr.append(c_surf)
        c_wound_arr.append(c_wound)
        c_sys_arr.append(c_sys)

        if t >= t_end_h:
            break

        # ODEs
        dC_surf = -(k_pen + _K_EVAP) * c_surf
        dC_wound = (
            k_pen * c_surf * _V_SURFACE_ML / _V_WOUND_G - _K_WOUND_ELIM * c_wound - _K_WS * c_wound
        )
        dC_sys = _K_WS * c_wound * _V_WOUND_G / vd_L - ke * c_sys

        c_surf = max(0.0, c_surf + dC_surf * dt_h)
        c_wound = max(0.0, c_wound + dC_wound * dt_h)
        c_sys = max(0.0, c_sys + dC_sys * dt_h)

        t = min(t + dt_h, t_end_h)

    # Derived metrics
    cmax_wound = max(c_wound_arr)
    tmax_wound = times[c_wound_arr.index(cmax_wound)]
    auc_wound = _trapz_auc(times, c_wound_arr)
    cmax_systemic = max(c_sys_arr)

    # Hours wound concentration exceeds MIC
    mic_coverage = sum(dt_h for i in range(1, len(times)) if c_wound_arr[i] > mic_mg_g)
    mic_coverage = max(0.0, mic_coverage)

    notes = (
        f"wound_depth={wound_depth}, pen_factor={pen_factor}, "
        f"k_pen={k_pen:.4f}/h, ke={ke:.4f}/h, "
        f"viscosity={formulation_viscosity_cp:.1f} cP."
    )

    return WoundPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        wound_depth=wound_depth,
        times_h=times,
        c_surface_mg_mL=c_surf_arr,
        c_wound_mg_g=c_wound_arr,
        c_systemic_mg_L=c_sys_arr,
        cmax_wound_mg_g=cmax_wound,
        tmax_wound_h=tmax_wound,
        auc_wound_mg_h_per_g=auc_wound,
        cmax_systemic_mg_L=cmax_systemic,
        mic_coverage_h=mic_coverage,
        notes=notes,
    )


def compare_wound_depths(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    formulation_viscosity_cp: float = 1.0,
    t_end_h: float = 24.0,
    mic_mg_g: float = _DEFAULT_MIC_MG_G,
) -> list[WoundPKResult]:
    """Simulate wound PK for all three wound depths, sorted by wound AUC descending.

    Returns list of 3 WoundPKResult objects:
    [full_thickness, partial_thickness, superficial] in descending wound AUC order.
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0; got {dose_mg}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0; got {cl_L_per_h}")

    results = []
    for depth in _PENETRATION_FACTORS:
        result = simulate_wound_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            wound_depth=depth,
            formulation_viscosity_cp=formulation_viscosity_cp,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            t_end_h=t_end_h,
            mic_mg_g=mic_mg_g,
        )
        results.append(result)

    results.sort(key=lambda r: r.auc_wound_mg_h_per_g, reverse=True)
    return results
