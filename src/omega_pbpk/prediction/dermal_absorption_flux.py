"""Phase 334 — Dermal Absorption Flux Calculator.

Calculates transdermal permeability and absorption flux using the
Potts-Guy equation. Estimates lag time, steady-state flux, daily
absorption, and systemic concentration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DermalAbsorptionResult:
    """Result of transdermal permeability and absorption flux calculation."""

    drug_name: str
    logP: float
    mw_Da: float
    kp_cm_per_h: float
    lag_time_h: float
    steady_state_flux_mg_cm2_per_h: float
    daily_absorption_mg: float
    css_systemic_mg_L: float
    permeability_class: str
    absorption_risk: str
    notes: str


_LOGP_MIN = -5.0
_LOGP_MAX = 10.0


def _classify_permeability(kp: float) -> str:
    """Return permeability class based on Kp (cm/h)."""
    if kp < 1e-4:
        return "low"
    if kp < 1e-3:
        return "moderate"
    return "high"


def _classify_absorption_risk(daily_absorption_mg: float) -> str:
    """Return absorption risk class based on daily absorption in mg."""
    if daily_absorption_mg < 0.1:
        return "negligible"
    if daily_absorption_mg < 1.0:
        return "low"
    if daily_absorption_mg <= 10.0:
        return "moderate"
    return "high"


def calculate_dermal_absorption(
    drug_name: str,
    logP: float,
    mw_Da: float,
    c_donor_mg_mL: float,
    skin_area_cm2: float = 100.0,
    cl_sys_L_per_h: float = 1.0,
    membrane_thickness_cm: float = 0.05,
) -> DermalAbsorptionResult:
    """Calculate transdermal permeability and absorption flux.

    Uses the Potts-Guy equation for permeability prediction and a
    simplified diffusivity model for lag time estimation.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    logP:
        Octanol-water partition coefficient (log units). Valid range: [-5, 10].
    mw_Da:
        Molecular weight in Daltons.
    c_donor_mg_mL:
        Donor compartment drug concentration in mg/mL.
    skin_area_cm2:
        Skin surface area exposed in cm^2 (default 100 cm^2).
    cl_sys_L_per_h:
        Systemic clearance in L/h used to estimate steady-state concentration.
    membrane_thickness_cm:
        Effective skin membrane thickness in cm (default 0.05 cm).

    Returns
    -------
    DermalAbsorptionResult
        Frozen dataclass containing all permeability and absorption metrics.
    """
    # --- Input validation ---
    if mw_Da <= 0:
        raise ValueError(f"mw_Da must be > 0, got {mw_Da}")
    if c_donor_mg_mL <= 0:
        raise ValueError(f"c_donor_mg_mL must be > 0, got {c_donor_mg_mL}")
    if skin_area_cm2 <= 0:
        raise ValueError(f"skin_area_cm2 must be > 0, got {skin_area_cm2}")
    if cl_sys_L_per_h <= 0:
        raise ValueError(f"cl_sys_L_per_h must be > 0, got {cl_sys_L_per_h}")
    if membrane_thickness_cm <= 0:
        raise ValueError(f"membrane_thickness_cm must be > 0, got {membrane_thickness_cm}")
    if logP < _LOGP_MIN or logP > _LOGP_MAX:
        raise ValueError(f"logP must be in [{_LOGP_MIN}, {_LOGP_MAX}], got {logP}")

    # --- Potts-Guy permeability (cm/h) ---
    log_kp = 0.71 * logP - 0.0061 * mw_Da - 6.3
    kp = 10.0**log_kp  # cm/h

    # --- Diffusivity and lag time ---
    # D = 6.0e-3 * exp(-0.01 * MW) cm^2/h
    D = 6.0e-3 * math.exp(-0.01 * mw_Da)  # cm^2/h
    h = membrane_thickness_cm
    t_lag_h = (h**2) / (6.0 * D)  # h

    # --- Steady-state flux ---
    j_ss = kp * c_donor_mg_mL  # mg/cm^2/h

    # --- Daily absorption ---
    daily_absorption = j_ss * 24.0 * skin_area_cm2  # mg/day

    # --- Systemic steady-state concentration estimate ---
    # Css = daily_absorption / (CL * 1000)  [mg/L]
    # CL converted from L/h: daily_absorption / 24h gives mg/h absorbed
    # Css = (daily_absorption / 24) / (CL * 1000 / 24) => simplifies
    # Actually: rate in = J_ss * skin_area in mg/h; Css = rate_in / CL
    rate_in_mg_per_h = j_ss * skin_area_cm2  # mg/h
    css = rate_in_mg_per_h / (cl_sys_L_per_h * 1000.0)  # mg/L

    permeability_class = _classify_permeability(kp)
    absorption_risk = _classify_absorption_risk(daily_absorption)

    notes_parts = [
        f"log_Kp={log_kp:.4f}",
        f"Kp={kp:.2e} cm/h",
        f"D={D:.2e} cm^2/h",
        f"t_lag={t_lag_h:.3f} h",
        f"J_ss={j_ss:.4f} mg/cm^2/h",
        f"daily_absorption={daily_absorption:.4f} mg",
        f"Css={css:.4e} mg/L",
    ]
    notes = "; ".join(notes_parts)

    return DermalAbsorptionResult(
        drug_name=drug_name,
        logP=logP,
        mw_Da=mw_Da,
        kp_cm_per_h=kp,
        lag_time_h=t_lag_h,
        steady_state_flux_mg_cm2_per_h=j_ss,
        daily_absorption_mg=daily_absorption,
        css_systemic_mg_L=css,
        permeability_class=permeability_class,
        absorption_risk=absorption_risk,
        notes=notes,
    )
