"""
Phase 745 — Drug Precipitation in GI Tract

Models the dissolution-precipitation dynamics of poorly soluble drugs in the GI tract.
Supersaturated solutions or high-solubility salt forms can precipitate in the small
intestine (pH 6.8), reducing bioavailability.

3-state ODE model:
  A_dissolved (mg)    : drug in solution (available for absorption)
  A_precipitated (mg) : crystallized precipitate (not absorbed)
  C_plasma (mg/L)     : systemic drug concentration
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class GIPrecipitationResult:
    """Result of GI precipitation PK simulation."""

    drug_name: str
    dose_mg: float
    times_h: list
    c_plasma_mg_L: list
    a_dissolved_mg: list
    a_precipitated_mg: list
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    peak_precipitate_mg: float
    fraction_precipitated: float
    fraction_absorbed: float
    t_half_h: float
    notes: str


def _validate_params(
    dose_mg: float,
    c_sat_mg_L: float,
    v_gi_L: float,
    cl_L_per_h: float,
    vd_L: float,
) -> None:
    """Validate simulation parameters."""
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if c_sat_mg_L <= 0:
        raise ValueError(f"c_sat_mg_L must be > 0, got {c_sat_mg_L}")
    if v_gi_L <= 0:
        raise ValueError(f"v_gi_L must be > 0, got {v_gi_L}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")


def simulate_gi_precipitation_pk(
    drug_name: str,
    dose_mg: float,
    c_sat_mg_L: float = 5.0,
    k_abs_per_h: float = 1.5,
    k_precip_per_h: float = 2.0,
    k_rediss_per_h: float = 0.1,
    v_gi_L: float = 0.25,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    t_end_h: float = 12.0,
    dt_h: float = 0.02,
) -> GIPrecipitationResult:
    """
    Simulate GI precipitation PK using forward Euler integration.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Oral dose in mg.
    c_sat_mg_L : float
        Solubility at intestinal pH in mg/L (default 5.0 mg/L).
    k_abs_per_h : float
        Intestinal absorption rate constant (/h).
    k_precip_per_h : float
        Precipitation rate constant (/h).
    k_rediss_per_h : float
        Re-dissolution rate constant (/h).
    v_gi_L : float
        GI fluid volume in L.
    cl_L_per_h : float
        Systemic clearance in L/h.
    vd_L : float
        Volume of distribution in L.
    t_end_h : float
        Simulation end time in hours.
    dt_h : float
        Forward Euler time step in hours.

    Returns
    -------
    GIPrecipitationResult
    """
    _validate_params(dose_mg, c_sat_mg_L, v_gi_L, cl_L_per_h, vd_L)

    ke = cl_L_per_h / vd_L  # elimination rate constant (/h)
    saturation_mass_mg = c_sat_mg_L * v_gi_L  # max dissolved mass at saturation

    # Initial conditions
    a_dis = dose_mg
    a_prec = 0.0
    c_pl = 0.0

    n_steps = int(math.ceil(t_end_h / dt_h))
    times_h: list = []
    c_plasma_list: list = []
    a_dissolved_list: list = []
    a_precipitated_list: list = []

    t = 0.0
    for _ in range(n_steps + 1):
        times_h.append(t)
        c_plasma_list.append(c_pl)
        a_dissolved_list.append(a_dis)
        a_precipitated_list.append(a_prec)

        # Supersaturation driving force (only precipitate when above saturation)
        excess = max(0.0, a_dis - saturation_mass_mg)

        # ODEs
        d_a_dis = -k_abs_per_h * a_dis - k_precip_per_h * excess + k_rediss_per_h * a_prec
        d_a_prec = k_precip_per_h * excess - k_rediss_per_h * a_prec
        d_c_pl = k_abs_per_h * a_dis / vd_L - ke * c_pl

        # Forward Euler update
        a_dis = max(0.0, a_dis + d_a_dis * dt_h)
        a_prec = max(0.0, a_prec + d_a_prec * dt_h)
        c_pl = max(0.0, c_pl + d_c_pl * dt_h)

        t += dt_h

    # Compute summary metrics
    cmax_mg_L = max(c_plasma_list)
    tmax_idx = c_plasma_list.index(cmax_mg_L)
    tmax_h = times_h[tmax_idx]

    # Trapezoidal AUC
    auc = 0.0
    for i in range(1, len(times_h)):
        auc += 0.5 * (c_plasma_list[i - 1] + c_plasma_list[i]) * (times_h[i] - times_h[i - 1])

    peak_precipitate_mg = max(a_precipitated_list)
    fraction_precipitated = peak_precipitate_mg / dose_mg if dose_mg > 0 else 0.0
    # Clamp to [0, 1]
    fraction_precipitated = min(1.0, max(0.0, fraction_precipitated))

    # Estimated fraction absorbed = CL × AUC / dose (assuming 100% systemic bioavailability of absorbed drug)
    fraction_absorbed = cl_L_per_h * auc / dose_mg if dose_mg > 0 else 0.0
    fraction_absorbed = min(1.0, max(0.0, fraction_absorbed))

    t_half_h = math.log(2.0) / ke if ke > 0 else float("inf")

    # Build notes
    do_number = dose_mg / (c_sat_mg_L * v_gi_L) if (c_sat_mg_L * v_gi_L) > 0 else float("inf")
    if do_number < 1:
        risk = "low"
    elif do_number <= 5:
        risk = "moderate"
    else:
        risk = "high"

    notes = (
        f"Dose number (Do) = {do_number:.2f}; precipitation risk = {risk}; "
        f"fraction_precipitated = {fraction_precipitated:.3f}; "
        f"saturation_mass = {saturation_mass_mg:.2f} mg."
    )

    return GIPrecipitationResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_list,
        a_dissolved_mg=a_dissolved_list,
        a_precipitated_mg=a_precipitated_list,
        cmax_mg_L=cmax_mg_L,
        tmax_h=tmax_h,
        auc_mg_h_per_L=auc,
        peak_precipitate_mg=peak_precipitate_mg,
        fraction_precipitated=fraction_precipitated,
        fraction_absorbed=fraction_absorbed,
        t_half_h=t_half_h,
        notes=notes,
    )


def assess_precipitation_risk(
    dose_mg: float,
    c_sat_mg_L: float,
    v_gi_L: float = 0.25,
) -> dict:
    """
    Assess GI precipitation risk based on the Dose Number (Do).

    Do = dose / (C_sat × V_GI)

    Parameters
    ----------
    dose_mg : float
        Oral dose in mg.
    c_sat_mg_L : float
        Solubility at intestinal pH in mg/L.
    v_gi_L : float
        GI fluid volume in L (default 0.25 L).

    Returns
    -------
    dict with keys:
        dose_number (float) : Do
        risk_level (str)    : "low", "moderate", or "high"
        interpretation (str): human-readable description
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if c_sat_mg_L <= 0:
        raise ValueError(f"c_sat_mg_L must be > 0, got {c_sat_mg_L}")
    if v_gi_L <= 0:
        raise ValueError(f"v_gi_L must be > 0, got {v_gi_L}")

    dose_number = dose_mg / (c_sat_mg_L * v_gi_L)

    if dose_number < 1:
        risk_level = "low"
        interpretation = (
            f"Do = {dose_number:.2f} < 1: drug is below solubility limit; "
            "minimal precipitation expected."
        )
    elif dose_number <= 5:
        risk_level = "moderate"
        interpretation = (
            f"Do = {dose_number:.2f} (1–5): partial precipitation possible; "
            "formulation optimization recommended."
        )
    else:
        risk_level = "high"
        interpretation = (
            f"Do = {dose_number:.2f} > 5: significant precipitation expected; "
            "bioavailability likely reduced."
        )

    return {
        "dose_mg": dose_mg,
        "c_sat_mg_L": c_sat_mg_L,
        "v_gi_L": v_gi_L,
        "dose_number": dose_number,
        "risk_level": risk_level,
        "interpretation": interpretation,
    }
