"""Transdermal iontophoresis PK model — Phase 925.

Electrically-assisted transdermal drug delivery via iontophoresis,
extending passive transdermal absorption with a 3-compartment ODE model
(skin/SC+VE, dermis/blood vessels, systemic plasma).

Two modes:
  - passive     : no electric current applied
  - iontophoresis : current enhances skin permeation via electroosmosis/electrophoresis

Enhancement factor:
  enhancement_factor = 1 + current_mA * charge_factor
  where charge_factor ~ 5 (cationic), 2 (anionic), 0.5 (neutral)

ODEs (Forward Euler):
  dA_skin/dt   = -k_skin * enhancement * A_skin
  dA_dermis/dt =  k_skin * enhancement * A_skin - k_d2p * A_dermis
  dC_plasma/dt =  k_d2p * A_dermis / Vd - k_elim * C_plasma
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "TransdermalIontophoresisPKResult",
    "simulate_iontophoresis",
    "compare_modes",
]

# Charge factors for enhancement factor calculation
_CHARGE_FACTORS: dict[str, float] = {
    "cationic": 5.0,
    "anionic": 2.0,
    "neutral": 0.5,
}

_VALID_ROUTES = {"iontophoresis", "passive"}
_VALID_CHARGE_TYPES = {"cationic", "anionic", "neutral"}


def _trapz(y: list[float], x: list[float]) -> float:
    """Manual trapezoidal integration."""
    if len(x) < 2:
        return 0.0
    total = 0.0
    for i in range(len(x) - 1):
        total += 0.5 * (y[i] + y[i + 1]) * (x[i + 1] - x[i])
    return total


@dataclass
class TransdermalIontophoresisPKResult:
    """Result of transdermal iontophoresis PK simulation."""

    drug_name: str
    route: str
    current_mA: float
    charge_type: str
    dose_mg: float
    times_h: list[float]
    c_plasma_mg_L: list[float]
    a_skin_mg: list[float]
    a_dermis_mg: list[float]
    cmax_plasma_mg_L: float
    tmax_plasma_h: float
    auc_plasma_mg_h_per_L: float
    enhancement_factor: float
    bioavailability_pct: float
    flux_peak_mg_h: float
    notes: str


def simulate_iontophoresis(
    drug_name: str,
    dose_mg: float,
    route: str = "iontophoresis",
    current_mA: float = 0.5,
    charge_type: str = "cationic",
    k_skin_per_h: float = 0.02,
    k_d2p_per_h: float = 0.5,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> TransdermalIontophoresisPKResult:
    """Simulate transdermal drug delivery with optional iontophoresis.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Dose applied to the skin surface (mg). Must be > 0.
    route : str
        Delivery route: 'iontophoresis' or 'passive'.
    current_mA : float
        Applied current (mA). Must be >= 0.
    charge_type : str
        Drug charge: 'cationic', 'anionic', or 'neutral'.
    k_skin_per_h : float
        Passive skin permeation rate constant (1/h).
    k_d2p_per_h : float
        Dermis-to-plasma transfer rate constant (1/h).
    cl_L_per_h : float
        Systemic clearance (L/h). Must be > 0.
    vd_L : float
        Volume of distribution (L). Must be > 0.
    t_end_h : float
        Simulation duration (h).
    dt_h : float
        Forward Euler time step (h).

    Returns
    -------
    TransdermalIontophoresisPKResult
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if current_mA < 0:
        raise ValueError("current_mA must be >= 0")
    if route not in _VALID_ROUTES:
        raise ValueError(f"route must be one of {_VALID_ROUTES}")
    if charge_type not in _VALID_CHARGE_TYPES:
        raise ValueError(f"charge_type must be one of {_VALID_CHARGE_TYPES}")

    # Compute enhancement factor
    if route == "passive" or current_mA == 0.0:
        enhancement_factor = 1.0
    else:
        charge_factor = _CHARGE_FACTORS[charge_type]
        enhancement_factor = 1.0 + current_mA * charge_factor

    k_elim = cl_L_per_h / vd_L

    # Initial conditions (amounts in mg)
    a_skin = dose_mg
    a_dermis = 0.0
    c_plasma = 0.0  # concentration in mg/L
    a_eliminated = 0.0

    n_steps = int(math.ceil(t_end_h / dt_h))
    times: list[float] = []
    a_skin_list: list[float] = []
    a_dermis_list: list[float] = []
    c_plasma_list: list[float] = []

    # Track flux for peak flux calculation
    flux_list: list[float] = []  # rate of drug entering dermis (mg/h)

    for step in range(n_steps + 1):
        t = step * dt_h
        times.append(t)
        a_skin_list.append(a_skin)
        a_dermis_list.append(a_dermis)
        c_plasma_list.append(c_plasma)

        # Rate of drug entering dermis
        flux = k_skin_per_h * enhancement_factor * a_skin
        flux_list.append(flux)

        # Forward Euler update
        d_skin = k_skin_per_h * enhancement_factor * a_skin * dt_h
        d_dermis_out = k_d2p_per_h * a_dermis * dt_h
        d_plasma_elim = k_elim * c_plasma * dt_h

        a_plasma_mg = c_plasma * vd_L
        d_plasma_in = k_d2p_per_h * a_dermis * dt_h

        a_skin = a_skin - d_skin
        a_dermis = a_dermis + d_skin - d_dermis_out
        # Update plasma concentration
        a_plasma_mg = a_plasma_mg + d_plasma_in - d_plasma_elim
        c_plasma = a_plasma_mg / vd_L
        a_eliminated += d_plasma_elim

        # Clamp negatives due to Euler approximation
        a_skin = max(0.0, a_skin)
        a_dermis = max(0.0, a_dermis)
        c_plasma = max(0.0, c_plasma)

    # PK metrics
    cmax_plasma_mg_L = max(c_plasma_list)
    if cmax_plasma_mg_L > 0:
        tmax_idx = c_plasma_list.index(cmax_plasma_mg_L)
        tmax_plasma_h = times[tmax_idx]
    else:
        tmax_plasma_h = 0.0

    auc_plasma_mg_h_per_L = _trapz(c_plasma_list, times)

    # Bioavailability: fraction of dose reaching plasma
    # Total absorbed = eliminated + still in plasma
    total_in_plasma = c_plasma_list[-1] * vd_L
    total_absorbed = a_eliminated + total_in_plasma
    bioavailability_pct = min(100.0, (total_absorbed / dose_mg) * 100.0) if dose_mg > 0 else 0.0

    # Peak flux (mg/h)
    flux_peak_mg_h = max(flux_list) if flux_list else 0.0

    # Build notes
    notes_parts = [
        f"Route: {route}",
        f"Enhancement factor: {enhancement_factor:.3f}",
        f"charge_type: {charge_type}",
        f"k_skin_per_h (effective): {k_skin_per_h * enhancement_factor:.4f} 1/h",
        f"k_elim: {k_elim:.4f} 1/h",
        f"Cmax: {cmax_plasma_mg_L:.4f} mg/L at {tmax_plasma_h:.2f} h",
        f"AUC: {auc_plasma_mg_h_per_L:.4f} mg*h/L",
        f"Bioavailability: {bioavailability_pct:.1f}%",
    ]
    notes = "; ".join(notes_parts)

    return TransdermalIontophoresisPKResult(
        drug_name=drug_name,
        route=route,
        current_mA=current_mA,
        charge_type=charge_type,
        dose_mg=dose_mg,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        a_skin_mg=a_skin_list,
        a_dermis_mg=a_dermis_list,
        cmax_plasma_mg_L=cmax_plasma_mg_L,
        tmax_plasma_h=tmax_plasma_h,
        auc_plasma_mg_h_per_L=auc_plasma_mg_h_per_L,
        enhancement_factor=enhancement_factor,
        bioavailability_pct=bioavailability_pct,
        flux_peak_mg_h=flux_peak_mg_h,
        notes=notes,
    )


def compare_modes(
    drug_name: str,
    dose_mg: float,
    current_mA: float = 0.5,
    charge_type: str = "cationic",
    k_skin_per_h: float = 0.02,
    k_d2p_per_h: float = 0.5,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> list[TransdermalIontophoresisPKResult]:
    """Compare passive vs iontophoresis delivery modes.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Dose applied to skin (mg).
    current_mA : float
        Current to apply in iontophoresis mode (mA).
    charge_type : str
        Drug charge type for enhancement calculation.
    k_skin_per_h : float
        Passive skin permeation rate (1/h).
    k_d2p_per_h : float
        Dermis-to-plasma rate (1/h).
    cl_L_per_h : float
        Systemic clearance (L/h).
    vd_L : float
        Volume of distribution (L).
    t_end_h : float
        Simulation end time (h).
    dt_h : float
        Time step (h).

    Returns
    -------
    list[TransdermalIontophoresisPKResult]
        Two results [passive, iontophoresis], sorted by AUC descending
        (iontophoresis typically first).
    """
    common_kwargs = dict(
        dose_mg=dose_mg,
        charge_type=charge_type,
        k_skin_per_h=k_skin_per_h,
        k_d2p_per_h=k_d2p_per_h,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
        t_end_h=t_end_h,
        dt_h=dt_h,
    )

    passive_result = simulate_iontophoresis(
        drug_name=drug_name,
        route="passive",
        current_mA=0.0,
        **common_kwargs,
    )
    ionto_result = simulate_iontophoresis(
        drug_name=drug_name,
        route="iontophoresis",
        current_mA=current_mA,
        **common_kwargs,
    )

    results = [passive_result, ionto_result]
    results.sort(key=lambda r: r.auc_plasma_mg_h_per_L, reverse=True)
    return results
