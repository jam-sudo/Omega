"""Efflux transporter kinetics — ABC transporter-mediated drug efflux (P-gp, BCRP, MRP2).

Models Michaelis-Menten efflux kinetics and bi-directional Caco-2 transport simulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "NetFluxResult",
    "Caco2TransportResult",
    "efflux_rate",
    "net_flux",
    "simulate_caco2_transport",
    "classify_substrate",
]

# Efflux ratio thresholds (FDA/EMA guidance)
_ER_MODERATE_LOW = 2.0
_ER_STRONG_LOW = 5.0

# Default Caco-2 monolayer area (cm^2) for a standard 24-well insert
_DEFAULT_AREA_CM2 = 1.0


@dataclass(frozen=True)
class NetFluxResult:
    """Result of net flux calculation across an epithelial barrier."""

    c_apical_mg_L: float
    c_basal_mg_L: float
    papp_ab_cm_s: float
    papp_ba_cm_s: float
    vmax_mg_per_h: float
    km_mg_L: float
    efflux_ratio: float
    substrate_class: str
    notes: str


@dataclass(frozen=True)
class Caco2TransportResult:
    """Result of a bi-directional Caco-2 transport simulation."""

    times_h: list[float]
    c_apical: list[float]
    c_basolateral: list[float]
    papp_apparent: float
    efflux_ratio: float
    substrate_class: str
    notes: str


# ---------------------------------------------------------------------------
# Core kinetics
# ---------------------------------------------------------------------------


def efflux_rate(c_intracellular_mg_L: float, vmax_mg_per_h: float, km_mg_L: float) -> float:
    """Compute Michaelis-Menten efflux rate.

    Parameters
    ----------
    c_intracellular_mg_L:
        Intracellular drug concentration (mg/L). Must be >= 0.
    vmax_mg_per_h:
        Maximum efflux rate (mg/h). Must be > 0.
    km_mg_L:
        Michaelis constant (mg/L). Must be > 0.

    Returns
    -------
    float
        Efflux rate in mg/h: Vmax * C / (Km + C).
    """
    if vmax_mg_per_h <= 0:
        raise ValueError(f"vmax_mg_per_h must be > 0, got {vmax_mg_per_h}")
    if km_mg_L <= 0:
        raise ValueError(f"km_mg_L must be > 0, got {km_mg_L}")
    if c_intracellular_mg_L < 0:
        raise ValueError(f"c_intracellular_mg_L must be >= 0, got {c_intracellular_mg_L}")

    return vmax_mg_per_h * c_intracellular_mg_L / (km_mg_L + c_intracellular_mg_L)


def classify_substrate(efflux_ratio: float) -> str:
    """Classify a compound as a transporter substrate based on efflux ratio.

    Parameters
    ----------
    efflux_ratio:
        ER = Papp_BA / Papp_AB.

    Returns
    -------
    str
        'not_substrate' (ER < 2), 'moderate' (2 <= ER <= 5), 'strong' (ER > 5).
    """
    if efflux_ratio < _ER_MODERATE_LOW:
        return "not_substrate"
    if efflux_ratio <= _ER_STRONG_LOW:
        return "moderate"
    return "strong"


def net_flux(
    c_apical_mg_L: float,
    c_basal_mg_L: float,
    papp_ab_cm_s: float,
    papp_ba_cm_s: float,
    vmax_mg_per_h: float,
    km_mg_L: float,
    area_cm2: float = _DEFAULT_AREA_CM2,
) -> NetFluxResult:
    """Compute net flux and efflux ratio across an epithelial monolayer.

    Parameters
    ----------
    c_apical_mg_L:
        Drug concentration on the apical side (mg/L). Must be >= 0.
    c_basal_mg_L:
        Drug concentration on the basolateral side (mg/L). Must be >= 0.
    papp_ab_cm_s:
        Apparent permeability A→B (passive, cm/s). Must be > 0.
    papp_ba_cm_s:
        Apparent permeability B→A including efflux (cm/s). Must be > 0.
    vmax_mg_per_h:
        Transporter Vmax (mg/h). Must be > 0.
    km_mg_L:
        Transporter Km (mg/L). Must be > 0.
    area_cm2:
        Monolayer area (cm^2). Must be > 0.

    Returns
    -------
    NetFluxResult
    """
    if c_apical_mg_L < 0:
        raise ValueError(f"c_apical_mg_L must be >= 0, got {c_apical_mg_L}")
    if c_basal_mg_L < 0:
        raise ValueError(f"c_basal_mg_L must be >= 0, got {c_basal_mg_L}")
    if papp_ab_cm_s <= 0:
        raise ValueError(f"papp_ab_cm_s must be > 0, got {papp_ab_cm_s}")
    if papp_ba_cm_s <= 0:
        raise ValueError(f"papp_ba_cm_s must be > 0, got {papp_ba_cm_s}")
    if vmax_mg_per_h <= 0:
        raise ValueError(f"vmax_mg_per_h must be > 0, got {vmax_mg_per_h}")
    if km_mg_L <= 0:
        raise ValueError(f"km_mg_L must be > 0, got {km_mg_L}")
    if area_cm2 <= 0:
        raise ValueError(f"area_cm2 must be > 0, got {area_cm2}")

    er = papp_ba_cm_s / papp_ab_cm_s
    substrate_class = classify_substrate(er)

    notes = (
        f"ER={er:.2f}; substrate class: {substrate_class}; "
        f"efflux rate at basal conc = {efflux_rate(c_basal_mg_L, vmax_mg_per_h, km_mg_L):.4f} mg/h"
    )

    return NetFluxResult(
        c_apical_mg_L=c_apical_mg_L,
        c_basal_mg_L=c_basal_mg_L,
        papp_ab_cm_s=papp_ab_cm_s,
        papp_ba_cm_s=papp_ba_cm_s,
        vmax_mg_per_h=vmax_mg_per_h,
        km_mg_L=km_mg_L,
        efflux_ratio=er,
        substrate_class=substrate_class,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Caco-2 bi-directional transport simulation
# ---------------------------------------------------------------------------

# cm/s to L/h conversion factor for a given area:
#   flux (mg/h) = Papp (cm/s) * area (cm^2) * C (mg/L) * 3600 (s/h) / 1000 (cm^3→L)
_CM_S_TO_L_H = 3600.0 / 1000.0  # = 3.6


def simulate_caco2_transport(
    c_donor_mg_L: float,
    papp_cm_s: float,
    vmax_mg_per_h: float,
    km_mg_L: float,
    volume_donor_mL: float = 0.4,
    volume_acceptor_mL: float = 1.0,
    t_end_h: float = 2.0,
    dt_h: float = 0.01,
) -> Caco2TransportResult:
    """Simulate bi-directional Caco-2 transport with apical efflux (Forward Euler).

    The monolayer area is fixed at 1.12 cm^2 (standard 12-well insert).

    States:
      A (apical, donor side)   — units: mg/L
      B (basolateral, acceptor) — units: mg/L

    ODEs (simplified; efflux modelled on the basolateral compartment driving
    drug back toward apical):

        dA/dt = -Papp_AB * area * A / vol_A  +  efflux_rate(B) * area / vol_A
        dB/dt = +Papp_AB * area * A / vol_A  -  efflux_rate(B) * area / vol_B

    The apparent Papp is estimated from the linear increase in acceptor
    concentration early in the simulation (first 10% of time points).

    Parameters
    ----------
    c_donor_mg_L:
        Initial donor-side concentration (mg/L). Must be > 0.
    papp_cm_s:
        Passive A→B permeability (cm/s). Must be > 0.
    vmax_mg_per_h:
        Efflux transporter Vmax (mg/h). Must be > 0.
    km_mg_L:
        Efflux transporter Km (mg/L). Must be > 0.
    volume_donor_mL:
        Donor compartment volume (mL). Must be > 0.
    volume_acceptor_mL:
        Acceptor compartment volume (mL). Must be > 0.
    t_end_h:
        Simulation end time (h). Must be > 0.
    dt_h:
        Integration time step (h). Must be > 0.

    Returns
    -------
    Caco2TransportResult
    """
    if c_donor_mg_L <= 0:
        raise ValueError(f"c_donor_mg_L must be > 0, got {c_donor_mg_L}")
    if papp_cm_s <= 0:
        raise ValueError(f"papp_cm_s must be > 0, got {papp_cm_s}")
    if vmax_mg_per_h <= 0:
        raise ValueError(f"vmax_mg_per_h must be > 0, got {vmax_mg_per_h}")
    if km_mg_L <= 0:
        raise ValueError(f"km_mg_L must be > 0, got {km_mg_L}")
    if volume_donor_mL <= 0:
        raise ValueError(f"volume_donor_mL must be > 0, got {volume_donor_mL}")
    if volume_acceptor_mL <= 0:
        raise ValueError(f"volume_acceptor_mL must be > 0, got {volume_acceptor_mL}")
    if t_end_h <= 0:
        raise ValueError(f"t_end_h must be > 0, got {t_end_h}")
    if dt_h <= 0:
        raise ValueError(f"dt_h must be > 0, got {dt_h}")

    # Standard 12-well insert area
    area_cm2 = 1.12

    # Convert volumes from mL to L
    vol_a_L = volume_donor_mL / 1000.0
    vol_b_L = volume_acceptor_mL / 1000.0

    # Passive permeability coefficient: Papp_cm_s * area_cm2 * 3.6 => L/h
    perm_L_per_h = papp_cm_s * area_cm2 * _CM_S_TO_L_H

    # Efflux area factor: transporter flux = efflux_rate(C_B) * (area factor)
    # Efflux rate is in mg/h; distributing back apically via same area scaling
    efflux_area = area_cm2 / 1.0  # normalised; the ODE uses per-compartment volume

    n_steps = max(1, int(round(t_end_h / dt_h)))
    times: list[float] = []
    c_a_list: list[float] = []
    c_b_list: list[float] = []

    c_a = c_donor_mg_L
    c_b = 0.0
    t = 0.0

    for _ in range(n_steps + 1):
        times.append(t)
        c_a_list.append(c_a)
        c_b_list.append(c_b)

        if t >= t_end_h:
            break

        # Passive A→B flux (mg/h total = Papp_L_h * C_A)
        passive_flux = perm_L_per_h * c_a  # mg/h  (concentration × vol-normalised rate)

        # Efflux of B back to A (mg/h per area, then scaled by efflux_area)
        efflux = efflux_rate(c_b, vmax_mg_per_h, km_mg_L) * efflux_area  # mg/h

        # Concentration rates (mg/L/h)
        dc_a = (-passive_flux + efflux) / vol_a_L
        dc_b = (passive_flux - efflux) / vol_b_L

        c_a = max(0.0, c_a + dc_a * dt_h)
        c_b = max(0.0, c_b + dc_b * dt_h)
        t = min(t + dt_h, t_end_h)

    # Compute apparent Papp from the final mass transferred
    # Papp_apparent (cm/s) = dC_B/dt * vol_B / (area * C_A0)
    # Use linear regression on acceptor concentration increase
    times_arr = np.array(times)
    c_b_arr = np.array(c_b_list)
    # Slope of C_B vs time (mg/L/h) — take first half for linearity
    half = max(2, len(times_arr) // 2)
    if half > 1 and times_arr[half - 1] > times_arr[0]:
        slope = float(np.polyfit(times_arr[:half], c_b_arr[:half], 1)[0])
    else:
        slope = (c_b_list[-1] - c_b_list[0]) / max(t_end_h, 1e-9)

    # Papp = slope * vol_B (L) / (area_cm2 * C_donor) in L/h/cm^2 → convert to cm/s
    # L/h/cm^2 → cm/s: divide by 3600/1000 = 3.6
    papp_apparent_cm_s = (slope * vol_b_L) / (area_cm2 * c_donor_mg_L * _CM_S_TO_L_H)
    papp_apparent_cm_s = max(0.0, papp_apparent_cm_s)

    # Efflux ratio: compare effective BA vs AB permeability
    # BA is dominated by efflux; use the ratio of passive vs active
    # For simplicity, ER = (C_B_steady / C_A0 expected with no efflux) ratio
    # Use ratio of Vmax contribution vs passive
    er_passive = papp_cm_s
    # Effective BA = passive + efflux contribution
    efflux_at_half = efflux_rate(c_donor_mg_L / 2.0, vmax_mg_per_h, km_mg_L)
    efflux_papp_equiv = efflux_at_half * efflux_area / (area_cm2 * c_donor_mg_L * _CM_S_TO_L_H)
    papp_ba_equiv = er_passive + efflux_papp_equiv
    er = papp_ba_equiv / er_passive if er_passive > 0 else 1.0

    substrate_class = classify_substrate(er)

    notes = (
        f"Simulated {n_steps} steps; final C_A={c_a_list[-1]:.3f} mg/L, "
        f"C_B={c_b_list[-1]:.3f} mg/L; ER={er:.2f}; class={substrate_class}"
    )

    return Caco2TransportResult(
        times_h=times,
        c_apical=c_a_list,
        c_basolateral=c_b_list,
        papp_apparent=papp_apparent_cm_s,
        efflux_ratio=er,
        substrate_class=substrate_class,
        notes=notes,
    )
