"""Spleen PK model for nanoparticles and macrophage-targeted drugs.

Models drug distribution in the spleen as a perfusion-limited compartment
with irreversible phagocytic uptake (MPS sequestration).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from omega_pbpk._compat import np_trapz


@dataclass
class SpleenPKResult:
    """Result of spleen PK simulation."""

    drug_name: str
    dose_mg: float
    kp_spleen: float
    phagocytosis_rate_per_h: float
    times_h: list[float] = field(default_factory=list)
    c_plasma_mg_L: list[float] = field(default_factory=list)
    c_spleen_mg_L: list[float] = field(default_factory=list)
    m_phagocytosed_mg: list[float] = field(default_factory=list)
    cmax_plasma: float = 0.0
    auc_plasma: float = 0.0
    f_phagocytosed_24h: float = 0.0
    notes: str = ""


def simulate_spleen_pk(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    spleen_blood_flow_L_per_h: float = 0.9,
    kp_spleen: float = 3.0,
    phagocytosis_rate_per_h: float = 0.1,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> SpleenPKResult:
    """Simulate drug distribution in the spleen with phagocytic sequestration.

    Two-compartment model: plasma + spleen.
    Perfusion-limited transfer between plasma and spleen.
    Irreversible phagocytic uptake (macrophage sequestration).

    Parameters
    ----------
    drug_name : str
        Name of the drug or nanoparticle.
    dose_mg : float
        IV bolus dose (mg).
    cl_L_per_h : float
        Systemic clearance (L/h).
    vd_L : float
        Volume of distribution (L).
    spleen_blood_flow_L_per_h : float
        Blood flow to spleen (L/h), default ~0.9 L/h (typical human).
    kp_spleen : float
        Spleen tissue-to-plasma partition coefficient (dimensionless).
    phagocytosis_rate_per_h : float
        First-order rate of macrophage phagocytosis (1/h).
    t_end_h : float
        Simulation end time (h).
    dt_h : float
        Forward Euler time step (h).

    Returns
    -------
    SpleenPKResult
        Plasma, spleen, and phagocytosed mass time courses with PK metrics.
    """
    if not drug_name or not isinstance(drug_name, str):
        raise ValueError("drug_name must be a non-empty string")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if spleen_blood_flow_L_per_h <= 0:
        raise ValueError("spleen_blood_flow_L_per_h must be > 0")
    if kp_spleen <= 0:
        raise ValueError("kp_spleen must be > 0")
    if phagocytosis_rate_per_h < 0:
        raise ValueError("phagocytosis_rate_per_h must be >= 0")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be > 0")
    if dt_h <= 0 or dt_h > t_end_h:
        raise ValueError("dt_h must be > 0 and <= t_end_h")

    # Spleen volume: ~0.3% of body volume, scaled by Kp
    vd_spleen = vd_L * 0.003 * kp_spleen
    vd_plasma = max(vd_L - vd_spleen, 1.0)

    q = spleen_blood_flow_L_per_h  # blood flow to spleen
    ke = cl_L_per_h / vd_plasma  # elimination rate constant from plasma

    n_steps = max(int(math.ceil(t_end_h / dt_h)), 1)
    times = np.linspace(0.0, t_end_h, n_steps + 1)

    # State variables: mass in plasma and spleen (mg)
    a_plasma = np.zeros(n_steps + 1)
    a_spleen = np.zeros(n_steps + 1)
    a_phago = np.zeros(n_steps + 1)

    # IV bolus initial condition
    a_plasma[0] = dose_mg

    for i in range(n_steps):
        c_plasma = a_plasma[i] / vd_plasma  # mg/L
        c_spleen = a_spleen[i] / vd_spleen  # mg/L

        # Perfusion-limited transfer: Q * (C_plasma - C_spleen / Kp)
        transfer = q * (c_plasma - c_spleen / kp_spleen)

        # Phagocytic uptake (irreversible)
        phago_rate = phagocytosis_rate_per_h * c_spleen * vd_spleen

        # ODEs
        da_plasma = -ke * a_plasma[i] - transfer
        da_spleen = transfer - phago_rate
        da_phago = phago_rate

        a_plasma[i + 1] = max(a_plasma[i] + dt_h * da_plasma, 0.0)
        a_spleen[i + 1] = max(a_spleen[i] + dt_h * da_spleen, 0.0)
        a_phago[i + 1] = a_phago[i] + dt_h * da_phago

    c_plasma_arr = a_plasma / vd_plasma
    c_spleen_arr = a_spleen / vd_spleen

    # PK metrics
    cmax_plasma = float(np.max(c_plasma_arr))
    auc_plasma = float(np_trapz(c_plasma_arr, times))

    # Fraction phagocytosed at 24h (or t_end if < 24h)
    total_recovered = a_plasma[-1] + a_spleen[-1] + a_phago[-1]
    f_phago = float(a_phago[-1] / dose_mg) if dose_mg > 0 else 0.0

    notes_parts = []
    if kp_spleen > 5:
        notes_parts.append("High Kp suggests nanoparticle or lipophilic drug")
    if phagocytosis_rate_per_h > 1.0:
        notes_parts.append("Rapid MPS sequestration expected")
    if f_phago > 0.5:
        notes_parts.append("Majority of dose sequestered by MPS")
    if total_recovered < dose_mg * 0.05:
        notes_parts.append("Drug substantially eliminated by end of simulation")
    notes = "; ".join(notes_parts) if notes_parts else "Normal spleen PK"

    return SpleenPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        kp_spleen=kp_spleen,
        phagocytosis_rate_per_h=phagocytosis_rate_per_h,
        times_h=times.tolist(),
        c_plasma_mg_L=c_plasma_arr.tolist(),
        c_spleen_mg_L=c_spleen_arr.tolist(),
        m_phagocytosed_mg=a_phago.tolist(),
        cmax_plasma=cmax_plasma,
        auc_plasma=auc_plasma,
        f_phagocytosed_24h=f_phago,
        notes=notes,
    )


def mps_uptake_analysis(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    phagocytosis_rates: list[float] | None = None,
) -> list[dict]:
    """Sweep phagocytosis rates and return fraction sequestered in MPS at simulation end.

    Parameters
    ----------
    drug_name : str
        Name of the drug or nanoparticle.
    dose_mg : float
        IV bolus dose (mg).
    cl_L_per_h : float
        Systemic clearance (L/h).
    vd_L : float
        Volume of distribution (L).
    phagocytosis_rates : list[float], optional
        Phagocytosis rates (1/h) to sweep. Defaults to [0, 0.5, 1.0, 2.0, 5.0].

    Returns
    -------
    list[dict]
        Each dict contains phagocytosis_rate, f_sequestered, and classification.
    """
    if not drug_name or not isinstance(drug_name, str):
        raise ValueError("drug_name must be a non-empty string")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")

    if phagocytosis_rates is None:
        phagocytosis_rates = [0.0, 0.5, 1.0, 2.0, 5.0]

    rates = list(phagocytosis_rates)
    if not rates:
        raise ValueError("phagocytosis_rates must not be empty")
    for r in rates:
        if r < 0:
            raise ValueError(f"phagocytosis rate {r} must be >= 0")

    results = []
    for rate in rates:
        sim = simulate_spleen_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            phagocytosis_rate_per_h=rate,
            t_end_h=24.0,
            dt_h=0.1,
        )
        f_seq = sim.f_phagocytosed_24h

        if f_seq < 0.1:
            classification = "Low MPS uptake"
        elif f_seq < 0.3:
            classification = "Moderate MPS uptake"
        elif f_seq < 0.6:
            classification = "High MPS uptake"
        else:
            classification = "Extensive MPS sequestration"

        results.append(
            {
                "phagocytosis_rate_per_h": rate,
                "f_sequestered": f_seq,
                "classification": classification,
                "cmax_plasma_mg_L": sim.cmax_plasma,
                "auc_plasma_mg_h_L": sim.auc_plasma,
            }
        )

    return results


__all__ = [
    "SpleenPKResult",
    "simulate_spleen_pk",
    "mps_uptake_analysis",
]
