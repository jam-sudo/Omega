"""
Phase 210 — Lymphatic Drug Transport

Models lymphatic drug transport and thoracic duct delivery for lipophilic
drugs using a 3-compartment Forward Euler ODE system.

Compartments
------------
A_lymph_gut (mg)        Drug in mesenteric lymph (chylomicron-associated)
A_lymph_thoracic (mg)   Drug in thoracic duct
C_systemic (mg/L)       Systemic plasma concentration

Lymphatic fraction
------------------
f_lymph = 1 / (1 + exp(-0.8 * (logP - 4)))
    logP < 3 → f_lymph << 0.5 (portal dominant)
    logP > 5 → f_lymph >> 0.5 (lymphatic dominant)

Default rate constants
----------------------
k_transit  = 0.3 /h  (mesenteric lymph → thoracic duct)
k_delivery = 0.5 /h  (thoracic duct → systemic circulation)
ka         = 1.0 /h  (GI absorption rate)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass (plain – has list fields)
# ---------------------------------------------------------------------------


@dataclass
class LymphaticTransportResult:
    """Full simulation result for lymphatic drug transport."""

    drug_name: str
    dose_mg: float
    logP: float
    f_lymph_fraction: float  # logistic(0.8*(logP-4))
    times_h: list[float]  # simulation time grid
    c_systemic_mg_L: list[float]  # plasma concentration trajectory
    a_lymph_thoracic_mg: list[float]  # thoracic duct drug amount trajectory
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    lymphatic_auc_contribution_pct: float
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_K_TRANSIT = 0.3  # /h  mesenteric → thoracic duct
_K_DELIVERY = 0.5  # /h  thoracic duct → systemic
_KA_DEFAULT = 1.0  # /h  oral absorption


def _logistic(x: float) -> float:
    """Numerically stable logistic function."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def _lymphatic_fraction(logP: float) -> float:
    """f_lymph based on logP (Sigmoid centred at logP = 4, slope 0.8)."""
    return _logistic(0.8 * (logP - 4.0))


def _validate(dose_mg: float, cl_L_per_h: float) -> None:
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0; got {dose_mg}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0; got {cl_L_per_h}")


def _trapz_auc(times: list[float], conc: list[float]) -> float:
    """Manual trapezoidal AUC."""
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        auc += 0.5 * (conc[i] + conc[i - 1]) * dt
    return auc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def simulate_lymphatic_transport(
    drug_name: str,
    dose_mg: float,
    logP: float,
    cl_L_per_h: float,
    vd_L: float,
    *,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
    ka: float = _KA_DEFAULT,
    k_transit: float = _K_TRANSIT,
    k_delivery: float = _K_DELIVERY,
    notes: str = "",
) -> LymphaticTransportResult:
    """
    Simulate 3-compartment lymphatic transport model using Forward Euler.

    Parameters
    ----------
    drug_name : str
    dose_mg : float
        Oral dose administered at t = 0.
    logP : float
        Octanol-water partition coefficient.
    cl_L_per_h : float
        Systemic clearance (L/h).
    vd_L : float
        Volume of distribution (L).
    t_end_h : float
        Simulation end time (h).
    dt_h : float
        Integration step size (h).
    ka : float
        Oral absorption rate constant (/h).
    k_transit : float
        Mesenteric lymph → thoracic duct rate (/h).
    k_delivery : float
        Thoracic duct → systemic rate (/h).
    notes : str
        Free-text notes passed through to result.

    Returns
    -------
    LymphaticTransportResult
    """
    _validate(dose_mg, cl_L_per_h)

    f_lymph = _lymphatic_fraction(logP)
    ke = cl_L_per_h / vd_L  # elimination rate (/h)

    # State variables  (GI depot tracked via exp(-ka*t) analytical solution)
    a_lymph_gut = 0.0
    a_lymph_thor = 0.0
    c_sys = 0.0

    times: list[float] = []
    c_sys_list: list[float] = []
    a_thor_list: list[float] = []

    n_steps = int(round(t_end_h / dt_h)) + 1

    for i in range(n_steps):
        t = i * dt_h

        times.append(t)
        c_sys_list.append(c_sys)
        a_thor_list.append(a_lymph_thor)

        # Absorption flux from GI (first-order, dose available at t=0)
        # Total absorption rate = ka * dose_mg * exp(-ka * t)
        absorption_flux = ka * dose_mg * math.exp(-ka * t)

        # ODEs
        d_lymph_gut = f_lymph * absorption_flux - k_transit * a_lymph_gut
        d_lymph_thor = k_transit * a_lymph_gut - k_delivery * a_lymph_thor
        d_c_sys = (
            (1.0 - f_lymph) * absorption_flux / vd_L + k_delivery * a_lymph_thor / vd_L - ke * c_sys
        )

        # Forward Euler update
        a_lymph_gut += d_lymph_gut * dt_h
        if a_lymph_gut < 0.0:
            a_lymph_gut = 0.0
        a_lymph_thor += d_lymph_thor * dt_h
        if a_lymph_thor < 0.0:
            a_lymph_thor = 0.0
        c_sys = c_sys + d_c_sys * dt_h
        if c_sys < 0.0:
            c_sys = 0.0

    # Post-process
    cmax = max(c_sys_list)
    tmax = times[c_sys_list.index(cmax)]
    auc_total = _trapz_auc(times, c_sys_list)

    # Lymphatic AUC contribution:
    # Reconstruct systemic concentration coming only from lymphatic pathway
    # by simulating with f_lymph = 1 (portal contribution = 0) to get the
    # share, then express as percentage of total AUC.
    if f_lymph > 0.0:
        a_lg2 = 0.0
        a_lt2 = 0.0
        c_s2 = 0.0
        c_lymph_only: list[float] = []
        for i in range(n_steps):
            t = i * dt_h
            c_lymph_only.append(c_s2)
            flux2 = ka * dose_mg * math.exp(-ka * t)
            d_lg2 = flux2 - k_transit * a_lg2  # f_lymph=1
            d_lt2 = k_transit * a_lg2 - k_delivery * a_lt2
            d_cs2 = k_delivery * a_lt2 / vd_L - ke * c_s2
            a_lg2 += d_lg2 * dt_h
            if a_lg2 < 0.0:
                a_lg2 = 0.0
            a_lt2 += d_lt2 * dt_h
            if a_lt2 < 0.0:
                a_lt2 = 0.0
            c_s2 = c_s2 + d_cs2 * dt_h
            if c_s2 < 0.0:
                c_s2 = 0.0
        auc_lymph_only = _trapz_auc(times, c_lymph_only)
        # Scale by actual f_lymph
        auc_lymph_contrib = auc_lymph_only * f_lymph
        lymph_pct = (auc_lymph_contrib / auc_total * 100.0) if auc_total > 0.0 else 0.0
    else:
        lymph_pct = 0.0

    default_notes = (
        f"logP={logP:.1f}, f_lymph={f_lymph:.3f}; "
        f"{'lymphatic absorption dominant' if f_lymph > 0.5 else 'portal absorption dominant'}."
    )

    return LymphaticTransportResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        logP=logP,
        f_lymph_fraction=f_lymph,
        times_h=times,
        c_systemic_mg_L=c_sys_list,
        a_lymph_thoracic_mg=a_thor_list,
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc_mg_h_per_L=auc_total,
        lymphatic_auc_contribution_pct=lymph_pct,
        notes=notes or default_notes,
    )


def compare_logp_effect(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    logp_values: list[float],
    *,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> list[LymphaticTransportResult]:
    """
    Simulate lymphatic transport for multiple logP values and return results
    sorted by lymphatic_fraction descending (most lipophilic first).

    Parameters
    ----------
    drug_name : str
    dose_mg : float
    cl_L_per_h : float
    vd_L : float
    logp_values : list of float
    t_end_h : float
    dt_h : float

    Returns
    -------
    list of LymphaticTransportResult sorted by f_lymph_fraction descending.
    """
    _validate(dose_mg, cl_L_per_h)

    results: list[LymphaticTransportResult] = []
    for lp in logp_values:
        r = simulate_lymphatic_transport(
            drug_name=drug_name,
            dose_mg=dose_mg,
            logP=lp,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            t_end_h=t_end_h,
            dt_h=dt_h,
        )
        results.append(r)

    results.sort(key=lambda r: r.f_lymph_fraction, reverse=True)
    return results
