"""
Phase 317 — Hepatic Sinusoidal Uptake Transporter Model

Models active hepatic uptake via OATP1B1/1B3 transporters using
Michaelis-Menten kinetics. Relevant for statins, repaglinide, and
other OATP substrates.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HepaticUptakeResult:
    """Result of a hepatic sinusoidal uptake simulation."""

    drug_name: str
    dose_mg: float
    route: str
    vmax_oatp_mg_per_h: float
    km_oatp_mg_per_L: float
    times_h: list
    c_plasma_mg_L: list
    c_hepatic_mg_L: list
    auc_plasma_mg_h_per_L: float
    cmax_plasma_mg_L: float
    auc_hepatic_mg_h_per_L: float
    cmax_hepatic_mg_L: float
    hepatic_to_plasma_ratio: float
    notes: str


def _validate_inputs(
    drug_name: str,
    dose_mg: float,
    route: str,
    cl_passive_L_per_h: float,
    vmax_oatp_mg_per_h: float,
    km_oatp_mg_per_L: float,
    vd_plasma_L: float,
    vd_hepatic_L: float,
    t_end_h: float,
) -> None:
    """Validate inputs for simulate_hepatic_uptake."""
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_passive_L_per_h < 0:
        raise ValueError(f"cl_passive_L_per_h must be >= 0, got {cl_passive_L_per_h}")
    if vmax_oatp_mg_per_h <= 0:
        raise ValueError(f"vmax_oatp_mg_per_h must be > 0, got {vmax_oatp_mg_per_h}")
    if km_oatp_mg_per_L <= 0:
        raise ValueError(f"km_oatp_mg_per_L must be > 0, got {km_oatp_mg_per_L}")
    if vd_plasma_L <= 0:
        raise ValueError(f"vd_plasma_L must be > 0, got {vd_plasma_L}")
    if vd_hepatic_L <= 0:
        raise ValueError(f"vd_hepatic_L must be > 0, got {vd_hepatic_L}")
    if t_end_h <= 0:
        raise ValueError(f"t_end_h must be > 0, got {t_end_h}")
    if route not in {"iv_bolus", "oral"}:
        raise ValueError(f"route must be 'iv_bolus' or 'oral', got '{route}'")


def _trapezoidal_auc(x: list, y: list) -> float:
    """Compute AUC using manual trapezoidal rule."""
    return sum(0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]) for i in range(1, len(x)))


def simulate_hepatic_uptake(
    drug_name: str,
    dose_mg: float,
    route: str,
    cl_passive_L_per_h: float,
    vmax_oatp_mg_per_h: float,
    km_oatp_mg_per_L: float,
    vd_plasma_L: float,
    vd_hepatic_L: float,
    t_end_h: float = 24.0,
    dt_h: float = 0.01,
) -> HepaticUptakeResult:
    """
    Simulate hepatic sinusoidal uptake via OATP1B1/1B3 transporters.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Dose in mg.
    route : str
        'iv_bolus' or 'oral' (F=0.6, ka=1.0/h).
    cl_passive_L_per_h : float
        Passive hepatic clearance (L/h).
    vmax_oatp_mg_per_h : float
        Maximum OATP uptake rate (mg/h).
    km_oatp_mg_per_L : float
        Michaelis constant for OATP (mg/L).
    vd_plasma_L : float
        Volume of distribution in plasma compartment (L).
    vd_hepatic_L : float
        Volume of distribution in hepatic compartment (L).
    t_end_h : float
        Simulation end time (h).
    dt_h : float
        Time step for forward Euler (h).

    Returns
    -------
    HepaticUptakeResult
    """
    _validate_inputs(
        drug_name,
        dose_mg,
        route,
        cl_passive_L_per_h,
        vmax_oatp_mg_per_h,
        km_oatp_mg_per_L,
        vd_plasma_L,
        vd_hepatic_L,
        t_end_h,
    )

    # Biliary excretion rate constant from liver (h^-1)
    k_biliary = 0.5

    # Oral bioavailability and absorption rate constant
    F_oral = 0.6
    ka = 1.0  # h^-1

    # Initial conditions
    if route == "iv_bolus":
        a_plasma = dose_mg
        a_gut = 0.0
    else:
        # oral: absorbed fraction via gut
        a_plasma = 0.0
        a_gut = dose_mg * F_oral

    a_hepatic = 0.0

    times_h = []
    c_plasma_list = []
    c_hepatic_list = []

    t = 0.0
    n_steps = int(t_end_h / dt_h) + 1

    for _ in range(n_steps):
        c_plasma = a_plasma / vd_plasma_L
        c_hepatic = a_hepatic / vd_hepatic_L

        times_h.append(t)
        c_plasma_list.append(c_plasma)
        c_hepatic_list.append(c_hepatic)

        # Active OATP uptake rate (Michaelis-Menten)
        cl_oatp_rate = vmax_oatp_mg_per_h * c_plasma / (km_oatp_mg_per_L + c_plasma)

        # Passive hepatic clearance flux
        passive_flux = cl_passive_L_per_h * c_plasma

        # Oral input into plasma
        oral_input = ka * a_gut if route == "oral" else 0.0

        # ODE: dA_plasma/dt
        d_plasma = oral_input - cl_oatp_rate - passive_flux

        # ODE: dA_hepatic/dt
        d_hepatic = cl_oatp_rate + passive_flux - k_biliary * a_hepatic

        # ODE: dA_gut/dt (oral only)
        d_gut = -ka * a_gut if route == "oral" else 0.0

        # Forward Euler update
        a_plasma += d_plasma * dt_h
        a_hepatic += d_hepatic * dt_h
        a_gut += d_gut * dt_h

        # Guard against negative amounts
        if a_plasma < 0.0:
            a_plasma = 0.0
        if a_hepatic < 0.0:
            a_hepatic = 0.0
        if a_gut < 0.0:
            a_gut = 0.0

        t += dt_h

    auc_plasma = _trapezoidal_auc(times_h, c_plasma_list)
    auc_hepatic = _trapezoidal_auc(times_h, c_hepatic_list)
    cmax_plasma = max(c_plasma_list)
    cmax_hepatic = max(c_hepatic_list)

    ratio = cmax_hepatic / cmax_plasma if cmax_plasma > 0.0 else 0.0

    notes = (
        f"OATP1B1/1B3 active uptake model; route={route}; "
        f"Vmax={vmax_oatp_mg_per_h} mg/h; Km={km_oatp_mg_per_L} mg/L; "
        f"k_biliary={k_biliary}/h"
    )

    return HepaticUptakeResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route=route,
        vmax_oatp_mg_per_h=vmax_oatp_mg_per_h,
        km_oatp_mg_per_L=km_oatp_mg_per_L,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_list,
        c_hepatic_mg_L=c_hepatic_list,
        auc_plasma_mg_h_per_L=auc_plasma,
        cmax_plasma_mg_L=cmax_plasma,
        auc_hepatic_mg_h_per_L=auc_hepatic,
        cmax_hepatic_mg_L=cmax_hepatic,
        hepatic_to_plasma_ratio=ratio,
        notes=notes,
    )


def compare_oatp_inhibition(
    drug_name: str,
    dose_mg: float,
    inhibition_fractions: list,
    cl_passive_L_per_h: float,
    vmax_oatp_mg_per_h: float,
    km_oatp_mg_per_L: float,
    vd_plasma_L: float,
    vd_hepatic_L: float,
    route: str = "oral",
    t_end_h: float = 24.0,
    dt_h: float = 0.01,
) -> list:
    """
    Compare OATP inhibition scenarios for a substrate drug.

    Parameters
    ----------
    drug_name : str
        Name of the substrate drug.
    dose_mg : float
        Dose in mg.
    inhibition_fractions : list
        List of floats 0-1. Each value is the fraction of OATP inhibited
        by a perpetrator drug (0 = no inhibition, 1 = complete inhibition).
    cl_passive_L_per_h : float
        Passive hepatic clearance (L/h).
    vmax_oatp_mg_per_h : float
        Maximum OATP uptake rate (mg/h).
    km_oatp_mg_per_L : float
        Michaelis constant (mg/L).
    vd_plasma_L : float
        Plasma volume of distribution (L).
    vd_hepatic_L : float
        Hepatic volume of distribution (L).
    route : str
        'iv_bolus' or 'oral'.
    t_end_h : float
        Simulation end time (h).
    dt_h : float
        Time step (h).

    Returns
    -------
    list of HepaticUptakeResult sorted by auc_plasma_mg_h_per_L descending.
    """
    results = []
    for frac in inhibition_fractions:
        # Inhibition reduces effective Vmax
        effective_vmax = vmax_oatp_mg_per_h * (1.0 - frac)
        # Ensure Vmax stays above a tiny positive value to avoid validation errors
        if effective_vmax <= 0.0:
            effective_vmax = 1e-9

        result = simulate_hepatic_uptake(
            drug_name=drug_name,
            dose_mg=dose_mg,
            route=route,
            cl_passive_L_per_h=cl_passive_L_per_h,
            vmax_oatp_mg_per_h=effective_vmax,
            km_oatp_mg_per_L=km_oatp_mg_per_L,
            vd_plasma_L=vd_plasma_L,
            vd_hepatic_L=vd_hepatic_L,
            t_end_h=t_end_h,
            dt_h=dt_h,
        )
        results.append(result)

    # Sort by plasma AUC descending (higher inhibition → higher plasma AUC)
    results.sort(key=lambda r: r.auc_plasma_mg_h_per_L, reverse=True)
    return results
