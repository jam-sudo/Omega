"""Renal elimination saturation model for active tubular secretion.

Models nonlinear renal tubular secretion for drugs excreted via OAT/OCT
transporters.  Combines glomerular filtration (linear) with Michaelis-Menten
active secretion to produce time-varying apparent renal clearance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz


@dataclass(frozen=True)
class RenalSaturationResult:
    """Results from a renal saturation simulation."""

    drug_name: str
    dose_mg: float
    route: str  # "iv" or "oral"

    gfr_mL_per_min: float
    vmax_mg_per_h: float
    km_mg_L: float

    times_h: list[float]
    c_plasma_mg_L: list[float]
    cl_renal_apparent_L_per_h: list[float]  # time-varying apparent CL
    cumulative_excreted_mg: list[float]

    # Summary
    t_half_h: float
    auc_mg_h_per_L: float
    fe_urine: float  # fraction excreted unchanged
    cl_renal_initial_L_per_h: float  # CLrenal at t=0 (low concentrations)
    cl_renal_at_cmax_L_per_h: float  # CLrenal at Cmax (may be saturated)
    saturation_pct_at_cmax: float  # % of Vmax capacity used at Cmax
    dose_proportional: bool  # True if saturation < 20% at Cmax
    notes: str


def _terminal_half_life(times: np.ndarray, conc: np.ndarray) -> float:
    """Estimate terminal half-life from the last 30% of the simulation."""
    n = len(times)
    start_idx = int(n * 0.70)
    # Ensure at least 3 points for reliable regression
    if n - start_idx < 3:
        start_idx = max(0, n - 3)

    t_seg = times[start_idx:]
    c_seg = conc[start_idx:]

    # Filter out zero/negative concentrations
    mask = c_seg > 1e-12
    if mask.sum() < 2:
        return float("nan")

    t_fit = t_seg[mask]
    c_fit = c_seg[mask]

    # Linear regression on log(C) vs t
    log_c = np.log(c_fit)
    slope, _ = np.polyfit(t_fit, log_c, 1)
    if slope >= 0:
        return float("nan")
    return -math.log(2.0) / slope


def simulate_renal_saturation(
    drug_name: str,
    dose_mg: float,
    cl_nonrenal_L_per_h: float,
    vd_L: float,
    gfr_mL_per_min: float = 120.0,
    fu_plasma: float = 0.5,
    vmax_mg_per_h: float = 50.0,
    km_mg_L: float = 2.0,
    route: str = "iv",
    ka_per_h: float = 1.5,
    f_oral: float = 0.9,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> RenalSaturationResult:
    """Simulate nonlinear renal tubular secretion saturation.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Administered dose (mg).
    cl_nonrenal_L_per_h : float
        Non-renal (e.g. hepatic) clearance (L/h), must be > 0.
    vd_L : float
        Volume of distribution (L), must be > 0.
    gfr_mL_per_min : float
        Glomerular filtration rate (mL/min), default 120.
    fu_plasma : float
        Fraction unbound in plasma (0, 1].
    vmax_mg_per_h : float
        Maximum rate of active tubular secretion (mg/h).
    km_mg_L : float
        Michaelis constant for tubular secretion (mg/L).
    route : str
        "iv" or "oral".
    ka_per_h : float
        First-order absorption rate constant for oral route (h⁻¹).
    f_oral : float
        Oral bioavailability fraction (0, 1].
    t_end_h : float
        Simulation end time (h).
    dt_h : float
        Integration time step (h).

    Returns
    -------
    RenalSaturationResult
    """
    # --- Input validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_nonrenal_L_per_h <= 0:
        raise ValueError("cl_nonrenal_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if gfr_mL_per_min < 0:
        raise ValueError("gfr_mL_per_min must be >= 0")
    if not (0 < fu_plasma <= 1.0):
        raise ValueError("fu_plasma must be in (0, 1]")
    if vmax_mg_per_h <= 0:
        raise ValueError("vmax_mg_per_h must be > 0")
    if km_mg_L <= 0:
        raise ValueError("km_mg_L must be > 0")
    if route not in ("iv", "oral"):
        raise ValueError("route must be 'iv' or 'oral'")

    # --- Derived parameters ---
    # Filtration clearance (L/h)
    cl_filtration = (gfr_mL_per_min * 60.0 / 1000.0) * fu_plasma

    def cl_renal_at(c: float) -> float:
        """Total renal CL at plasma concentration c (mg/L)."""
        c_pos = max(c, 0.0)
        cl_sec = vmax_mg_per_h / (km_mg_L + c_pos)
        return cl_filtration + cl_sec

    # --- Time grid ---
    n_steps = max(int(t_end_h / dt_h) + 1, 2)
    times = np.linspace(0.0, t_end_h, n_steps)

    conc = np.zeros(n_steps)
    cl_renal_arr = np.zeros(n_steps)
    excreted = np.zeros(n_steps)

    # Regional excretion tracking per segment
    x_urine = 0.0  # cumulative excreted (mg)

    if route == "iv":
        c0 = dose_mg / vd_L
        conc[0] = c0
        cl_renal_arr[0] = cl_renal_at(c0)
        excreted[0] = 0.0

        for i in range(1, n_steps):
            c = conc[i - 1]
            cl_r = cl_renal_at(c)
            cl_total = cl_nonrenal_L_per_h + cl_r
            dC = -(cl_total * c / vd_L) * dt_h
            c_new = max(c + dC, 0.0)
            conc[i] = c_new
            cl_renal_arr[i] = cl_renal_at(c_new)
            x_urine += cl_r * c * dt_h
            excreted[i] = x_urine

    else:  # oral: 1-cpt with absorption depot
        dose_absorbed = dose_mg * f_oral
        m_gut = dose_absorbed
        c0 = 0.0
        conc[0] = c0
        cl_renal_arr[0] = cl_renal_at(c0)
        excreted[0] = 0.0

        for i in range(1, n_steps):
            c = conc[i - 1]
            cl_r = cl_renal_at(c)
            cl_total = cl_nonrenal_L_per_h + cl_r
            absorption_flux = ka_per_h * m_gut * dt_h
            dC = (absorption_flux / vd_L) - (cl_total * c / vd_L) * dt_h
            m_gut = max(m_gut - absorption_flux, 0.0)
            c_new = max(c + dC, 0.0)
            conc[i] = c_new
            cl_renal_arr[i] = cl_renal_at(c_new)
            x_urine += cl_r * c * dt_h
            excreted[i] = x_urine

    # --- Summary metrics ---
    auc = float(np_trapz(conc, times))

    t_half = _terminal_half_life(times, conc)

    cmax = float(conc.max())

    cl_renal_initial = cl_renal_at(conc[0])
    cl_renal_at_cmax = cl_renal_at(cmax)

    # Saturation: fraction of secretory capacity used at Cmax
    # pct = Cmax / (Km + Cmax) * 100
    saturation_pct = cmax / (km_mg_L + cmax) * 100.0 if cmax > 0 else 0.0

    dose_proportional = saturation_pct < 20.0

    fe_urine = float(excreted[-1]) / dose_mg
    fe_urine = min(fe_urine, 1.0)  # cap at 1 for numerical safety

    notes_parts = []
    if saturation_pct >= 80.0:
        notes_parts.append("Severe saturation: highly nonlinear kinetics expected.")
    elif saturation_pct >= 20.0:
        notes_parts.append("Partial saturation: dose-proportionality not expected.")
    else:
        notes_parts.append("Minimal saturation: near-linear kinetics.")
    if route == "oral":
        notes_parts.append(f"Oral route, F={f_oral:.2f}.")
    notes = " ".join(notes_parts)

    return RenalSaturationResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route=route,
        gfr_mL_per_min=gfr_mL_per_min,
        vmax_mg_per_h=vmax_mg_per_h,
        km_mg_L=km_mg_L,
        times_h=times.tolist(),
        c_plasma_mg_L=conc.tolist(),
        cl_renal_apparent_L_per_h=cl_renal_arr.tolist(),
        cumulative_excreted_mg=excreted.tolist(),
        t_half_h=t_half,
        auc_mg_h_per_L=auc,
        fe_urine=fe_urine,
        cl_renal_initial_L_per_h=cl_renal_initial,
        cl_renal_at_cmax_L_per_h=cl_renal_at_cmax,
        saturation_pct_at_cmax=saturation_pct,
        dose_proportional=dose_proportional,
        notes=notes,
    )


def dose_proportionality_check(
    drug_name: str,
    doses_mg: list[float],
    cl_nonrenal_L_per_h: float,
    vd_L: float,
    gfr_mL_per_min: float = 120.0,
    fu_plasma: float = 0.5,
    vmax_mg_per_h: float = 50.0,
    km_mg_L: float = 2.0,
) -> dict:
    """Check dose proportionality across multiple doses.

    Parameters
    ----------
    drug_name : str
        Drug name.
    doses_mg : list[float]
        List of doses to evaluate.
    cl_nonrenal_L_per_h : float
        Non-renal clearance (L/h).
    vd_L : float
        Volume of distribution (L).
    gfr_mL_per_min : float
        GFR (mL/min).
    fu_plasma : float
        Fraction unbound in plasma.
    vmax_mg_per_h : float
        Vmax for active secretion (mg/h).
    km_mg_L : float
        Km for active secretion (mg/L).

    Returns
    -------
    dict with keys: doses_mg, auc_values, dose_normalized_auc, proportional
    """
    auc_values = []
    for dose in doses_mg:
        result = simulate_renal_saturation(
            drug_name=drug_name,
            dose_mg=dose,
            cl_nonrenal_L_per_h=cl_nonrenal_L_per_h,
            vd_L=vd_L,
            gfr_mL_per_min=gfr_mL_per_min,
            fu_plasma=fu_plasma,
            vmax_mg_per_h=vmax_mg_per_h,
            km_mg_L=km_mg_L,
            route="iv",
        )
        auc_values.append(result.auc_mg_h_per_L)

    doses_arr = np.array(doses_mg, dtype=float)
    auc_arr = np.array(auc_values, dtype=float)
    dose_norm_auc = auc_arr / doses_arr  # AUC / dose

    # Proportional if max deviation from mean < 20%
    mean_norm = float(dose_norm_auc.mean())
    if mean_norm > 0:
        max_dev = float(np.abs(dose_norm_auc - mean_norm).max() / mean_norm)
    else:
        max_dev = 0.0
    proportional = max_dev < 0.20

    return {
        "doses_mg": list(doses_mg),
        "auc_values": auc_values,
        "dose_normalized_auc": dose_norm_auc.tolist(),
        "proportional": proportional,
    }
