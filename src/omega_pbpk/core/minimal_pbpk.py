"""Minimal PBPK (mPBPK) model with 3 tissue compartments + systemic blood.

A simplified whole-body PBPK alternative to full 35-compartment models,
useful for rapid predictions when full PBPK is unnecessary.

Compartments
------------
1. Richly-perfused tissue (RPT): liver, kidney, heart, lung, brain
2. Slowly-perfused tissue (SPT): muscle, skin, bone
3. Fat compartment
4. Central plasma (blood)

References
----------
- Cao Y & Bhatt DK, CPT Pharmacometrics Syst Pharmacol. 2016
- Berezhkovskiy LM, J Pharm Sci. 2004;93:1628–40
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz

# ---------------------------------------------------------------------------
# Physiological constants (70 kg reference adult)
# ---------------------------------------------------------------------------
Q_CARDIAC_L_H = 330.0  # cardiac output, L/h (5.5 L/min)
F_RPT = 0.65  # fraction of CO to RPT
F_SPT = 0.30  # fraction of CO to SPT
F_FAT = 0.05  # fraction of CO to fat

Q_RPT = Q_CARDIAC_L_H * F_RPT
Q_SPT = Q_CARDIAC_L_H * F_SPT
Q_FAT = Q_CARDIAC_L_H * F_FAT

V_BLOOD = 5.0  # L
V_RPT = 3.5  # L
V_SPT = 22.0  # L
V_FAT = 15.0  # L


@dataclass(frozen=True)
class MinimalPBPKResult:
    """Results from minimal PBPK simulation.

    Attributes
    ----------
    drug_name : Name of the drug.
    route : Administration route ('iv' or 'oral').
    dose_mg : Dose (mg).
    fup : Fraction unbound in plasma.
    times_h : Simulation time points (h).
    c_blood_mg_L : Blood concentration time-course (mg/L).
    c_rpt_mg_L : RPT concentration time-course (mg/L).
    c_spt_mg_L : SPT concentration time-course (mg/L).
    c_fat_mg_L : Fat concentration time-course (mg/L).
    cmax_blood : Peak blood concentration (mg/L).
    tmax_h : Time of peak blood concentration (h).
    auc_blood : AUC blood 0→t (mg·h/L).
    vd_ss_apparent_L : Apparent steady-state volume of distribution (L).
    t_half_h : Terminal half-life (h).
    tissue_distribution : Peak Cmax for each tissue.
    notes : Summary text.
    """

    drug_name: str
    route: str
    dose_mg: float
    fup: float
    times_h: list[float]
    c_blood_mg_L: list[float]
    c_rpt_mg_L: list[float]
    c_spt_mg_L: list[float]
    c_fat_mg_L: list[float]
    cmax_blood: float
    tmax_h: float
    auc_blood: float
    vd_ss_apparent_L: float
    t_half_h: float
    tissue_distribution: dict[str, float]
    notes: str


def vss_calculator(
    fup: float = 0.1,
    kp_rpt: float = 2.0,
    kp_spt: float = 1.0,
    kp_fat: float = 5.0,
) -> float:
    """Calculate steady-state volume of distribution from Kp values.

    Vss = V_blood + kp_rpt*V_rpt + kp_spt*V_spt + kp_fat*V_fat

    Parameters
    ----------
    fup : Fraction unbound in plasma (unused in formula but kept for API).
    kp_rpt : Tissue:plasma Kp for richly-perfused tissue.
    kp_spt : Tissue:plasma Kp for slowly-perfused tissue.
    kp_fat : Tissue:plasma Kp for fat.

    Returns
    -------
    Vss in litres.
    """
    return V_BLOOD + kp_rpt * V_RPT + kp_spt * V_SPT + kp_fat * V_FAT


def _estimate_terminal_half_life(times: np.ndarray, conc: np.ndarray) -> float:
    """Estimate terminal half-life from the last 30% of blood concentration."""
    n = len(conc)
    start = int(n * 0.7)
    t_seg = times[start:]
    c_seg = conc[start:]
    # Filter positive concentrations for log
    mask = c_seg > 0
    if np.sum(mask) < 2:
        return 0.0
    log_c = np.log(c_seg[mask])
    t_filt = t_seg[mask]
    if len(t_filt) < 2:
        return 0.0
    # Linear regression on log(C) vs t
    slope, _ = np.polyfit(t_filt, log_c, 1)
    if slope >= 0:
        return 0.0
    return float(np.log(2) / (-slope))


def simulate_minimal_pbpk(
    drug_name: str,
    dose_mg: float,
    cl_hepatic_L_per_h: float,
    fup: float = 0.1,
    kp_rpt: float = 2.0,
    kp_spt: float = 1.0,
    kp_fat: float = 5.0,
    route: str = "iv",
    ka_per_h: float = 1.0,
    f_oral: float = 1.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> MinimalPBPKResult:
    """Run a minimal 3-compartment PBPK simulation.

    Parameters
    ----------
    drug_name : Drug name.
    dose_mg : Dose (mg).
    cl_hepatic_L_per_h : Hepatic clearance (L/h).
    fup : Fraction unbound in plasma (0, 1].
    kp_rpt : Tissue:plasma partition coefficient for RPT.
    kp_spt : Tissue:plasma partition coefficient for SPT.
    kp_fat : Tissue:plasma partition coefficient for fat.
    route : 'iv' or 'oral'.
    ka_per_h : Oral absorption rate constant (h⁻¹).
    f_oral : Oral bioavailability fraction.
    t_end_h : Simulation duration (h).
    dt_h : Time step (h).

    Returns
    -------
    MinimalPBPKResult

    Raises
    ------
    ValueError : If any parameter is invalid.
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_hepatic_L_per_h <= 0:
        raise ValueError("cl_hepatic_L_per_h must be > 0")
    if fup <= 0 or fup > 1:
        raise ValueError("fup must be in (0, 1]")
    if kp_rpt <= 0 or kp_spt <= 0 or kp_fat <= 0:
        raise ValueError("All Kp values must be > 0")

    # Determine stable internal time step (max rate ~ sum(Q/V) for blood)
    max_rate = (Q_RPT + Q_SPT + Q_FAT + cl_hepatic_L_per_h) / V_BLOOD
    dt_internal = min(dt_h, 0.5 / max_rate)

    n_steps = max(int(t_end_h / dt_h), 1)
    n_sub = max(int(np.ceil(dt_h / dt_internal)), 1)
    dt_sub = dt_h / n_sub

    t = np.linspace(0.0, t_end_h, n_steps + 1)

    c_blood = np.zeros(n_steps + 1)
    c_rpt = np.zeros(n_steps + 1)
    c_spt = np.zeros(n_steps + 1)
    c_fat = np.zeros(n_steps + 1)

    # Oral absorption: gut compartment amount
    a_gut = 0.0
    if route.lower() == "iv":
        c_blood[0] = dose_mg / V_BLOOD
    else:
        a_gut = dose_mg * f_oral

    cl = cl_hepatic_L_per_h

    for i in range(n_steps):
        cb = c_blood[i]
        cr = c_rpt[i]
        cs = c_spt[i]
        cf = c_fat[i]

        for _s in range(n_sub):
            # Absorption from gut (oral)
            absorption = 0.0
            if route.lower() == "oral" and a_gut > 0:
                absorbed = ka_per_h * a_gut * dt_sub
                absorbed = min(absorbed, a_gut)
                a_gut -= absorbed
                absorption = absorbed / (V_BLOOD * dt_sub)

            flux_rpt = Q_RPT * (cb - cr / kp_rpt)
            flux_spt = Q_SPT * (cb - cs / kp_spt)
            flux_fat = Q_FAT * (cb - cf / kp_fat)

            cr = max(0.0, cr + flux_rpt / V_RPT * dt_sub)
            cs = max(0.0, cs + flux_spt / V_SPT * dt_sub)
            cf = max(0.0, cf + flux_fat / V_FAT * dt_sub)
            dc_blood = (-flux_rpt - flux_spt - flux_fat - cl * cb) / V_BLOOD + absorption
            cb = max(0.0, cb + dc_blood * dt_sub)

        c_blood[i + 1] = cb
        c_rpt[i + 1] = cr
        c_spt[i + 1] = cs
        c_fat[i + 1] = cf

    cmax_blood = float(np.max(c_blood))
    tmax = float(t[int(np.argmax(c_blood))])
    auc = float(np_trapz(c_blood, t))
    vss = vss_calculator(fup, kp_rpt, kp_spt, kp_fat)
    t_half = _estimate_terminal_half_life(t, c_blood)

    tissue_dist = {
        "rpt": float(np.max(c_rpt)),
        "spt": float(np.max(c_spt)),
        "fat": float(np.max(c_fat)),
    }

    notes = (
        f"Minimal PBPK: {drug_name}, {dose_mg}mg {route}. "
        f"Cmax={cmax_blood:.4f} mg/L at {tmax:.2f}h. "
        f"AUC={auc:.3f} mg·h/L. Vss={vss:.1f}L. t½={t_half:.2f}h."
    )

    return MinimalPBPKResult(
        drug_name=drug_name,
        route=route,
        dose_mg=dose_mg,
        fup=fup,
        times_h=t.tolist(),
        c_blood_mg_L=c_blood.tolist(),
        c_rpt_mg_L=c_rpt.tolist(),
        c_spt_mg_L=c_spt.tolist(),
        c_fat_mg_L=c_fat.tolist(),
        cmax_blood=cmax_blood,
        tmax_h=tmax,
        auc_blood=auc,
        vd_ss_apparent_L=vss,
        t_half_h=t_half,
        tissue_distribution=tissue_dist,
        notes=notes,
    )


__all__ = [
    "MinimalPBPKResult",
    "simulate_minimal_pbpk",
    "vss_calculator",
]
