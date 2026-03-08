"""Biliary excretion and enterohepatic circulation (EHC) model.

Models the biliary excretion of drugs and subsequent re-absorption via
enterohepatic circulation.  EHC causes multiple plasma concentration peaks
and prolongs drug half-life due to periodic gallbladder emptying that
recycles drug back into the gut lumen.

References
----------
- Roberts MS et al., J Pharmacokinet Biopharm. 2002;30(2):97-127 (EHC modelling)
- Gao Y et al., Xenobiotica. 2014;44(1):1-14 (biliary excretion)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class EnterohepatiCResult:
    """Result container for enterohepatic circulation simulation."""

    drug_name: str
    dose_mg: float
    times_h: list[float] = field(default_factory=list)
    c_plasma_mg_L: list[float] = field(default_factory=list)
    c_gut_mg: list[float] = field(default_factory=list)
    c_bile_mg: list[float] = field(default_factory=list)
    cmax_mg_L: float = 0.0
    tmax_h: float = 0.0
    auc_mg_h_per_L: float = 0.0
    t_half_h: float = 0.0
    n_secondary_peaks: int = 0
    ehc_fraction: float = 0.0
    bile_excretion_pct: float = 0.0
    notes: str = ""


def simulate_enterohepatic(
    drug_name: str,
    dose_mg: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    f_bile: float,
    k_bile_per_h: float = 0.1,
    k_reabs_per_h: float = 0.5,
    t_gallbladder_h: float = 6.0,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
) -> EnterohepatiCResult:
    """Simulate enterohepatic circulation of a drug.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Oral dose in mg.
    cl_sys_L_per_h : float
        Systemic clearance (L/h).
    vd_sys_L : float
        Volume of distribution (L).
    f_bile : float
        Fraction of eliminated drug that enters bile (0–1 exclusive).
    k_bile_per_h : float
        Biliary excretion rate constant (1/h).
    k_reabs_per_h : float
        Re-absorption rate constant from bile intestine (1/h).  Unused in
        simplified model but kept for API compatibility.
    t_gallbladder_h : float
        Gallbladder emptying interval (h).
    t_end_h : float
        Simulation end time (h).
    dt_h : float
        Time step (h).

    Returns
    -------
    EnterohepatiCResult
    """
    # --- validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    if f_bile < 0 or f_bile >= 1:
        raise ValueError("f_bile must be in [0, 1)")
    if t_gallbladder_h <= 0:
        raise ValueError("t_gallbladder_h must be > 0")

    # --- constants ---
    ka = 1.0  # oral absorption rate (1/h)
    k_el = cl_sys_L_per_h / vd_sys_L  # total elimination rate
    k_nonbile_el = k_el * (1.0 - f_bile)

    # --- state variables ---
    a_gut = dose_mg  # drug amount in gut lumen (mg)
    c_plasma = 0.0  # plasma concentration (mg/L)
    a_gallbladder = 0.0  # drug amount in gallbladder (mg)

    n_steps = int(math.ceil(t_end_h / dt_h)) + 1

    times: list[float] = []
    c_plasma_list: list[float] = []
    c_gut_list: list[float] = []
    c_bile_list: list[float] = []

    for i in range(n_steps):
        t = i * dt_h

        times.append(t)
        c_plasma_list.append(max(0.0, c_plasma))
        c_gut_list.append(max(0.0, a_gut))
        c_bile_list.append(max(0.0, a_gallbladder))

        # --- gallbladder emptying (pulsatile) ---
        if i > 0 and t_gallbladder_h > 0:
            if (t % t_gallbladder_h) < dt_h:
                a_gut += a_gallbladder
                a_gallbladder = 0.0

        # --- ODEs (Forward Euler) ---
        da_gut = -ka * a_gut
        dc_plasma = (
            ka * a_gut / vd_sys_L - k_nonbile_el * c_plasma - k_bile_per_h * f_bile * c_plasma
        )
        da_gb = k_bile_per_h * f_bile * c_plasma * vd_sys_L

        a_gut += da_gut * dt_h
        c_plasma += dc_plasma * dt_h
        a_gallbladder += da_gb * dt_h

        # clamp
        a_gut = max(0.0, a_gut)
        c_plasma = max(0.0, c_plasma)
        a_gallbladder = max(0.0, a_gallbladder)

    # --- derived metrics ---
    # Cmax / Tmax
    cmax = 0.0
    tmax_idx = 0
    for idx, c in enumerate(c_plasma_list):
        if c > cmax:
            cmax = c
            tmax_idx = idx
    tmax = times[tmax_idx]

    # AUC (trapezoidal)
    auc = sum(
        0.5 * (c_plasma_list[j] + c_plasma_list[j - 1]) * (times[j] - times[j - 1])
        for j in range(1, len(times))
    )

    # t_half
    t_half = 0.693 / k_el if k_el > 0 else 0.0

    # secondary peaks (after first peak)
    n_secondary = 0
    for idx in range(tmax_idx + 2, len(c_plasma_list) - 1):
        if (
            c_plasma_list[idx] > c_plasma_list[idx - 1]
            and c_plasma_list[idx] > c_plasma_list[idx + 1]
        ):
            n_secondary += 1

    # bile excretion pct
    bile_excretion_pct = f_bile * 100.0

    return EnterohepatiCResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        c_gut_mg=c_gut_list,
        c_bile_mg=c_bile_list,
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc_mg_h_per_L=auc,
        t_half_h=t_half,
        n_secondary_peaks=n_secondary,
        ehc_fraction=f_bile,
        bile_excretion_pct=bile_excretion_pct,
        notes="Enterohepatic circulation simulation with periodic gallbladder emptying.",
    )
