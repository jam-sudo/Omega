"""Sublingual rapid absorption model for fast-onset drugs.

Models rapid sublingual drug absorption for fast-onset drugs such as
nitroglycerin, buprenorphine, and fentanyl citrate.  Focuses on
time-to-onset (time to reach minimum effective concentration) and
therapeutic window characterisation.

Unlike the general sublingual_pbpk module, this model targets ultra-rapid
absorption kinetics with logP-dependent absorption rate and simplified
2-state pharmacokinetics.

References
----------
- Weinberg DS et al., Clin Pharmacol Ther. 1988;44(3):335-42 (sublingual NTG)
- Mendelson J et al., Clin Pharmacol Ther. 1997;62(4):410-8 (sublingual buprenorphine)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SublingualRapidResult:
    """Result container for sublingual rapid absorption simulation."""

    drug_name: str
    dose_mg: float
    logP: float
    times_h: list[float] = field(default_factory=list)
    c_plasma_mg_L: list[float] = field(default_factory=list)
    c_sublingual_mg: list[float] = field(default_factory=list)
    cmax_mg_L: float = 0.0
    tmax_h: float = 0.0
    auc_mg_h_per_L: float = 0.0
    t_onset_h: float = 0.0
    t_offset_h: float = 0.0
    time_in_therapeutic_window_h: float = 0.0
    bioavailability_pct: float = 0.0
    ka_sublingual_per_h: float = 0.0
    notes: str = ""


def simulate_sublingual_rapid(
    drug_name: str,
    dose_mg: float,
    logP: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    mec_mg_L: float = 0.001,
    mtc_mg_L: float | None = None,
    t_end_h: float = 4.0,
    dt_h: float = 0.01,
) -> SublingualRapidResult:
    """Simulate rapid sublingual drug absorption.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Administered dose in mg.
    logP : float
        Octanol-water partition coefficient (log scale).
    cl_sys_L_per_h : float
        Systemic clearance (L/h).
    vd_sys_L : float
        Volume of distribution (L).
    mec_mg_L : float
        Minimum effective concentration (mg/L).
    mtc_mg_L : float or None
        Maximum tolerated concentration (mg/L). None = no upper bound.
    t_end_h : float
        Simulation end time (h).
    dt_h : float
        Time step (h).

    Returns
    -------
    SublingualRapidResult
    """
    # --- validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    if mec_mg_L <= 0:
        raise ValueError("mec_mg_L must be > 0")

    # --- absorption rate (logP-dependent) ---
    if logP > 0:
        ka_sl = 0.5 + logP * 1.5
    else:
        ka_sl = 0.5
    ka_sl = max(0.5, min(15.0, ka_sl))

    # --- sublingual bioavailability ---
    if logP > 0:
        f_sl = 0.6 + logP * 0.05
    else:
        f_sl = 0.6
    f_sl = max(0.3, min(0.95, f_sl))

    # --- elimination rate ---
    k_el = cl_sys_L_per_h / vd_sys_L

    # --- initial conditions ---
    a_sl = dose_mg * f_sl  # sublingual amount (mg)
    c_plasma = 0.0  # plasma concentration (mg/L)

    # --- storage ---
    n_steps = int(t_end_h / dt_h) + 1
    times: list[float] = []
    c_plasma_list: list[float] = []
    c_sublingual_list: list[float] = []

    cmax = 0.0
    tmax = 0.0

    for i in range(n_steps):
        t = i * dt_h
        times.append(t)
        c_plasma_list.append(c_plasma)
        c_sublingual_list.append(a_sl)

        if c_plasma > cmax:
            cmax = c_plasma
            tmax = t

        # Forward Euler
        da_sl = -ka_sl * a_sl
        dc_plasma = ka_sl * a_sl / vd_sys_L - k_el * c_plasma

        a_sl += da_sl * dt_h
        c_plasma += dc_plasma * dt_h

    # --- AUC (manual trapezoidal) ---
    auc = sum(
        0.5 * (c_plasma_list[i] + c_plasma_list[i - 1]) * (times[i] - times[i - 1])
        for i in range(1, len(times))
    )

    # --- onset / offset ---
    t_onset = t_end_h
    t_offset = 0.0
    for i, cp in enumerate(c_plasma_list):
        if cp >= mec_mg_L:
            t_onset = times[i]
            break

    for i in range(len(c_plasma_list) - 1, -1, -1):
        if c_plasma_list[i] >= mec_mg_L:
            t_offset = times[i]
            break

    # --- therapeutic window duration ---
    time_in_window = 0.0
    for i in range(1, len(times)):
        cp = c_plasma_list[i]
        above_mec = cp >= mec_mg_L
        below_mtc = True if mtc_mg_L is None else cp <= mtc_mg_L
        if above_mec and below_mtc:
            time_in_window += times[i] - times[i - 1]

    bioavailability_pct = min(100.0, f_sl * 100.0)

    notes_parts: list[str] = []
    notes_parts.append(f"ka_sl={ka_sl:.2f}/h, f_sl={f_sl:.2f}")
    if mtc_mg_L is not None:
        notes_parts.append(f"mtc={mtc_mg_L} mg/L")

    return SublingualRapidResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        logP=logP,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        c_sublingual_mg=c_sublingual_list,
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc_mg_h_per_L=auc,
        t_onset_h=t_onset,
        t_offset_h=t_offset,
        time_in_therapeutic_window_h=time_in_window,
        bioavailability_pct=bioavailability_pct,
        ka_sublingual_per_h=ka_sl,
        notes="; ".join(notes_parts),
    )
