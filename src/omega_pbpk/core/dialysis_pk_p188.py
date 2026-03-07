"""Peritoneal dialysis and hemodialysis PK simulation with dose supplementation — Phase 188."""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "DialysisSessionResult",
    "simulate_dialysis_session",
    "compare_dialysis_modalities",
]


@dataclass
class DialysisSessionResult:
    """Result from dialysis session PK simulation."""

    drug_name: str
    dose_mg: float
    modality: str  # "HD", "PD", or "none"
    vd_L: float
    cl_dialysis_L_per_h: float
    times_h: list
    c_plasma_mg_L: list
    cmax_mg_L: float
    cmin_pre_dialysis_mg_L: float
    cmin_post_dialysis_mg_L: float
    fraction_removed_per_session: float
    supplement_dose_mg: float
    notes: str


def simulate_dialysis_session(
    drug_name: str,
    dose_mg: float,
    cl_renal_L_per_h: float,
    vd_L: float,
    cl_dialysis_L_per_h: float,
    session_duration_h: float,
    interdialysis_h: float,
    n_sessions: int,
    modality: str = "HD",
    target_Cmin_mg_L: float = 0.0,
    dt_h: float = 0.1,
) -> DialysisSessionResult:
    """Simulate drug PK during repeated dialysis sessions.

    The simulation alternates between:
    - Interdialytic phase: C(t) = C0 * exp(-ke_off * t)
      where ke_off = cl_renal / vd_L
    - Dialysis session: C(t) = C0 * exp(-ke_on * t)
      where ke_on = (cl_renal + cl_dialysis) / vd_L

    Starts with an IV bolus dose at t=0, then enters the first
    interdialytic phase, then alternates session / off phases.

    Args:
        drug_name: Drug name.
        dose_mg: Initial IV bolus dose in mg. Must be > 0.
        cl_renal_L_per_h: Residual renal clearance (L/h). Must be >= 0.
        vd_L: Volume of distribution (L). Must be > 0.
        cl_dialysis_L_per_h: Dialysis clearance (L/h). Must be >= 0.
        session_duration_h: Duration of each dialysis session (h).
        interdialysis_h: Duration between sessions (off-dialysis) (h).
        n_sessions: Number of dialysis sessions to simulate.
        modality: Modality label ("HD", "PD", or "none").
        target_Cmin_mg_L: Target minimum concentration (mg/L).
            If post-dialysis conc < target, a supplemental dose is recommended.
        dt_h: Time step for output resolution (h).

    Returns:
        DialysisSessionResult

    Raises:
        ValueError: On invalid inputs.
    """
    if dose_mg <= 0.0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if vd_L <= 0.0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if cl_renal_L_per_h < 0.0:
        raise ValueError(f"cl_renal_L_per_h must be >= 0, got {cl_renal_L_per_h}")
    if cl_dialysis_L_per_h < 0.0:
        raise ValueError(f"cl_dialysis_L_per_h must be >= 0, got {cl_dialysis_L_per_h}")
    if session_duration_h <= 0.0:
        raise ValueError(f"session_duration_h must be > 0, got {session_duration_h}")
    if interdialysis_h <= 0.0:
        raise ValueError(f"interdialysis_h must be > 0, got {interdialysis_h}")
    if n_sessions < 1:
        raise ValueError(f"n_sessions must be >= 1, got {n_sessions}")

    ke_off = cl_renal_L_per_h / vd_L  # elimination rate off-dialysis (1/h)
    ke_on = (cl_renal_L_per_h + cl_dialysis_L_per_h) / vd_L  # on-dialysis

    # Initial concentration after IV bolus
    c_current = dose_mg / vd_L

    times_h: list[float] = []
    c_plasma: list[float] = []
    t_abs = 0.0  # absolute time

    # Helper to append a phase
    def _append_phase(c0: float, ke: float, duration_h: float) -> float:
        """Append concentration profile for one phase, return end concentration."""
        nonlocal t_abs
        n = max(1, int(round(duration_h / dt_h)))
        for i in range(n + 1):
            frac = i / n
            t_local = frac * duration_h
            c = c0 * math.exp(-ke * t_local)
            t_point = t_abs + t_local
            # Avoid duplicate time points at phase boundaries
            if times_h and abs(t_point - times_h[-1]) < 1e-9:
                continue
            times_h.append(t_point)
            c_plasma.append(c)
        t_abs += duration_h
        return c0 * math.exp(-ke * duration_h)

    # Record initial point
    times_h.append(0.0)
    c_plasma.append(c_current)

    cmin_pre_list: list[float] = []
    cmin_post_list: list[float] = []

    # Simulate: for each session, do interdialytic phase then session
    for _ in range(n_sessions):
        # Interdialytic phase
        c_pre = c_current * math.exp(-ke_off * interdialysis_h)
        _append_phase(c_current, ke_off, interdialysis_h)
        cmin_pre_list.append(c_pre)
        c_current = c_pre

        # Dialysis session
        c_post = c_current * math.exp(-ke_on * session_duration_h)
        _append_phase(c_current, ke_on, session_duration_h)
        cmin_post_list.append(c_post)
        c_current = c_post

    cmax = max(c_plasma) if c_plasma else 0.0
    cmin_pre = min(cmin_pre_list) if cmin_pre_list else c_plasma[-1] if c_plasma else 0.0
    cmin_post = min(cmin_post_list) if cmin_post_list else c_plasma[-1] if c_plasma else 0.0

    # Fraction removed per session: (C_pre - C_post) / C_pre for last session
    # Use last session values
    if cmin_pre_list and cmin_post_list:
        c_pre_last = cmin_pre_list[-1]
        c_post_last = cmin_post_list[-1]
        if c_pre_last > 1e-12:
            fraction_removed = (c_pre_last - c_post_last) / c_pre_last
        else:
            fraction_removed = 0.0
        fraction_removed = max(0.0, min(1.0, fraction_removed))
    else:
        fraction_removed = 0.0

    # Supplemental dose recommendation
    supplement_dose = 0.0
    if target_Cmin_mg_L > 0.0 and cmin_post < target_Cmin_mg_L:
        deficit_mg_L = target_Cmin_mg_L - cmin_post
        supplement_dose = deficit_mg_L * vd_L

    notes = (
        f"Modality={modality}; "
        f"ke_off={ke_off:.4f}/h; ke_on={ke_on:.4f}/h; "
        f"n_sessions={n_sessions}; "
        f"fraction_removed/session={fraction_removed:.3f}; "
        f"Cmax={cmax:.3f} mg/L; "
        f"Cmin_pre={cmin_pre:.3f} mg/L; Cmin_post={cmin_post:.3f} mg/L"
    )

    return DialysisSessionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        modality=modality,
        vd_L=vd_L,
        cl_dialysis_L_per_h=cl_dialysis_L_per_h,
        times_h=times_h,
        c_plasma_mg_L=c_plasma,
        cmax_mg_L=cmax,
        cmin_pre_dialysis_mg_L=cmin_pre,
        cmin_post_dialysis_mg_L=cmin_post,
        fraction_removed_per_session=fraction_removed,
        supplement_dose_mg=supplement_dose,
        notes=notes,
    )


def compare_dialysis_modalities(
    drug_name: str,
    dose_mg: float,
    vd_L: float,
    cl_renal_L_per_h: float = 0.0,
    hd_cl_L_per_h: float = 15.0,
    pd_cl_L_per_h: float = 3.0,
    hd_session_h: float = 4.0,
    pd_session_h: float = 8.0,
    interdialysis_h: float = 48.0,
    n_sessions: int = 3,
    target_Cmin_mg_L: float = 0.0,
) -> list[DialysisSessionResult]:
    """Compare drug PK across hemodialysis (HD), peritoneal dialysis (PD), and no dialysis.

    Args:
        drug_name: Drug name.
        dose_mg: Initial dose in mg.
        vd_L: Volume of distribution (L).
        cl_renal_L_per_h: Residual renal clearance (L/h).
        hd_cl_L_per_h: Hemodialysis clearance (L/h).
        pd_cl_L_per_h: Peritoneal dialysis clearance (L/h).
        hd_session_h: HD session duration (h).
        pd_session_h: PD session duration (h).
        interdialysis_h: Interdialytic period (h).
        n_sessions: Number of sessions.
        target_Cmin_mg_L: Target minimum concentration (mg/L).

    Returns:
        List of 3 DialysisSessionResult (HD, PD, no-dialysis).
    """
    hd_result = simulate_dialysis_session(
        drug_name=drug_name,
        dose_mg=dose_mg,
        cl_renal_L_per_h=cl_renal_L_per_h,
        vd_L=vd_L,
        cl_dialysis_L_per_h=hd_cl_L_per_h,
        session_duration_h=hd_session_h,
        interdialysis_h=interdialysis_h,
        n_sessions=n_sessions,
        modality="HD",
        target_Cmin_mg_L=target_Cmin_mg_L,
    )

    pd_result = simulate_dialysis_session(
        drug_name=drug_name,
        dose_mg=dose_mg,
        cl_renal_L_per_h=cl_renal_L_per_h,
        vd_L=vd_L,
        cl_dialysis_L_per_h=pd_cl_L_per_h,
        session_duration_h=pd_session_h,
        interdialysis_h=interdialysis_h,
        n_sessions=n_sessions,
        modality="PD",
        target_Cmin_mg_L=target_Cmin_mg_L,
    )

    # No dialysis baseline (zero dialysis clearance, minimal session duration)
    no_dialysis_result = simulate_dialysis_session(
        drug_name=drug_name,
        dose_mg=dose_mg,
        cl_renal_L_per_h=cl_renal_L_per_h,
        vd_L=vd_L,
        cl_dialysis_L_per_h=0.0,
        session_duration_h=hd_session_h,
        interdialysis_h=interdialysis_h,
        n_sessions=n_sessions,
        modality="none",
        target_Cmin_mg_L=target_Cmin_mg_L,
    )

    return [hd_result, pd_result, no_dialysis_result]
