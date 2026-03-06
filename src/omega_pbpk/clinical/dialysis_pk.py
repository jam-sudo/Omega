"""Phase 220 — Dialysis / Haemodialysis PK.

Models drug removal during haemodialysis sessions.  Important for CKD
patients on dialysis who require post-dialysis dose supplementation.

Model
-----
Simple 1-compartment ODE with an added dialysis clearance term that is
active only during scheduled sessions:

    dC/dt = -(CL_residual + CL_dialysis(t)) / Vd * C

CL_dialysis = Qb * E  where E = fup * sieving_coeff  (capped at 0.95).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def dialysis_extraction(
    fup: float,
    sieving_coeff: float,
    qb_L_per_h: float = 18.0,
) -> float:
    """Return the dialysis extraction ratio E and clearance CL_dialysis.

    Parameters
    ----------
    fup:
        Unbound fraction in plasma (0, 1].
    sieving_coeff:
        Membrane sieving coefficient [0, 1].
    qb_L_per_h:
        Blood flow through dialyzer (L/h).  Typical 300 mL/min = 18 L/h.

    Returns
    -------
    Extraction ratio E in [0, 0.95].
    """
    if not (0 < fup <= 1):
        raise ValueError("fup must be in (0, 1]")
    if not (0 <= sieving_coeff <= 1):
        raise ValueError("sieving_coeff must be in [0, 1]")
    if qb_L_per_h <= 0:
        raise ValueError("qb_L_per_h must be > 0")

    E = fup * sieving_coeff
    return min(E, 0.95)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class DialysisPKResult:
    """Result of a multi-session haemodialysis PK simulation."""

    drug_name: str
    dose_mg: float
    fup: float
    sieving_coeff: float

    times_h: list[float] = field(default_factory=list)
    c_plasma_mg_L: list[float] = field(default_factory=list)
    is_dialysis_time: list[bool] = field(default_factory=list)

    cmax_mg_L: float = 0.0
    cmin_predialysis_mg_L: float = 0.0
    fraction_removed_per_session: float = 0.0
    cl_dialysis_L_per_h: float = 0.0
    supplemental_dose_mg: float = 0.0
    notes: str = ""


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def simulate_dialysis_pk(
    drug_name: str,
    dose_mg: float,
    cl_residual_L_per_h: float,
    vd_L: float,
    fup: float,
    sieving_coeff: float = 0.8,
    qb_L_per_h: float = 18.0,
    session_duration_h: float = 4.0,
    interdialytic_interval_h: float = 48.0,
    n_sessions: int = 3,
    route: str = "iv",
    t_end_h: float | None = None,
    dt_h: float = 0.1,
) -> DialysisPKResult:
    """Simulate multi-session haemodialysis PK.

    Parameters
    ----------
    drug_name:
        Identifier string.
    dose_mg:
        Initial dose (mg).
    cl_residual_L_per_h:
        Residual (non-dialysis) clearance (L/h).
    vd_L:
        Volume of distribution (L).
    fup:
        Unbound plasma fraction (0, 1].
    sieving_coeff:
        Dialyser membrane sieving coefficient [0, 1].
    qb_L_per_h:
        Blood flow through dialyser (L/h).
    session_duration_h:
        Duration of each dialysis session (h).
    interdialytic_interval_h:
        Gap between end of one session and start of next (h).
    n_sessions:
        Number of dialysis sessions.
    route:
        ``"iv"`` (instantaneous bolus) — only supported route.
    t_end_h:
        Total simulation time.  Defaults to
        ``n_sessions * (session_duration_h + interdialytic_interval_h)``.
    dt_h:
        Integration step (h).

    Returns
    -------
    :class:`DialysisPKResult`
    """
    # --- input validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_residual_L_per_h < 0:
        raise ValueError("cl_residual_L_per_h must be >= 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if not (0 < fup <= 1):
        raise ValueError("fup must be in (0, 1]")
    if not (0 <= sieving_coeff <= 1):
        raise ValueError("sieving_coeff must be in [0, 1]")
    if qb_L_per_h <= 0:
        raise ValueError("qb_L_per_h must be > 0")
    if session_duration_h <= 0:
        raise ValueError("session_duration_h must be > 0")
    if interdialytic_interval_h <= 0:
        raise ValueError("interdialytic_interval_h must be > 0")
    if n_sessions < 1:
        raise ValueError("n_sessions must be >= 1")

    cycle_h = session_duration_h + interdialytic_interval_h
    if t_end_h is None:
        t_end_h = n_sessions * cycle_h

    # Dialysis parameters
    E = dialysis_extraction(fup, sieving_coeff, qb_L_per_h)
    cl_dialysis = qb_L_per_h * E

    # Build list of session windows: (start, end)
    sessions: list[tuple[float, float]] = []
    t_sess_start = 0.0
    for _ in range(n_sessions):
        sessions.append((t_sess_start, t_sess_start + session_duration_h))
        t_sess_start += cycle_h

    def _in_dialysis(t: float) -> bool:
        for s, e in sessions:
            if s <= t < e:
                return True
        return False

    # Forward Euler integration
    n_steps = int(math.ceil(t_end_h / dt_h))
    times: list[float] = []
    concs: list[float] = []
    is_dial: list[bool] = []

    # IV bolus: C0 = dose / Vd
    C = dose_mg / vd_L

    t = 0.0
    for _ in range(n_steps + 1):
        times.append(round(t, 6))
        concs.append(max(C, 0.0))
        is_dial.append(_in_dialysis(t))

        if t >= t_end_h:
            break

        cl_total = cl_residual_L_per_h + (cl_dialysis if _in_dialysis(t) else 0.0)
        _ke = cl_total / vd_L
        dC = -_ke * C * dt_h
        C = max(C + dC, 0.0)
        t = round(t + dt_h, 6)

    cmax = max(concs) if concs else 0.0

    # Pre-dialysis troughs: concentration just before each session start
    cmin_values: list[float] = []
    for s_start, _ in sessions:
        # Find the index closest to (but not exceeding) s_start
        idx = None
        for i, tt in enumerate(times):
            if tt <= s_start:
                idx = i
            else:
                break
        if idx is not None:
            cmin_values.append(concs[idx])

    cmin_predialysis = min(cmin_values) if cmin_values else 0.0

    # Fraction removed per session (average)
    frac_list: list[float] = []
    for s_start, s_end in sessions:
        # C before session
        idx_before = None
        for i, tt in enumerate(times):
            if tt <= s_start:
                idx_before = i
            else:
                break
        # C after session
        idx_after = None
        for i, tt in enumerate(times):
            if tt <= s_end:
                idx_after = i
            else:
                break
        if idx_before is not None and idx_after is not None:
            c_before = concs[idx_before]
            c_after = concs[idx_after]
            if c_before > 0:
                frac_list.append((c_before - c_after) / c_before)

    frac_removed = sum(frac_list) / len(frac_list) if frac_list else 0.0
    frac_removed = max(0.0, min(frac_removed, 1.0))

    # Supplemental dose
    supp = supplemental_dose(
        drug_name,
        dose_mg,
        fup,
        sieving_coeff,
        qb_L_per_h,
        session_duration_h,
        cl_residual_L_per_h,
        vd_L,
    )

    notes = f"E={E:.3f}, CL_dialysis={cl_dialysis:.2f} L/h, frac_removed≈{frac_removed:.2f}/session"

    return DialysisPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        fup=fup,
        sieving_coeff=sieving_coeff,
        times_h=times,
        c_plasma_mg_L=concs,
        is_dialysis_time=is_dial,
        cmax_mg_L=cmax,
        cmin_predialysis_mg_L=cmin_predialysis,
        fraction_removed_per_session=frac_removed,
        cl_dialysis_L_per_h=cl_dialysis,
        supplemental_dose_mg=supp,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Supplemental dose
# ---------------------------------------------------------------------------


def supplemental_dose(
    drug_name: str,
    dose_mg: float,
    fup: float,
    sieving_coeff: float,
    qb_L_per_h: float,
    session_duration_h: float,
    cl_L_per_h: float,
    vd_L: float,
) -> float:
    """Estimate post-dialysis supplemental dose to restore pre-dialysis Cmax.

    Strategy: compute the fraction of drug removed during a single session,
    then return ``dose_mg * fraction_removed`` as the replacement dose.

    Parameters
    ----------
    drug_name:
        Identifier (unused, kept for API consistency).
    dose_mg:
        Original dose (mg).
    fup:
        Unbound plasma fraction.
    sieving_coeff:
        Membrane sieving coefficient.
    qb_L_per_h:
        Blood flow through dialyser (L/h).
    session_duration_h:
        Dialysis session length (h).
    cl_L_per_h:
        Total clearance (residual + any other) during interdialytic period.
    vd_L:
        Volume of distribution (L).

    Returns
    -------
    Supplemental dose in mg (>= 0).
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if not (0 < fup <= 1):
        raise ValueError("fup must be in (0, 1]")
    if not (0 <= sieving_coeff <= 1):
        raise ValueError("sieving_coeff must be in [0, 1]")
    if qb_L_per_h <= 0:
        raise ValueError("qb_L_per_h must be > 0")
    if session_duration_h <= 0:
        raise ValueError("session_duration_h must be > 0")
    if cl_L_per_h < 0:
        raise ValueError("cl_L_per_h must be >= 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")

    E = dialysis_extraction(fup, sieving_coeff, qb_L_per_h)
    cl_dialysis = qb_L_per_h * E
    cl_total = cl_L_per_h + cl_dialysis
    _ke = cl_total / vd_L

    # Fraction removed = 1 - C_after / C_before = 1 - exp(-ke_dial * t)
    # Using ke_dial = cl_dialysis / Vd for the dialysis-attributable removal
    ke_dial = cl_dialysis / vd_L
    frac_removed_dialysis = 1.0 - math.exp(-ke_dial * session_duration_h)

    # Fraction also eliminated by residual clearance during session
    ke_res = cl_total / vd_L
    frac_total = 1.0 - math.exp(-ke_res * session_duration_h)

    # Supplemental dose proportional to amount lost due to dialysis
    # (dialysis contribution to total fraction removed)
    if frac_total > 0:
        dial_fraction = frac_removed_dialysis / max(frac_total, 1e-12)
    else:
        dial_fraction = 0.0

    supp = dose_mg * frac_total * min(dial_fraction, 1.0)
    return max(supp, 0.0)
