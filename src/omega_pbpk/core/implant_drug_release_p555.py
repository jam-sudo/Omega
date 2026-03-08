"""Phase 555 - Drug Release from Implants.

Models drug release from subcutaneous/intramuscular implants
(rods, pellets, microspheres) using Forward Euler integration.
"""

import math
from dataclasses import dataclass, field

VALID_IMPLANT_TYPES = ("rod", "pellet", "microsphere")


@dataclass
class ImplantReleaseResult:
    """Result of implant drug release simulation."""

    drug_name: str
    dose_mg: float
    implant_type: str
    times_h: list = field(default_factory=list)
    c_systemic_mg_L: list = field(default_factory=list)
    cumulative_released_pct: list = field(default_factory=list)
    cmax_mg_L: float = 0.0
    tmax_h: float = 0.0
    auc_mg_h_per_L: float = 0.0
    t50_released_h: float = 0.0
    t90_released_h: float = math.inf
    duration_days: float = 0.0
    notes: str = ""


def simulate_implant_release(
    drug_name: str,
    dose_mg: float,
    implant_type: str,
    cl_L_per_h: float,
    vd_L: float,
    t_end_h: float = 720.0,
    dt_h: float = 1.0,
) -> ImplantReleaseResult:
    """Simulate drug release from an implant using Forward Euler.

    Args:
        drug_name: Name of the drug.
        dose_mg: Total drug dose in implant (mg).
        implant_type: One of "rod", "pellet", "microsphere".
        cl_L_per_h: Systemic clearance (L/h).
        vd_L: Volume of distribution (L).
        t_end_h: Simulation end time (h).
        dt_h: Time step (h).

    Returns:
        ImplantReleaseResult with time-course and summary PK.
    """
    # Validation
    if implant_type not in VALID_IMPLANT_TYPES:
        raise ValueError(f"implant_type must be one of {VALID_IMPLANT_TYPES}, got '{implant_type}'")
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")

    # State variables
    m_implant = dose_mg  # drug remaining in implant (mg)
    c_sys = 0.0  # systemic concentration (mg/L)

    times_h = []
    c_systemic_mg_L = []
    cumulative_released_pct = []

    cmax = 0.0
    tmax = 0.0
    t50 = 0.0
    t90 = math.inf
    found_t50 = False
    found_t90 = False

    n_steps = int(t_end_h / dt_h) + 1

    for i in range(n_steps):
        t = i * dt_h

        times_h.append(t)
        c_systemic_mg_L.append(c_sys)
        cum_pct = (dose_mg - m_implant) / dose_mg * 100.0
        cumulative_released_pct.append(cum_pct)

        if c_sys > cmax:
            cmax = c_sys
            tmax = t

        if not found_t50 and cum_pct >= 50.0:
            t50 = t
            found_t50 = True
        if not found_t90 and cum_pct >= 90.0:
            t90 = t
            found_t90 = True

        # Compute release rate
        if implant_type == "rod":
            # Zero-order release: dose_mg / 500 per hour, stop when depleted
            zero_order_rate = dose_mg / 500.0
            release_rate = min(zero_order_rate, m_implant / dt_h)
        elif implant_type == "pellet":
            k_release = 0.005
            release_rate = k_release * m_implant
        else:  # microsphere
            k = 0.02 if t < 72.0 else 0.003
            release_rate = k * m_implant

        # Forward Euler update
        dm = -release_rate * dt_h
        dc = (release_rate / vd_L - (cl_L_per_h / vd_L) * c_sys) * dt_h

        m_implant = max(0.0, m_implant + dm)
        c_sys = max(0.0, c_sys + dc)

    # AUC via trapezoidal rule
    auc = sum(
        0.5 * (c_systemic_mg_L[i] + c_systemic_mg_L[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )

    # Duration above 10% of Cmax (in days)
    threshold = 0.1 * cmax
    above_times = [times_h[i] for i in range(len(times_h)) if c_systemic_mg_L[i] >= threshold]
    if above_times:
        duration_days = (above_times[-1] - above_times[0]) / 24.0
    else:
        duration_days = 0.0

    if not found_t90:
        t90 = math.inf

    notes = (
        f"Forward Euler simulation of {implant_type} implant release; dt={dt_h}h, t_end={t_end_h}h"
    )

    return ImplantReleaseResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        implant_type=implant_type,
        times_h=times_h,
        c_systemic_mg_L=c_systemic_mg_L,
        cumulative_released_pct=cumulative_released_pct,
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc_mg_h_per_L=auc,
        t50_released_h=t50,
        t90_released_h=t90,
        duration_days=duration_days,
        notes=notes,
    )
