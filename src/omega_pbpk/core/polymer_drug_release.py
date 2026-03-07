"""Polymer drug release — erosion-controlled vs diffusion-controlled release models."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class PolymerReleaseResult:
    """Result of polymer drug release simulation."""

    drug_name: str
    dose_mg: float
    mechanism: str
    k_release: float
    times_h: list[float] = field(default_factory=list)
    m_remaining_mg: list[float] = field(default_factory=list)
    m_released_mg: list[float] = field(default_factory=list)
    c_plasma_mg_L: list[float] = field(default_factory=list)
    cmax_mg_L: float = 0.0
    tmax_h: float = 0.0
    auc_mg_h_per_L: float = 0.0
    t90_h: float = 0.0
    f_released: float = 0.0
    notes: str = ""


_VALID_MECHANISMS = ("erosion", "diffusion", "combined")


def simulate_polymer_release(
    drug_name: str,
    dose_mg: float,
    mechanism: str,
    k_release: float,
    t_end_h: float,
    dt_h: float = 0.01,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
) -> PolymerReleaseResult:
    """Simulate drug release from a polymer matrix and resulting plasma PK.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Total drug dose loaded in the polymer matrix (mg).
    mechanism : str
        Release mechanism: "erosion" (zero-order), "diffusion" (Higuchi), or "combined".
    k_release : float
        Release rate constant. For erosion: fractional rate constant (1/h);
        for diffusion: Higuchi constant (dimensionless, scales fraction released).
    t_end_h : float
        Simulation end time (h).
    dt_h : float
        ODE integration time step (h).
    cl_L_per_h : float
        Drug clearance (L/h).
    vd_L : float
        Volume of distribution (L).

    Returns
    -------
    PolymerReleaseResult
    """
    if not drug_name:
        raise ValueError("drug_name must not be empty")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if mechanism not in _VALID_MECHANISMS:
        raise ValueError(f"mechanism must be one of {_VALID_MECHANISMS}")
    if k_release <= 0:
        raise ValueError("k_release must be > 0")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be > 0")
    if dt_h <= 0:
        raise ValueError("dt_h must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")

    ke = cl_L_per_h / vd_L  # elimination rate constant (1/h)

    times: list[float] = []
    m_remaining: list[float] = []
    m_released: list[float] = []
    c_plasma: list[float] = []

    # State variables
    m_rem = dose_mg  # drug remaining in polymer (mg)
    m_rel = 0.0  # cumulative drug released (mg)
    c = 0.0  # plasma concentration (mg/L)

    t = 0.0
    n_steps = int(round(t_end_h / dt_h)) + 1

    cmax = 0.0
    tmax = 0.0
    t90 = t_end_h  # default if 90% not reached

    # Track whether t90 has been found
    _t90_found = False

    for _i in range(n_steps):
        times.append(t)
        m_remaining.append(m_rem)
        m_released.append(m_rel)
        c_plasma.append(c)

        if c > cmax:
            cmax = c
            tmax = t

        if not _t90_found and m_rel >= 0.9 * dose_mg:
            t90 = t
            _t90_found = True

        # Compute release rate (mg/h) for each mechanism
        if mechanism == "erosion":
            # Zero-order approximation: constant fractional rate limited by remaining mass
            rate_erosion = min(k_release * dose_mg, m_rem / dt_h if dt_h > 0 else m_rem)
            release_rate = rate_erosion
        elif mechanism == "diffusion":
            # Higuchi: dm/dt = dose * k / (2 * sqrt(k * t + eps))
            release_rate = dose_mg * k_release / (2.0 * math.sqrt(k_release * t + 1e-9))
            # Clamp release to available drug
            release_rate = min(release_rate, m_rem / dt_h if m_rem > 0 else 0.0)
        else:  # combined
            rate_erosion = min(k_release * dose_mg, m_rem / dt_h if dt_h > 0 else m_rem)
            rate_diffusion = dose_mg * k_release / (2.0 * math.sqrt(k_release * t + 1e-9))
            rate_diffusion = min(rate_diffusion, m_rem / dt_h if m_rem > 0 else 0.0)
            release_rate = 0.5 * (rate_erosion + rate_diffusion)

        # Forward Euler ODE step
        # Mass balance: dm_remaining/dt = -release_rate
        # dc/dt = release_rate / vd_L - ke * c
        dm_rem = -release_rate * dt_h
        dc = (release_rate / vd_L - ke * c) * dt_h

        # Clamp: remaining mass cannot go below 0
        new_m_rem = max(0.0, m_rem + dm_rem)
        actual_released = m_rem - new_m_rem
        m_rem = new_m_rem
        m_rel = min(dose_mg, m_rel + actual_released)
        c = max(0.0, c + dc)
        t = round(t + dt_h, 10)

    # AUC via trapezoidal rule
    auc = 0.0
    for j in range(1, len(times)):
        auc += 0.5 * (c_plasma[j] + c_plasma[j - 1]) * (times[j] - times[j - 1])

    f_released = m_released[-1] / dose_mg

    notes = (
        f"Mechanism: {mechanism}. k_release={k_release:.4f}. "
        f"Final f_released={f_released:.3f}. "
        f"t90={'not reached' if not _t90_found else f'{t90:.2f} h'}."
    )

    return PolymerReleaseResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        mechanism=mechanism,
        k_release=k_release,
        times_h=times,
        m_remaining_mg=m_remaining,
        m_released_mg=m_released,
        c_plasma_mg_L=c_plasma,
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc_mg_h_per_L=auc,
        t90_h=t90,
        f_released=f_released,
        notes=notes,
    )


def compare_mechanisms(
    drug_name: str,
    dose_mg: float,
    k_release: float,
    cl_L_per_h: float,
    vd_L: float,
    t_end_h: float,
    dt_h: float = 0.01,
) -> dict[str, PolymerReleaseResult]:
    """Compare drug release across all three mechanisms.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Total dose (mg).
    k_release : float
        Release rate constant (shared across mechanisms).
    cl_L_per_h : float
        Clearance (L/h).
    vd_L : float
        Volume of distribution (L).
    t_end_h : float
        Simulation duration (h).
    dt_h : float
        Integration step (h).

    Returns
    -------
    dict with keys "erosion", "diffusion", "combined"
    """
    results: dict[str, PolymerReleaseResult] = {}
    for mech in _VALID_MECHANISMS:
        results[mech] = simulate_polymer_release(
            drug_name=drug_name,
            dose_mg=dose_mg,
            mechanism=mech,
            k_release=k_release,
            t_end_h=t_end_h,
            dt_h=dt_h,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
        )
    return results
