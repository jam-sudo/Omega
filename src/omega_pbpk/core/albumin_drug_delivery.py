"""Albumin-mediated drug delivery model.

Models drug transport facilitated by albumin in tissues using
the dissociation/facilitated transport hypothesis:
  - Albumin-bound drug can be delivered to tissues when free drug is consumed
  - Effective delivery includes unbound + fraction of albumin-bound drug

References: Weisiger 2002, Jansen et al. 2004 (facilitated transport theory)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz


@dataclass(frozen=True)
class AlbuminBindingResult:
    """Result of albumin binding calculation at a single concentration."""

    c_total: float  # total drug concentration (mg/L)
    c_free: float  # free (unbound) drug concentration (mg/L)
    c_bound: float  # albumin-bound drug concentration (mg/L)
    fu: float  # unbound fraction = c_free / c_total
    notes: str


@dataclass(frozen=True)
class AlbuminDeliveryResult:
    """Result of albumin-facilitated PK simulation."""

    times_h: list[float]  # time points (h)
    c_total: list[float]  # total plasma concentration profile (mg/L)
    c_free_profile: list[float]  # free drug concentration profile (mg/L)
    fu_profile: list[float]  # unbound fraction over time
    cmax_total: float  # peak total concentration (mg/L)
    tmax_h: float  # time of peak concentration (h)
    auc_total: float  # AUC of total concentration (mg*h/L)
    facilitation_factor: float  # fraction of bound drug delivered (0-1)
    notes: str


def albumin_binding(
    c_total_mg_L: float,
    ka_L_per_mg: float,
    bmax_albumin_mg_L: float = 630.0,
) -> AlbuminBindingResult:
    """Compute albumin binding at a given total drug concentration.

    Uses Langmuir (one-site) binding isotherm:
        c_bound = bmax * ka * c_free / (1 + ka * c_free)
        c_total = c_free + c_bound

    Solving for c_free yields a quadratic:
        ka * c_free^2 + (1 + ka*bmax - ka*c_total) * c_free - c_total = 0

    Parameters
    ----------
    c_total_mg_L : float
        Total drug concentration in plasma (mg/L).
    ka_L_per_mg : float
        Albumin-drug association constant (L/mg).
    bmax_albumin_mg_L : float
        Maximum binding capacity of albumin in plasma (mg/L).
        Default 630 mg/L corresponds to ~0.63 mM albumin (42 g/L / 66.5 kDa).

    Returns
    -------
    AlbuminBindingResult
    """
    if c_total_mg_L < 0:
        raise ValueError("c_total_mg_L must be >= 0")
    if ka_L_per_mg <= 0:
        raise ValueError("ka_L_per_mg must be > 0")
    if bmax_albumin_mg_L <= 0:
        raise ValueError("bmax_albumin_mg_L must be > 0")

    if c_total_mg_L == 0.0:
        return AlbuminBindingResult(
            c_total=0.0,
            c_free=0.0,
            c_bound=0.0,
            fu=1.0,
            notes="Zero concentration: no binding.",
        )

    # Quadratic: ka*cf^2 + (1 + ka*bmax - ka*ctot)*cf - ctot = 0
    a_coeff = ka_L_per_mg
    b_coeff = 1.0 + ka_L_per_mg * bmax_albumin_mg_L - ka_L_per_mg * c_total_mg_L
    c_coeff = -c_total_mg_L

    discriminant = b_coeff * b_coeff - 4.0 * a_coeff * c_coeff
    discriminant = max(discriminant, 0.0)  # numerical guard
    c_free = (-b_coeff + math.sqrt(discriminant)) / (2.0 * a_coeff)
    c_free = float(np.clip(c_free, 0.0, c_total_mg_L))

    c_bound = c_total_mg_L - c_free
    fu = c_free / c_total_mg_L

    notes = (
        f"Albumin binding: ctot={c_total_mg_L:.4g} mg/L, "
        f"cfree={c_free:.4g} mg/L, cbound={c_bound:.4g} mg/L, "
        f"fu={fu:.4f}, ka={ka_L_per_mg:.4g} L/mg, bmax={bmax_albumin_mg_L:.4g} mg/L"
    )

    return AlbuminBindingResult(
        c_total=c_total_mg_L,
        c_free=c_free,
        c_bound=c_bound,
        fu=fu,
        notes=notes,
    )


def simulate_albumin_facilitated_pk(
    dose_mg: float,
    vd_L: float,
    cl_intrinsic_L_per_h: float,
    ka_albumin_L_per_mg: float,
    bmax_albumin_mg_L: float = 630.0,
    facilitation_factor: float = 0.3,
    ka_abs_per_h: float = 1.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> AlbuminDeliveryResult:
    """Simulate 1-compartment oral PK with concentration-dependent albumin binding.

    Effective clearance accounts for albumin-mediated facilitated transport:
        cl_eff = cl_intrinsic * fu + facilitation_factor * cl_intrinsic * (1 - fu)

    This models the scenario where albumin-bound drug contributes partially
    to tissue uptake (facilitation_factor=0: only free drug cleared;
    facilitation_factor=1: bound drug fully cleared at same rate as free).

    ODE system (Forward Euler):
        dAgut/dt    = -ka_abs * Agut
        dAcentral/dt = ka_abs * Agut - cl_eff(C) * C

    where C = Acentral / Vd and cl_eff depends on C via albumin binding.

    Parameters
    ----------
    dose_mg : float
        Oral dose (mg).
    vd_L : float
        Volume of distribution (L).
    cl_intrinsic_L_per_h : float
        Intrinsic clearance based on free drug (L/h).
    ka_albumin_L_per_mg : float
        Albumin association constant for this drug (L/mg).
    bmax_albumin_mg_L : float
        Albumin binding capacity (mg/L), default 630 mg/L.
    facilitation_factor : float
        Fraction of bound drug delivered to tissue (0=no facilitation, 1=full).
    ka_abs_per_h : float
        First-order oral absorption rate constant (1/h).
    t_end_h : float
        Simulation end time (h).
    dt_h : float
        Forward Euler time step (h).

    Returns
    -------
    AlbuminDeliveryResult
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if cl_intrinsic_L_per_h <= 0:
        raise ValueError("cl_intrinsic_L_per_h must be > 0")
    if ka_albumin_L_per_mg <= 0:
        raise ValueError("ka_albumin_L_per_mg must be > 0")
    if bmax_albumin_mg_L <= 0:
        raise ValueError("bmax_albumin_mg_L must be > 0")
    if not (0.0 <= facilitation_factor <= 1.0):
        raise ValueError("facilitation_factor must be in [0, 1]")
    if ka_abs_per_h <= 0:
        raise ValueError("ka_abs_per_h must be > 0")

    n_steps = max(int(t_end_h / dt_h), 1)
    times = np.linspace(0.0, t_end_h, n_steps + 1)

    c_total_arr = np.zeros(n_steps + 1)
    c_free_arr = np.zeros(n_steps + 1)
    fu_arr = np.ones(n_steps + 1)  # fu=1 when C=0

    # Initial state
    a_gut = dose_mg  # drug amount in gut (mg)
    a_central = 0.0  # drug amount in central compartment (mg)

    for i in range(n_steps):
        c_t = a_central / vd_L  # total concentration (mg/L)

        # Albumin binding at current concentration
        bind = albumin_binding(c_t, ka_albumin_L_per_mg, bmax_albumin_mg_L)
        fu_t = bind.fu

        # Effective clearance (L/h): free + facilitated bound contributions
        cl_eff = cl_intrinsic_L_per_h * (fu_t + facilitation_factor * (1.0 - fu_t))

        # Forward Euler ODE step
        d_gut = -ka_abs_per_h * a_gut
        d_central = ka_abs_per_h * a_gut - cl_eff * c_t  # mg/h (cl_eff*c_t = mg/L * L/h)

        a_gut = max(a_gut + d_gut * dt_h, 0.0)
        a_central = max(a_central + d_central * dt_h, 0.0)

        # Record state at next time point
        c_new = a_central / vd_L
        bind_new = albumin_binding(c_new, ka_albumin_L_per_mg, bmax_albumin_mg_L)
        c_total_arr[i + 1] = c_new
        c_free_arr[i + 1] = bind_new.c_free
        fu_arr[i + 1] = bind_new.fu

    auc_total = float(np_trapz(c_total_arr, times))
    cmax_total = float(np.max(c_total_arr))
    tmax_h = float(times[int(np.argmax(c_total_arr))])

    notes = (
        f"Albumin-facilitated PK: dose={dose_mg} mg, Vd={vd_L} L, "
        f"CL_int={cl_intrinsic_L_per_h} L/h, ff={facilitation_factor:.2f}, "
        f"Cmax={cmax_total:.4g} mg/L, AUC={auc_total:.4g} mg*h/L"
    )

    return AlbuminDeliveryResult(
        times_h=times.tolist(),
        c_total=c_total_arr.tolist(),
        c_free_profile=c_free_arr.tolist(),
        fu_profile=fu_arr.tolist(),
        cmax_total=cmax_total,
        tmax_h=tmax_h,
        auc_total=auc_total,
        facilitation_factor=facilitation_factor,
        notes=notes,
    )


def compare_facilitation(
    dose_mg: float,
    vd_L: float,
    cl_intrinsic_L_per_h: float,
    ka_albumin: float,
    bmax_albumin: float = 630.0,
) -> dict[str, AlbuminDeliveryResult]:
    """Compare albumin-facilitated PK at three facilitation levels.

    Simulates facilitation_factor = 0.0 (no facilitation), 0.3 (partial),
    and 1.0 (full facilitation) to illustrate the impact on drug exposure.

    Parameters
    ----------
    dose_mg : float
        Oral dose (mg).
    vd_L : float
        Volume of distribution (L).
    cl_intrinsic_L_per_h : float
        Intrinsic clearance based on free drug (L/h).
    ka_albumin : float
        Albumin association constant (L/mg).
    bmax_albumin : float
        Albumin binding capacity (mg/L).

    Returns
    -------
    dict with keys "no_facilitation", "partial_facilitation", "full_facilitation"
    """
    results: dict[str, AlbuminDeliveryResult] = {}

    for label, ff in [
        ("no_facilitation", 0.0),
        ("partial_facilitation", 0.3),
        ("full_facilitation", 1.0),
    ]:
        results[label] = simulate_albumin_facilitated_pk(
            dose_mg=dose_mg,
            vd_L=vd_L,
            cl_intrinsic_L_per_h=cl_intrinsic_L_per_h,
            ka_albumin_L_per_mg=ka_albumin,
            bmax_albumin_mg_L=bmax_albumin,
            facilitation_factor=ff,
        )

    return results


__all__ = [
    "AlbuminBindingResult",
    "AlbuminDeliveryResult",
    "albumin_binding",
    "simulate_albumin_facilitated_pk",
    "compare_facilitation",
]
