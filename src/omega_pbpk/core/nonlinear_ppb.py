"""Nonlinear (saturable) plasma protein binding model.

Models concentration-dependent free fraction (fu) using a Langmuir isotherm
(single-class saturable binding sites). At high drug concentrations, binding
sites become saturated and fu increases toward 1.0.

References
----------
Scatchard 1949 (Ann NY Acad Sci), Benet & Hoener 2002 (Clin Pharmacol Ther),
Kratochwil 2002 (Pharm Res).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "NonlinearPPBResult",
    "NonlinearPPBPKResult",
    "nonlinear_fu",
    "fu_concentration_profile",
    "simulate_nonlinear_ppb_pk",
]

# Bisection tolerance
_BISECT_TOL = 1e-12
_BISECT_MAX_ITER = 200


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NonlinearPPBResult:
    """Result of fu profile across a concentration range."""

    c_range: tuple  # concentration values (mg/L)
    fu_values: tuple  # fu at each concentration
    fu_low_conc: float  # fu at low concentration limit
    fu_high_conc: float  # fu at high concentration limit
    notes: str


@dataclass(frozen=True)
class NonlinearPPBPKResult:
    """Result of 1-compartment PK simulation with nonlinear PPB."""

    times_h: tuple  # time array (h)
    c_plasma: tuple  # total plasma concentration (mg/L)
    fu_profile: tuple  # fu at each time point
    cmax: float  # peak total plasma concentration (mg/L)
    tmax_h: float  # time of Cmax (h)
    auc: float  # AUC_0-t using trapezoidal rule (mg/L * h)
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _solve_cfree_bisection(
    c_total: float,
    bmax: float,
    kd: float,
    n_sites: float,
) -> float:
    """Solve for c_free given c_total using bisection on Langmuir isotherm.

    Mass balance: c_total = c_free + bound
    Langmuir:     bound = n_sites * bmax * c_free / (kd + c_free)

    Returns c_free in [0, c_total].
    """
    if c_total <= 0.0:
        return 0.0

    def residual(cf: float) -> float:
        bound = n_sites * bmax * cf / (kd + cf)
        return cf + bound - c_total

    lo = 0.0
    hi = c_total

    f_lo = residual(lo)  # = 0 + 0 - c_total < 0
    f_hi = residual(hi)  # >= hi - c_total; could be positive or negative

    # If f_hi < 0 all c_total is free (binding capacity exhausted)
    if f_hi <= 0.0:
        return c_total

    for _ in range(_BISECT_MAX_ITER):
        mid = 0.5 * (lo + hi)
        f_mid = residual(mid)
        if abs(f_mid) < _BISECT_TOL or (hi - lo) < _BISECT_TOL:
            return mid
        if f_lo * f_mid < 0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid

    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def nonlinear_fu(
    c_total_mg_L: float,
    bmax_mg_L: float,
    kd_mg_L: float,
    n_sites: float = 1.0,
) -> float:
    """Compute free fraction (fu) at a given total plasma concentration.

    Uses a Langmuir isotherm (saturable, single-class binding):

        bound = n_sites * bmax * c_free / (kd + c_free)
        c_total = c_free + bound
        fu = c_free / c_total

    At c_total -> 0, fu approaches the linear fu (low-concentration limit).
    At c_total >> kd, binding sites saturate and fu -> 1.

    Parameters
    ----------
    c_total_mg_L : float
        Total drug concentration in plasma (mg/L). Must be >= 0.
    bmax_mg_L : float
        Maximum binding capacity of the protein (mg/L, i.e., protein
        concentration * stoichiometry). Must be > 0.
    kd_mg_L : float
        Dissociation constant (mg/L). Must be > 0.
    n_sites : float
        Number of equivalent binding site classes (default 1.0).

    Returns
    -------
    float
        Unbound fraction fu in [0, 1].
    """
    if bmax_mg_L <= 0:
        raise ValueError(f"bmax_mg_L must be > 0, got {bmax_mg_L}")
    if kd_mg_L <= 0:
        raise ValueError(f"kd_mg_L must be > 0, got {kd_mg_L}")
    if n_sites <= 0:
        raise ValueError(f"n_sites must be > 0, got {n_sites}")
    if c_total_mg_L < 0:
        raise ValueError(f"c_total_mg_L must be >= 0, got {c_total_mg_L}")

    if c_total_mg_L == 0.0:
        # fu at zero concentration: use low-conc approximation
        # bound ~ n_sites * bmax * c_free / kd  =>  c_free ~ c_total / (1 + n_sites*bmax/kd)
        fu_lin = 1.0 / (1.0 + n_sites * bmax_mg_L / kd_mg_L)
        return fu_lin

    c_free = _solve_cfree_bisection(c_total_mg_L, bmax_mg_L, kd_mg_L, n_sites)
    fu = c_free / c_total_mg_L
    # Clamp for numerical safety
    return float(max(0.0, min(1.0, fu)))


def fu_concentration_profile(
    c_range_mg_L: list[float] | np.ndarray,
    bmax_mg_L: float,
    kd_mg_L: float,
    n_sites: float = 1.0,
) -> NonlinearPPBResult:
    """Compute fu across a concentration range.

    Parameters
    ----------
    c_range_mg_L : list or array
        Concentration values (mg/L) at which to compute fu.
    bmax_mg_L : float
        Maximum binding capacity (mg/L).
    kd_mg_L : float
        Dissociation constant (mg/L).
    n_sites : float
        Number of binding site classes.

    Returns
    -------
    NonlinearPPBResult
    """
    c_arr = list(c_range_mg_L)
    if not c_arr:
        raise ValueError("c_range_mg_L must not be empty")

    fu_vals = [nonlinear_fu(c, bmax_mg_L, kd_mg_L, n_sites) for c in c_arr]
    fu_low = nonlinear_fu(1e-6, bmax_mg_L, kd_mg_L, n_sites)
    fu_high = fu_vals[-1]

    notes = (
        f"Langmuir PPB: bmax={bmax_mg_L} mg/L, Kd={kd_mg_L} mg/L, "
        f"n_sites={n_sites}; fu_low={fu_low:.4f}, fu_high={fu_high:.4f}"
    )

    return NonlinearPPBResult(
        c_range=tuple(c_arr),
        fu_values=tuple(fu_vals),
        fu_low_conc=fu_low,
        fu_high_conc=fu_high,
        notes=notes,
    )


def simulate_nonlinear_ppb_pk(
    dose_mg: float,
    vd_L: float,
    cl_L_per_h: float,
    bmax_mg_L: float,
    kd_mg_L: float,
    ka_per_h: float = 1.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> NonlinearPPBPKResult:
    """Simulate 1-compartment oral PK with concentration-dependent fu.

    The effective clearance at each time step is scaled by the ratio of
    instantaneous fu to the linear (low-concentration) fu:

        cl_eff(t) = cl_L_per_h * fu(c(t)) / fu_linear

    This captures the increase in total clearance at high concentrations
    where more drug is unbound and available for elimination.

    ODE (Forward Euler):
        dA_gut/dt = -ka * A_gut
        dc/dt     = ka * A_gut / Vd - cl_eff(c) * c / Vd

    Parameters
    ----------
    dose_mg : float
        Oral dose in mg.
    vd_L : float
        Volume of distribution (L). Must be > 0.
    cl_L_per_h : float
        Total clearance at linear PPB conditions (L/h). Must be > 0.
    bmax_mg_L : float
        Maximum protein binding capacity (mg/L). Must be > 0.
    kd_mg_L : float
        Dissociation constant (mg/L). Must be > 0.
    ka_per_h : float
        First-order oral absorption rate constant (h^-1). Default 1.0.
    t_end_h : float
        Simulation end time (h). Default 24.0.
    dt_h : float
        Forward-Euler step size (h). Default 0.1.

    Returns
    -------
    NonlinearPPBPKResult
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if ka_per_h <= 0:
        raise ValueError(f"ka_per_h must be > 0, got {ka_per_h}")
    if t_end_h <= 0:
        raise ValueError(f"t_end_h must be > 0, got {t_end_h}")
    if dt_h <= 0:
        raise ValueError(f"dt_h must be > 0, got {dt_h}")

    # fu at low concentration (linear limit)
    fu_linear = nonlinear_fu(1e-6, bmax_mg_L, kd_mg_L)

    n_steps = int(round(t_end_h / dt_h))
    times = [0.0] * (n_steps + 1)
    c_arr = [0.0] * (n_steps + 1)
    fu_arr = [fu_linear] * (n_steps + 1)

    a_gut = dose_mg  # amount in gut compartment (mg)
    c = 0.0  # plasma concentration (mg/L)

    for i in range(n_steps):
        times[i] = i * dt_h

        fu_t = nonlinear_fu(c, bmax_mg_L, kd_mg_L)
        fu_arr[i] = fu_t

        # Scale clearance by fu ratio
        cl_eff = cl_L_per_h * fu_t / fu_linear if fu_linear > 0 else cl_L_per_h

        # Forward Euler
        da_gut = -ka_per_h * a_gut * dt_h
        dc = (ka_per_h * a_gut / vd_L - cl_eff * c / vd_L) * dt_h

        a_gut = max(0.0, a_gut + da_gut)
        c = max(0.0, c + dc)

        c_arr[i + 1] = c

    times[n_steps] = n_steps * dt_h
    fu_arr[n_steps] = nonlinear_fu(c_arr[n_steps], bmax_mg_L, kd_mg_L)

    # PK metrics
    c_np = np.array(c_arr)
    t_np = np.array(times)

    cmax = float(np.max(c_np))
    tmax_idx = int(np.argmax(c_np))
    tmax_h = float(t_np[tmax_idx])
    auc = float(np.trapz(c_np, t_np))

    notes = (
        f"1-cpt oral nonlinear PPB PK: dose={dose_mg} mg, Vd={vd_L} L, "
        f"CL={cl_L_per_h} L/h, bmax={bmax_mg_L} mg/L, Kd={kd_mg_L} mg/L; "
        f"fu_linear={fu_linear:.4f}, Cmax={cmax:.4f} mg/L, "
        f"Tmax={tmax_h:.2f} h, AUC={auc:.4f} mg/L*h"
    )

    return NonlinearPPBPKResult(
        times_h=tuple(times),
        c_plasma=tuple(c_arr),
        fu_profile=tuple(fu_arr),
        cmax=cmax,
        tmax_h=tmax_h,
        auc=auc,
        notes=notes,
    )
