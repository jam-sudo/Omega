"""Phase 280 — Cardiac Drug Distribution PK Model.

3-compartment PK model for cardiac drugs: plasma + cardiac tissue + peripheral.
Includes cardiac partitioning coefficient and effect-site (heart) concentration
for PD effect estimation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

_VALID_ROUTES = frozenset(["iv_bolus", "oral"])

# Default oral absorption parameters
_KA_DEFAULT = 1.0  # per hour
_F_ORAL_DEFAULT = 0.7  # oral bioavailability fraction


@dataclass
class CardiacPKResult:
    """Result of cardiac PK simulation."""

    drug_name: str
    dose_mg: float
    route: str
    kp_cardiac: float
    times_h: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    c_cardiac_mg_L: list = field(default_factory=list)
    auc_plasma_mg_h_per_L: float = 0.0
    cmax_plasma_mg_L: float = 0.0
    tmax_plasma_h: float = 0.0
    auc_cardiac_mg_h_per_L: float = 0.0
    cmax_cardiac_mg_L: float = 0.0
    tmax_cardiac_h: float = 0.0
    cardiac_to_plasma_ratio: float = 0.0
    notes: str = ""


def _trapezoidal_auc(x: list[float], y: list[float]) -> float:
    """Manual trapezoidal AUC integration."""
    return sum(0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]) for i in range(1, len(x)))


def simulate_cardiac_pk(
    drug_name: str,
    dose_mg: float,
    route: str,
    cl_L_per_h: float,
    vd_central_L: float,
    vd_cardiac_L: float,
    kp_cardiac: float,
    k12_per_h: float,
    k21_per_h: float,
    t_end_h: float,
    dt_h: float,
    ka_per_h: float = _KA_DEFAULT,
    f_oral: float = _F_ORAL_DEFAULT,
) -> CardiacPKResult:
    """Simulate cardiac drug distribution using a 3-compartment PK model.

    Compartments:
        - Central (plasma): A_central
        - Cardiac tissue: A_cardiac
        - (Gut absorption depot for oral route)

    ODE system (Forward Euler):
        Oral gut: dA_gut/dt = -ka * A_gut
        dA_central/dt = -CL/Vd_central * A_central - k12 * A_central + k21 * A_cardiac
                        [+ ka * A_gut for oral]
        dA_cardiac/dt = k12 * A_central - k21 * A_cardiac

    Concentrations:
        C_plasma = A_central / vd_central
        C_cardiac_tissue = A_cardiac / vd_cardiac

    Args:
        drug_name: Drug identifier.
        dose_mg: Dose in mg.
        route: "iv_bolus" or "oral".
        cl_L_per_h: Total clearance in L/h.
        vd_central_L: Volume of distribution of central compartment in L.
        vd_cardiac_L: Volume of distribution of cardiac compartment in L.
        kp_cardiac: Cardiac tissue partitioning coefficient (heart/plasma at equilibrium).
        k12_per_h: Transfer rate from central to cardiac (per hour).
        k21_per_h: Transfer rate from cardiac to central (per hour).
        t_end_h: Simulation end time in hours.
        dt_h: Integration time step in hours.
        ka_per_h: Oral absorption rate constant (per hour, default 1.0).
        f_oral: Oral bioavailability fraction (default 0.7).

    Returns:
        CardiacPKResult with time series and summary PK parameters.

    Raises:
        ValueError: On invalid parameters.
    """
    # Validation
    if dose_mg <= 0.0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if route not in _VALID_ROUTES:
        raise ValueError(f"route must be one of {sorted(_VALID_ROUTES)}, got {route!r}")
    if cl_L_per_h <= 0.0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_central_L <= 0.0:
        raise ValueError(f"vd_central_L must be > 0, got {vd_central_L}")
    if vd_cardiac_L <= 0.0:
        raise ValueError(f"vd_cardiac_L must be > 0, got {vd_cardiac_L}")
    if kp_cardiac <= 0.0:
        raise ValueError(f"kp_cardiac must be > 0, got {kp_cardiac}")
    if k12_per_h <= 0.0:
        raise ValueError(f"k12_per_h must be > 0, got {k12_per_h}")
    if k21_per_h <= 0.0:
        raise ValueError(f"k21_per_h must be > 0, got {k21_per_h}")
    if t_end_h <= 0.0:
        raise ValueError(f"t_end_h must be > 0, got {t_end_h}")
    if dt_h <= 0.0:
        raise ValueError(f"dt_h must be > 0, got {dt_h}")

    # Initial conditions
    if route == "iv_bolus":
        a_central = dose_mg
        a_gut = 0.0
    else:  # oral
        a_central = 0.0
        a_gut = dose_mg * f_oral

    a_cardiac = 0.0

    ke = cl_L_per_h / vd_central_L  # elimination rate from central

    times: list[float] = []
    c_plasma_list: list[float] = []
    c_cardiac_list: list[float] = []

    t = 0.0
    n_steps = int(math.ceil(t_end_h / dt_h))

    for _ in range(n_steps + 1):
        # Record state
        c_plasma = a_central / vd_central_L
        c_cardiac = a_cardiac / vd_cardiac_L
        times.append(t)
        c_plasma_list.append(c_plasma)
        c_cardiac_list.append(c_cardiac)

        if t >= t_end_h:
            break

        # Effective dt (last step may be smaller)
        effective_dt = min(dt_h, t_end_h - t)

        # Derivatives
        if route == "oral":
            d_gut = -ka_per_h * a_gut
            absorption_flux = ka_per_h * a_gut
        else:
            d_gut = 0.0
            absorption_flux = 0.0

        d_central = (
            -ke * a_central - k12_per_h * a_central + k21_per_h * a_cardiac + absorption_flux
        )
        d_cardiac = k12_per_h * a_central - k21_per_h * a_cardiac

        # Forward Euler update
        a_gut += d_gut * effective_dt
        a_central += d_central * effective_dt
        a_cardiac += d_cardiac * effective_dt

        # Clamp to non-negative
        a_gut = max(0.0, a_gut)
        a_central = max(0.0, a_central)
        a_cardiac = max(0.0, a_cardiac)

        t += effective_dt

    # AUC by trapezoidal rule
    auc_plasma = _trapezoidal_auc(times, c_plasma_list)
    auc_cardiac = _trapezoidal_auc(times, c_cardiac_list)

    # Cmax and Tmax
    cmax_plasma = max(c_plasma_list)
    tmax_plasma_idx = c_plasma_list.index(cmax_plasma)
    tmax_plasma = times[tmax_plasma_idx]

    cmax_cardiac = max(c_cardiac_list)
    tmax_cardiac_idx = c_cardiac_list.index(cmax_cardiac)
    tmax_cardiac = times[tmax_cardiac_idx]

    # Cardiac to plasma ratio
    ratio = cmax_cardiac / cmax_plasma if cmax_plasma > 0.0 else 0.0

    notes = (
        f"3-cpt cardiac PK model; route={route}; "
        f"kp_cardiac={kp_cardiac:.2f}; "
        f"k12={k12_per_h:.3f}/h; k21={k21_per_h:.3f}/h"
    )

    return CardiacPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route=route,
        kp_cardiac=kp_cardiac,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        c_cardiac_mg_L=c_cardiac_list,
        auc_plasma_mg_h_per_L=auc_plasma,
        cmax_plasma_mg_L=cmax_plasma,
        tmax_plasma_h=tmax_plasma,
        auc_cardiac_mg_h_per_L=auc_cardiac,
        cmax_cardiac_mg_L=cmax_cardiac,
        tmax_cardiac_h=tmax_cardiac,
        cardiac_to_plasma_ratio=ratio,
        notes=notes,
    )


def compare_cardiac_drugs(
    drug_name: str,
    dose_mg: float,
    kp_cardiac_list: list[float],
    cl_L_per_h: float,
    vd_central_L: float,
    vd_cardiac_L: float = 5.0,
    k12_per_h: float = 0.5,
    k21_per_h: float = 0.3,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
    route: str = "iv_bolus",
) -> list[CardiacPKResult]:
    """Compare cardiac drug distribution across different kp_cardiac values.

    Args:
        drug_name: Drug identifier.
        dose_mg: Dose in mg.
        kp_cardiac_list: List of cardiac partitioning coefficients to compare.
        cl_L_per_h: Total clearance in L/h.
        vd_central_L: Volume of distribution of central compartment in L.
        vd_cardiac_L: Volume of distribution of cardiac compartment in L.
        k12_per_h: Transfer rate from central to cardiac (per hour).
        k21_per_h: Transfer rate from cardiac to central (per hour).
        t_end_h: Simulation end time in hours.
        dt_h: Integration time step in hours.
        route: Dosing route ("iv_bolus" or "oral").

    Returns:
        List of CardiacPKResult sorted by cmax_cardiac descending.
    """
    results = [
        simulate_cardiac_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            route=route,
            cl_L_per_h=cl_L_per_h,
            vd_central_L=vd_central_L,
            vd_cardiac_L=vd_cardiac_L,
            kp_cardiac=kp,
            k12_per_h=k12_per_h,
            k21_per_h=k21_per_h,
            t_end_h=t_end_h,
            dt_h=dt_h,
        )
        for kp in kp_cardiac_list
    ]
    return sorted(results, key=lambda r: r.cmax_cardiac_mg_L, reverse=True)
