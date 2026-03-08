"""Phase 232 — Extravascular PK Model.

Models extravascular drug distribution (subcutaneous, intramuscular,
intraperitoneal, sublingual) with depot and systemic compartments using
a 2-compartment Forward Euler ODE system.

Compartments
------------
- A_depot (mg)  : drug at injection/absorption site
- C_plasma (mg/L): systemic plasma concentration

ODEs
----
    dA_depot/dt  = -ka * A_depot
    dC_plasma/dt = F * ka * A_depot / Vd - ke * C_plasma
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Route-specific bioavailability and absorption rate constants
_ROUTE_PARAMS: dict[str, dict[str, float]] = {
    "subcutaneous": {"F": 0.85, "ka": 0.3},
    "intramuscular": {"F": 0.95, "ka": 0.8},
    "intraperitoneal": {"F": 0.90, "ka": 0.5},
    "sublingual": {"F": 0.70, "ka": 2.0},
}

VALID_ROUTES = frozenset(_ROUTE_PARAMS.keys())


@dataclass
class ExtravascularPKResult:
    """Result of extravascular PK simulation."""

    drug_name: str
    dose_mg: float
    route: str
    bioavailability: float
    ka_per_h: float
    times_h: list
    c_plasma_mg_L: list
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    t_half_absorption_h: float
    t_half_elimination_h: float
    notes: str


def _trapz(y: list, x: list) -> float:
    """Trapezoidal integration (manual, no numpy)."""
    total = 0.0
    for i in range(1, len(x)):
        total += 0.5 * (y[i - 1] + y[i]) * (x[i] - x[i - 1])
    return total


def _validate_inputs(
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    route: str,
) -> None:
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if route not in VALID_ROUTES:
        raise ValueError(f"route '{route}' not recognised. Valid options: {sorted(VALID_ROUTES)}")


def simulate_extravascular_pk(
    drug_name: str,
    dose_mg: float,
    route: str,
    cl_L_per_h: float,
    vd_L: float,
    t_end_h: float = 24.0,
    dt_h: float = 0.01,
    notes: str = "",
) -> ExtravascularPKResult:
    """Simulate extravascular PK using a 2-compartment Forward Euler ODE.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    dose_mg:
        Administered dose in mg (must be > 0).
    route:
        Absorption route: 'subcutaneous', 'intramuscular',
        'intraperitoneal', or 'sublingual'.
    cl_L_per_h:
        Systemic clearance in L/h (must be > 0).
    vd_L:
        Volume of distribution in L (must be > 0).
    t_end_h:
        Simulation end time in hours (default 24 h).
    dt_h:
        Forward Euler step size in hours (default 0.01 h).
    notes:
        Optional free-text notes.

    Returns
    -------
    ExtravascularPKResult
    """
    _validate_inputs(dose_mg, cl_L_per_h, vd_L, route)

    params = _ROUTE_PARAMS[route]
    F = params["F"]
    ka = params["ka"]
    ke = cl_L_per_h / vd_L  # elimination rate constant (1/h)

    # --- Forward Euler integration ---
    n_steps = max(1, int(round(t_end_h / dt_h)))
    times: list[float] = []
    c_plasma: list[float] = []

    a_depot = dose_mg
    c_sys = 0.0
    t = 0.0

    for _ in range(n_steps + 1):
        times.append(t)
        c_plasma.append(c_sys)

        # Derivatives
        da_depot = -ka * a_depot
        dc_sys = F * ka * a_depot / vd_L - ke * c_sys

        # Update state
        a_depot = a_depot + dt_h * da_depot
        c_sys = c_sys + dt_h * dc_sys
        t = t + dt_h

    # --- Derived PK metrics ---
    cmax = max(c_plasma)
    tmax = times[c_plasma.index(cmax)]
    auc = _trapz(c_plasma, times)
    t_half_abs = math.log(2.0) / ka
    t_half_elim = math.log(2.0) / ke

    result_notes = notes if notes else (f"F={F}, ka={ka}/h, ke={ke:.4f}/h")

    return ExtravascularPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route=route,
        bioavailability=F,
        ka_per_h=ka,
        times_h=times,
        c_plasma_mg_L=c_plasma,
        cmax_mg_L=round(cmax, 6),
        tmax_h=round(tmax, 4),
        auc_mg_h_per_L=round(auc, 6),
        t_half_absorption_h=round(t_half_abs, 6),
        t_half_elimination_h=round(t_half_elim, 6),
        notes=result_notes,
    )


def compare_extravascular_routes(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    t_end_h: float = 24.0,
    dt_h: float = 0.01,
) -> list[ExtravascularPKResult]:
    """Simulate all four extravascular routes and return results sorted by AUC descending.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    dose_mg:
        Administered dose in mg (must be > 0).
    cl_L_per_h:
        Systemic clearance in L/h (must be > 0).
    vd_L:
        Volume of distribution in L (must be > 0).
    t_end_h:
        Simulation end time in hours (default 24 h).
    dt_h:
        Forward Euler step size in hours (default 0.01 h).

    Returns
    -------
    list of ExtravascularPKResult sorted by auc_mg_h_per_L descending (4 items).
    """
    results = []
    for route in sorted(VALID_ROUTES):
        result = simulate_extravascular_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            route=route,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            t_end_h=t_end_h,
            dt_h=dt_h,
        )
        results.append(result)

    results.sort(key=lambda r: r.auc_mg_h_per_L, reverse=True)
    return results
