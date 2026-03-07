"""Subcutaneous and intramuscular depot injection PK model.

Supports zero/first-order absorption from depot with formulation-specific
absorption rate constants.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Formulation default ka values (per hour)
_FORMULATION_KA: dict[str, float] = {
    "aqueous": 0.5,
    "suspension": 0.1,
    "oil": 0.02,
    "polymer_microsphere": 0.005,
}

# Route modifier on ka
_ROUTE_KA_MODIFIER: dict[str, float] = {
    "sc": 1.0,
    "im": 1.5,
}

_VALID_ROUTES = {"sc", "im"}
_VALID_FORMULATIONS = set(_FORMULATION_KA.keys())


@dataclass
class DepotInjectionResult:
    """Result of a depot injection simulation."""

    drug_name: str
    dose_mg: float
    route: str
    formulation: str
    times_h: list = field(default_factory=list)
    c_depot_mg: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    cmax_mg_L: float = 0.0
    tmax_h: float = 0.0
    auc_mg_h_L: float = 0.0
    t_half_absorption_h: float = 0.0
    t_half_elimination_h: float = 0.0
    f_absorbed_pct: float = 0.0
    notes: str = ""


def simulate_depot_injection(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    route: str = "sc",
    formulation: str = "aqueous",
    ka_per_h: float | None = None,
    f_bioavail: float = 1.0,
    lag_time_h: float = 0.0,
    t_end_h: float = 72.0,
    dt_h: float = 0.1,
) -> DepotInjectionResult:
    """Simulate depot injection PK using first-order absorption model.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Administered dose in mg.
    cl_L_per_h:
        Total clearance in L/h.
    vd_L:
        Volume of distribution in L.
    route:
        Injection route: "sc" (subcutaneous) or "im" (intramuscular).
    formulation:
        Formulation type: "aqueous", "suspension", "oil", "polymer_microsphere".
    ka_per_h:
        Absorption rate constant (1/h). Auto-calculated from formulation if None.
    f_bioavail:
        Fraction bioavailable (0, 1].
    lag_time_h:
        Lag time before absorption begins (h).
    t_end_h:
        Simulation end time (h).
    dt_h:
        Forward Euler time step (h).

    Returns
    -------
    DepotInjectionResult
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if route not in _VALID_ROUTES:
        raise ValueError(f"route must be one of {_VALID_ROUTES}, got '{route}'")
    if formulation not in _VALID_FORMULATIONS:
        raise ValueError(f"formulation must be one of {_VALID_FORMULATIONS}, got '{formulation}'")
    if not (0.0 < f_bioavail <= 1.0):
        raise ValueError(f"f_bioavail must be in (0, 1], got {f_bioavail}")
    if t_end_h <= 0:
        raise ValueError(f"t_end_h must be > 0, got {t_end_h}")

    # Determine effective ka
    base_ka = _FORMULATION_KA[formulation] if ka_per_h is None else ka_per_h
    ka_eff = base_ka * _ROUTE_KA_MODIFIER[route]

    # Derived half-lives
    t_half_absorption = math.log(2) / ka_eff
    t_half_elimination = math.log(2) * vd_L / cl_L_per_h

    # Initial conditions
    a_depot_init = dose_mg * f_bioavail
    a_depot = a_depot_init
    c_plasma = 0.0

    # Storage
    n_steps = int(t_end_h / dt_h) + 1
    times_h: list[float] = []
    c_depot_mg: list[float] = []
    c_plasma_mg_L: list[float] = []

    # Forward Euler integration
    t = 0.0
    for _ in range(n_steps):
        times_h.append(t)
        c_depot_mg.append(a_depot)
        c_plasma_mg_L.append(c_plasma)

        # Absorption only after lag time
        absorbing = t >= lag_time_h

        da_depot = -ka_eff * a_depot if absorbing else 0.0
        dc_plasma = (ka_eff * a_depot / vd_L if absorbing else 0.0) - (cl_L_per_h / vd_L) * c_plasma

        a_depot = max(0.0, a_depot + da_depot * dt_h)
        c_plasma = max(0.0, c_plasma + dc_plasma * dt_h)
        t += dt_h

    # Compute metrics
    cmax_mg_L = max(c_plasma_mg_L)
    tmax_h = times_h[c_plasma_mg_L.index(cmax_mg_L)]

    # Trapezoidal AUC
    auc = 0.0
    for i in range(1, len(times_h)):
        auc += 0.5 * (c_plasma_mg_L[i - 1] + c_plasma_mg_L[i]) * (times_h[i] - times_h[i - 1])

    # Fraction absorbed
    final_depot = c_depot_mg[-1]
    f_absorbed_pct = max(0.0, 100.0 * (1.0 - final_depot / a_depot_init))

    route_label = "Subcutaneous" if route == "sc" else "Intramuscular"
    notes = (
        f"{route_label} {formulation} depot injection. "
        f"ka_eff={ka_eff:.4f} 1/h, "
        f"t1/2_abs={t_half_absorption:.1f}h, "
        f"t1/2_elim={t_half_elimination:.1f}h."
    )

    return DepotInjectionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route=route,
        formulation=formulation,
        times_h=times_h,
        c_depot_mg=c_depot_mg,
        c_plasma_mg_L=c_plasma_mg_L,
        cmax_mg_L=cmax_mg_L,
        tmax_h=tmax_h,
        auc_mg_h_L=auc,
        t_half_absorption_h=t_half_absorption,
        t_half_elimination_h=t_half_elimination,
        f_absorbed_pct=f_absorbed_pct,
        notes=notes,
    )


def compare_formulations(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    route: str = "sc",
    t_end_h: float = 168.0,
) -> list:
    """Return list of DepotInjectionResult for all 4 formulations.

    Results are sorted by tmax (ascending), which corresponds to the natural
    order: aqueous < suspension < oil < polymer_microsphere.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Administered dose in mg.
    cl_L_per_h:
        Total clearance in L/h.
    vd_L:
        Volume of distribution in L.
    route:
        Injection route: "sc" or "im".
    t_end_h:
        Simulation end time in hours.

    Returns
    -------
    list of DepotInjectionResult sorted by tmax ascending
    """
    results = []
    for formulation in ("aqueous", "suspension", "oil", "polymer_microsphere"):
        result = simulate_depot_injection(
            drug_name=drug_name,
            dose_mg=dose_mg,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            route=route,
            formulation=formulation,
            t_end_h=t_end_h,
        )
        results.append(result)

    # Sort by tmax ascending
    results.sort(key=lambda r: r.tmax_h)
    return results
