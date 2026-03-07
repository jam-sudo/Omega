"""
Phase 522 — Multi-Route Bioavailability Comparison

Compare drug bioavailability and PK across different administration routes
using 1-compartment forward Euler simulation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

SUPPORTED_ROUTES = {"iv", "oral", "sc", "im"}


@dataclass
class RoutePKResult:
    """PK simulation result for a single administration route."""

    drug_name: str
    route: str
    dose_mg: float
    times_h: list
    c_plasma_mg_L: list
    cmax: float
    tmax_h: float
    auc: float
    f_bioavailable: float
    t_half_h: float
    notes: str


def _trapezoidal_auc(times: list[float], concentrations: list[float]) -> float:
    """Compute AUC using the trapezoidal rule."""
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        auc += 0.5 * (concentrations[i] + concentrations[i - 1]) * dt
    return auc


def simulate_route_pk(
    drug_name: str,
    dose_mg: float,
    route: str,
    cl_L_per_h: float,
    vd_L: float,
    ka_per_h: float = 1.0,
    f_oral: float = 0.9,
    f_sc: float = 0.8,
    lag_time_h: float = 0.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> RoutePKResult:
    """
    Simulate 1-compartment PK for a given administration route.

    Routes
    ------
    - 'iv'   : Instantaneous bolus (f=1)
    - 'oral' : First-order absorption with ka_per_h and f_oral
    - 'sc'   : Subcutaneous; ka_per_h/2, f_sc, lag_time_h applied
    - 'im'   : Intramuscular; ka_per_h*1.5, f=0.95

    Parameters
    ----------
    drug_name   : Drug identifier string.
    dose_mg     : Dose in mg.
    route       : One of 'iv', 'oral', 'sc', 'im'.
    cl_L_per_h  : Clearance (L/h).
    vd_L        : Volume of distribution (L).
    ka_per_h    : Absorption rate constant (1/h) for oral route.
    f_oral      : Oral bioavailability fraction.
    f_sc        : Subcutaneous bioavailability fraction.
    lag_time_h  : Lag time before absorption starts (h).
    t_end_h     : Simulation end time (h).
    dt_h        : Time step (h).

    Returns
    -------
    RoutePKResult
    """
    if route not in SUPPORTED_ROUTES:
        raise ValueError(f"route must be one of {sorted(SUPPORTED_ROUTES)}; got '{route}'")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")

    ke = cl_L_per_h / vd_L  # elimination rate constant (1/h)
    t_half_h = math.log(2.0) / ke

    # Determine route-specific parameters
    if route == "iv":
        f = 1.0
        ka = None  # not used
        lag = 0.0
    elif route == "oral":
        f = f_oral
        ka = ka_per_h
        lag = lag_time_h
    elif route == "sc":
        f = f_sc
        ka = ka_per_h / 2.0
        lag = lag_time_h
    else:  # im
        f = 0.95
        ka = ka_per_h * 1.5
        lag = 0.0

    # Build time array
    n_steps = int(round(t_end_h / dt_h))
    times: list[float] = [i * dt_h for i in range(n_steps + 1)]

    concentrations: list[float] = []

    if route == "iv":
        # C(0) = dose_mg / vd_L (instantaneous bolus)
        c0 = dose_mg / vd_L
        for t in times:
            concentrations.append(c0 * math.exp(-ke * t))
    else:
        # Forward Euler for depot + central compartment
        absorbed_dose = f * dose_mg  # mg
        # depot state: amount in depot (mg); central: amount in plasma (mg)
        depot = absorbed_dose
        central = 0.0  # mg

        conc_list: list[float] = []
        conc_list.append(central / vd_L)

        for i in range(1, n_steps + 1):
            t_current = (i - 1) * dt_h
            # lag time: no absorption until t > lag
            if t_current < lag:
                eff_ka = 0.0
            else:
                eff_ka = ka  # type: ignore[assignment]

            absorption_rate = eff_ka * depot  # mg/h
            elimination_rate = ke * central  # mg/h

            # Forward Euler step
            depot_new = depot - absorption_rate * dt_h
            central_new = central + (absorption_rate - elimination_rate) * dt_h

            # Clamp negatives
            depot = max(0.0, depot_new)
            central = max(0.0, central_new)

            conc_list.append(central / vd_L)

        concentrations = conc_list

    auc = _trapezoidal_auc(times, concentrations)

    cmax = max(concentrations)
    tmax_h = times[concentrations.index(cmax)]

    return RoutePKResult(
        drug_name=drug_name,
        route=route,
        dose_mg=dose_mg,
        times_h=times,
        c_plasma_mg_L=concentrations,
        cmax=cmax,
        tmax_h=tmax_h,
        auc=auc,
        f_bioavailable=f,
        t_half_h=t_half_h,
        notes=f"1-cpt {route} simulation; ke={ke:.4f}/h; t½={t_half_h:.2f}h",
    )


def compare_routes(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    routes: list[str] | None = None,
    **kwargs,
) -> list[RoutePKResult]:
    """
    Simulate PK for multiple administration routes and compare.

    Parameters
    ----------
    drug_name   : Drug identifier string.
    dose_mg     : Dose in mg (same for all routes).
    cl_L_per_h  : Clearance (L/h).
    vd_L        : Volume of distribution (L).
    routes      : List of routes to simulate. Defaults to ['iv','oral','sc','im'].
    **kwargs    : Additional keyword arguments passed to simulate_route_pk.

    Returns
    -------
    list[RoutePKResult]
        Sorted by AUC descending.
    """
    if routes is None:
        routes = ["iv", "oral", "sc", "im"]

    results: list[RoutePKResult] = []
    for route in routes:
        result = simulate_route_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            route=route,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            **kwargs,
        )
        results.append(result)

    results.sort(key=lambda r: r.auc, reverse=True)
    return results


def relative_bioavailability(
    test_result: RoutePKResult,
    reference_result: RoutePKResult,
) -> dict:
    """
    Calculate relative bioavailability of test vs reference route.

    F_rel = (AUC_test / dose_test) / (AUC_ref / dose_ref) * 100

    Parameters
    ----------
    test_result      : PK result for the test route.
    reference_result : PK result for the reference route.

    Returns
    -------
    dict with keys: F_rel_pct, cmax_ratio, auc_ratio, notes
    """
    if reference_result.auc == 0:
        raise ValueError("Reference AUC is zero; cannot compute relative bioavailability.")

    auc_ratio = test_result.auc / reference_result.auc
    dose_normalized_test = test_result.auc / test_result.dose_mg
    dose_normalized_ref = reference_result.auc / reference_result.dose_mg

    if dose_normalized_ref == 0:
        f_rel_pct = 0.0
    else:
        f_rel_pct = dose_normalized_test / dose_normalized_ref * 100.0

    cmax_ratio = test_result.cmax / reference_result.cmax if reference_result.cmax != 0 else 0.0

    notes = (
        f"F_rel({test_result.route} vs {reference_result.route}) = {f_rel_pct:.1f}%; "
        f"AUC ratio = {auc_ratio:.3f}; Cmax ratio = {cmax_ratio:.3f}"
    )

    return {
        "F_rel_pct": f_rel_pct,
        "cmax_ratio": cmax_ratio,
        "auc_ratio": auc_ratio,
        "notes": notes,
    }
