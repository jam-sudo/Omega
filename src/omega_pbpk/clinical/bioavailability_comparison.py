"""
Multi-Route Bioavailability Comparison.

Simulates 1-compartment PK for IV, oral, SC, IM, rectal, and intranasal
routes and compares absolute/relative bioavailability, Cmax, Tmax, AUC.
"""

from __future__ import annotations

from dataclasses import dataclass

VALID_ROUTES = ("iv", "oral", "sc", "im", "rectal", "intranasal")


@dataclass(frozen=True)
class BiAvailRouteResult:
    """Bioavailability result for one route."""

    drug_name: str
    route: str
    dose_mg: float
    f_absolute: float
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    relative_bioavailability_pct: float
    notes: str


def _route_f_and_ka(
    route: str,
    fh: float,
    fa: float,
    f_nasal: float = 0.5,
    f_swallowed_fraction: float = 0.3,
) -> tuple[float, float | None]:
    """
    Return (F_absolute, ka_per_h) for a route.

    ka=None means instantaneous (IV).
    """
    route = route.lower()
    if route == "iv":
        return 1.0, None
    elif route == "oral":
        f = fa * fh
        return f, 1.0
    elif route == "sc":
        f = 0.7 * fh
        return f, 0.25
    elif route == "im":
        f = 0.9 * fh
        return f, 0.5
    elif route == "rectal":
        f_bypass = 0.5
        f = f_bypass + (1.0 - f_bypass) * fh
        return f, 1.5
    elif route == "intranasal":
        f = f_nasal + (1.0 - f_nasal) * f_swallowed_fraction * fh
        return f, 1.5
    else:
        raise ValueError(f"Unknown route: {route}")


def _simulate_1cpt(
    dose_mg: float,
    f_abs: float,
    ka: float | None,
    cl_L_per_h: float,
    vd_L: float,
    t_end_h: float,
    dt_h: float,
) -> tuple[float, float, float]:
    """
    Simulate 1-compartment PK with forward Euler.

    Returns (cmax_mg_L, tmax_h, auc_mg_h_per_L).
    """
    ke = cl_L_per_h / vd_L

    if ka is None:
        # IV bolus: instantaneous dose
        c0 = (dose_mg * f_abs) / vd_L
        c = c0
        cmax = c0
        tmax = 0.0
        auc = 0.0
        t = 0.0
        c_prev = c0
        while t < t_end_h:
            c_next = c * (1.0 - ke * dt_h)
            if c_next < 0.0:
                c_next = 0.0
            auc += 0.5 * (c_prev + c_next) * dt_h
            c_prev = c_next
            c = c_next
            t += dt_h
        return cmax, tmax, auc
    else:
        # 1-cpt with first-order absorption
        dose_abs = dose_mg * f_abs  # mg in depot
        x_depot = dose_abs
        x_central = 0.0  # mg in central
        cmax = 0.0
        tmax = 0.0
        auc = 0.0
        t = 0.0
        c_prev = 0.0
        while t <= t_end_h:
            c = x_central / vd_L
            if c > cmax:
                cmax = c
                tmax = t
            auc += 0.5 * (c_prev + c) * dt_h if t > 0.0 else 0.0
            # Euler step
            dxd = -ka * x_depot
            dxc = ka * x_depot - ke * x_central
            x_depot += dxd * dt_h
            x_central += dxc * dt_h
            if x_depot < 0.0:
                x_depot = 0.0
            if x_central < 0.0:
                x_central = 0.0
            c_prev = x_central / vd_L
            t += dt_h
        return cmax, tmax, auc


def compare_bioavailability(
    drug_name: str,
    dose_mg: float,
    fh: float = 0.7,
    fa: float = 0.9,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    routes: list[str] | None = None,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> list[BiAvailRouteResult]:
    """
    Compare bioavailability across administration routes.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Dose in mg (same for all routes).
    fh:
        Hepatic bioavailability (0 < fh <= 1).
    fa:
        Fraction absorbed from GI (0 < fa <= 1).
    cl_L_per_h:
        Total clearance (L/h).
    vd_L:
        Volume of distribution (L).
    routes:
        Subset of routes to simulate. Default: all 6.
    t_end_h:
        Simulation end time (h).
    dt_h:
        Forward Euler time step (h).

    Returns
    -------
    List of BiAvailRouteResult sorted descending by f_absolute.
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be positive, got {dose_mg}")
    if not (0 < fh <= 1):
        raise ValueError(f"fh must be in (0, 1], got {fh}")
    if not (0 < fa <= 1):
        raise ValueError(f"fa must be in (0, 1], got {fa}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be positive, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be positive, got {vd_L}")

    if routes is None:
        routes = list(VALID_ROUTES)

    routes_lower = [r.lower() for r in routes]
    for r in routes_lower:
        if r not in VALID_ROUTES:
            raise ValueError(f"Unknown route: {r}. Valid: {VALID_ROUTES}")

    # Always compute IV for reference AUC
    f_iv, ka_iv = _route_f_and_ka("iv", fh, fa)
    _, _, auc_iv = _simulate_1cpt(dose_mg, f_iv, ka_iv, cl_L_per_h, vd_L, t_end_h, dt_h)

    results = []
    for route in routes_lower:
        f_abs, ka = _route_f_and_ka(route, fh, fa)
        cmax, tmax, auc = _simulate_1cpt(dose_mg, f_abs, ka, cl_L_per_h, vd_L, t_end_h, dt_h)

        rel_ba_pct = (auc / auc_iv * 100.0) if auc_iv > 0.0 else 0.0

        note_parts = [
            f"Route: {route.upper()}",
            f"F_abs={f_abs:.3f}, ke={cl_L_per_h / vd_L:.4f}/h",
            f"Cmax={cmax:.4f} mg/L at Tmax={tmax:.2f}h",
            f"AUC={auc:.3f} mg*h/L, F_rel={rel_ba_pct:.1f}% vs IV",
        ]
        if route == "iv":
            note_parts.append("Reference route (F=1.0)")
        elif route == "oral":
            note_parts.append(f"First-pass extraction: fa={fa:.2f}, fh={fh:.2f}")
        elif route == "sc":
            note_parts.append("Subcutaneous slow absorption (ka=0.25/h)")
        elif route == "im":
            note_parts.append("Intramuscular moderate absorption (ka=0.5/h)")
        elif route == "rectal":
            note_parts.append("Partial portal bypass (50% avoids first pass)")
        elif route == "intranasal":
            note_parts.append("Nasal fraction direct + swallowed fraction via oral")

        results.append(
            BiAvailRouteResult(
                drug_name=drug_name,
                route=route,
                dose_mg=dose_mg,
                f_absolute=f_abs,
                cmax_mg_L=cmax,
                tmax_h=tmax,
                auc_mg_h_per_L=auc,
                relative_bioavailability_pct=rel_ba_pct,
                notes="; ".join(note_parts),
            )
        )

    results.sort(key=lambda r: r.f_absolute, reverse=True)
    return results


def rank_routes(
    drug_name: str,
    dose_mg: float,
    fh: float = 0.7,
    fa: float = 0.9,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    routes: list[str] | None = None,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> dict:
    """
    Rank routes and identify best and worst for the drug.

    Returns
    -------
    Dict with keys: best_route, worst_route, results (list[BiAvailRouteResult]).
    """
    results = compare_bioavailability(
        drug_name=drug_name,
        dose_mg=dose_mg,
        fh=fh,
        fa=fa,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
        routes=routes,
        t_end_h=t_end_h,
        dt_h=dt_h,
    )
    best_route = results[0].route if results else ""
    worst_route = results[-1].route if results else ""
    return {
        "best_route": best_route,
        "worst_route": worst_route,
        "results": results,
    }
