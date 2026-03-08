"""Phase 281 — Drug Prodrug Activation Kinetics.

Simulate prodrug -> active drug conversion kinetics using a two-compartment
sequential model: prodrug in systemic -> active drug in systemic, with
first-pass conversion and forward-Euler ODE integration.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProdrugPKResult:
    """Result of a prodrug activation PK simulation."""

    drug_name: str
    dose_mg: float
    route: str
    k_conv_per_h: float
    times_h: list
    c_prodrug_mg_L: list
    c_active_mg_L: list
    auc_prodrug_mg_h_per_L: float
    cmax_prodrug_mg_L: float
    tmax_prodrug_h: float
    auc_active_mg_h_per_L: float
    cmax_active_mg_L: float
    tmax_active_h: float
    activation_efficiency: float
    lag_time_h: float
    notes: str


def _trapezoid_auc(x: list, y: list) -> float:
    """Manual trapezoidal AUC calculation."""
    return sum(0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]) for i in range(1, len(x)))


def _validate_prodrug_inputs(
    dose_mg: float,
    cl_prodrug_L_per_h: float,
    vd_prodrug_L: float,
    k_conv_per_h: float,
    cl_active_L_per_h: float,
    vd_active_L: float,
    t_end_h: float,
    f_oral_prodrug: float,
    route: str,
) -> None:
    """Validate all inputs for prodrug simulation."""
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_prodrug_L_per_h <= 0:
        raise ValueError(f"cl_prodrug_L_per_h must be > 0, got {cl_prodrug_L_per_h}")
    if vd_prodrug_L <= 0:
        raise ValueError(f"vd_prodrug_L must be > 0, got {vd_prodrug_L}")
    if k_conv_per_h <= 0:
        raise ValueError(f"k_conv_per_h must be > 0, got {k_conv_per_h}")
    if cl_active_L_per_h <= 0:
        raise ValueError(f"cl_active_L_per_h must be > 0, got {cl_active_L_per_h}")
    if vd_active_L <= 0:
        raise ValueError(f"vd_active_L must be > 0, got {vd_active_L}")
    if t_end_h <= 0:
        raise ValueError(f"t_end_h must be > 0, got {t_end_h}")
    if not (0 < f_oral_prodrug <= 1):
        raise ValueError(f"f_oral_prodrug must be in (0, 1], got {f_oral_prodrug}")
    if route not in {"oral", "iv_bolus"}:
        raise ValueError(f"route must be 'oral' or 'iv_bolus', got {route!r}")


def simulate_prodrug_pk(
    drug_name: str,
    dose_mg: float,
    route: str,
    f_oral_prodrug: float,
    cl_prodrug_L_per_h: float,
    vd_prodrug_L: float,
    k_conv_per_h: float,
    cl_active_L_per_h: float,
    vd_active_L: float,
    t_end_h: float = 24.0,
    dt_h: float = 0.01,
) -> ProdrugPKResult:
    """Simulate prodrug -> active drug conversion kinetics.

    Parameters
    ----------
    drug_name:
        Name of the prodrug compound.
    dose_mg:
        Administered dose in milligrams.
    route:
        Route of administration: "oral" or "iv_bolus".
    f_oral_prodrug:
        Oral bioavailability fraction of the prodrug (0 < f <= 1).
    cl_prodrug_L_per_h:
        Clearance of prodrug (L/h).
    vd_prodrug_L:
        Volume of distribution of prodrug (L).
    k_conv_per_h:
        First-order conversion rate prodrug -> active drug (1/h).
    cl_active_L_per_h:
        Clearance of active drug (L/h).
    vd_active_L:
        Volume of distribution of active drug (L).
    t_end_h:
        Simulation end time in hours.
    dt_h:
        Forward-Euler step size in hours.

    Returns
    -------
    ProdrugPKResult
        Dataclass with concentration-time profiles and PK metrics.
    """
    _validate_prodrug_inputs(
        dose_mg,
        cl_prodrug_L_per_h,
        vd_prodrug_L,
        k_conv_per_h,
        cl_active_L_per_h,
        vd_active_L,
        t_end_h,
        f_oral_prodrug,
        route,
    )

    ka = 1.0  # absorption rate constant (1/h)

    # Derived rate constants
    ke_prodrug = cl_prodrug_L_per_h / vd_prodrug_L
    ke_active = cl_active_L_per_h / vd_active_L

    # Initial conditions
    if route == "oral":
        a_gut = dose_mg * f_oral_prodrug
        a_prodrug = 0.0
    else:  # iv_bolus
        a_gut = 0.0
        a_prodrug = dose_mg

    a_active = 0.0

    times_h: list = []
    c_prodrug_mg_L: list = []
    c_active_mg_L: list = []

    t = 0.0
    n_steps = int(round(t_end_h / dt_h))

    for _ in range(n_steps + 1):
        c_prod = a_prodrug / vd_prodrug_L
        c_act = a_active / vd_active_L

        times_h.append(t)
        c_prodrug_mg_L.append(c_prod)
        c_active_mg_L.append(c_act)

        if _ == n_steps:
            break

        # Forward Euler derivatives
        if route == "oral":
            dA_gut = -ka * a_gut
            dA_prodrug = ka * a_gut - ke_prodrug * a_prodrug - k_conv_per_h * a_prodrug
        else:
            dA_gut = 0.0
            dA_prodrug = -ke_prodrug * a_prodrug - k_conv_per_h * a_prodrug

        dA_active = k_conv_per_h * a_prodrug - ke_active * a_active

        a_gut += dA_gut * dt_h
        if a_gut < 0.0:
            a_gut = 0.0
        a_prodrug += dA_prodrug * dt_h
        if a_prodrug < 0.0:
            a_prodrug = 0.0
        a_active += dA_active * dt_h
        if a_active < 0.0:
            a_active = 0.0

        t += dt_h

    # PK metrics for prodrug
    auc_prodrug = _trapezoid_auc(times_h, c_prodrug_mg_L)
    cmax_prodrug = max(c_prodrug_mg_L)
    tmax_prodrug = times_h[c_prodrug_mg_L.index(cmax_prodrug)]

    # PK metrics for active drug
    auc_active = _trapezoid_auc(times_h, c_active_mg_L)
    cmax_active = max(c_active_mg_L)
    tmax_active = times_h[c_active_mg_L.index(cmax_active)]

    # Activation efficiency
    total_auc = auc_prodrug + auc_active
    if total_auc > 0:
        activation_efficiency = auc_active / total_auc
    else:
        activation_efficiency = 0.0

    # Lag time: time for active drug to reach 10% of cmax_active
    threshold = 0.1 * cmax_active
    lag_time_h = 0.0
    if threshold > 0:
        for i, c in enumerate(c_active_mg_L):
            if c >= threshold:
                lag_time_h = times_h[i]
                break

    notes = (
        f"Route: {route}. "
        f"k_conv={k_conv_per_h:.3f}/h. "
        f"Activation efficiency={activation_efficiency:.3f}."
    )

    return ProdrugPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route=route,
        k_conv_per_h=k_conv_per_h,
        times_h=times_h,
        c_prodrug_mg_L=c_prodrug_mg_L,
        c_active_mg_L=c_active_mg_L,
        auc_prodrug_mg_h_per_L=auc_prodrug,
        cmax_prodrug_mg_L=cmax_prodrug,
        tmax_prodrug_h=tmax_prodrug,
        auc_active_mg_h_per_L=auc_active,
        cmax_active_mg_L=cmax_active,
        tmax_active_h=tmax_active,
        activation_efficiency=activation_efficiency,
        lag_time_h=lag_time_h,
        notes=notes,
    )


def compare_conversion_rates(
    drug_name: str,
    dose_mg: float,
    k_conv_list: list,
    cl_prodrug_L_per_h: float,
    cl_active_L_per_h: float,
    vd_prodrug_L: float,
    vd_active_L: float,
    route: str = "oral",
    f_oral_prodrug: float = 0.9,
    t_end_h: float = 24.0,
    dt_h: float = 0.01,
) -> list:
    """Compare prodrug activation across multiple conversion rates.

    Parameters
    ----------
    drug_name:
        Name of the prodrug compound.
    dose_mg:
        Administered dose in milligrams.
    k_conv_list:
        List of first-order conversion rate constants (1/h) to compare.
    cl_prodrug_L_per_h:
        Clearance of prodrug (L/h).
    cl_active_L_per_h:
        Clearance of active drug (L/h).
    vd_prodrug_L:
        Volume of distribution of prodrug (L).
    vd_active_L:
        Volume of distribution of active drug (L).
    route:
        Route of administration.
    f_oral_prodrug:
        Oral bioavailability fraction.
    t_end_h:
        Simulation end time in hours.
    dt_h:
        Forward-Euler step size in hours.

    Returns
    -------
    list[ProdrugPKResult]
        Results sorted by cmax_active descending.
    """
    results = []
    for k_conv in k_conv_list:
        result = simulate_prodrug_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            route=route,
            f_oral_prodrug=f_oral_prodrug,
            cl_prodrug_L_per_h=cl_prodrug_L_per_h,
            vd_prodrug_L=vd_prodrug_L,
            k_conv_per_h=k_conv,
            cl_active_L_per_h=cl_active_L_per_h,
            vd_active_L=vd_active_L,
            t_end_h=t_end_h,
            dt_h=dt_h,
        )
        results.append(result)

    results.sort(key=lambda r: r.cmax_active_mg_L, reverse=True)
    return results
