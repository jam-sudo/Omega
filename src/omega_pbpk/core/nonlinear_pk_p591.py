"""Phase 591 — Dose-Dependent Nonlinear PK with Capacity-Limited Elimination."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NonlinearPKResult:
    """Result of a nonlinear PK simulation."""

    drug_name: str
    dose_mg: float
    vmax_mg_h: float
    km_mg_L: float
    vd_L: float
    times_h: list
    c_plasma_mg_L: list
    auc_mg_h_per_L: float
    cmax_mg_L: float
    tmax_h: float
    t_half_effective_h: float
    dose_proportionality_index: float
    nonlinearity_class: str
    notes: str


def _trapz(x: list, y: list) -> float:
    """Manual trapezoidal integration."""
    return sum(0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]) for i in range(1, len(x)))


def _simulate_core(
    dose_mg: float,
    vmax_mg_h: float,
    km_mg_L: float,
    vd_L: float,
    cl_linear_L_per_h: float,
    route: str,
    ka_per_h: float,
    f_oral: float,
    t_end_h: float,
    dt_h: float,
) -> tuple[list, list]:
    """Run forward Euler simulation; returns (times_h, c_plasma_mg_L)."""
    n = max(int(round(t_end_h / dt_h)), 1)
    times = [i * dt_h for i in range(n + 1)]
    conc = [0.0] * (n + 1)

    if route == "iv":
        conc[0] = dose_mg / vd_L
        for i in range(n):
            c = conc[i]
            mm_elim = vmax_mg_h * c / (km_mg_L + c) if (km_mg_L + c) > 0 else 0.0
            lin_elim = cl_linear_L_per_h * c
            dC = -(mm_elim + lin_elim) / vd_L
            conc[i + 1] = max(c + dC * dt_h, 0.0)
    else:
        # oral
        a_gut = dose_mg * f_oral
        conc[0] = 0.0
        for i in range(n):
            abs_rate = ka_per_h * a_gut
            a_gut = max(a_gut - abs_rate * dt_h, 0.0)
            c = conc[i]
            mm_elim = vmax_mg_h * c / (km_mg_L + c) if (km_mg_L + c) > 0 else 0.0
            lin_elim = cl_linear_L_per_h * c
            dC = abs_rate / vd_L - (mm_elim + lin_elim) / vd_L
            conc[i + 1] = max(c + dC * dt_h, 0.0)

    return times, conc


def simulate_nonlinear_pk(
    drug_name: str,
    dose_mg: float,
    vmax_mg_h: float,
    km_mg_L: float,
    vd_L: float,
    cl_linear_L_per_h: float = 0.0,
    route: str = "iv",
    ka_per_h: float = 1.0,
    f_oral: float = 0.9,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> NonlinearPKResult:
    """Simulate 1-compartment PK with Michaelis-Menten (saturable) elimination.

    Parameters
    ----------
    drug_name : str
    dose_mg : float
        Dose in mg.
    vmax_mg_h : float
        Maximum elimination rate (mg/h).
    km_mg_L : float
        Michaelis constant (mg/L).
    vd_L : float
        Volume of distribution (L).
    cl_linear_L_per_h : float
        Additional linear clearance (L/h), default 0.
    route : str
        "iv" or "oral".
    ka_per_h : float
        First-order absorption rate (oral only, per h).
    f_oral : float
        Oral bioavailability fraction (0, 1].
    t_end_h : float
        Simulation end time (h).
    dt_h : float
        Integration step (h).
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if vmax_mg_h <= 0:
        raise ValueError("vmax_mg_h must be > 0")
    if km_mg_L <= 0:
        raise ValueError("km_mg_L must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if cl_linear_L_per_h < 0:
        raise ValueError("cl_linear_L_per_h must be >= 0")
    if not (0 < f_oral <= 1.0):
        raise ValueError("f_oral must be in (0, 1]")
    if route not in ("iv", "oral"):
        raise ValueError("route must be 'iv' or 'oral'")

    times, conc = _simulate_core(
        dose_mg,
        vmax_mg_h,
        km_mg_L,
        vd_L,
        cl_linear_L_per_h,
        route,
        ka_per_h,
        f_oral,
        t_end_h,
        dt_h,
    )

    auc = _trapz(times, conc)
    cmax = max(conc)
    tmax_idx = conc.index(cmax)
    tmax = times[tmax_idx]

    # Effective half-life: time from tmax to when conc drops to Cmax/2
    half_cmax = cmax / 2.0
    t_half_eff = t_end_h  # default if never reached
    for i in range(tmax_idx, len(conc)):
        if conc[i] <= half_cmax:
            # interpolate
            if i > 0 and conc[i - 1] > half_cmax:
                dt_seg = times[i] - times[i - 1]
                dc = conc[i - 1] - conc[i]
                if dc > 0:
                    frac = (conc[i - 1] - half_cmax) / dc
                    t_half_eff = times[i - 1] + frac * dt_seg - tmax
                else:
                    t_half_eff = times[i] - tmax
            else:
                t_half_eff = times[i] - tmax
            t_half_eff = max(t_half_eff, 0.0)
            break

    # Nonlinearity: simulate at 2× dose
    times2, conc2 = _simulate_core(
        dose_mg * 2,
        vmax_mg_h,
        km_mg_L,
        vd_L,
        cl_linear_L_per_h,
        route,
        ka_per_h,
        f_oral,
        t_end_h,
        dt_h,
    )
    auc_double = _trapz(times2, conc2)

    ratio = (auc_double / (2.0 * auc)) if auc > 0 else 1.0
    if ratio < 1.1:
        nonlinearity_class = "linear"
    elif ratio < 1.5:
        nonlinearity_class = "weakly_nonlinear"
    else:
        nonlinearity_class = "strongly_nonlinear"

    dose_prop_index = auc / dose_mg if dose_mg > 0 else 0.0

    notes = (
        f"AUC ratio at 2x dose: {ratio:.3f}. "
        f"Route: {route}. "
        f"Nonlinearity class: {nonlinearity_class}."
    )

    return NonlinearPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        vmax_mg_h=vmax_mg_h,
        km_mg_L=km_mg_L,
        vd_L=vd_L,
        times_h=times,
        c_plasma_mg_L=conc,
        auc_mg_h_per_L=auc,
        cmax_mg_L=cmax,
        tmax_h=tmax,
        t_half_effective_h=t_half_eff,
        dose_proportionality_index=dose_prop_index,
        nonlinearity_class=nonlinearity_class,
        notes=notes,
    )


def dose_proportionality_analysis(
    drug_name: str,
    doses_mg: list,
    vmax_mg_h: float,
    km_mg_L: float,
    vd_L: float,
) -> list:
    """Run nonlinear PK simulation across multiple doses.

    Returns list of NonlinearPKResult sorted by dose ascending.
    """
    results = []
    for d in doses_mg:
        r = simulate_nonlinear_pk(
            drug_name=drug_name,
            dose_mg=d,
            vmax_mg_h=vmax_mg_h,
            km_mg_L=km_mg_L,
            vd_L=vd_L,
        )
        results.append(r)
    results.sort(key=lambda r: r.dose_mg)
    return results


__all__ = [
    "NonlinearPKResult",
    "simulate_nonlinear_pk",
    "dose_proportionality_analysis",
]
