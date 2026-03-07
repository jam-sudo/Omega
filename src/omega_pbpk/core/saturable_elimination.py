"""Saturable (Michaelis-Menten) elimination PK with dose-dependent behavior."""

from __future__ import annotations

import math


def mm_elimination_rate(
    concentration_mg_L: float,
    vmax_mg_L_per_h: float,
    km_mg_L: float,
) -> float:
    """Michaelis-Menten elimination rate.

    v = Vmax * C / (Km + C)

    Returns rate in mg/L/h.
    """
    if vmax_mg_L_per_h <= 0:
        raise ValueError("vmax_mg_L_per_h must be > 0")
    if km_mg_L <= 0:
        raise ValueError("km_mg_L must be > 0")
    if concentration_mg_L < 0:
        raise ValueError("concentration_mg_L must be >= 0")

    return vmax_mg_L_per_h * concentration_mg_L / (km_mg_L + concentration_mg_L)


def apparent_ke(
    concentration_mg_L: float,
    vmax_mg_L_per_h: float,
    km_mg_L: float,
    vd_L: float,
) -> float:
    """Instantaneous apparent elimination rate constant.

    ke_apparent = Vmax / (Vd * (Km + C))

    Returns ke in 1/h.
    """
    if vmax_mg_L_per_h <= 0:
        raise ValueError("vmax_mg_L_per_h must be > 0")
    if km_mg_L <= 0:
        raise ValueError("km_mg_L must be > 0")
    if concentration_mg_L < 0:
        raise ValueError("concentration_mg_L must be >= 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")

    return vmax_mg_L_per_h / (vd_L * (km_mg_L + concentration_mg_L))


def simulate_mm_1cpt(
    drug_name: str,
    dose_mg: float,
    vd_L: float,
    vmax_mg_L_per_h: float,
    km_mg_L: float,
    ka_per_h: float = 0.0,
    route: str = "iv",
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> dict:
    """Simulate 1-compartment PK with Michaelis-Menten elimination.

    Routes:
      'iv'   - IV bolus: Cp0 = dose / Vd
      'oral' - 1-cpt with gut absorption: dA_gut/dt = -ka*A_gut,
               dCp/dt = ka*A_gut/Vd - vmax*Cp/(Km+Cp)

    dCp/dt uses vmax in mg/L/h:
      dCp/dt = input_per_Vd - vmax_mg_L_per_h * Cp / (km_mg_L + Cp)
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if vmax_mg_L_per_h <= 0:
        raise ValueError("vmax_mg_L_per_h must be > 0")
    if km_mg_L <= 0:
        raise ValueError("km_mg_L must be > 0")
    if route not in ("iv", "oral"):
        raise ValueError("route must be 'iv' or 'oral'")
    if route == "oral" and ka_per_h <= 0:
        raise ValueError("ka_per_h must be > 0 for oral route")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be > 0")
    if dt_h <= 0:
        raise ValueError("dt_h must be > 0")

    n_steps = int(math.ceil(t_end_h / dt_h))
    times_h: list[float] = []
    c_plasma_mg_L: list[float] = []

    if route == "iv":
        cp = dose_mg / vd_L
        a_gut = 0.0
    else:
        cp = 0.0
        a_gut = dose_mg

    t = 0.0
    times_h.append(t)
    c_plasma_mg_L.append(cp)

    for _ in range(n_steps):
        # Elimination rate
        elim = vmax_mg_L_per_h * cp / (km_mg_L + cp) if cp > 0 else 0.0

        if route == "oral":
            dA_gut = -ka_per_h * a_gut
            dCp = (ka_per_h * a_gut / vd_L) - elim
            a_gut = max(0.0, a_gut + dA_gut * dt_h)
        else:
            dCp = -elim

        cp = max(0.0, cp + dCp * dt_h)
        t += dt_h
        times_h.append(t)
        c_plasma_mg_L.append(cp)

    # Cmax and Tmax
    cmax = max(c_plasma_mg_L)
    tmax_h = times_h[c_plasma_mg_L.index(cmax)]

    # AUC via trapezoidal rule
    auc = 0.0
    for i in range(1, len(times_h)):
        auc += 0.5 * (c_plasma_mg_L[i - 1] + c_plasma_mg_L[i]) * (times_h[i] - times_h[i - 1])

    # Initial apparent half-life (at Cp0 for IV, or at Cmax for oral)
    c_ref = dose_mg / vd_L if route == "iv" else cmax
    if c_ref > 0:
        ke_init = vmax_mg_L_per_h / (vd_L * (km_mg_L + c_ref))
        t_half_initial_h = math.log(2.0) / ke_init if ke_init > 0 else float("inf")
    else:
        t_half_initial_h = float("inf")

    return {
        "drug_name": drug_name,
        "route": route,
        "dose_mg": dose_mg,
        "times_h": times_h,
        "c_plasma_mg_L": c_plasma_mg_L,
        "cmax": cmax,
        "tmax_h": tmax_h,
        "auc": auc,
        "t_half_initial_h": t_half_initial_h,
    }


def dose_proportionality_check(
    doses: list,
    vd_L: float,
    vmax_mg_L_per_h: float,
    km_mg_L: float,
    t_end_h: float = 24.0,
) -> list:
    """Check dose proportionality by simulating IV bolus for each dose.

    Returns list of dicts with: dose, cmax, auc, dose_normalized_cmax,
    dose_normalized_auc, non_linear (bool if dose_normalized_auc varies > 20%).
    """
    if not doses:
        raise ValueError("doses list must not be empty")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if vmax_mg_L_per_h <= 0:
        raise ValueError("vmax_mg_L_per_h must be > 0")
    if km_mg_L <= 0:
        raise ValueError("km_mg_L must be > 0")

    results = []
    for dose in doses:
        if dose <= 0:
            raise ValueError(f"All doses must be > 0, got {dose}")
        sim = simulate_mm_1cpt(
            drug_name="compound",
            dose_mg=dose,
            vd_L=vd_L,
            vmax_mg_L_per_h=vmax_mg_L_per_h,
            km_mg_L=km_mg_L,
            route="iv",
            t_end_h=t_end_h,
        )
        results.append(
            {
                "dose": dose,
                "cmax": sim["cmax"],
                "auc": sim["auc"],
                "dose_normalized_cmax": sim["cmax"] / dose,
                "dose_normalized_auc": sim["auc"] / dose,
            }
        )

    # Non-linear flag: dose_normalized_auc varies > 20%
    aucs_norm = [r["dose_normalized_auc"] for r in results]
    auc_min = min(aucs_norm)
    auc_max = max(aucs_norm)
    auc_mean = sum(aucs_norm) / len(aucs_norm)
    non_linear = ((auc_max - auc_min) / auc_mean) > 0.20 if auc_mean > 0 else False

    for r in results:
        r["non_linear"] = non_linear

    return results
