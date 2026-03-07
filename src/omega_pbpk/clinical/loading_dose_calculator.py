"""Phase 748 — Drug Loading Dose Calculator.

Calculates optimal loading dose to rapidly achieve therapeutic steady-state
concentrations, then verifies via 1-compartment forward Euler simulation.
"""

from dataclasses import dataclass


@dataclass
class LoadingDoseResult:
    """Result of loading dose calculation and simulation."""

    drug_name: str
    target_css_avg_mg_L: float
    loading_dose_mg: float
    maintenance_dose_mg: float
    dosing_interval_h: float
    freq_per_day: int
    route: str
    times_h: list
    c_plasma_mg_L: list
    css_max_mg_L: float
    css_min_mg_L: float
    css_avg_mg_L: float
    time_to_therapeutic_h: float
    loading_ratio: float
    notes: str


_VALID_FREQ = {1, 2, 3, 4}


def _validate_inputs(
    target_css_avg_mg_L: float,
    cl_L_per_h: float,
    vd_L: float,
    freq_per_day: int,
) -> None:
    if target_css_avg_mg_L <= 0:
        raise ValueError(f"target_css_avg_mg_L must be > 0, got {target_css_avg_mg_L}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if freq_per_day not in _VALID_FREQ:
        raise ValueError(f"freq_per_day must be one of {_VALID_FREQ}, got {freq_per_day}")


def _simulate(
    loading_dose_mg: float,
    maintenance_dose_mg: float,
    tau_h: float,
    cl_L_per_h: float,
    vd_L: float,
    f_oral: float,
    ka_per_h: float,
    route: str,
    t_end_h: float,
    dt_h: float,
) -> tuple:
    """Forward Euler 1-compartment simulation.

    Returns (times, c_plasma).
    """
    ke = cl_L_per_h / vd_L  # elimination rate constant

    n_steps = int(t_end_h / dt_h) + 1
    times = [i * dt_h for i in range(n_steps)]
    c_plasma = [0.0] * n_steps

    # State: A_central (mg), A_gut (mg)
    a_central = 0.0
    a_gut = 0.0

    # Apply loading dose at t=0
    if route == "iv":
        a_central += loading_dose_mg
    else:
        a_gut += loading_dose_mg * f_oral

    # Track dosing events (maintenance at each interval after t=0)
    # Dose times: tau_h, 2*tau_h, 3*tau_h, ...
    next_dose_time = tau_h

    c_plasma[0] = a_central / vd_L

    for i in range(1, n_steps):
        t = times[i]

        # Apply maintenance dose at each interval
        if t >= next_dose_time - dt_h * 0.5:
            if route == "iv":
                a_central += maintenance_dose_mg
            else:
                a_gut += maintenance_dose_mg * f_oral
            next_dose_time += tau_h

        # ODE: oral
        if route == "oral":
            d_gut = -ka_per_h * a_gut
            d_central = ka_per_h * a_gut - ke * a_central
            a_gut = max(0.0, a_gut + d_gut * dt_h)
            a_central = max(0.0, a_central + d_central * dt_h)
        else:
            # IV: no gut compartment
            d_central = -ke * a_central
            a_central = max(0.0, a_central + d_central * dt_h)

        c_plasma[i] = a_central / vd_L

    return times, c_plasma


def _extract_last_interval_stats(
    times: list,
    c_plasma: list,
    tau_h: float,
    t_end_h: float,
) -> tuple:
    """Extract Cmax, Cmin, Cavg from the last dosing interval."""
    t_start = t_end_h - tau_h
    interval_c = [c for t, c in zip(times, c_plasma) if t >= t_start]
    if not interval_c:
        interval_c = c_plasma[-10:] if len(c_plasma) >= 10 else c_plasma

    css_max = max(interval_c)
    css_min = min(interval_c)
    # Trapezoidal AUC / interval length for average
    n = len(interval_c)
    dt_approx = tau_h / max(n - 1, 1)
    auc = 0.0
    for j in range(n - 1):
        auc += 0.5 * (interval_c[j] + interval_c[j + 1]) * dt_approx
    css_avg = auc / tau_h if tau_h > 0 else 0.5 * (css_max + css_min)
    return css_max, css_min, css_avg


def _time_to_therapeutic(
    times: list,
    c_plasma: list,
    threshold: float,
) -> float:
    """Return first time C >= threshold, or t_end if never reached."""
    for t, c in zip(times, c_plasma):
        if c >= threshold:
            return t
    return times[-1]


def calculate_loading_dose(
    drug_name: str,
    target_css_avg_mg_L: float,
    cl_L_per_h: float,
    vd_L: float,
    f_oral: float = 1.0,
    ka_per_h: float = 1.0,
    freq_per_day: int = 2,
    route: str = "oral",
    t_half_h: float = None,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
) -> LoadingDoseResult:
    """Calculate loading and maintenance doses to achieve target Css.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    target_css_avg_mg_L:
        Target average steady-state plasma concentration (mg/L).
    cl_L_per_h:
        Total clearance (L/h, must be > 0).
    vd_L:
        Volume of distribution (L, must be > 0).
    f_oral:
        Oral bioavailability fraction (0–1).
    ka_per_h:
        First-order oral absorption rate constant (1/h).
    freq_per_day:
        Dosing frequency per day (1, 2, 3, or 4).
    route:
        "oral" or "iv".
    t_half_h:
        Half-life (h). If None, computed from CL/Vd.
    t_end_h:
        Simulation duration (h).
    dt_h:
        Forward Euler step size (h).

    Returns
    -------
    LoadingDoseResult
    """
    _validate_inputs(target_css_avg_mg_L, cl_L_per_h, vd_L, freq_per_day)

    tau_h = 24.0 / freq_per_day
    ke = cl_L_per_h / vd_L
    t_half = t_half_h if t_half_h is not None else (0.693147 / ke)

    # Maintenance dose: Css_avg = F * Dose_m / (CL * tau)
    # => Dose_m = target_css_avg * CL * tau / F
    bioavail = f_oral if route == "oral" else 1.0
    maintenance_dose_mg = target_css_avg_mg_L * cl_L_per_h * tau_h / bioavail

    # Loading dose: instantly achieve target_css_avg in the central compartment
    # IV: C(0) = Loading_dose / Vd => Loading_dose = target * Vd
    # Oral: accounts for bioavailability => Loading_dose = target * Vd / F
    if route == "iv":
        loading_dose_mg = target_css_avg_mg_L * vd_L
    else:
        loading_dose_mg = target_css_avg_mg_L * vd_L / bioavail

    # Ensure loading >= maintenance (for drugs with long t_half / small tau)
    loading_dose_mg = max(loading_dose_mg, maintenance_dose_mg)

    loading_ratio = loading_dose_mg / maintenance_dose_mg if maintenance_dose_mg > 0 else 1.0

    # Simulate
    times, c_plasma = _simulate(
        loading_dose_mg=loading_dose_mg,
        maintenance_dose_mg=maintenance_dose_mg,
        tau_h=tau_h,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
        f_oral=bioavail,
        ka_per_h=ka_per_h,
        route=route,
        t_end_h=t_end_h,
        dt_h=dt_h,
    )

    css_max, css_min, css_avg = _extract_last_interval_stats(times, c_plasma, tau_h, t_end_h)
    therapeutic_threshold = target_css_avg_mg_L * 0.9
    ttt = _time_to_therapeutic(times, c_plasma, therapeutic_threshold)

    notes = (
        f"t_half={t_half:.2f} h; tau={tau_h:.1f} h; "
        f"ke={ke:.4f} /h; maintenance={maintenance_dose_mg:.2f} mg; "
        f"loading={loading_dose_mg:.2f} mg; route={route}"
    )

    return LoadingDoseResult(
        drug_name=drug_name,
        target_css_avg_mg_L=target_css_avg_mg_L,
        loading_dose_mg=loading_dose_mg,
        maintenance_dose_mg=maintenance_dose_mg,
        dosing_interval_h=tau_h,
        freq_per_day=freq_per_day,
        route=route,
        times_h=times,
        c_plasma_mg_L=c_plasma,
        css_max_mg_L=css_max,
        css_min_mg_L=css_min,
        css_avg_mg_L=css_avg,
        time_to_therapeutic_h=ttt,
        loading_ratio=loading_ratio,
        notes=notes,
    )


def compare_loading_strategies(
    drug_name: str,
    target_css_avg: float,
    cl: float,
    vd: float,
    strategies: list,
) -> list:
    """Compare multiple dosing strategies for the same drug.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    target_css_avg:
        Target average steady-state concentration (mg/L).
    cl:
        Total clearance (L/h).
    vd:
        Volume of distribution (L).
    strategies:
        List of dicts with optional keys matching `calculate_loading_dose`
        keyword arguments (e.g., freq_per_day, route, f_oral).

    Returns
    -------
    list[LoadingDoseResult] in the same order as strategies.
    """
    results = []
    for s in strategies:
        kwargs = dict(s)
        result = calculate_loading_dose(
            drug_name=drug_name,
            target_css_avg_mg_L=target_css_avg,
            cl_L_per_h=cl,
            vd_L=vd,
            **kwargs,
        )
        results.append(result)
    return results
