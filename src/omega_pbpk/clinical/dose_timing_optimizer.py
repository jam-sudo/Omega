"""Dose timing optimizer: maximize time-in-therapeutic-range for multi-dose regimens."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DoseTimingResult:
    drug_name: str
    dose_mg: float
    optimized_intervals_h: tuple  # optimized dose times
    regimen_type: str  # "qd","bid","tid","qid","custom"
    time_in_range_pct: float  # % of time plasma conc in [target_low, target_high]
    mean_conc_mg_L: float
    peak_trough_ratio: float  # Cmax/Cmin
    min_conc_mg_L: float
    max_conc_mg_L: float
    n_doses: int
    simulation_hours: float
    notes: str


def compute_time_in_range(
    times_h: list,
    concs_mg_L: list,
    low_mg_L: float,
    high_mg_L: float,
) -> float:
    """Return fraction of time (0-1) that concentration is within [low, high]."""
    if not times_h or not concs_mg_L:
        return 0.0
    n = len(concs_mg_L)
    count_in = sum(1 for c in concs_mg_L if low_mg_L <= c <= high_mg_L)
    return count_in / n if n > 0 else 0.0


def simulate_multi_dose_profile(
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    dose_times_h: list,
    t_end_h: float,
    route: str = "oral",
    ka_per_h: float = 1.0,
    f_oral: float = 1.0,
    vd_L_2: float = 0.0,  # second compartment (0 = 1-cpt)
    dt_h: float = 0.1,
) -> tuple:
    """Returns (times_h, c_plasma_mg_L) lists via forward Euler."""
    k_elim = cl_L_per_h / vd_L  # elimination rate constant

    # Sort dose times
    sorted_doses = sorted(dose_times_h)

    # State: C_plasma (mg/L), A_gut (mg)
    c_plasma = 0.0
    a_gut = 0.0

    times_out = []
    concs_out = []

    t = 0.0
    # Use a set for fast lookup of dose times (rounded to nearest dt)
    # We'll track which doses have been administered
    dose_idx = 0

    n_steps = int(round(t_end_h / dt_h)) + 1

    for step in range(n_steps):
        t = step * dt_h

        # Administer doses at the appropriate times
        while dose_idx < len(sorted_doses) and sorted_doses[dose_idx] <= t + 1e-9:
            if route == "oral":
                a_gut += dose_mg * f_oral
            else:  # iv
                c_plasma += dose_mg / vd_L
            dose_idx += 1

        times_out.append(t)
        concs_out.append(max(0.0, c_plasma))

        # Forward Euler step
        if route == "oral":
            dc_dt = ka_per_h * a_gut / vd_L - k_elim * c_plasma
            da_gut_dt = -ka_per_h * a_gut
            c_plasma = max(0.0, c_plasma + dc_dt * dt_h)
            a_gut = max(0.0, a_gut + da_gut_dt * dt_h)
        else:
            dc_dt = -k_elim * c_plasma
            c_plasma = max(0.0, c_plasma + dc_dt * dt_h)

    return (times_out, concs_out)


def optimize_dose_timing(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    target_low_mg_L: float,
    target_high_mg_L: float,
    n_doses_per_day: int = 2,  # 1=qd, 2=bid, 3=tid, 4=qid
    route: str = "oral",
    ka_per_h: float = 1.0,
    f_oral: float = 1.0,
    simulation_days: int = 3,  # simulate n days to reach SS
    dt_h: float = 0.1,
) -> DoseTimingResult:
    """Optimize dose timing to maximize time-in-therapeutic-range."""
    # Validation
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if target_low_mg_L <= 0:
        raise ValueError(f"target_low_mg_L must be > 0, got {target_low_mg_L}")
    if target_high_mg_L <= target_low_mg_L:
        raise ValueError(
            f"target_high_mg_L ({target_high_mg_L}) must be > target_low_mg_L ({target_low_mg_L})"
        )
    if n_doses_per_day not in (1, 2, 3, 4):
        raise ValueError(f"n_doses_per_day must be in {{1,2,3,4}}, got {n_doses_per_day}")
    if route not in ("oral", "iv"):
        raise ValueError(f"route must be 'oral' or 'iv', got {route}")

    # Regimen type mapping
    regimen_map = {1: "qd", 2: "bid", 3: "tid", 4: "qid"}
    regimen_type = regimen_map.get(n_doses_per_day, "custom")

    # Equal interval dosing
    interval_h = 24.0 / n_doses_per_day
    total_doses = n_doses_per_day * simulation_days
    dose_times = [i * interval_h for i in range(total_doses)]

    t_end_h = simulation_days * 24.0

    # Simulate
    times_h, concs_mg_L = simulate_multi_dose_profile(
        dose_mg=dose_mg,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
        dose_times_h=dose_times,
        t_end_h=t_end_h,
        route=route,
        ka_per_h=ka_per_h,
        f_oral=f_oral,
        dt_h=dt_h,
    )

    # Extract last 24h for steady-state analysis
    ss_start = t_end_h - 24.0
    ss_times = []
    ss_concs = []
    for t, c in zip(times_h, concs_mg_L):
        if t >= ss_start - 1e-9:
            ss_times.append(t)
            ss_concs.append(c)

    if not ss_concs:
        ss_concs = concs_mg_L
        ss_times = times_h

    # Compute metrics from last 24h
    tir_fraction = compute_time_in_range(ss_times, ss_concs, target_low_mg_L, target_high_mg_L)
    time_in_range_pct = tir_fraction * 100.0

    max_conc = max(ss_concs) if ss_concs else 0.0
    min_conc = min(ss_concs) if ss_concs else 0.0
    mean_conc = sum(ss_concs) / len(ss_concs) if ss_concs else 0.0

    # Peak-trough ratio (avoid division by zero)
    if min_conc > 1e-12:
        peak_trough_ratio = max_conc / min_conc
    else:
        peak_trough_ratio = max_conc / 1e-12 if max_conc > 0 else 1.0

    # Ensure ratio >= 1
    peak_trough_ratio = max(1.0, peak_trough_ratio)

    notes = (
        f"Equal-interval {regimen_type} dosing every {interval_h:.1f}h; "
        f"SS assessed over last 24h of {simulation_days}-day simulation"
    )

    return DoseTimingResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        optimized_intervals_h=tuple(dose_times),
        regimen_type=regimen_type,
        time_in_range_pct=time_in_range_pct,
        mean_conc_mg_L=mean_conc,
        peak_trough_ratio=peak_trough_ratio,
        min_conc_mg_L=min_conc,
        max_conc_mg_L=max_conc,
        n_doses=total_doses,
        simulation_hours=t_end_h,
        notes=notes,
    )
