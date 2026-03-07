"""Phase 653 — Deep Tissue PK Model.

3-compartment model: plasma (central), shallow tissue (well-perfused),
deep tissue (poorly-perfused: bone, fat, muscle) creating a depot effect.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class DeepTissuePKResult:
    drug_name: str
    dose_mg: float
    times_h: list
    c_plasma_mg_L: list
    c_shallow_mg_L: list
    c_deep_mg_L: list
    cmax_plasma: float
    tmax_plasma_h: float
    auc_plasma: float
    auc_shallow: float
    auc_deep: float
    deep_tissue_half_life_h: float
    terminal_phase_t_half_h: float
    accumulation_in_deep: float
    washout_time_90pct_h: float
    notes: str


def _validate_inputs(
    dose_mg: float,
    cl_plasma_L_per_h: float,
    vd_plasma_L: float,
    vd_shallow_L: float,
    vd_deep_L: float,
    q_shallow_L_per_h: float,
    q_deep_L_per_h: float,
    route: str,
    f_oral: float,
) -> None:
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_plasma_L_per_h <= 0:
        raise ValueError("cl_plasma_L_per_h must be > 0")
    if vd_plasma_L <= 0:
        raise ValueError("vd_plasma_L must be > 0")
    if vd_shallow_L <= 0:
        raise ValueError("vd_shallow_L must be > 0")
    if vd_deep_L <= 0:
        raise ValueError("vd_deep_L must be > 0")
    if q_shallow_L_per_h <= 0:
        raise ValueError("q_shallow_L_per_h must be > 0")
    if q_deep_L_per_h <= 0:
        raise ValueError("q_deep_L_per_h must be > 0")
    if route not in ("iv", "oral"):
        raise ValueError(f"route must be 'iv' or 'oral', got '{route}'")
    if not (0 < f_oral <= 1):
        raise ValueError("f_oral must be in (0, 1]")


def _trapezoidal_auc(times: list, concs: list) -> float:
    auc = 0.0
    for i in range(len(times) - 1):
        auc += (concs[i] + concs[i + 1]) / 2.0 * (times[i + 1] - times[i])
    return auc


def _compute_terminal_half_life(times: list, concs: list) -> float:
    """Estimate terminal half-life from last 30% of time points using log-linear regression."""
    n = len(times)
    start_idx = max(1, int(n * 0.7))
    # Filter positive concentrations
    pts = [(times[i], concs[i]) for i in range(start_idx, n) if concs[i] > 0]
    if len(pts) < 2:
        # fallback: use last two positive points from full series
        pts = [(times[i], concs[i]) for i in range(n) if concs[i] > 0]
        if len(pts) < 2:
            return float("nan")
    log_c = [math.log(c) for _, c in pts]
    t_vals = [t for t, _ in pts]
    n_pts = len(pts)
    mean_t = sum(t_vals) / n_pts
    mean_lc = sum(log_c) / n_pts
    num = sum((t_vals[i] - mean_t) * (log_c[i] - mean_lc) for i in range(n_pts))
    den = sum((t_vals[i] - mean_t) ** 2 for i in range(n_pts))
    if den == 0 or num >= 0:
        return float("nan")
    slope = num / den  # should be negative
    return math.log(2) / (-slope)


def simulate_deep_tissue_pk(
    drug_name: str,
    dose_mg: float,
    cl_plasma_L_per_h: float = 5.0,
    vd_plasma_L: float = 10.0,
    vd_shallow_L: float = 30.0,
    vd_deep_L: float = 100.0,
    q_shallow_L_per_h: float = 20.0,
    q_deep_L_per_h: float = 1.0,
    route: str = "iv",
    ka_per_h: float = 1.0,
    f_oral: float = 0.8,
    t_end_h: float = 168.0,
    dt_h: float = 0.5,
) -> DeepTissuePKResult:
    """Simulate deep tissue PK using a 3-compartment forward Euler model."""
    _validate_inputs(
        dose_mg,
        cl_plasma_L_per_h,
        vd_plasma_L,
        vd_shallow_L,
        vd_deep_L,
        q_shallow_L_per_h,
        q_deep_L_per_h,
        route,
        f_oral,
    )

    # CFL-like stability condition: dt < Vp / (CL + Qs + Qd)
    dt_max = vd_plasma_L / (cl_plasma_L_per_h + q_shallow_L_per_h + q_deep_L_per_h)
    # Determine integer sub-steps (up to 10)
    n_sub = 1
    while dt_h / n_sub > dt_max and n_sub < 10:
        n_sub += 1
    dt_eff = dt_h / n_sub

    n_steps = int(round(t_end_h / dt_h))
    times: list = [0.0]
    c_plasma: list = []
    c_shallow: list = []
    c_deep: list = []

    # Initial amounts (mg)
    if route == "iv":
        Ap = dose_mg
        A_gut = 0.0
    else:
        Ap = 0.0
        A_gut = dose_mg * f_oral  # absorbed fraction enters gut depot

    As = 0.0
    Ad = 0.0

    # Record initial concentrations
    c_plasma.append(Ap / vd_plasma_L)
    c_shallow.append(As / vd_shallow_L)
    c_deep.append(Ad / vd_deep_L)

    for step in range(n_steps):
        # Use sub-stepping for stability
        for _ in range(n_sub):
            Cp = Ap / vd_plasma_L
            Cs = As / vd_shallow_L
            Cd = Ad / vd_deep_L

            # Oral absorption input
            if route == "oral":
                oral_input = ka_per_h * A_gut
                dA_gut = -ka_per_h * A_gut
            else:
                oral_input = 0.0
                dA_gut = 0.0

            # ODE
            dAp = (
                -cl_plasma_L_per_h * Cp
                - q_shallow_L_per_h * (Cp - Cs)
                - q_deep_L_per_h * (Cp - Cd)
                + oral_input
            )
            dAs = q_shallow_L_per_h * (Cp - Cs)
            dAd = q_deep_L_per_h * (Cp - Cd)

            Ap = max(0.0, Ap + dAp * dt_eff)
            As = max(0.0, As + dAs * dt_eff)
            Ad = max(0.0, Ad + dAd * dt_eff)
            if route == "oral":
                A_gut = max(0.0, A_gut + dA_gut * dt_eff)

        t = (step + 1) * dt_h
        times.append(t)
        c_plasma.append(Ap / vd_plasma_L)
        c_shallow.append(As / vd_shallow_L)
        c_deep.append(Ad / vd_deep_L)

    # Derived metrics
    cmax_plasma = max(c_plasma)
    tmax_plasma_h = times[c_plasma.index(cmax_plasma)]
    cmax_deep = max(c_deep)

    auc_plasma = _trapezoidal_auc(times, c_plasma)
    auc_shallow = _trapezoidal_auc(times, c_shallow)
    auc_deep = _trapezoidal_auc(times, c_deep)

    # Deep tissue apparent half-life from deep compartment decay after peak
    peak_deep_idx = c_deep.index(cmax_deep)
    t_deep_decay = times[peak_deep_idx:]
    c_deep_decay = c_deep[peak_deep_idx:]
    deep_tissue_half_life_h = _compute_terminal_half_life(t_deep_decay, c_deep_decay)
    if math.isnan(deep_tissue_half_life_h) or deep_tissue_half_life_h <= 0:
        # Fallback: analytical deep half-life = Vd_deep / Q_deep
        deep_tissue_half_life_h = vd_deep_L * math.log(2) / q_deep_L_per_h

    # Terminal plasma half-life from plasma concentration decay
    terminal_phase_t_half_h = _compute_terminal_half_life(times, c_plasma)
    if math.isnan(terminal_phase_t_half_h) or terminal_phase_t_half_h <= 0:
        # Fallback: use simple 1-cpt approximation
        cl_total = cl_plasma_L_per_h
        vd_total = vd_plasma_L + vd_shallow_L + vd_deep_L
        terminal_phase_t_half_h = math.log(2) * vd_total / cl_total

    # Accumulation in deep tissue
    accumulation_in_deep = cmax_deep / cmax_plasma if cmax_plasma > 0 else 0.0

    # Washout time: time for deep tissue to drop to 10% of peak (90% loss)
    target_90pct = cmax_deep * 0.1
    washout_time_90pct_h = t_end_h  # default if not reached
    for i in range(peak_deep_idx, len(c_deep)):
        if c_deep[i] <= target_90pct:
            washout_time_90pct_h = times[i] - times[peak_deep_idx]
            break

    notes = (
        f"3-cpt deep tissue PK ({route}); "
        f"Q_deep={q_deep_L_per_h} L/h, Vd_deep={vd_deep_L} L; "
        f"sub-steps={n_sub} for numerical stability"
    )

    return DeepTissuePKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times,
        c_plasma_mg_L=c_plasma,
        c_shallow_mg_L=c_shallow,
        c_deep_mg_L=c_deep,
        cmax_plasma=cmax_plasma,
        tmax_plasma_h=tmax_plasma_h,
        auc_plasma=auc_plasma,
        auc_shallow=auc_shallow,
        auc_deep=auc_deep,
        deep_tissue_half_life_h=deep_tissue_half_life_h,
        terminal_phase_t_half_h=terminal_phase_t_half_h,
        accumulation_in_deep=accumulation_in_deep,
        washout_time_90pct_h=washout_time_90pct_h,
        notes=notes,
    )


def compare_tissue_distribution(
    drug_name: str,
    dose_mg: float,
    scenarios: list[dict],
) -> list[DeepTissuePKResult]:
    """Compare multiple deep tissue distribution scenarios."""
    results = []
    for scenario in scenarios:
        label = scenario.get("label", "unnamed")
        q_deep = scenario.get("q_deep_L_per_h", 1.0)
        vd_deep = scenario.get("vd_deep_L", 100.0)
        result = simulate_deep_tissue_pk(
            drug_name=f"{drug_name} [{label}]",
            dose_mg=dose_mg,
            q_deep_L_per_h=q_deep,
            vd_deep_L=vd_deep,
        )
        results.append(result)
    return results
