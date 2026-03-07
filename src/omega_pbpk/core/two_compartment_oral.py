"""Two-compartment oral absorption PK model (gut depot -> central -> peripheral)."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class TwoCompartmentOralResult:
    drug_name: str
    dose_mg: float
    route: str  # "oral" or "iv"
    times_h: list
    c_central_mg_L: list  # central compartment concentration
    c_peripheral_mg_L: list  # peripheral compartment concentration
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_L: float
    t_half_terminal_h: float  # terminal half-life
    vss_L: float  # volume of distribution at steady state
    notes: str


def compute_vss(vd1_L: float, vd2_L: float) -> float:
    """Vss = Vd1 + Vd2."""
    return vd1_L + vd2_L


def estimate_terminal_half_life(times_h: list, c_central_mg_L: list, frac: float = 0.2) -> float:
    """Estimate terminal t½ from the last frac of the profile using log-linear regression."""
    n = len(times_h)
    if n < 4:
        return float("nan")

    start_idx = int(n * (1.0 - frac))
    if start_idx >= n - 2:
        start_idx = n - 3

    # Collect points with positive concentration in the terminal phase
    pts = []
    for i in range(start_idx, n):
        c = c_central_mg_L[i]
        if c > 0:
            pts.append((times_h[i], math.log(c)))

    if len(pts) < 2:
        return float("nan")

    # Linear regression on log(C) vs t
    sx = sum(p[0] for p in pts)
    sy = sum(p[1] for p in pts)
    sxy = sum(p[0] * p[1] for p in pts)
    sxx = sum(p[0] ** 2 for p in pts)
    np_ = len(pts)

    denom = np_ * sxx - sx * sx
    if abs(denom) < 1e-12:
        return float("nan")

    slope = (np_ * sxy - sx * sy) / denom

    if slope >= 0:
        # Non-negative slope — use a rough estimate from endpoint
        c_start = c_central_mg_L[start_idx]
        c_end = c_central_mg_L[-1]
        dt = times_h[-1] - times_h[start_idx]
        if c_start > 0 and c_end > 0 and dt > 0:
            lam = (math.log(c_start) - math.log(c_end)) / dt
            if lam > 0:
                return math.log(2) / lam
        return float("nan")

    lam_z = -slope
    return math.log(2) / lam_z


def simulate_two_cpt_oral(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd1_L: float,
    vd2_L: float,
    q12_L_per_h: float,
    route: str = "oral",
    ka_per_h: float = 1.0,
    f_oral: float = 1.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> TwoCompartmentOralResult:
    """Simulate 2-compartment PK with oral or IV administration via Forward Euler."""
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd1_L <= 0:
        raise ValueError(f"vd1_L must be > 0, got {vd1_L}")
    if vd2_L <= 0:
        raise ValueError(f"vd2_L must be > 0, got {vd2_L}")
    if q12_L_per_h <= 0:
        raise ValueError(f"q12_L_per_h must be > 0, got {q12_L_per_h}")
    if route not in {"oral", "iv"}:
        raise ValueError(f"route must be 'oral' or 'iv', got {route!r}")
    if not (0 < f_oral <= 1.0):
        raise ValueError(f"f_oral must be in (0, 1], got {f_oral}")
    if t_end_h <= 0:
        raise ValueError(f"t_end_h must be > 0, got {t_end_h}")
    if dt_h <= 0:
        raise ValueError(f"dt_h must be > 0, got {dt_h}")

    # Micro-rate constants
    ke = cl_L_per_h / vd1_L
    k12 = q12_L_per_h / vd1_L
    k21 = q12_L_per_h / vd2_L

    # Initial conditions
    if route == "oral":
        a_gut = dose_mg * f_oral
        c1 = 0.0
        c2 = 0.0
    else:
        a_gut = 0.0
        c1 = dose_mg / vd1_L
        c2 = 0.0

    times_h: list = []
    c_central: list = []
    c_peripheral: list = []

    t = 0.0
    n_steps = int(t_end_h / dt_h) + 1

    for _ in range(n_steps):
        times_h.append(t)
        c_central.append(c1)
        c_peripheral.append(c2)

        # Forward Euler derivatives
        dc1_dt = (ka_per_h * a_gut / vd1_L) - (ke + k12) * c1 + k21 * c2
        dc2_dt = k12 * c1 - k21 * c2
        da_gut_dt = -ka_per_h * a_gut

        # Update state
        c1 = c1 + dc1_dt * dt_h
        c2 = c2 + dc2_dt * dt_h
        a_gut = a_gut + da_gut_dt * dt_h

        # Clamp non-negative
        if c1 < 0:
            c1 = 0.0
        if c2 < 0:
            c2 = 0.0
        if a_gut < 0:
            a_gut = 0.0

        t = round(t + dt_h, 10)

    # Derived PK metrics
    cmax = max(c_central)
    tmax_idx = c_central.index(cmax)
    tmax = times_h[tmax_idx]

    # Trapezoidal AUC
    auc = 0.0
    for i in range(1, len(times_h)):
        auc += 0.5 * (c_central[i - 1] + c_central[i]) * (times_h[i] - times_h[i - 1])

    t_half = estimate_terminal_half_life(times_h, c_central)
    vss = compute_vss(vd1_L, vd2_L)

    notes = (
        f"2-cpt {route} model: ke={ke:.4f}/h, k12={k12:.4f}/h, k21={k21:.4f}/h"
        f"{', ka=' + str(round(ka_per_h, 4)) + '/h' if route == 'oral' else ''}"
    )

    return TwoCompartmentOralResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route=route,
        times_h=times_h,
        c_central_mg_L=c_central,
        c_peripheral_mg_L=c_peripheral,
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc_mg_h_L=auc,
        t_half_terminal_h=t_half,
        vss_L=vss,
        notes=notes,
    )
