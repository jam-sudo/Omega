"""Unified PK simulation engine — Phase 496."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PKSimResult:
    """Result from a PK simulation.

    Attributes:
        drug_name: Drug identifier.
        n_compartments: Number of compartments used (1 or 2).
        route: Dosing route ('iv' or 'oral').
        times_h: Simulation time vector (h).
        c_plasma_mg_L: Central compartment concentration (mg/L).
        c_peripheral_mg_L: Peripheral compartment concentration (mg/L); empty for 1-cpt.
        cmax_mg_L: Peak plasma concentration (mg/L).
        tmax_h: Time of peak plasma concentration (h).
        auc_mg_h_L: Area under the curve by trapezoidal rule (mg·h/L).
        t_half_h: Terminal elimination half-life (h).
        vss_L: Volume of distribution at steady state (L).
        notes: Simulation notes or warnings.
    """

    drug_name: str
    n_compartments: int
    route: str
    times_h: list[float] = field(default_factory=list)
    c_plasma_mg_L: list[float] = field(default_factory=list)
    c_peripheral_mg_L: list[float] = field(default_factory=list)
    cmax_mg_L: float = 0.0
    tmax_h: float = 0.0
    auc_mg_h_L: float = 0.0
    t_half_h: float = 0.0
    vss_L: float = 0.0
    notes: str = ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _trapezoidal_auc(times: list[float], concs: list[float]) -> float:
    """Compute AUC using the trapezoidal rule."""
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        auc += 0.5 * (concs[i] + concs[i - 1]) * dt
    return auc


def _terminal_half_life(times: list[float], concs: list[float]) -> float:
    """Estimate terminal half-life from log-linear regression on the last 30% of data."""
    n = len(times)
    if n < 4:
        return float("nan")

    start_idx = max(1, int(0.7 * n))
    # Filter to positive concentrations
    segment_t: list[float] = []
    segment_c: list[float] = []
    for i in range(start_idx, n):
        if concs[i] > 0:
            segment_t.append(times[i])
            segment_c.append(math.log(concs[i]))

    if len(segment_t) < 2:
        return float("nan")

    # Simple linear regression: log(C) = a + b*t
    n_seg = len(segment_t)
    sum_t = sum(segment_t)
    sum_c = sum(segment_c)
    sum_t2 = sum(x * x for x in segment_t)
    sum_tc = sum(segment_t[i] * segment_c[i] for i in range(n_seg))

    denom = n_seg * sum_t2 - sum_t * sum_t
    if abs(denom) < 1e-12:
        return float("nan")

    slope = (n_seg * sum_tc - sum_t * sum_c) / denom
    if slope >= 0:
        return float("nan")

    return math.log(2.0) / (-slope)


# ---------------------------------------------------------------------------
# Dosing regimen helpers
# ---------------------------------------------------------------------------


def create_dosing_regimen(
    dose_mg: float,
    interval_h: float,
    n_doses: int,
    start_h: float = 0.0,
) -> list[dict]:
    """Create a multi-dose regimen.

    Parameters
    ----------
    dose_mg:
        Dose size (mg).
    interval_h:
        Dosing interval (h).
    n_doses:
        Number of doses.
    start_h:
        Time of first dose (h); default 0.

    Returns
    -------
    list[dict]
        Each entry: {'time_h': float, 'dose_mg': float, 'route': str}.
        Route defaults to 'oral'; call ``simulate_pk`` with route='iv' to override.
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if interval_h <= 0:
        raise ValueError("interval_h must be > 0")
    if n_doses < 1:
        raise ValueError("n_doses must be >= 1")

    return [
        {"time_h": start_h + i * interval_h, "dose_mg": dose_mg, "route": "oral"}
        for i in range(n_doses)
    ]


def apply_dose(
    state: dict,
    dose_mg: float,
    route: str,
    vd_L: float,
    f_oral: float = 1.0,
) -> dict:
    """Apply a dose to the current compartment state (immutable — returns new dict).

    Parameters
    ----------
    state:
        Current ODE state dict with keys: 'c1', 'a2', 'gut'.
    dose_mg:
        Dose amount (mg).
    route:
        'iv' or 'oral'.
    vd_L:
        Central volume of distribution (L).
    f_oral:
        Oral bioavailability fraction [0, 1].

    Returns
    -------
    dict
        Updated state dict.
    """
    new_state = dict(state)
    if route == "iv":
        absorbed_mg_L = (dose_mg / vd_L) if vd_L > 0 else 0.0
        new_state["c1"] = state.get("c1", 0.0) + absorbed_mg_L
    elif route == "oral":
        new_state["gut"] = state.get("gut", 0.0) + dose_mg * f_oral
    else:
        raise ValueError(f"Unknown route '{route}'. Use 'iv' or 'oral'.")
    return new_state


def calculate_summary_stats(
    times_h: list[float],
    concs_mg_L: list[float],
    dose_mg: float,
) -> dict:
    """Calculate summary PK statistics from a concentration-time profile.

    Parameters
    ----------
    times_h:
        Time vector (h).
    concs_mg_L:
        Plasma concentration vector (mg/L).
    dose_mg:
        Total administered dose (mg) — used for CL calculation.

    Returns
    -------
    dict
        Keys: cmax, tmax, auc, t_half, clearance_apparent.
    """
    if not times_h or not concs_mg_L:
        return {
            "cmax": 0.0,
            "tmax": 0.0,
            "auc": 0.0,
            "t_half": float("nan"),
            "clearance_apparent": float("nan"),
        }

    cmax = max(concs_mg_L)
    tmax = times_h[concs_mg_L.index(cmax)]
    auc = _trapezoidal_auc(times_h, concs_mg_L)
    t_half = _terminal_half_life(times_h, concs_mg_L)
    cl_apparent = dose_mg / auc if auc > 0 else float("nan")

    return {
        "cmax": cmax,
        "tmax": tmax,
        "auc": auc,
        "t_half": t_half,
        "clearance_apparent": cl_apparent,
    }


# ---------------------------------------------------------------------------
# Main simulation function
# ---------------------------------------------------------------------------


def simulate_pk(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    route: str = "iv",
    n_compartments: int = 1,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
    ka_per_h: float = 1.0,
    f_oral: float = 1.0,
    q12_L_per_h: float = 0.0,
    vd2_L: float = 0.0,
    dosing_regimen: list[dict] | None = None,
) -> PKSimResult:
    """Simulate plasma concentration-time profile using forward Euler integration.

    Supports 1- and 2-compartment models with IV or oral dosing and
    multi-dose regimens.

    Parameters
    ----------
    drug_name:
        Drug identifier string.
    dose_mg:
        Dose (mg). Ignored if dosing_regimen is provided.
    cl_L_per_h:
        Total clearance (L/h).
    vd_L:
        Central volume of distribution (L).
    route:
        'iv' or 'oral'.
    n_compartments:
        1 or 2.
    t_end_h:
        Simulation end time (h).
    dt_h:
        Integration step size (h).
    ka_per_h:
        Absorption rate constant for oral dosing (h⁻¹).
    f_oral:
        Oral bioavailability (fraction).
    q12_L_per_h:
        Inter-compartmental clearance (L/h); 2-cpt only.
    vd2_L:
        Peripheral volume (L); 2-cpt only.
    dosing_regimen:
        Optional list of dose events: [{'time_h', 'dose_mg', 'route'}].
        Overrides dose_mg/route if provided.

    Returns
    -------
    PKSimResult
    """
    # ---- Input validation -----------------------------------------------
    if not drug_name:
        raise ValueError("drug_name must be a non-empty string")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if route not in ("iv", "oral"):
        raise ValueError("route must be 'iv' or 'oral'")
    if n_compartments not in (1, 2):
        raise ValueError("n_compartments must be 1 or 2")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be > 0")
    if dt_h <= 0:
        raise ValueError("dt_h must be > 0")
    if not (0.0 <= f_oral <= 1.0):
        raise ValueError("f_oral must be in [0, 1]")
    if n_compartments == 2 and (q12_L_per_h < 0 or vd2_L <= 0):
        raise ValueError("For 2-cpt: q12_L_per_h >= 0 and vd2_L > 0 required")

    # ---- Kinetic rate constants -----------------------------------------
    ke = cl_L_per_h / vd_L  # elimination rate constant (h⁻¹)

    k12 = q12_L_per_h / vd_L if n_compartments == 2 else 0.0
    k21 = q12_L_per_h / vd2_L if n_compartments == 2 and vd2_L > 0 else 0.0

    # ---- Build dosing schedule ------------------------------------------
    if dosing_regimen is not None:
        schedule = sorted(dosing_regimen, key=lambda e: e["time_h"])
    else:
        first_route = route
        schedule = [{"time_h": 0.0, "dose_mg": dose_mg, "route": first_route}]

    # ---- State initialisation ------------------------------------------
    # c1: central conc (mg/L), a2: peripheral amount (mg), gut: gut depot (mg)
    state: dict[str, float] = {"c1": 0.0, "a2": 0.0, "gut": 0.0}

    # Apply time-zero doses
    for event in schedule:
        if event["time_h"] == 0.0:
            state = apply_dose(state, event["dose_mg"], event["route"], vd_L, f_oral)

    # ---- ODE integration (forward Euler) --------------------------------
    n_steps = int(math.ceil(t_end_h / dt_h))
    times: list[float] = []
    c_plasma: list[float] = []
    c_periph: list[float] = []

    # Remaining dose events (time > 0)
    pending = [e for e in schedule if e["time_h"] > 0.0]

    for step in range(n_steps + 1):
        t = step * dt_h

        # Record state
        times.append(t)
        c_plasma.append(state["c1"])
        c_periph.append(state["a2"] / vd2_L if n_compartments == 2 and vd2_L > 0 else 0.0)

        if step == n_steps:
            break

        # Apply pending doses at this time step
        t_next = (step + 1) * dt_h
        applied_indices = []
        for idx, event in enumerate(pending):
            if event["time_h"] <= t_next and event["time_h"] > t:
                state = apply_dose(state, event["dose_mg"], event["route"], vd_L, f_oral)
                applied_indices.append(idx)
        for idx in reversed(applied_indices):
            pending.pop(idx)

        # Derivatives
        c1 = state["c1"]
        a2 = state["a2"]
        gut = state["gut"]

        # Gut depot (oral absorption)
        d_gut = -ka_per_h * gut

        # Central compartment
        oral_input = (
            (ka_per_h * gut / vd_L)
            if route == "oral" or any(e["route"] == "oral" for e in schedule)
            else 0.0
        )
        if n_compartments == 1:
            d_c1 = -(ke) * c1 + oral_input
            d_a2 = 0.0
        else:
            d_c1 = -(ke + k12) * c1 + (k21 * a2 / vd_L) + oral_input
            d_a2 = k12 * c1 * vd_L - k21 * a2

        # Euler step
        state = {
            "c1": max(0.0, c1 + d_c1 * dt_h),
            "a2": max(0.0, a2 + d_a2 * dt_h),
            "gut": max(0.0, gut + d_gut * dt_h),
        }

    # ---- Summary statistics --------------------------------------------
    cmax = max(c_plasma) if c_plasma else 0.0
    tmax = times[c_plasma.index(cmax)] if c_plasma else 0.0
    auc = _trapezoidal_auc(times, c_plasma)
    t_half = _terminal_half_life(times, c_plasma)

    # Vss: for 1-cpt = Vd; for 2-cpt = Vd1 + Vd2
    vss = vd_L + (vd2_L if n_compartments == 2 else 0.0)

    return PKSimResult(
        drug_name=drug_name,
        n_compartments=n_compartments,
        route=route,
        times_h=times,
        c_plasma_mg_L=c_plasma,
        c_peripheral_mg_L=c_periph if n_compartments == 2 else [],
        cmax_mg_L=round(cmax, 6),
        tmax_h=round(tmax, 4),
        auc_mg_h_L=round(auc, 6),
        t_half_h=round(t_half, 4) if not math.isnan(t_half) else float("nan"),
        vss_L=vss,
        notes="",
    )
