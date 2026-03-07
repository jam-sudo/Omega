"""Phase 250: Microsphere depot PK model.

Models injectable microsphere controlled-release formulations using a
2-compartment ODE: depot + plasma. Supports biphasic release (burst +
first-order erosion).
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Polymer lookup table
# ---------------------------------------------------------------------------

_POLYMERS = {
    "PLGA_fast":  {"k_release": 0.02,  "burst_fraction": 0.20},
    "PLGA_slow":  {"k_release": 0.005, "burst_fraction": 0.10},
    "PLA":        {"k_release": 0.003, "burst_fraction": 0.05},
    "chitosan":   {"k_release": 0.04,  "burst_fraction": 0.25},
    "albumin":    {"k_release": 0.08,  "burst_fraction": 0.15},
}

_F_DEPOT = 0.95  # bioavailability fraction from depot site


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trapz(times: list[float], values: list[float]) -> float:
    """Trapezoidal integration."""
    return sum(
        0.5 * (values[i] + values[i - 1]) * (times[i] - times[i - 1])
        for i in range(1, len(times))
    )


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class MicrospherePKResult:
    """Result of microsphere PK simulation."""

    drug_name: str
    dose_mg: float
    polymer_type: str
    particle_size_um: float

    times_h: list[float] = field(default_factory=list)
    c_release_depot_mg: list[float] = field(default_factory=list)
    c_plasma_mg_L: list[float] = field(default_factory=list)

    cmax_plasma_mg_L: float = 0.0
    tmax_plasma_h: float = 0.0
    auc_plasma_mg_h_per_L: float = 0.0
    t50_release_h: float = 0.0
    t90_release_h: float = 0.0
    mean_residence_time_h: float = 0.0
    duration_of_action_h: float = 0.0
    notes: str = ""


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate_microsphere_pk(
    drug_name: str,
    dose_mg: float,
    polymer_type: str,
    particle_size_um: float = 50.0,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    mic_threshold_mg_L: float = 0.1,
    t_end_h: float = 720.0,
    dt_h: float = 1.0,
) -> MicrospherePKResult:
    """Simulate plasma concentration profile from an injectable microsphere depot.

    Parameters
    ----------
    drug_name:
        Drug identifier string.
    dose_mg:
        Total dose in milligrams.
    polymer_type:
        One of "PLGA_fast", "PLGA_slow", "PLA", "chitosan", "albumin".
    particle_size_um:
        Mean particle diameter in micrometres. Smaller particles release faster.
    cl_L_per_h:
        Systemic clearance in L/h.
    vd_L:
        Volume of distribution in L.
    mic_threshold_mg_L:
        Minimum effective (or inhibitory) concentration in mg/L used to
        compute duration_of_action_h.
    t_end_h:
        Simulation end time in hours.
    dt_h:
        Forward-Euler step size in hours.

    Returns
    -------
    MicrospherePKResult
    """
    # --- Validation ---
    if dose_mg <= 0.0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_L_per_h <= 0.0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0.0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if polymer_type not in _POLYMERS:
        raise ValueError(
            f"Unknown polymer_type '{polymer_type}'. Valid: {sorted(_POLYMERS)}"
        )
    if t_end_h <= 0.0:
        raise ValueError("t_end_h must be > 0")
    if dt_h <= 0.0:
        raise ValueError("dt_h must be > 0")

    polymer = _POLYMERS[polymer_type]
    burst_fraction = polymer["burst_fraction"]

    # Particle size correction: smaller particles release faster
    k_release = polymer["k_release"] * (50.0 / particle_size_um) ** 0.3

    ke = cl_L_per_h / vd_L  # elimination rate constant (1/h)

    # Initial conditions
    a_depot = dose_mg * (1.0 - burst_fraction)
    c_plasma = dose_mg * burst_fraction / vd_L  # immediate burst


    times: list[float] = []
    c_depot_list: list[float] = []
    c_plasma_list: list[float] = []

    n_steps = int(t_end_h / dt_h)

    for i in range(n_steps + 1):
        t = i * dt_h
        times.append(t)
        c_depot_list.append(a_depot)
        c_plasma_list.append(c_plasma)

        if i < n_steps:
            # ODE: Forward Euler
            da_depot = -k_release * a_depot
            dc_plasma = _F_DEPOT * k_release * a_depot / vd_L - ke * c_plasma

            a_depot = max(0.0, a_depot + da_depot * dt_h)
            c_plasma = max(0.0, c_plasma + dc_plasma * dt_h)

    # --- Derived PK metrics ---
    cmax = max(c_plasma_list)
    tmax = times[c_plasma_list.index(cmax)]
    auc = _trapz(times, c_plasma_list)

    # t50 / t90 release: fraction released from depot
    # Total released at time t = dose - a_depot(t)
    t50 = t_end_h  # default: not reached
    t90 = t_end_h
    for i, t in enumerate(times):
        fraction_released = (dose_mg - c_depot_list[i]) / dose_mg
        if t50 == t_end_h and fraction_released >= 0.50:
            t50 = t
        if t90 == t_end_h and fraction_released >= 0.90:
            t90 = t

    # Mean residence time via AUMC/AUC
    if auc > 0.0:
        aumc_vals = [times[i] * c_plasma_list[i] for i in range(len(times))]
        aumc = _trapz(times, aumc_vals)
        mrt = aumc / auc
    else:
        mrt = 0.0

    # Duration of action: total time plasma >= mic_threshold
    duration = 0.0
    for i in range(1, len(times)):
        # If both points are above threshold, count full interval
        above_start = c_plasma_list[i - 1] >= mic_threshold_mg_L
        above_end = c_plasma_list[i] >= mic_threshold_mg_L
        if above_start and above_end:
            duration += times[i] - times[i - 1]
        elif above_start or above_end:
            # Linear interpolation to find crossing point
            duration += (times[i] - times[i - 1]) * 0.5

    notes = (
        f"Polymer: {polymer_type}, k_release={k_release:.4f}/h "
        f"(burst {burst_fraction*100:.0f}%), ke={ke:.4f}/h"
    )

    return MicrospherePKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        polymer_type=polymer_type,
        particle_size_um=particle_size_um,
        times_h=times,
        c_release_depot_mg=c_depot_list,
        c_plasma_mg_L=c_plasma_list,
        cmax_plasma_mg_L=cmax,
        tmax_plasma_h=tmax,
        auc_plasma_mg_h_per_L=auc,
        t50_release_h=t50,
        t90_release_h=t90,
        mean_residence_time_h=mrt,
        duration_of_action_h=duration,
        notes=notes,
    )


def compare_polymers(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
) -> list:
    """Compare all 5 polymer types and sort by duration_of_action_h descending.

    Parameters
    ----------
    drug_name:
        Drug identifier.
    dose_mg:
        Total dose in milligrams.
    cl_L_per_h:
        Systemic clearance in L/h.
    vd_L:
        Volume of distribution in L.

    Returns
    -------
    list[MicrospherePKResult] sorted by duration_of_action_h descending.
    """
    results = [
        simulate_microsphere_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            polymer_type=polymer,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
        )
        for polymer in _POLYMERS
    ]
    results.sort(key=lambda r: r.duration_of_action_h, reverse=True)
    return results
