"""
Phase 252: Subcutaneous / Intramuscular Drug Implant PK Model

Simulates sustained drug release from implants over weeks to months.
Uses a 2-compartment (depot + plasma) ODE model with forward Euler integration,
time units in days to handle long release profiles.
"""

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Implant type lookup table
# ---------------------------------------------------------------------------

_IMPLANT_TYPES = {
    "reservoir": {
        "k_release_per_day": 0.01,
        "burst_fraction": 0.03,
    },
    "matrix_rod": {
        "k_release_per_day": 0.015,
        "burst_fraction": 0.05,
    },
    "osmotic_pump": {
        "k_release_per_day": 0.008,
        "burst_fraction": 0.0,
    },
    "bioerodible": {
        "k_release_per_day": 0.02,
        "burst_fraction": 0.08,
    },
    "hydrogel": {
        "k_release_per_day": 0.05,
        "burst_fraction": 0.15,
    },
}

VALID_IMPLANT_TYPES = list(_IMPLANT_TYPES.keys())


# ---------------------------------------------------------------------------
# Result dataclass (plain — has list fields)
# ---------------------------------------------------------------------------


@dataclass
class ImplantPKResult:
    """Result of an implant PK simulation."""

    drug_name: str
    dose_mg: float
    implant_type: str
    times_days: list[float]
    c_plasma_mg_L: list[float]
    cmax_plasma_mg_L: float
    tmax_days: float
    auc_plasma_mg_day_per_L: float
    t50_release_days: float
    t90_release_days: float
    duration_therapeutic_days: float
    mean_residence_time_days: float
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _trapz(times: list[float], values: list[float]) -> float:
    """Manual trapezoidal integration."""
    return sum(
        0.5 * (values[i] + values[i - 1]) * (times[i] - times[i - 1]) for i in range(1, len(times))
    )


def _find_time_for_fraction(
    times: list[float],
    a_depot_list: list[float],
    a_depot_0: float,
    fraction: float,
) -> float:
    """
    Find the time at which depot amount drops to (1-fraction)*a_depot_0,
    i.e., fraction of drug has been released from depot.
    Returns the last time point if threshold is never reached.
    """
    target = (1.0 - fraction) * a_depot_0
    for i in range(len(a_depot_list)):
        if a_depot_list[i] <= target:
            return times[i]
    return times[-1]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def simulate_implant_pk(
    drug_name: str,
    dose_mg: float,
    implant_type: str,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    mic_threshold_mg_L: float = 0.1,
    t_end_days: float = 180.0,
    dt_days: float = 0.5,
) -> ImplantPKResult:
    """
    Simulate plasma PK for a subcutaneous/intramuscular drug implant.

    Uses a 2-compartment forward-Euler ODE:
        dA_depot/dt = -k_release * A_depot
        dC_plasma/dt = k_release * A_depot / vd_L - ke * C_plasma

    where ke = cl_L_per_day / vd_L and cl_L_per_day = cl_L_per_h * 24.

    Parameters
    ----------
    drug_name : str
    dose_mg : float
        Total dose loaded in the implant (mg).
    implant_type : str
        One of: reservoir, matrix_rod, osmotic_pump, bioerodible, hydrogel.
    cl_L_per_h : float
        Systemic clearance (L/h). Default 5.0.
    vd_L : float
        Volume of distribution (L). Default 50.0.
    mic_threshold_mg_L : float
        Minimum inhibitory (or effective) concentration for duration calc.
    t_end_days : float
        Simulation end time (days). Default 180.
    dt_days : float
        Forward Euler step size (days). Default 0.5.

    Returns
    -------
    ImplantPKResult
    """
    if dose_mg <= 0.0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if implant_type not in _IMPLANT_TYPES:
        raise ValueError(f"implant_type must be one of {VALID_IMPLANT_TYPES}, got '{implant_type}'")

    props = _IMPLANT_TYPES[implant_type]
    k_release = props["k_release_per_day"]
    burst_fraction = props["burst_fraction"]

    # Convert clearance to per-day
    cl_L_per_day = cl_L_per_h * 24.0
    ke = cl_L_per_day / vd_L  # elimination rate constant (1/day)

    # Initial conditions
    a_depot_0 = dose_mg * (1.0 - burst_fraction)
    c_plasma_0 = dose_mg * burst_fraction / vd_L

    # Forward Euler simulation
    n_steps = int(t_end_days / dt_days) + 1
    times: list[float] = []
    c_plasma_list: list[float] = []
    a_depot_list: list[float] = []

    a_depot = a_depot_0
    c_plasma = c_plasma_0

    for step in range(n_steps):
        t = step * dt_days
        times.append(t)
        c_plasma_list.append(c_plasma)
        a_depot_list.append(a_depot)

        # ODE RHS
        da_depot = -k_release * a_depot
        dc_plasma = k_release * a_depot / vd_L - ke * c_plasma

        # Forward Euler update
        a_depot = a_depot + da_depot * dt_days
        c_plasma = c_plasma + dc_plasma * dt_days

        # Clamp to non-negative
        if a_depot < 0.0:
            a_depot = 0.0
        if c_plasma < 0.0:
            c_plasma = 0.0

    # Derived metrics
    cmax = max(c_plasma_list)
    tmax_days = times[c_plasma_list.index(cmax)]

    auc = _trapz(times, c_plasma_list)

    # t50 and t90 release (from depot depletion)
    t50_release = _find_time_for_fraction(times, a_depot_list, a_depot_0, 0.50)
    t90_release = _find_time_for_fraction(times, a_depot_list, a_depot_0, 0.90)

    # Duration above MIC
    duration_therapeutic = 0.0
    for i in range(1, len(times)):
        c_mid = 0.5 * (c_plasma_list[i] + c_plasma_list[i - 1])
        if c_mid >= mic_threshold_mg_L:
            duration_therapeutic += times[i] - times[i - 1]

    # Mean residence time: MRT = AUMC / AUC
    # AUMC = integral of t * C(t) dt
    tc_list = [times[i] * c_plasma_list[i] for i in range(len(times))]
    aumc = _trapz(times, tc_list)
    mrt = aumc / auc if auc > 0.0 else 0.0

    notes = (
        f"implant={implant_type}; k_release={k_release}/day; "
        f"burst={burst_fraction * 100:.0f}%; Cmax={cmax:.4f} mg/L at {tmax_days:.1f} d; "
        f"AUC={auc:.2f} mg*day/L; duration_above_MIC={duration_therapeutic:.1f} d"
    )

    return ImplantPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        implant_type=implant_type,
        times_days=times,
        c_plasma_mg_L=c_plasma_list,
        cmax_plasma_mg_L=cmax,
        tmax_days=tmax_days,
        auc_plasma_mg_day_per_L=auc,
        t50_release_days=t50_release,
        t90_release_days=t90_release,
        duration_therapeutic_days=duration_therapeutic,
        mean_residence_time_days=mrt,
        notes=notes,
    )


def compare_implant_types(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
) -> list[ImplantPKResult]:
    """
    Compare all 5 implant types for the given drug and dose.

    Returns a list sorted by duration_therapeutic_days descending
    (using default mic_threshold_mg_L=0.1 and t_end_days=180).

    Parameters
    ----------
    drug_name : str
    dose_mg : float
    cl_L_per_h : float
    vd_L : float

    Returns
    -------
    list of ImplantPKResult (5 items)
    """
    results = []
    for it in VALID_IMPLANT_TYPES:
        result = simulate_implant_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            implant_type=it,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
        )
        results.append(result)

    results.sort(key=lambda r: r.duration_therapeutic_days, reverse=True)
    return results
