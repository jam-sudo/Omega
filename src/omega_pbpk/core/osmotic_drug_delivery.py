"""
Phase 875: Osmotic Drug Delivery Model (OROS)
Models osmotic release oral system (OROS) drug release kinetics.
"""

from dataclasses import dataclass

VALID_OSMOTIC_TYPES = {"single_layer", "dual_layer", "push_pull"}


@dataclass
class OsmoticDrugDeliveryResult:
    """Result from osmotic drug delivery simulation."""

    drug_name: str
    total_dose_mg: float
    osmotic_type: str
    t_release_h: float
    lag_h: float
    times_h: list[float]
    c_plasma_mg_L: list[float]
    a_core_mg: list[float]
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    fraction_released: float
    zero_order_duration_h: float
    t_half_effective_h: float
    notes: str


def _compute_auc_trapezoidal(times: list[float], conc: list[float]) -> float:
    """Manual trapezoidal AUC."""
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        auc += 0.5 * (conc[i] + conc[i - 1]) * dt
    return auc


def simulate_osmotic_delivery(
    drug_name: str,
    total_dose_mg: float,
    t_release_h: float = 8.0,
    lag_h: float = 1.0,
    vd_L: float = 50.0,
    cl_L_per_h: float = 5.0,
    osmotic_type: str = "single_layer",
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> OsmoticDrugDeliveryResult:
    """
    Simulate OROS drug delivery using forward Euler ODE integration.

    Parameters
    ----------
    drug_name : str
    total_dose_mg : float
    t_release_h : float
        Target zero-order release duration (h).
    lag_h : float
        Coating lag time before release begins (h).
    vd_L : float
        Volume of distribution (L).
    cl_L_per_h : float
        Clearance (L/h).
    osmotic_type : str
        "single_layer", "dual_layer", or "push_pull".
    t_end_h : float
        Simulation end time (h).
    dt_h : float
        Integration step (h).

    Returns
    -------
    OsmoticDrugDeliveryResult
    """
    # Validation
    if total_dose_mg <= 0:
        raise ValueError("total_dose_mg must be > 0")
    if t_release_h <= 0:
        raise ValueError("t_release_h must be > 0")
    if lag_h < 0:
        raise ValueError("lag_h must be >= 0")
    if osmotic_type not in VALID_OSMOTIC_TYPES:
        raise ValueError(
            f"osmotic_type must be one of {sorted(VALID_OSMOTIC_TYPES)}, got '{osmotic_type}'"
        )
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")

    # Adjust parameters for push_pull
    effective_lag = lag_h
    effective_t_release = t_release_h
    if osmotic_type == "push_pull":
        effective_lag = lag_h + 1.0
        # pump_rate *= 0.8 means it takes longer to release the same dose
        # effective t_release is extended to accommodate slower rate
        # We model this by scaling: same dose over same nominal release but
        # apply rate *= 0.8 so the actual release duration is longer
        # Duration = total_dose_mg / k_pump; k_pump_pp = 0.8 * k_pump_sl
        # => duration_pp = total_dose_mg / (0.8 * k_pump_sl) = t_release / 0.8
        effective_t_release = t_release_h / 0.8

    # Compute base pump rate from target release duration
    # k_pump = total_dose / t_release (constant zero-order rate)
    k_pump = total_dose_mg / effective_t_release  # mg/h

    ke = cl_L_per_h / vd_L  # 1/h elimination rate constant

    # State variables
    a_core = total_dose_mg  # drug remaining in core (mg)
    c_plasma = 0.0  # plasma concentration (mg/L)

    times_h: list[float] = []
    c_plasma_list: list[float] = []
    a_core_list: list[float] = []

    t = 0.0
    n_steps = int(round(t_end_h / dt_h)) + 1

    # Tracking
    dose_released = 0.0
    half_dose = total_dose_mg / 2.0
    t_half_effective = t_end_h  # fallback if never reached
    release_active_time = 0.0

    for _step in range(n_steps):
        times_h.append(t)
        c_plasma_list.append(c_plasma)
        a_core_list.append(a_core)

        # Determine release rate
        if t > effective_lag and a_core > 0.0:
            release_rate = k_pump
        else:
            release_rate = 0.0

        # Clamp: don't release more than what's in core
        if release_rate * dt_h > a_core:
            release_rate = a_core / dt_h

        # Track when half dose is released
        prev_released = dose_released
        dose_released += release_rate * dt_h
        if prev_released < half_dose <= dose_released:
            # Interpolate
            frac = (half_dose - prev_released) / max(release_rate * dt_h, 1e-12)
            t_half_effective = t + frac * dt_h

        # Track zero-order duration
        if release_rate > 0.0:
            release_active_time += dt_h

        # Forward Euler update
        dc_dt = (release_rate / vd_L) - ke * c_plasma
        da_core_dt = -release_rate

        c_plasma = c_plasma + dc_dt * dt_h
        if c_plasma < 0.0:
            c_plasma = 0.0
        a_core = a_core + da_core_dt * dt_h
        if a_core < 0.0:
            a_core = 0.0

        t += dt_h

    # PK metrics
    cmax = max(c_plasma_list)
    tmax = times_h[c_plasma_list.index(cmax)]
    auc = _compute_auc_trapezoidal(times_h, c_plasma_list)
    fraction_released = dose_released / total_dose_mg
    if fraction_released > 1.0:
        fraction_released = 1.0

    notes_parts = []
    if osmotic_type == "push_pull":
        notes_parts.append("Push-pull design: rate *= 0.8, lag += 1 h.")
    if osmotic_type == "dual_layer":
        notes_parts.append("Dual-layer design: same zero-order pump rate.")
    if fraction_released < 0.95:
        notes_parts.append(
            f"Only {100 * fraction_released:.1f}% released within simulation window."
        )
    notes = " ".join(notes_parts) if notes_parts else "Standard osmotic delivery."

    return OsmoticDrugDeliveryResult(
        drug_name=drug_name,
        total_dose_mg=total_dose_mg,
        osmotic_type=osmotic_type,
        t_release_h=t_release_h,
        lag_h=lag_h,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_list,
        a_core_mg=a_core_list,
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc_mg_h_per_L=auc,
        fraction_released=fraction_released,
        zero_order_duration_h=release_active_time,
        t_half_effective_h=t_half_effective,
        notes=notes,
    )


def compare_osmotic_designs(
    drug_name: str,
    total_dose_mg: float,
    designs: list[dict],
) -> list[OsmoticDrugDeliveryResult]:
    """
    Compare multiple OROS design configurations.

    Parameters
    ----------
    drug_name : str
    total_dose_mg : float
    designs : list[dict]
        Each dict may contain keys: "osmotic_type", "t_release_h", "lag_h"

    Returns
    -------
    list[OsmoticDrugDeliveryResult] sorted by auc_mg_h_per_L descending
    """
    results = []
    for design in designs:
        osmotic_type = design.get("osmotic_type", "single_layer")
        t_release_h = float(design.get("t_release_h", 8.0))
        lag_h = float(design.get("lag_h", 1.0))
        result = simulate_osmotic_delivery(
            drug_name=drug_name,
            total_dose_mg=total_dose_mg,
            t_release_h=t_release_h,
            lag_h=lag_h,
            osmotic_type=osmotic_type,
        )
        results.append(result)

    # Sort by AUC descending
    results.sort(key=lambda r: r.auc_mg_h_per_L, reverse=True)
    return results
