"""Transdermal patch PK simulation (Phase 251).

Models reservoir and matrix transdermal drug delivery patches using a
simplified 2-compartment system: patch depot -> skin/plasma -> systemic.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TransdermalPatchResult:
    """Result of a transdermal patch PK simulation."""

    drug_name: str
    dose_mg: float
    patch_type: str  # "reservoir" or "matrix"
    release_area_cm2: float
    patch_duration_h: float

    times_h: list[float] = field(default_factory=list)
    m_patch_mg: list[float] = field(default_factory=list)  # Drug remaining in patch
    c_plasma_mg_L: list[float] = field(default_factory=list)  # Plasma concentration

    cmax_mg_L: float = 0.0
    tmax_h: float = 0.0
    auc_mg_h_per_L: float = 0.0
    f_delivered: float = 0.0  # Fraction of dose actually delivered
    fluctuation_pct: float = 0.0  # (Cmax - Cmin) / Cavg * 100 during steady delivery
    delivery_rate_mg_per_h: float = 0.0  # Average delivery rate during patch
    notes: str = ""


def simulate_transdermal_patch(
    drug_name: str,
    dose_mg: float,
    patch_type: str = "reservoir",
    release_area_cm2: float = 20.0,
    patch_duration_h: float = 72.0,
    flux_mg_per_cm2_per_h: float | None = None,
    cl_L_per_h: float = 5.0,
    vd_L: float = 100.0,
    k_skin_per_h: float = 0.5,
    lag_time_h: float = 2.0,
    t_end_h: float = 96.0,
    dt_h: float = 0.1,
) -> TransdermalPatchResult:
    """Simulate transdermal patch PK.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Total drug dose in the patch (mg).
    patch_type:
        "reservoir" (near zero-order) or "matrix" (first-order-like).
    release_area_cm2:
        Active skin contact area of the patch (cm²).
    patch_duration_h:
        Duration the patch is worn (h).
    flux_mg_per_cm2_per_h:
        Skin flux rate. If None, computed from dose / area / duration.
    cl_L_per_h:
        Systemic clearance (L/h).
    vd_L:
        Volume of distribution (L).
    k_skin_per_h:
        First-order rate constant for skin absorption into plasma (h⁻¹).
    lag_time_h:
        Lag time before drug reaches systemic circulation (h).
    t_end_h:
        Simulation end time (h).
    dt_h:
        Integration time step (h).

    Returns
    -------
    TransdermalPatchResult
    """
    # --- Input validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if release_area_cm2 <= 0:
        raise ValueError("release_area_cm2 must be > 0")
    if patch_duration_h <= 0:
        raise ValueError("patch_duration_h must be > 0")
    if patch_type not in ("reservoir", "matrix"):
        raise ValueError("patch_type must be 'reservoir' or 'matrix'")

    # Compute flux if not provided
    if flux_mg_per_cm2_per_h is None:
        # Assumes 80% bioavailability through skin for baseline estimate
        flux_mg_per_cm2_per_h = dose_mg / (release_area_cm2 * patch_duration_h)

    # Derived PK parameters
    ke = cl_L_per_h / vd_L  # Elimination rate constant (h⁻¹)

    # For matrix patch: first-order release constant such that integrated
    # release over patch_duration equals dose (approx.)
    # k_release for matrix: dose * k_rel * exp(-k_rel*t) integrated = dose * (1 - exp(-k_rel*T))
    # We target ~90% release by patch_duration -> k_rel ≈ 2.3 / patch_duration
    k_matrix = 2.3 / patch_duration_h  # h⁻¹, gives ~90% release by patch_duration

    # Total flux for reservoir
    total_flux_reservoir = flux_mg_per_cm2_per_h * release_area_cm2  # mg/h

    # Simulation
    n_steps = int(t_end_h / dt_h) + 1
    times = [i * dt_h for i in range(n_steps)]

    m_patch = [0.0] * n_steps  # Drug remaining in patch (mg)
    m_skin = [0.0] * n_steps  # Drug in skin compartment (mg)
    c_plasma = [0.0] * n_steps  # Plasma concentration (mg/L)

    m_patch[0] = dose_mg
    m_skin[0] = 0.0
    c_plasma[0] = 0.0

    m_absorbed_total = 0.0  # Total drug absorbed into systemic circulation

    for i in range(1, n_steps):
        t = times[i - 1]  # Current time
        mp = m_patch[i - 1]
        ms = m_skin[i - 1]
        cp = c_plasma[i - 1]

        # Release from patch to skin (after lag time, while patch is on)
        if t >= lag_time_h and t < patch_duration_h and mp > 0:
            if patch_type == "reservoir":
                # Constant flux until reservoir depletes
                release_rate = min(total_flux_reservoir, mp / dt_h)
            else:
                # Matrix: exponentially declining flux
                release_rate = k_matrix * mp
        else:
            release_rate = 0.0

        # Drug absorbed from skin to plasma
        absorption_rate = k_skin_per_h * ms  # mg/h

        # Drug eliminated from plasma
        elimination_rate = ke * cp * vd_L  # mg/h

        # ODE update (forward Euler)
        dm_patch = -release_rate * dt_h
        dm_skin = (release_rate - absorption_rate) * dt_h
        dc_plasma = (absorption_rate - elimination_rate) * dt_h / vd_L

        # Apply changes with bounds
        m_patch[i] = max(0.0, mp + dm_patch)
        m_skin[i] = max(0.0, ms + dm_skin)
        c_plasma[i] = max(0.0, cp + dc_plasma)

        # Track absorbed drug (from skin to plasma)
        m_absorbed_total += absorption_rate * dt_h

    # Compute summary statistics
    cmax = max(c_plasma)
    tmax = times[c_plasma.index(cmax)]

    # AUC via trapezoidal rule
    auc = 0.0
    for i in range(1, n_steps):
        auc += 0.5 * (c_plasma[i - 1] + c_plasma[i]) * dt_h

    # f_delivered: fraction of dose absorbed systemically
    f_delivered = min(1.0, max(0.0, m_absorbed_total / dose_mg))

    # Average delivery rate during patch application
    delivery_rate = m_absorbed_total / patch_duration_h

    # Fluctuation during the main delivery phase (lag to end of patch)
    # Use concentrations from lag+delivery_onset to patch_duration
    delivery_start_idx = int((lag_time_h + 2.0) / dt_h)  # Allow 2h to reach quasi-SS
    delivery_end_idx = min(int(patch_duration_h / dt_h), n_steps - 1)
    if delivery_end_idx > delivery_start_idx + 2:
        delivery_concs = c_plasma[delivery_start_idx:delivery_end_idx]
        c_max_d = max(delivery_concs)
        c_min_d = min(delivery_concs)
        c_avg_d = sum(delivery_concs) / len(delivery_concs)
        if c_avg_d > 0:
            fluctuation_pct = (c_max_d - c_min_d) / c_avg_d * 100.0
        else:
            fluctuation_pct = 0.0
    else:
        fluctuation_pct = 0.0

    notes = (
        f"{patch_type.capitalize()} patch: {dose_mg} mg over {patch_duration_h} h, "
        f"area={release_area_cm2} cm², lag={lag_time_h} h"
    )

    return TransdermalPatchResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        patch_type=patch_type,
        release_area_cm2=release_area_cm2,
        patch_duration_h=patch_duration_h,
        times_h=times,
        m_patch_mg=m_patch,
        c_plasma_mg_L=c_plasma,
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc_mg_h_per_L=auc,
        f_delivered=f_delivered,
        fluctuation_pct=fluctuation_pct,
        delivery_rate_mg_per_h=delivery_rate,
        notes=notes,
    )


def compare_patch_types(
    drug_name: str,
    dose_mg: float,
    release_area_cm2: float,
    patch_duration_h: float,
    cl_L_per_h: float,
    vd_L: float,
    **kwargs,
) -> dict:
    """Compare reservoir vs matrix patch types.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Total drug dose in the patch (mg).
    release_area_cm2:
        Active skin contact area (cm²).
    patch_duration_h:
        Duration the patch is worn (h).
    cl_L_per_h:
        Systemic clearance (L/h).
    vd_L:
        Volume of distribution (L).
    **kwargs:
        Additional keyword arguments passed to simulate_transdermal_patch.

    Returns
    -------
    dict with keys: "reservoir", "matrix", "auc_ratio" (reservoir/matrix),
    "cmax_ratio", "fluctuation_ratio" (reservoir/matrix, lower = more controlled)
    """
    reservoir_result = simulate_transdermal_patch(
        drug_name=drug_name,
        dose_mg=dose_mg,
        patch_type="reservoir",
        release_area_cm2=release_area_cm2,
        patch_duration_h=patch_duration_h,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
        **kwargs,
    )
    matrix_result = simulate_transdermal_patch(
        drug_name=drug_name,
        dose_mg=dose_mg,
        patch_type="matrix",
        release_area_cm2=release_area_cm2,
        patch_duration_h=patch_duration_h,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
        **kwargs,
    )

    auc_ratio = (
        reservoir_result.auc_mg_h_per_L / matrix_result.auc_mg_h_per_L
        if matrix_result.auc_mg_h_per_L > 0
        else float("nan")
    )
    cmax_ratio = (
        reservoir_result.cmax_mg_L / matrix_result.cmax_mg_L
        if matrix_result.cmax_mg_L > 0
        else float("nan")
    )
    fluctuation_ratio = (
        reservoir_result.fluctuation_pct / matrix_result.fluctuation_pct
        if matrix_result.fluctuation_pct > 0
        else float("nan")
    )

    return {
        "reservoir": reservoir_result,
        "matrix": matrix_result,
        "auc_ratio": auc_ratio,
        "cmax_ratio": cmax_ratio,
        "fluctuation_ratio": fluctuation_ratio,
    }
