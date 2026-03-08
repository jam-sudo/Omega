"""Phase 272: Mechanistic skin permeation model using Fick's diffusion.

Models transdermal drug permeation through skin layers (vehicle -> stratum
corneum -> dermis -> plasma) using forward Euler integration.
"""

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
A_SKIN_CM2 = 100.0  # Application area (cm^2)
V_VEHICLE_ML = 10.0  # Volume of applied formulation (mL)
V_SC_ML = 0.5  # Hydrated stratum corneum volume (mL)
V_DERMIS_ML = 2.0  # Dermis volume (mL)
V_PLASMA_L = 3.0  # Plasma volume (L)

# Vehicle parameters: (Kv_sc, permeation enhancement factor)
VEHICLE_PARAMS = {
    "solution": (10.0, 1.0),
    "cream": (5.0, 0.8),
    "gel": (8.0, 0.9),
    "ointment": (3.0, 0.7),
}

BASE_PERM_CM_PER_H = 1e-3  # Base permeability (cm/h)
K_SD_PER_H = 0.1  # SC -> dermis first-order rate constant (1/h)
K_DP_PER_H = 0.05  # Dermis -> plasma first-order rate constant (1/h)

PLASMA_REACH_THRESHOLD = 0.01  # Fraction of Cmax for tlag determination


@dataclass
class SkinPermeationResult:
    """Result of skin permeation simulation."""

    drug_name: str
    dose_applied_mg: float
    vehicle_type: str
    times_h: list[float]
    c_vehicle_mg_mL: list[float]
    c_sc_mg_g: list[float]
    c_dermis_mg_mL: list[float]
    c_plasma_mg_L: list[float]
    flux_peak_ug_cm2_h: float
    tlag_h: float
    auc_dermis_mg_h_per_mL: float
    auc_plasma_mg_h_per_L: float
    cumulative_absorbed_mg: float
    skin_retention_pct: float
    notes: str


def _trapz(times: list[float], values: list[float]) -> float:
    """Trapezoidal integration."""
    return sum(
        0.5 * (values[i] + values[i - 1]) * (times[i] - times[i - 1]) for i in range(1, len(times))
    )


def simulate_skin_permeation(
    drug_name: str,
    dose_applied_mg: float,
    logp: float,
    mw: float,
    vehicle_type: str = "solution",
    cl_L_per_h: float = 5.0,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
) -> SkinPermeationResult:
    """Simulate transdermal drug permeation through skin layers.

    Uses a 3-compartment forward Euler model:
        vehicle -> stratum corneum (SC) -> dermis -> plasma

    Args:
        drug_name: Name of the drug.
        dose_applied_mg: Applied dose in mg.
        logp: Octanol-water partition coefficient (influences permeability).
        mw: Molecular weight in g/mol.
        vehicle_type: One of "solution", "cream", "gel", "ointment".
        cl_L_per_h: Systemic clearance in L/h.
        t_end_h: Simulation end time in hours.
        dt_h: Time step in hours.

    Returns:
        SkinPermeationResult with time profiles and summary metrics.

    Raises:
        ValueError: If inputs are invalid.
    """
    if dose_applied_mg <= 0:
        raise ValueError(f"dose_applied_mg must be > 0, got {dose_applied_mg}")
    if vehicle_type not in VEHICLE_PARAMS:
        raise ValueError(
            f"vehicle_type must be one of {list(VEHICLE_PARAMS.keys())}, got '{vehicle_type}'"
        )
    if t_end_h <= 0:
        raise ValueError(f"t_end_h must be > 0, got {t_end_h}")
    if dt_h <= 0 or dt_h >= t_end_h:
        raise ValueError(f"dt_h must be in (0, t_end_h), got {dt_h}")

    _kv_sc, enhance = VEHICLE_PARAMS[vehicle_type]

    # Permeability coefficient (cm/h) — logP influences SC partitioning
    # logP optimal around 2-3 for skin; use simple logP-based correction
    # logP_factor: peak near logP=2, penalized for very high/low values
    logp_factor = 1.0 / (1.0 + abs(logp - 2.0) * 0.1)
    perm_cm_h = enhance * BASE_PERM_CM_PER_H * (1.0 + logp_factor)

    # Rate constants
    # vehicle -> SC: based on Fick's law (concentration gradient vehicle -> SC surface)
    k_vs = perm_cm_h * A_SKIN_CM2 / V_VEHICLE_ML  # 1/h, driving vehicle concentration
    k_sd = K_SD_PER_H  # 1/h
    k_dp = K_DP_PER_H  # 1/h

    # Systemic elimination from plasma
    ke = cl_L_per_h / V_PLASMA_L  # 1/h (plasma)

    # Initial conditions
    c_veh = dose_applied_mg / V_VEHICLE_ML  # mg/mL
    c_sc = 0.0  # mg/mL (simplified — mass in SC / V_SC)
    c_der = 0.0  # mg/mL
    c_pla = 0.0  # mg/L

    # Storage
    n_steps = int(t_end_h / dt_h) + 1
    times: list[float] = []
    c_veh_arr: list[float] = []
    c_sc_arr: list[float] = []
    c_der_arr: list[float] = []
    c_pla_arr: list[float] = []
    flux_arr: list[float] = []
    cum_absorbed_arr: list[float] = []

    cumulative_absorbed = 0.0

    for step in range(n_steps):
        t = step * dt_h

        # Flux from vehicle to SC (ug/cm2/h) — for reporting
        # flux = perm * c_veh [mg/mL = mg/cm3] * 1000 [ug/mg] = ug/cm3/h * A...
        # Simplified: flux = perm_cm_h * c_veh * 1000 (ug/cm2/h)
        flux_ug_cm2_h = perm_cm_h * c_veh * 1000.0

        times.append(t)
        c_veh_arr.append(c_veh)
        c_sc_arr.append(c_sc)
        c_der_arr.append(c_der)
        c_pla_arr.append(c_pla)
        flux_arr.append(flux_ug_cm2_h)
        cum_absorbed_arr.append(cumulative_absorbed)

        # Rate of change
        # mass transferred vehicle -> SC per h (mg/h)
        rate_vs_mass = k_vs * c_veh * V_VEHICLE_ML  # mg/h (mass leaving vehicle)

        dc_veh = -k_vs * c_veh  # mg/mL/h
        dc_sc = (rate_vs_mass / V_SC_ML) - k_sd * c_sc  # mg/mL/h
        dc_der = (k_sd * c_sc * V_SC_ML / V_DERMIS_ML) - k_dp * c_der  # mg/mL/h

        # mass entering plasma per h (mg/h): k_dp * c_der * V_DERMIS_ML
        # c_plasma in mg/L: convert V_DERMIS from mL to L
        rate_dp_mass_mg_h = k_dp * c_der * V_DERMIS_ML  # mg/h
        dc_pla = (rate_dp_mass_mg_h / (V_PLASMA_L * 1000.0)) * 1000.0 - ke * c_pla
        # Simplify: dc_pla (mg/L/h) = rate_dp_mass_mg_h / V_PLASMA_L [mg/L] - ke * c_pla
        dc_pla = rate_dp_mass_mg_h / V_PLASMA_L - ke * c_pla  # mg/L/h

        # Cumulative absorbed (mass entering plasma)
        cumulative_absorbed += rate_dp_mass_mg_h * dt_h

        # Forward Euler update
        c_veh = max(0.0, c_veh + dc_veh * dt_h)
        c_sc = max(0.0, c_sc + dc_sc * dt_h)
        c_der = max(0.0, c_der + dc_der * dt_h)
        c_pla = max(0.0, c_pla + dc_pla * dt_h)

    # Summary metrics
    flux_peak = max(flux_arr) if flux_arr else 0.0

    # tlag: time for plasma to reach 1% of its eventual maximum
    c_pla_max = max(c_pla_arr) if c_pla_arr else 0.0
    tlag = t_end_h  # default if never reached
    threshold = PLASMA_REACH_THRESHOLD * c_pla_max
    for i, cp in enumerate(c_pla_arr):
        if cp >= threshold and threshold > 0.0:
            tlag = times[i]
            break

    auc_dermis = _trapz(times, c_der_arr)
    auc_plasma = _trapz(times, c_pla_arr)

    skin_retention_pct = max(0.0, min(100.0, (1.0 - cumulative_absorbed / dose_applied_mg) * 100.0))

    notes_parts = []
    if skin_retention_pct > 80.0:
        notes_parts.append("High skin retention — consider formulation optimization.")
    if flux_peak < 0.1:
        notes_parts.append("Low peak flux — poor transdermal permeation.")
    notes = " ".join(notes_parts) if notes_parts else "Simulation complete."

    return SkinPermeationResult(
        drug_name=drug_name,
        dose_applied_mg=dose_applied_mg,
        vehicle_type=vehicle_type,
        times_h=times,
        c_vehicle_mg_mL=c_veh_arr,
        c_sc_mg_g=c_sc_arr,
        c_dermis_mg_mL=c_der_arr,
        c_plasma_mg_L=c_pla_arr,
        flux_peak_ug_cm2_h=flux_peak,
        tlag_h=tlag,
        auc_dermis_mg_h_per_mL=auc_dermis,
        auc_plasma_mg_h_per_L=auc_plasma,
        cumulative_absorbed_mg=cumulative_absorbed,
        skin_retention_pct=skin_retention_pct,
        notes=notes,
    )


def compare_vehicles(
    drug_name: str,
    dose_applied_mg: float,
    logp: float,
    mw: float,
    cl_L_per_h: float = 5.0,
) -> list[SkinPermeationResult]:
    """Compare all vehicle types for transdermal drug delivery.

    Args:
        drug_name: Name of the drug.
        dose_applied_mg: Applied dose in mg.
        logp: Octanol-water partition coefficient.
        mw: Molecular weight in g/mol.
        cl_L_per_h: Systemic clearance in L/h.

    Returns:
        List of SkinPermeationResult for all vehicles, sorted by
        auc_plasma descending (best delivery first).
    """
    results = []
    for vehicle in VEHICLE_PARAMS:
        result = simulate_skin_permeation(
            drug_name=drug_name,
            dose_applied_mg=dose_applied_mg,
            logp=logp,
            mw=mw,
            vehicle_type=vehicle,
            cl_L_per_h=cl_L_per_h,
        )
        results.append(result)

    # Sort by plasma AUC descending
    results.sort(key=lambda r: r.auc_plasma_mg_h_per_L, reverse=True)
    return results
