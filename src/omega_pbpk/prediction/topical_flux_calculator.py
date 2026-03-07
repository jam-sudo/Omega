"""Phase 289 — Topical Formulation Flux Calculator.

Calculates steady-state percutaneous flux and cumulative permeation for
topical formulations using Fick's first law with vehicle-specific partitioning.
"""

from dataclasses import dataclass

VALID_FORMULATIONS = {"cream", "gel", "ointment", "lotion"}

# Enhancement factors relative to cream baseline
ENHANCEMENT_FACTORS: dict[str, float] = {
    "cream": 1.0,
    "gel": 0.8,
    "ointment": 1.3,
    "lotion": 0.7,
}


@dataclass(frozen=True)
class TopicalFluxResult:
    """Result of topical flux calculation."""

    drug_name: str
    logp: float
    mw: float
    concentration_pct: float
    formulation_type: str
    kp_cm_per_h: float
    kp_eff_cm_per_h: float
    enhancement_factor: float
    steady_state_flux_ug_per_cm2_per_h: float
    cumulative_24h_ug: float
    lag_time_h: float
    f_sys_estimate: float
    permeability_class: str
    notes: str


def _validate_inputs(
    logp: float,
    mw: float,
    concentration_pct: float,
    application_area_cm2: float,
    vehicle_skin_partition: float,
    formulation_type: str,
) -> None:
    """Validate all input parameters."""
    if not (-5 < logp < 10):
        raise ValueError(f"logp must be in (-5, 10), got {logp}")
    if mw <= 0:
        raise ValueError(f"mw must be > 0, got {mw}")
    if not (0 < concentration_pct <= 100):
        raise ValueError(f"concentration_pct must be in (0, 100], got {concentration_pct}")
    if application_area_cm2 <= 0:
        raise ValueError(f"application_area_cm2 must be > 0, got {application_area_cm2}")
    if vehicle_skin_partition <= 0:
        raise ValueError(f"vehicle_skin_partition must be > 0, got {vehicle_skin_partition}")
    if formulation_type not in VALID_FORMULATIONS:
        raise ValueError(
            f"formulation_type must be one of {VALID_FORMULATIONS}, got '{formulation_type}'"
        )


def calculate_topical_flux(
    drug_name: str,
    logp: float,
    mw: float,
    concentration_pct: float,
    formulation_type: str,
    application_area_cm2: float,
    vehicle_skin_partition: float = 1.0,
) -> TopicalFluxResult:
    """Calculate steady-state percutaneous flux for a topical formulation.

    Parameters
    ----------
    drug_name:
        Name of the drug compound.
    logp:
        Octanol-water partition coefficient (log scale). Must be in (-5, 10).
    mw:
        Molecular weight in g/mol. Must be > 0.
    concentration_pct:
        Drug concentration in formulation (% w/w). Must be in (0, 100].
    formulation_type:
        One of "cream", "gel", "ointment", "lotion".
    application_area_cm2:
        Skin surface area of application in cm2. Must be > 0.
    vehicle_skin_partition:
        Partition coefficient from vehicle into skin (Kv). Default 1.0.
        Must be > 0.

    Returns
    -------
    TopicalFluxResult
        Frozen dataclass with flux, cumulative permeation, and classification.
    """
    _validate_inputs(
        logp, mw, concentration_pct, application_area_cm2, vehicle_skin_partition, formulation_type
    )

    # Potts-Guy QSAR: log10(Kp_cm/h) = -2.7 + 0.71*logp - 0.0061*mw
    log_kp = -2.7 + 0.71 * logp - 0.0061 * mw
    kp_cm_per_h = 10.0**log_kp

    enhancement_factor = ENHANCEMENT_FACTORS[formulation_type]
    kp_eff_cm_per_h = kp_cm_per_h * enhancement_factor * vehicle_skin_partition

    # Drug concentration in vehicle: % w/w * 10 ≈ mg/mL
    c_vehicle_mg_per_ml = concentration_pct * 10.0

    # Steady-state flux: J_ss = Kp_eff * C_vehicle (mg/cm2/h) → µg/cm2/h
    j_ss_mg = kp_eff_cm_per_h * c_vehicle_mg_per_ml
    j_ss_ug = j_ss_mg * 1000.0

    # Cumulative 24h permeation (µg)
    cumulative_24h_ug = j_ss_ug * application_area_cm2 * 24.0

    # Approximate lag time
    t_lag_h = 1.0 / (kp_eff_cm_per_h + 0.1)

    # Systemic bioavailability estimate
    f_sys = min(1.0, kp_eff_cm_per_h / (kp_eff_cm_per_h + 10.0))

    # Permeability classification based on kp_eff
    if kp_eff_cm_per_h > 0.1:
        permeability_class = "high"
    elif kp_eff_cm_per_h >= 0.01:
        permeability_class = "moderate"
    else:
        permeability_class = "low"

    notes = (
        f"Potts-Guy QSAR Kp={kp_cm_per_h:.4e} cm/h; "
        f"enhancement={enhancement_factor:.1f}x ({formulation_type}); "
        f"Kv_skin={vehicle_skin_partition:.2f}; "
        f"C_vehicle={c_vehicle_mg_per_ml:.1f} mg/mL"
    )

    return TopicalFluxResult(
        drug_name=drug_name,
        logp=logp,
        mw=mw,
        concentration_pct=concentration_pct,
        formulation_type=formulation_type,
        kp_cm_per_h=kp_cm_per_h,
        kp_eff_cm_per_h=kp_eff_cm_per_h,
        enhancement_factor=enhancement_factor,
        steady_state_flux_ug_per_cm2_per_h=j_ss_ug,
        cumulative_24h_ug=cumulative_24h_ug,
        lag_time_h=t_lag_h,
        f_sys_estimate=f_sys,
        permeability_class=permeability_class,
        notes=notes,
    )


def compare_formulations(
    drug_name: str,
    logp: float,
    mw: float,
    concentration_pct: float,
    application_area_cm2: float,
    vehicle_skin_partition: float = 1.0,
) -> list[TopicalFluxResult]:
    """Compare all four formulation types, sorted by flux descending.

    Parameters
    ----------
    drug_name:
        Name of the drug compound.
    logp:
        Octanol-water partition coefficient.
    mw:
        Molecular weight in g/mol.
    concentration_pct:
        Drug concentration in formulation (% w/w).
    application_area_cm2:
        Skin surface area of application in cm2.
    vehicle_skin_partition:
        Partition coefficient from vehicle into skin. Default 1.0.

    Returns
    -------
    list[TopicalFluxResult]
        Results for all formulation types sorted by steady_state_flux
        descending (highest flux first).
    """
    results = []
    for ftype in VALID_FORMULATIONS:
        result = calculate_topical_flux(
            drug_name=drug_name,
            logp=logp,
            mw=mw,
            concentration_pct=concentration_pct,
            formulation_type=ftype,
            application_area_cm2=application_area_cm2,
            vehicle_skin_partition=vehicle_skin_partition,
        )
        results.append(result)

    results.sort(key=lambda r: r.steady_state_flux_ug_per_cm2_per_h, reverse=True)
    return results
