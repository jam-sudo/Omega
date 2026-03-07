"""
Phase 343 — Inhalation Toxicology Dosimetry

PBPK-based dosimetry model for inhaled toxicants following EPA/OSHA risk assessment
methodology. Computes regional deposition, systemic absorption, human equivalent
concentration (HEC), and margin of exposure (MOE).
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InhalationDosimetryResult:
    chemical_name: str
    airborne_conc_mg_per_m3: float
    exposure_duration_h: float
    minute_ventilation_L_per_min: float
    total_inhaled_mg_per_h: float
    alveolar_deposited_mg_per_h: float
    absorbed_systemic_mg_per_h: float
    total_absorbed_mg: float
    human_equivalent_conc_mg_per_kg: float
    moe: float
    risk_class: str  # "low" / "moderate" / "high"
    notes: str


# ---------------------------------------------------------------------------
# Core calculation
# ---------------------------------------------------------------------------

_DEFAULT_DEPOSITION: dict[str, float] = {"et": 0.3, "tb": 0.2, "al": 0.5}
_VALID_RISK = {"low", "moderate", "high"}


def calculate_inhalation_dosimetry(
    chemical_name: str,
    airborne_conc_mg_per_m3: float,
    exposure_duration_h: float,
    tidal_volume_mL: float = 500.0,
    respiratory_rate_per_min: float = 15.0,
    regional_deposition_fraction: dict[str, float] | None = None,
    absorption_fraction_al: float = 0.8,
    body_weight_kg: float = 70.0,
    noael_mg_per_kg: float = 100.0,
) -> InhalationDosimetryResult:
    """Compute inhalation dosimetry for a single exposure scenario.

    Parameters
    ----------
    chemical_name:
        Name or identifier for the chemical / toxicant.
    airborne_conc_mg_per_m3:
        Airborne concentration in mg/m3 (must be > 0).
    exposure_duration_h:
        Duration of inhalation exposure in hours (must be > 0).
    tidal_volume_mL:
        Tidal volume in mL (default 500 mL, typical adult at rest).
    respiratory_rate_per_min:
        Breaths per minute (default 15).
    regional_deposition_fraction:
        Dict with keys "et", "tb", "al" representing extrathoracic,
        tracheobronchial, and alveolar deposition fractions.  Must sum
        to 1.0 (±0.01).  Defaults to {"et": 0.3, "tb": 0.2, "al": 0.5}.
    absorption_fraction_al:
        Fraction of alveolar-deposited mass absorbed systemically (0–1,
        default 0.8).
    body_weight_kg:
        Subject body weight in kg (default 70 kg).
    noael_mg_per_kg:
        No-observed-adverse-effect level in mg/kg used for MOE (default
        100 mg/kg).

    Returns
    -------
    InhalationDosimetryResult
    """
    # --- Validation ---
    if airborne_conc_mg_per_m3 <= 0:
        raise ValueError(f"airborne_conc_mg_per_m3 must be > 0, got {airborne_conc_mg_per_m3}")
    if exposure_duration_h <= 0:
        raise ValueError(f"exposure_duration_h must be > 0, got {exposure_duration_h}")
    if body_weight_kg <= 0:
        raise ValueError(f"body_weight_kg must be > 0, got {body_weight_kg}")
    if not (0.0 <= absorption_fraction_al <= 1.0):
        raise ValueError(f"absorption_fraction_al must be in [0, 1], got {absorption_fraction_al}")

    if regional_deposition_fraction is None:
        dep = dict(_DEFAULT_DEPOSITION)
    else:
        dep = dict(regional_deposition_fraction)

    # Validate deposition fractions
    for key in ("et", "tb", "al"):
        if key not in dep:
            raise ValueError(f"regional_deposition_fraction missing key '{key}'")
    frac_sum = dep["et"] + dep["tb"] + dep["al"]
    if abs(frac_sum - 1.0) > 0.01:
        raise ValueError(
            f"regional_deposition_fraction must sum to 1.0 (±0.01), got {frac_sum:.4f}"
        )

    # --- Calculations ---
    # Minute ventilation
    mv_L_per_min = tidal_volume_mL * respiratory_rate_per_min / 1000.0

    # Total inhaled mass per hour (mg/h)
    # MV [L/min] * 60 [min/h] * conc [mg/m3] / 1000 [L/m3]
    total_inhaled_mg_per_h = mv_L_per_min * 60.0 * airborne_conc_mg_per_m3 / 1000.0

    # Regional deposition
    m_al_per_h = total_inhaled_mg_per_h * dep["al"]

    # Absorbed systemic dose per hour (alveolar absorption only)
    absorbed_mg_per_h = m_al_per_h * absorption_fraction_al

    # Total absorbed over exposure
    total_absorbed_mg = absorbed_mg_per_h * exposure_duration_h

    # Human Equivalent Concentration (normalised dose)
    hec = total_absorbed_mg / body_weight_kg

    # Margin of Exposure
    moe = noael_mg_per_kg / hec if hec > 0 else float("inf")

    # Risk classification
    if moe > 1000:
        risk_class = "low"
    elif moe >= 100:
        risk_class = "moderate"
    else:
        risk_class = "high"

    notes = (
        f"MV={mv_L_per_min:.2f} L/min; "
        f"ET={dep['et']:.2f}/TB={dep['tb']:.2f}/AL={dep['al']:.2f}; "
        f"NOAEL={noael_mg_per_kg} mg/kg; "
        f"MOE={moe:.1f} -> {risk_class} risk"
    )

    return InhalationDosimetryResult(
        chemical_name=chemical_name,
        airborne_conc_mg_per_m3=airborne_conc_mg_per_m3,
        exposure_duration_h=exposure_duration_h,
        minute_ventilation_L_per_min=mv_L_per_min,
        total_inhaled_mg_per_h=total_inhaled_mg_per_h,
        alveolar_deposited_mg_per_h=m_al_per_h,
        absorbed_systemic_mg_per_h=absorbed_mg_per_h,
        total_absorbed_mg=total_absorbed_mg,
        human_equivalent_conc_mg_per_kg=hec,
        moe=moe,
        risk_class=risk_class,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Exposure duration comparison utility
# ---------------------------------------------------------------------------


def compare_exposure_durations(
    chemical_name: str,
    airborne_conc_mg_per_m3: float,
    durations_h: list[float],
    tidal_volume_mL: float = 500.0,
    respiratory_rate_per_min: float = 15.0,
    regional_deposition_fraction: dict[str, float] | None = None,
    absorption_fraction_al: float = 0.8,
    body_weight_kg: float = 70.0,
    noael_mg_per_kg: float = 100.0,
) -> list[InhalationDosimetryResult]:
    """Compute dosimetry for multiple exposure durations and return results
    sorted by ``total_absorbed_mg`` in ascending order.

    Parameters
    ----------
    chemical_name, airborne_conc_mg_per_m3:
        Shared chemical / scenario parameters.
    durations_h:
        List of exposure durations (hours) to evaluate.
    (remaining parameters):
        Passed directly to :func:`calculate_inhalation_dosimetry`.

    Returns
    -------
    list[InhalationDosimetryResult]
        Results sorted by total_absorbed_mg ascending.
    """
    results = [
        calculate_inhalation_dosimetry(
            chemical_name=chemical_name,
            airborne_conc_mg_per_m3=airborne_conc_mg_per_m3,
            exposure_duration_h=d,
            tidal_volume_mL=tidal_volume_mL,
            respiratory_rate_per_min=respiratory_rate_per_min,
            regional_deposition_fraction=regional_deposition_fraction,
            absorption_fraction_al=absorption_fraction_al,
            body_weight_kg=body_weight_kg,
            noael_mg_per_kg=noael_mg_per_kg,
        )
        for d in durations_h
    ]
    return sorted(results, key=lambda r: r.total_absorbed_mg)
