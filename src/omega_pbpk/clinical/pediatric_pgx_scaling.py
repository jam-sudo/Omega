"""
Phase 378 — Pediatric Pharmacogenomics Scaling

Scale pharmacogenomics-based dose adjustments for pediatric patients accounting
for age-dependent enzyme maturation and genetic polymorphisms.
"""

from dataclasses import dataclass

# CYP enzyme maturation parameters (Hill equation)
_CYP_MATURATION: dict[str, dict[str, float]] = {
    "CYP3A4": {"t50_years": 0.5, "hill": 2.0},
    "CYP2D6": {"t50_years": 1.0, "hill": 3.0},
    "CYP2C9": {"t50_years": 1.5, "hill": 2.0},
    "CYP2C19": {"t50_years": 1.0, "hill": 2.0},
    "CYP1A2": {"t50_years": 2.0, "hill": 3.0},
    "CYP2E1": {"t50_years": 0.25, "hill": 2.0},
}

# Default maturation parameters for unknown CYP
_DEFAULT_MATURATION: dict[str, float] = {"t50_years": 1.0, "hill": 2.0}

# PGx intrinsic dose factors per phenotype
_PGX_FACTORS: dict[str, float] = {
    "PM": 0.25,
    "IM": 0.6,
    "NM": 1.0,
    "UM": 1.75,
}

_VALID_PHENOTYPES = {"PM", "IM", "NM", "UM"}

# Reference adult weight (kg)
_ADULT_WEIGHT_KG = 70.0


@dataclass(frozen=True)
class PediatricPGxResult:
    """Result of pediatric pharmacogenomics dose scaling."""

    drug_name: str
    cyp_enzyme: str
    metabolizer_phenotype: str
    age_years: float
    weight_kg: float
    adult_dose_mg: float
    pediatric_dose_mg: float
    pgx_adjusted_dose_mg: float
    allometric_factor: float
    enzyme_maturation_factor: float
    pgx_dose_factor: float
    combined_dose_factor: float
    dose_cap_applied: bool
    monitoring_required: bool
    clinical_guidance: str
    notes: str


def _validate_inputs(
    drug_name: str,
    cyp_enzyme: str,
    metabolizer_phenotype: str,
    age_years: float,
    weight_kg: float,
    adult_dose_mg: float,
    fraction_metabolized_by_cyp: float,
) -> None:
    """Validate all input parameters."""
    if age_years <= 0 or age_years > 18:
        raise ValueError(f"age_years must be in (0, 18], got {age_years}")
    if weight_kg <= 0:
        raise ValueError(f"weight_kg must be > 0, got {weight_kg}")
    if adult_dose_mg <= 0:
        raise ValueError(f"adult_dose_mg must be > 0, got {adult_dose_mg}")
    if metabolizer_phenotype not in _VALID_PHENOTYPES:
        raise ValueError(
            f"metabolizer_phenotype must be one of {_VALID_PHENOTYPES}, "
            f"got '{metabolizer_phenotype}'"
        )
    valid_cyps = set(_CYP_MATURATION.keys()) | {"OTHER"}
    if cyp_enzyme not in valid_cyps:
        raise ValueError(f"cyp_enzyme must be one of {valid_cyps}, got '{cyp_enzyme}'")
    if not (0.0 <= fraction_metabolized_by_cyp <= 1.0):
        raise ValueError(
            f"fraction_metabolized_by_cyp must be in [0, 1], got {fraction_metabolized_by_cyp}"
        )


def _compute_maturation_factor(cyp_enzyme: str, age_years: float) -> float:
    """Compute enzyme maturation factor using Hill equation."""
    params = _CYP_MATURATION.get(cyp_enzyme, _DEFAULT_MATURATION)
    t50 = params["t50_years"]
    hill = params["hill"]

    age_h = age_years**hill
    t50_h = t50**hill
    maturation = age_h / (age_h + t50_h)

    # Clamp to [0.05, 1.0]
    if maturation < 0.05:
        maturation = 0.05
    if maturation > 1.0:
        maturation = 1.0
    return maturation


def _build_clinical_guidance(
    metabolizer_phenotype: str,
    age_years: float,
) -> str:
    """Build clinical guidance string."""
    base_guidance = {
        "PM": "Significant reduction needed; risk of toxicity",
        "IM": "Standard dosing with allometric/maturation adjustments",
        "NM": "Standard dosing with allometric/maturation adjustments",
        "UM": "Higher dose may be needed; monitor efficacy",
    }
    guidance = base_guidance[metabolizer_phenotype]
    if age_years < 1:
        guidance += " Neonate: enhanced monitoring required."
    return guidance


def calculate_pediatric_pgx_dose(
    drug_name: str,
    cyp_enzyme: str,
    metabolizer_phenotype: str,
    age_years: float,
    weight_kg: float,
    adult_dose_mg: float,
    fraction_metabolized_by_cyp: float = 0.8,
) -> PediatricPGxResult:
    """
    Calculate age- and genotype-adjusted pediatric dose.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    cyp_enzyme : str
        CYP enzyme responsible for metabolism. One of:
        CYP3A4, CYP2D6, CYP2C9, CYP2C19, CYP1A2, CYP2E1, OTHER.
    metabolizer_phenotype : str
        Genetic metabolizer status: "PM", "IM", "NM", or "UM".
    age_years : float
        Patient age in years. Range (0, 18].
    weight_kg : float
        Patient weight in kg. Must be > 0.
    adult_dose_mg : float
        Standard adult dose in mg. Must be > 0.
    fraction_metabolized_by_cyp : float
        Fraction of drug metabolized by the specified CYP. Range [0, 1].
        Default 0.8.

    Returns
    -------
    PediatricPGxResult
        Frozen dataclass with all scaling factors and adjusted doses.
    """
    _validate_inputs(
        drug_name,
        cyp_enzyme,
        metabolizer_phenotype,
        age_years,
        weight_kg,
        adult_dose_mg,
        fraction_metabolized_by_cyp,
    )

    # Allometric weight scaling
    allometric_factor = (weight_kg / _ADULT_WEIGHT_KG) ** 0.75

    # Enzyme maturation
    enzyme_maturation_factor = _compute_maturation_factor(cyp_enzyme, age_years)

    # PGx dose factor (accounting for fraction metabolized by this CYP)
    intrinsic_pgx = _PGX_FACTORS[metabolizer_phenotype]
    pgx_dose_factor = 1.0 - fraction_metabolized_by_cyp * (1.0 - intrinsic_pgx)

    # Pediatric dose before PGx adjustment
    pediatric_dose_mg = adult_dose_mg * allometric_factor * enzyme_maturation_factor

    # Combined dose factor
    combined_dose_factor = allometric_factor * enzyme_maturation_factor * pgx_dose_factor

    # PGx-adjusted dose
    pgx_adjusted_dose_raw = adult_dose_mg * combined_dose_factor

    # Apply dose cap at adult dose
    dose_cap_applied = pgx_adjusted_dose_raw > adult_dose_mg
    pgx_adjusted_dose_mg = min(pgx_adjusted_dose_raw, adult_dose_mg)

    # Monitoring requirement
    monitoring_required = (
        metabolizer_phenotype in {"PM", "UM"}
        or age_years < 2
        or pgx_dose_factor < 0.5
        or pgx_dose_factor > 1.5
    )

    clinical_guidance = _build_clinical_guidance(metabolizer_phenotype, age_years)

    notes_parts = [
        f"allometric_factor={allometric_factor:.3f}",
        f"maturation_factor={enzyme_maturation_factor:.3f}",
        f"pgx_dose_factor={pgx_dose_factor:.3f}",
        f"combined_dose_factor={combined_dose_factor:.3f}",
    ]
    if dose_cap_applied:
        notes_parts.append("dose capped at adult dose")
    notes = "; ".join(notes_parts)

    return PediatricPGxResult(
        drug_name=drug_name,
        cyp_enzyme=cyp_enzyme,
        metabolizer_phenotype=metabolizer_phenotype,
        age_years=age_years,
        weight_kg=weight_kg,
        adult_dose_mg=adult_dose_mg,
        pediatric_dose_mg=pediatric_dose_mg,
        pgx_adjusted_dose_mg=pgx_adjusted_dose_mg,
        allometric_factor=allometric_factor,
        enzyme_maturation_factor=enzyme_maturation_factor,
        pgx_dose_factor=pgx_dose_factor,
        combined_dose_factor=combined_dose_factor,
        dose_cap_applied=dose_cap_applied,
        monitoring_required=monitoring_required,
        clinical_guidance=clinical_guidance,
        notes=notes,
    )


def compare_phenotypes(
    drug_name: str,
    cyp_enzyme: str,
    age_years: float,
    weight_kg: float,
    adult_dose_mg: float,
    fraction_metabolized_by_cyp: float = 0.8,
) -> list[PediatricPGxResult]:
    """
    Compare all four metabolizer phenotypes for a given drug and patient.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    cyp_enzyme : str
        CYP enzyme responsible for metabolism.
    age_years : float
        Patient age in years.
    weight_kg : float
        Patient weight in kg.
    adult_dose_mg : float
        Standard adult dose in mg.
    fraction_metabolized_by_cyp : float
        Fraction metabolized by the CYP. Default 0.8.

    Returns
    -------
    list of PediatricPGxResult
        Results for all 4 phenotypes (PM, IM, NM, UM), sorted by
        pgx_adjusted_dose_mg ascending.
    """
    results = []
    for phenotype in ("PM", "IM", "NM", "UM"):
        result = calculate_pediatric_pgx_dose(
            drug_name=drug_name,
            cyp_enzyme=cyp_enzyme,
            metabolizer_phenotype=phenotype,
            age_years=age_years,
            weight_kg=weight_kg,
            adult_dose_mg=adult_dose_mg,
            fraction_metabolized_by_cyp=fraction_metabolized_by_cyp,
        )
        results.append(result)

    results.sort(key=lambda r: r.pgx_adjusted_dose_mg)
    return results
