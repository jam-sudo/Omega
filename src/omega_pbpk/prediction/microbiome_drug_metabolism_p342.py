"""Phase 342 — Microbiome Drug Metabolism Model.

Models gut microbiome-mediated drug metabolism affecting oral bioavailability.
The microbiome can reduce systemic drug exposure by metabolizing drug
in the GI lumen prior to absorption.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_VALID_DRUG_CLASSES = {"antibiotic", "NSAID", "cardiovascular", "other"}


def _validate_inputs(
    k_microbial_metabolism_per_h: float,
    f_abs_without_microbiome: float,
    transit_time_h: float,
    drug_class: str,
) -> None:
    if not (0.0 <= k_microbial_metabolism_per_h <= 5.0):
        raise ValueError(
            f"k_microbial_metabolism_per_h must be in [0, 5], got {k_microbial_metabolism_per_h}"
        )
    if not (0.0 < f_abs_without_microbiome <= 1.0):
        raise ValueError(
            f"f_abs_without_microbiome must be in (0, 1], got {f_abs_without_microbiome}"
        )
    if transit_time_h <= 0.0:
        raise ValueError(f"transit_time_h must be > 0, got {transit_time_h}")
    if drug_class not in _VALID_DRUG_CLASSES:
        raise ValueError(f"drug_class must be one of {_VALID_DRUG_CLASSES}, got '{drug_class}'")


@dataclass(frozen=True)
class MicrobiomeDrugMetabolismResult:
    """Result of microbiome drug metabolism analysis."""

    drug_name: str
    dose_mg: float
    microbiome_density_CFU_mL: float
    effective_microbial_rate_per_h: float
    fraction_metabolized_by_microbiome: float
    f_abs_without_microbiome: float
    f_abs_with_microbiome: float
    bioavailability_reduction_pct: float
    metabolite_fraction: float
    metabolite_auc_fraction: float
    dysbiosis_impact: str
    clinical_implication: str
    drug_class: str
    notes: str


def simulate_microbiome_drug_metabolism(
    drug_name: str,
    dose_mg: float,
    k_microbial_metabolism_per_h: float,
    f_abs_without_microbiome: float,
    microbiome_density: float = 1e11,
    transit_time_h: float = 6.0,
    drug_class: str = "other",
    gut_dysbiosis: bool = False,
) -> MicrobiomeDrugMetabolismResult:
    """Simulate microbiome-mediated drug metabolism in the GI tract.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Administered dose (mg).
    k_microbial_metabolism_per_h:
        Microbial metabolic rate constant per hour (0-5).
    f_abs_without_microbiome:
        Baseline absorption fraction without microbiome influence (0, 1].
    microbiome_density:
        Gut microbiome density in CFU/mL (default 1e11).
    transit_time_h:
        GI transit time in hours (default 6.0).
    drug_class:
        Drug class: 'antibiotic', 'NSAID', 'cardiovascular', or 'other'.
    gut_dysbiosis:
        Whether patient has gut dysbiosis (reduces microbiome activity).

    Returns
    -------
    MicrobiomeDrugMetabolismResult
        Comprehensive microbiome drug metabolism result.
    """
    _validate_inputs(
        k_microbial_metabolism_per_h,
        f_abs_without_microbiome,
        transit_time_h,
        drug_class,
    )

    # Scale microbial rate by actual microbiome density relative to reference
    effective_k = k_microbial_metabolism_per_h * (microbiome_density / 1e11)

    # Dysbiosis reduces microbiome activity
    if gut_dysbiosis:
        effective_k *= 0.3

    # Antibiotic depletes its own microbiome (self-depletion)
    if drug_class == "antibiotic":
        effective_k *= 0.5

    # Fraction metabolized by microbiome (first-order, clamped to 0.9 max)
    raw_frac_metabolized = 1.0 - math.exp(-effective_k * transit_time_h)
    fraction_metabolized_by_microbiome = max(0.0, min(0.9, raw_frac_metabolized))

    # Effective absorption fraction
    f_abs_effective = f_abs_without_microbiome * (1.0 - fraction_metabolized_by_microbiome)

    # Bioavailability reduction
    if f_abs_without_microbiome > 0.0:
        bioavailability_reduction_pct = (
            (f_abs_without_microbiome - f_abs_effective) / f_abs_without_microbiome
        ) * 100.0
    else:
        bioavailability_reduction_pct = 0.0

    # Metabolite production
    metabolite_fraction = fraction_metabolized_by_microbiome * 0.7

    # Systemic metabolite exposure
    metabolite_auc_fraction = metabolite_fraction * f_abs_effective

    # Dysbiosis impact label
    if not gut_dysbiosis:
        dysbiosis_impact = "none"
    else:
        dysbiosis_impact = "reduced_metabolism"

    # Clinical implication based on bioavailability reduction
    if bioavailability_reduction_pct < 10.0:
        clinical_implication = "negligible"
    elif bioavailability_reduction_pct < 30.0:
        clinical_implication = "minor"
    else:
        clinical_implication = "significant"

    # Build notes
    notes_parts = [
        f"Effective microbial rate: {effective_k:.4f} /h.",
        f"Fraction metabolized by microbiome: {fraction_metabolized_by_microbiome:.3f}.",
        f"Bioavailability reduction: {bioavailability_reduction_pct:.1f}%.",
    ]
    if gut_dysbiosis:
        notes_parts.append("Gut dysbiosis reduces microbial metabolism (30% of normal).")
    if drug_class == "antibiotic":
        notes_parts.append("Antibiotic self-depletes microbiome (50% reduction in effective rate).")
    if clinical_implication == "significant":
        notes_parts.append("Significant bioavailability reduction — consider dose adjustment.")
    elif clinical_implication == "minor":
        notes_parts.append("Minor bioavailability reduction — monitor drug levels.")

    notes = " ".join(notes_parts)

    return MicrobiomeDrugMetabolismResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        microbiome_density_CFU_mL=microbiome_density,
        effective_microbial_rate_per_h=effective_k,
        fraction_metabolized_by_microbiome=fraction_metabolized_by_microbiome,
        f_abs_without_microbiome=f_abs_without_microbiome,
        f_abs_with_microbiome=f_abs_effective,
        bioavailability_reduction_pct=bioavailability_reduction_pct,
        metabolite_fraction=metabolite_fraction,
        metabolite_auc_fraction=metabolite_auc_fraction,
        dysbiosis_impact=dysbiosis_impact,
        clinical_implication=clinical_implication,
        drug_class=drug_class,
        notes=notes,
    )
