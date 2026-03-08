"""Drug metabolism by gut microbiome (Phase 559).

Models colonic bacterial metabolism relevant for prodrugs and compounds
with azo/nitro reductions, glucuronide deconjugation, and general inactivation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

DRUG_CLASSES = ("prodrug_azo", "prodrug_nitro", "glucuronide_conjugate", "other")


@dataclass(frozen=True)
class GutMicrobiomeMetabolismResult:
    """Result of gut microbiome metabolism prediction."""

    drug_name: str
    drug_class: str  # "prodrug_azo", "prodrug_nitro", "glucuronide_conjugate", "other"
    microbial_cl_mL_per_min: float
    fraction_metabolized_colon: float
    fa_effective: float
    active_metabolite_fraction: float
    gut_transit_time_h: float
    microbiome_effect: str  # "activation", "inactivation", "no_effect"
    notes: str


def predict_microbiome_metabolism(
    drug_name: str,
    drug_class: str,
    dose_mg: float,
    intrinsic_clearance_microbiome_mL_per_min: float = 1.0,
    gut_transit_time_h: float = 6.0,
    colonic_volume_mL: float = 500.0,
) -> GutMicrobiomeMetabolismResult:
    """Predict drug metabolism by gut microbiome in the colon.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    drug_class : str
        One of "prodrug_azo", "prodrug_nitro", "glucuronide_conjugate", "other".
    dose_mg : float
        Dose in mg (must be > 0).
    intrinsic_clearance_microbiome_mL_per_min : float
        Intrinsic clearance by microbiome in mL/min (must be > 0).
    gut_transit_time_h : float
        Gut transit time in hours (must be > 0).
    colonic_volume_mL : float
        Colonic volume in mL (must be > 0).

    Returns
    -------
    GutMicrobiomeMetabolismResult
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if intrinsic_clearance_microbiome_mL_per_min <= 0:
        raise ValueError("intrinsic_clearance_microbiome_mL_per_min must be > 0")
    if gut_transit_time_h <= 0:
        raise ValueError("gut_transit_time_h must be > 0")
    if colonic_volume_mL <= 0:
        raise ValueError("colonic_volume_mL must be > 0")
    if drug_class not in DRUG_CLASSES:
        raise ValueError(f"drug_class must be one of {DRUG_CLASSES}, got '{drug_class}'")

    # Microbial first-order rate constant (per hour)
    k_microbial = intrinsic_clearance_microbiome_mL_per_min * 60.0 / colonic_volume_mL

    # Fraction metabolized during transit
    fraction_metabolized = 1.0 - math.exp(-k_microbial * gut_transit_time_h)

    # Determine effect and fractions based on drug class
    if drug_class in ("prodrug_azo", "prodrug_nitro"):
        microbiome_effect = "activation"
        active_metabolite_fraction = fraction_metabolized
        fa_effective = active_metabolite_fraction
        notes = (
            f"Prodrug ({drug_class}): microbiome activates drug via reduction. "
            f"{fraction_metabolized:.1%} converted to active form in colon."
        )
    elif drug_class == "glucuronide_conjugate":
        microbiome_effect = "activation"
        active_metabolite_fraction = fraction_metabolized
        fa_effective = active_metabolite_fraction
        notes = (
            "Glucuronide conjugate: microbiome deconjugation enables "
            f"enterohepatic recycling. {fraction_metabolized:.1%} deconjugated."
        )
    else:  # "other"
        microbiome_effect = "inactivation"
        active_metabolite_fraction = 0.0
        fa_effective = 1.0 - fraction_metabolized
        notes = (
            f"Non-prodrug: microbiome inactivates {fraction_metabolized:.1%} "
            "of drug in colon, reducing effective absorption."
        )

    return GutMicrobiomeMetabolismResult(
        drug_name=drug_name,
        drug_class=drug_class,
        microbial_cl_mL_per_min=intrinsic_clearance_microbiome_mL_per_min,
        fraction_metabolized_colon=fraction_metabolized,
        fa_effective=fa_effective,
        active_metabolite_fraction=active_metabolite_fraction,
        gut_transit_time_h=gut_transit_time_h,
        microbiome_effect=microbiome_effect,
        notes=notes,
    )


def compare_drug_classes(
    drug_name: str,
    dose_mg: float,
    intrinsic_clearance_microbiome_mL_per_min: float = 1.0,
    gut_transit_time_h: float = 6.0,
    colonic_volume_mL: float = 500.0,
) -> list:
    """Run prediction for all 4 drug classes and return sorted by fa_effective descending."""
    results = []
    for dc in DRUG_CLASSES:
        r = predict_microbiome_metabolism(
            drug_name=drug_name,
            drug_class=dc,
            dose_mg=dose_mg,
            intrinsic_clearance_microbiome_mL_per_min=intrinsic_clearance_microbiome_mL_per_min,
            gut_transit_time_h=gut_transit_time_h,
            colonic_volume_mL=colonic_volume_mL,
        )
        results.append(r)
    results.sort(key=lambda x: x.fa_effective, reverse=True)
    return results
