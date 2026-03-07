"""Polymorphic CYP substrate dosing model.

Models dose adjustments required for CYP enzyme polymorphisms based on
CPIC-style phenotype definitions. Supports CYP2D6, CYP2C19, CYP2C9,
and CYP3A5 with UM/EM/IM/PM phenotypes.

Reference: CPIC Clinical Pharmacogenomics Implementation Consortium guidelines.
"""

from __future__ import annotations

from dataclasses import dataclass

# Valid phenotypes and their ordering for screen_phenotypes output
_PHENOTYPES = ("EM", "IM", "PM", "UM")
_VALID_ENZYMES = ("CYP2D6", "CYP2C19", "CYP2C9", "CYP3A5")

# Star allele examples by phenotype (CYP2D6 notation as default)
_STAR_ALLELES: dict[str, str] = {
    "EM": "*1/*1",
    "IM": "*1/*4",
    "PM": "*4/*4",
    "UM": "*1xN/*1",
}


@dataclass(frozen=True)
class PolymorphicCYPDosingResult:
    """Result of polymorphic CYP dosing prediction for a single phenotype."""

    drug_name: str
    cyp_enzyme: str  # "CYP2D6", "CYP2C19", "CYP2C9", "CYP3A5"
    phenotype: str  # "UM", "EM", "IM", "PM"
    star_alleles: str  # e.g., "*1/*4"
    fm_cyp: float  # fraction metabolized by this CYP
    cl_ratio_vs_em: float  # CL ratio relative to EM
    auc_ratio_vs_em: float  # AUC ratio relative to EM (= 1 / cl_ratio)
    standard_dose_mg: float
    recommended_dose_mg: float
    dose_adjustment_factor: float
    expected_cmax_mg_L: float  # relative to EM at standard dose
    safety_concern: str  # "none", "toxicity_risk", "efficacy_risk"
    tdm_required: bool
    notes: str


def predict_polymorphic_dosing(
    drug_name: str,
    standard_dose_mg: float,
    cyp_enzyme: str,
    phenotype: str,
    fm_cyp: float,
    standard_cmax_mg_L: float,
) -> PolymorphicCYPDosingResult:
    """Predict dose adjustment for a CYP polymorphism phenotype.

    Args:
        drug_name: Name of the drug.
        standard_dose_mg: Standard dose for an EM patient (mg).
        cyp_enzyme: CYP enzyme name (e.g., "CYP2D6").
        phenotype: Metabolizer phenotype — "UM", "EM", "IM", or "PM".
        fm_cyp: Fraction of drug clearance attributable to this CYP (0–1).
        standard_cmax_mg_L: Expected Cmax in an EM patient at standard dose (mg/L).

    Returns:
        PolymorphicCYPDosingResult with dose recommendation and safety flags.

    Raises:
        ValueError: If fm_cyp not in [0, 1], phenotype invalid, or standard_dose_mg <= 0.
    """
    if not (0.0 <= fm_cyp <= 1.0):
        raise ValueError(f"fm_cyp must be in [0, 1], got {fm_cyp}")
    if phenotype not in _PHENOTYPES:
        raise ValueError(f"phenotype must be one of {_PHENOTYPES}, got {phenotype!r}")
    if standard_dose_mg <= 0:
        raise ValueError(f"standard_dose_mg must be > 0, got {standard_dose_mg}")

    # CL ratio relative to EM
    if phenotype == "UM":
        cl_ratio = 1.0 + fm_cyp * 0.3
    elif phenotype == "EM":
        cl_ratio = 1.0
    elif phenotype == "IM":
        cl_ratio = 1.0 - fm_cyp * 0.5
    else:  # PM
        raw = 1.0 - fm_cyp * 0.9
        cl_ratio = max(0.05, min(1.0, raw))

    # AUC ratio (inverse of CL ratio)
    auc_ratio = 1.0 / cl_ratio

    # Dose adjustment factor: to achieve same AUC as EM → scale with CL ratio
    dose_adjustment_factor = cl_ratio  # = 1 / auc_ratio

    # Recommended dose clamped to [10%, 500%] of standard dose
    recommended_dose_mg = standard_dose_mg * dose_adjustment_factor
    recommended_dose_mg = max(
        0.1 * standard_dose_mg, min(5.0 * standard_dose_mg, recommended_dose_mg)
    )

    # Expected Cmax relative to EM (higher AUC → proportionally higher Cmax approximation)
    expected_cmax_mg_L = standard_cmax_mg_L * auc_ratio

    # Safety concern classification
    if phenotype in ("PM", "IM") and fm_cyp > 0.7:
        safety_concern = "toxicity_risk"
    elif phenotype == "UM" and fm_cyp > 0.7:
        safety_concern = "efficacy_risk"
    else:
        safety_concern = "none"

    # TDM required if safety concern present or AUC differs substantially from EM
    tdm_required = safety_concern != "none" or abs(auc_ratio - 1.0) > 0.5

    # Star alleles
    star_alleles = _STAR_ALLELES[phenotype]

    # Build notes
    notes = (
        f"{cyp_enzyme} {phenotype}: CL ratio={cl_ratio:.3f}, AUC ratio={auc_ratio:.3f}. "
        f"fm_cyp={fm_cyp:.2f}. "
        f"Dose adjustment: {dose_adjustment_factor:.2f}x standard. "
        f"Safety: {safety_concern}."
    )

    return PolymorphicCYPDosingResult(
        drug_name=drug_name,
        cyp_enzyme=cyp_enzyme,
        phenotype=phenotype,
        star_alleles=star_alleles,
        fm_cyp=fm_cyp,
        cl_ratio_vs_em=cl_ratio,
        auc_ratio_vs_em=auc_ratio,
        standard_dose_mg=standard_dose_mg,
        recommended_dose_mg=recommended_dose_mg,
        dose_adjustment_factor=dose_adjustment_factor,
        expected_cmax_mg_L=expected_cmax_mg_L,
        safety_concern=safety_concern,
        tdm_required=tdm_required,
        notes=notes,
    )


def screen_phenotypes(
    drug_name: str,
    standard_dose_mg: float,
    cyp_enzyme: str,
    fm_cyp: float,
    standard_cmax_mg_L: float,
) -> list[PolymorphicCYPDosingResult]:
    """Screen all phenotypes for a drug-CYP combination.

    Returns results for EM, IM, PM, UM in that fixed order.

    Args:
        drug_name: Name of the drug.
        standard_dose_mg: Standard dose for an EM patient (mg).
        cyp_enzyme: CYP enzyme name.
        fm_cyp: Fraction of drug clearance attributable to this CYP (0–1).
        standard_cmax_mg_L: Expected Cmax in an EM patient at standard dose (mg/L).

    Returns:
        List of 4 PolymorphicCYPDosingResult in order: EM, IM, PM, UM.
    """
    return [
        predict_polymorphic_dosing(
            drug_name=drug_name,
            standard_dose_mg=standard_dose_mg,
            cyp_enzyme=cyp_enzyme,
            phenotype=phenotype,
            fm_cyp=fm_cyp,
            standard_cmax_mg_L=standard_cmax_mg_L,
        )
        for phenotype in _PHENOTYPES
    ]
