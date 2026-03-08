"""Oral absorption enhancement prediction model.

Models the effect of absorption enhancers on intestinal permeability and
fraction absorbed for drugs across all BCS classes.

Enhancers modeled
-----------------
- SDS (sodium dodecyl sulfate): tight junction disruption, solubilization
- Tween80: non-ionic surfactant, membrane fluidization
- EDTA: calcium chelator, reversible tight junction opening
- chitosan: mucoadhesive polymer, tight junction modulation
- piperine: P-gp inhibitor, enhances low-permeability drugs
- caprylic_acid: SCFA, transcellular and paracellular enhancement

References
----------
- Sun D et al, Pharm Res. 2004;21(6):979-88
- Maher S et al, Adv Drug Deliv Rev. 2016;106(Pt B):277-319
- Benet LZ, Pharmacol Rev. 2011;63(3):636-63
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class AbsorptionEnhancementResult:
    """Result of an absorption enhancement prediction.

    Attributes
    ----------
    drug_name : Name of the drug.
    enhancer : Name of the absorption enhancer.
    bcs_class : BCS classification (1-4).
    logP : Lipophilicity (log partition coefficient).
    peff_baseline_cm_s : Intestinal permeability without enhancer (cm/s).
    peff_enhanced_cm_s : Intestinal permeability with enhancer (cm/s).
    enhancement_ratio : peff_enhanced / peff_baseline.
    fa_baseline : Fraction absorbed without enhancer (0-1).
    fa_enhanced : Fraction absorbed with enhancer (0-1).
    delta_fa_pct : Percentage change in fa relative to baseline.
    bioavailability_improvement_pct : Proportional improvement in F (%).
    enhancer_concentration_pct : Enhancer concentration used (%).
    toxicity_concern : Safety level ('low', 'moderate', 'high').
    notes : Summary notes string.
    """

    drug_name: str
    enhancer: str
    bcs_class: int
    logP: float
    peff_baseline_cm_s: float
    peff_enhanced_cm_s: float
    enhancement_ratio: float
    fa_baseline: float
    fa_enhanced: float
    delta_fa_pct: float
    bioavailability_improvement_pct: float
    enhancer_concentration_pct: float
    toxicity_concern: str
    notes: str


def _compute_peff_baseline(logP: float, mw_Da: float) -> float:
    """Compute baseline intestinal permeability (cm/s).

    Uses an empirical model based on logP and molecular weight.
    Result is clamped to [1e-7, 1e-4] cm/s.

    Parameters
    ----------
    logP : Lipophilicity.
    mw_Da : Molecular weight (Da).

    Returns
    -------
    float
        Baseline Peff in cm/s.
    """
    peff = 10.0 ** (-4.0 + 0.5 * logP - 0.003 * mw_Da)
    return max(1e-7, min(1e-4, peff))


def _enhancer_factor(enhancer: str, enhancer_conc_pct: float, bcs_class: int) -> tuple:
    """Return (enhancement_factor, toxicity_concern) for a given enhancer.

    Parameters
    ----------
    enhancer : Enhancer name (case-insensitive key).
    enhancer_conc_pct : Enhancer concentration (%).
    bcs_class : BCS class (1-4).

    Returns
    -------
    tuple of (float, str)
        Enhancement factor (clamped 1.0-5.0) and toxicity level string.
    """
    key = enhancer.lower().strip()

    if key == "sds":
        factor = 2.0 + enhancer_conc_pct * 1.0
        toxicity = "high"
    elif key == "tween80":
        factor = 1.5 + enhancer_conc_pct * 0.5
        toxicity = "moderate"
    elif key == "edta":
        factor = 1.8 + enhancer_conc_pct * 0.8
        toxicity = "moderate"
    elif key == "chitosan":
        factor = 1.3 + enhancer_conc_pct * 0.3
        toxicity = "low"
    elif key == "piperine":
        factor = 1.2 if bcs_class in (1, 3) else 2.0
        toxicity = "low"
    elif key == "caprylic_acid":
        factor = 1.6 + enhancer_conc_pct * 0.4
        toxicity = "low"
    else:
        factor = 1.2
        toxicity = "low"

    factor = max(1.0, min(5.0, factor))
    return factor, toxicity


def _compute_fa(peff_cm_s: float) -> float:
    """Compute fraction absorbed using exponential absorption model.

    Parameters
    ----------
    peff_cm_s : Effective intestinal permeability (cm/s).

    Returns
    -------
    float
        Fraction absorbed (0-1).
    """
    return min(1.0, 1.0 - math.exp(-peff_cm_s * 1e4))


def predict_absorption_enhancement(
    drug_name: str,
    bcs_class: int,
    logP: float,
    mw_Da: float,
    enhancer: str,
    enhancer_conc_pct: float = 1.0,
    f_gut: float = 0.9,
    extraction_ratio: float = 0.2,
) -> AbsorptionEnhancementResult:
    """Predict the effect of an absorption enhancer on oral drug absorption.

    Parameters
    ----------
    drug_name : Name of the drug.
    bcs_class : BCS classification (1=high sol/high perm, 2=low sol/high perm,
                3=high sol/low perm, 4=low sol/low perm).
    logP : Lipophilicity (log octanol-water partition coefficient).
    mw_Da : Molecular weight (Da).
    enhancer : Name of the absorption enhancer.
    enhancer_conc_pct : Enhancer concentration (%). Default 1.0.
    f_gut : Gut wall extraction survival fraction (0-1). Default 0.9.
    extraction_ratio : Hepatic first-pass extraction ratio (0-1). Default 0.2.

    Returns
    -------
    AbsorptionEnhancementResult
        Prediction including permeability, fa, and improvement metrics.

    Raises
    ------
    ValueError
        If any input parameter is invalid.
    """
    # --- Validation ---
    if bcs_class not in (1, 2, 3, 4):
        raise ValueError("bcs_class must be 1, 2, 3, or 4")
    if mw_Da <= 0:
        raise ValueError("mw_Da must be positive")
    if enhancer_conc_pct <= 0:
        raise ValueError("enhancer_conc_pct must be positive")
    if not (0 < f_gut <= 1):
        raise ValueError("f_gut must be in (0, 1]")

    # --- Baseline permeability ---
    peff_baseline = _compute_peff_baseline(logP, mw_Da)

    # --- Enhancement factor ---
    factor, toxicity = _enhancer_factor(enhancer, enhancer_conc_pct, bcs_class)

    # --- Enhanced permeability ---
    peff_enhanced = min(1e-4, peff_baseline * factor)

    enhancement_ratio = peff_enhanced / peff_baseline

    # --- Fraction absorbed ---
    fa_baseline = _compute_fa(peff_baseline)
    fa_enhanced = _compute_fa(peff_enhanced)

    # --- Delta fa (%) ---
    delta_fa_pct = (fa_enhanced - fa_baseline) / max(0.001, fa_baseline) * 100.0

    # --- Bioavailability improvement ---
    # Since f_gut and (1-ER) are shared constants, proportional improvement = delta_fa_pct
    bioavailability_improvement_pct = delta_fa_pct

    notes = (
        f"BCS class {bcs_class} drug '{drug_name}' with {enhancer} "
        f"at {enhancer_conc_pct:.1f}%. "
        f"Peff: {peff_baseline:.2e} -> {peff_enhanced:.2e} cm/s "
        f"(x{enhancement_ratio:.2f}). "
        f"fa: {fa_baseline:.3f} -> {fa_enhanced:.3f}. "
        f"Toxicity concern: {toxicity}."
    )

    return AbsorptionEnhancementResult(
        drug_name=drug_name,
        enhancer=enhancer,
        bcs_class=bcs_class,
        logP=logP,
        peff_baseline_cm_s=peff_baseline,
        peff_enhanced_cm_s=peff_enhanced,
        enhancement_ratio=enhancement_ratio,
        fa_baseline=fa_baseline,
        fa_enhanced=fa_enhanced,
        delta_fa_pct=delta_fa_pct,
        bioavailability_improvement_pct=bioavailability_improvement_pct,
        enhancer_concentration_pct=enhancer_conc_pct,
        toxicity_concern=toxicity,
        notes=notes,
    )


def screen_enhancers(
    drug_name: str,
    bcs_class: int,
    logP: float,
    mw_Da: float,
    enhancer_list: list,
) -> list:
    """Screen multiple enhancers and rank by enhanced fraction absorbed.

    Parameters
    ----------
    drug_name : Name of the drug.
    bcs_class : BCS class (1-4).
    logP : Lipophilicity.
    mw_Da : Molecular weight (Da).
    enhancer_list : List of enhancer name strings to evaluate.

    Returns
    -------
    list of AbsorptionEnhancementResult
        Results sorted by fa_enhanced descending.
    """
    results = [
        predict_absorption_enhancement(drug_name, bcs_class, logP, mw_Da, enh)
        for enh in enhancer_list
    ]
    results.sort(key=lambda r: r.fa_enhanced, reverse=True)
    return results
