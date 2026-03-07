"""Solubility enhancement strategy prediction for poorly soluble drugs."""

from __future__ import annotations

import math
from dataclasses import dataclass

KNOWN_STRATEGIES = {
    "amorphous_dispersion",
    "cyclodextrin",
    "salt_form",
    "nanosizing",
    "lipid_formulation",
    "cosolvent",
}


@dataclass(frozen=True)
class SolubilityEnhancementResult:
    drug_name: str
    baseline_solubility_mg_mL: float
    strategy: str
    enhanced_solubility_mg_mL: float
    enhancement_ratio: float
    dose_number_enhanced: float
    bcs_class_enhanced: str
    feasibility_score: float
    notes: str


def _calc_enhancement_factor(
    strategy: str,
    logP: float,
    pka: float | None,
) -> float:
    """Calculate auto enhancement factor for a given strategy."""
    if strategy == "amorphous_dispersion":
        factor = 10 ** (0.5 * logP)
        return min(factor, 1000.0)
    elif strategy == "cyclodextrin":
        factor = 5 * (1 + logP)
        return min(max(factor, 5.0), 200.0)
    elif strategy == "salt_form":
        if pka is not None and (pka < 5 or pka > 9):
            return 50.0
        return 10.0
    elif strategy == "nanosizing":
        factor = 2 + 3 * max(0.0, logP - 1)
        return min(factor, 20.0)
    elif strategy == "lipid_formulation":
        factor = 5 * logP
        return min(max(factor, 1.0), 50.0)
    elif strategy == "cosolvent":
        factor = 2 + logP
        return min(max(factor, 2.0), 10.0)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def _calc_feasibility_score(
    strategy: str,
    logP: float,
    pka: float | None,
    enhancement_ratio: float,
    dose_number_enhanced: float,
) -> float:
    """Calculate feasibility score 0-100."""
    score = 50.0

    # Enhancement ratio bonus
    if enhancement_ratio > 0:
        score += min(30.0, 5 * math.log10(enhancement_ratio))

    # Dose number bonus
    if dose_number_enhanced <= 1.0:
        score += 20.0
    elif dose_number_enhanced <= 4.0:
        score += 10.0

    # logP suitability per strategy
    if strategy == "amorphous_dispersion":
        if logP > 3:
            score += 10.0
        elif logP < 1:
            score -= 10.0
    elif strategy == "cyclodextrin":
        if 1 < logP < 5:
            score += 10.0
        else:
            score -= 5.0
    elif strategy == "salt_form":
        if pka is not None:
            score += 10.0
        else:
            score -= 10.0
    elif strategy == "nanosizing":
        score += 5.0
    elif strategy == "lipid_formulation":
        if logP > 2:
            score += 10.0
        else:
            score -= 10.0
    elif strategy == "cosolvent":
        if logP < 3:
            score += 5.0
        else:
            score -= 5.0

    return max(0.0, min(100.0, score))


def _bcs_class(dose_number: float, logP: float) -> str:
    """Determine BCS class from dose number and logP-based Peff proxy."""
    high_peff = 0.0 <= logP <= 5.0
    if dose_number <= 1.0:
        return "I" if high_peff else "III"
    else:
        return "II" if high_peff else "IV"


def _strategy_notes(strategy: str) -> str:
    notes_map = {
        "amorphous_dispersion": (
            "Amorphous solid dispersion (ASD) using polymer carrier; best for high logP drugs."
        ),
        "cyclodextrin": (
            "Cyclodextrin complexation; suitable for moderate logP drugs with cavity-fitting geometry."
        ),
        "salt_form": ("Salt formation; effective for ionizable drugs (acids pKa<5, bases pKa>9)."),
        "nanosizing": (
            "Particle size reduction to nanoscale; universal applicability across drug classes."
        ),
        "lipid_formulation": (
            "Lipid-based drug delivery (LBDDS); optimal for lipophilic drugs (logP>2)."
        ),
        "cosolvent": (
            "Cosolvent system (e.g., PEG, propylene glycol); general approach for moderate logP drugs."
        ),
    }
    return notes_map[strategy]


def predict_enhancement(
    drug_name: str,
    baseline_solubility_mg_mL: float,
    dose_mg: float,
    logP: float,
    pka: float | None = None,
    strategy: str = "amorphous_dispersion",
    enhancement_factor: float | None = None,
) -> SolubilityEnhancementResult:
    """Predict solubility enhancement for a given strategy."""
    if baseline_solubility_mg_mL <= 0:
        raise ValueError("baseline_solubility_mg_mL must be > 0")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if strategy not in KNOWN_STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from {sorted(KNOWN_STRATEGIES)}")

    factor = (
        enhancement_factor
        if enhancement_factor is not None
        else _calc_enhancement_factor(strategy, logP, pka)
    )

    enhanced_solubility = baseline_solubility_mg_mL * factor
    enhancement_ratio = factor
    dose_number = dose_mg / (250.0 * enhanced_solubility)
    bcs_class = _bcs_class(dose_number, logP)
    feasibility = _calc_feasibility_score(strategy, logP, pka, enhancement_ratio, dose_number)
    notes = _strategy_notes(strategy)

    return SolubilityEnhancementResult(
        drug_name=drug_name,
        baseline_solubility_mg_mL=baseline_solubility_mg_mL,
        strategy=strategy,
        enhanced_solubility_mg_mL=enhanced_solubility,
        enhancement_ratio=enhancement_ratio,
        dose_number_enhanced=dose_number,
        bcs_class_enhanced=bcs_class,
        feasibility_score=feasibility,
        notes=notes,
    )


def rank_strategies(
    drug_name: str,
    baseline_solubility_mg_mL: float,
    dose_mg: float,
    logP: float,
    pka: float | None = None,
) -> list:
    """Evaluate all strategies and return sorted list by feasibility_score descending."""
    results = []
    for strategy in KNOWN_STRATEGIES:
        result = predict_enhancement(
            drug_name=drug_name,
            baseline_solubility_mg_mL=baseline_solubility_mg_mL,
            dose_mg=dose_mg,
            logP=logP,
            pka=pka,
            strategy=strategy,
        )
        results.append(result)
    results.sort(key=lambda r: r.feasibility_score, reverse=True)
    return results
