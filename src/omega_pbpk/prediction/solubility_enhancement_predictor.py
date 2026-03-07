"""Solubility enhancement strategy predictor for poorly soluble drugs.

Predicts and compares salt formation, cocrystal, amorphous dispersion,
cyclodextrin complexation, nanosizing, and parent drug solubility.
Pure Python stdlib only (math, dataclasses).
"""

from __future__ import annotations

from dataclasses import dataclass

_VALID_STRATEGIES = {"parent", "salt", "cocrystal", "amorphous", "cyclodextrin", "nano"}


@dataclass(frozen=True)
class SolubilityEnhancementResult:
    drug_name: str
    strategy: str
    intrinsic_solubility_mg_mL: float
    enhanced_solubility_mg_mL: float
    enhancement_fold: float
    predicted_bioavailability_pct: float
    physical_stability_score: float
    manufacturing_complexity: str
    notes: str


def _compute_enhancement_fold(
    strategy: str,
    logp: float,
    pka: float,
    mw_da: float,
) -> float:
    """Calculate solubility enhancement fold for a given strategy."""
    if strategy == "parent":
        return 1.0

    if strategy == "salt":
        if 2.0 <= pka <= 12.0:
            fold = 10.0 ** (0.5 * abs(pka - 7.0))
            fold = max(1.0, fold)
        else:
            fold = 1.5
        return min(fold, 100.0)

    if strategy == "cocrystal":
        fold = 2.0 + logp * 0.5
        fold = max(1.5, fold)
        fold = max(1.0, fold)
        return min(fold, 50.0)

    if strategy == "amorphous":
        fold = 10.0 ** (logp * 0.2)
        fold = max(2.0, fold)
        fold = max(1.0, fold)
        return min(fold, 1000.0)

    if strategy == "cyclodextrin":
        if logp > 0.0:
            fold = 1.0 + 2.0 * logp
            fold = max(1.0, fold)
        else:
            fold = 1.5
        return min(fold, 100.0)

    if strategy == "nano":
        fold = mw_da / 200.0
        fold = max(2.0, fold)
        fold = max(1.0, fold)
        return min(fold, 50.0)

    raise ValueError(f"Unknown strategy: {strategy}")


def _compute_bioavailability(enhanced_solubility: float) -> float:
    """BCS-like bioavailability estimate (1 mg/mL reference for BCS class II)."""
    return min(100.0, 20.0 + 80.0 * min(1.0, enhanced_solubility / 1.0))


_STABILITY_SCORES: dict[str, float] = {
    "parent": 90.0,
    "salt": 75.0,
    "cocrystal": 70.0,
    "amorphous": 40.0,
    "cyclodextrin": 80.0,
    "nano": 55.0,
}

_MFG_COMPLEXITY: dict[str, str] = {
    "parent": "low",
    "salt": "low",
    "cocrystal": "medium",
    "amorphous": "high",
    "cyclodextrin": "medium",
    "nano": "high",
}

_NOTES: dict[str, str] = {
    "parent": (
        "Unmodified parent drug; baseline reference. Suitable for BCS I/III drugs "
        "with adequate intrinsic solubility."
    ),
    "salt": (
        "Salt formation with counterion; effective for ionizable acids (pKa<7) and "
        "bases (pKa>7). Enhancement driven by pH-solubility relationship."
    ),
    "cocrystal": (
        "Pharmaceutical cocrystal with coformer; solubility gain depends on logP. "
        "Physical stability better than amorphous but requires coformer screening."
    ),
    "amorphous": (
        "Amorphous solid dispersion in polymer matrix (e.g., HPMC-AS, PVP-VA); "
        "highest enhancement for lipophilic drugs but limited physical stability."
    ),
    "cyclodextrin": (
        "Cyclodextrin inclusion complex (HP-beta-CD, gamma-CD); best for lipophilic "
        "molecules (logP>0) that fit the cyclodextrin cavity."
    ),
    "nano": (
        "Nanosizing by wet milling or precipitation; enhancement scales with MW. "
        "Improves dissolution rate; suitable across drug classes."
    ),
}


def predict_solubility_enhancement(
    drug_name: str,
    intrinsic_solubility_mg_mL: float,
    logp: float,
    pka: float,
    mw_da: float,
    strategy: str,
) -> SolubilityEnhancementResult:
    """Predict solubility enhancement for a given formulation strategy.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    intrinsic_solubility_mg_mL:
        Aqueous intrinsic (thermodynamic) solubility of the neutral form (mg/mL).
    logp:
        Octanol-water partition coefficient.
    pka:
        Most relevant pKa value.
    mw_da:
        Molecular weight in Daltons.
    strategy:
        One of "parent", "salt", "cocrystal", "amorphous", "cyclodextrin", "nano".

    Returns
    -------
    SolubilityEnhancementResult
    """
    if intrinsic_solubility_mg_mL <= 0.0:
        raise ValueError("intrinsic_solubility_mg_mL must be > 0")
    if not (-5.0 <= logp <= 10.0):
        raise ValueError("logp must be in [-5, 10]")
    if not (-5.0 <= pka <= 14.0):
        raise ValueError("pka must be in [-5, 14]")
    if mw_da <= 0.0:
        raise ValueError("mw_da must be > 0")
    if strategy not in _VALID_STRATEGIES:
        raise ValueError(f"strategy must be one of {sorted(_VALID_STRATEGIES)}, got '{strategy}'")

    fold = _compute_enhancement_fold(strategy, logp, pka, mw_da)
    enhanced_sol = intrinsic_solubility_mg_mL * fold
    ba_pct = _compute_bioavailability(enhanced_sol)

    return SolubilityEnhancementResult(
        drug_name=drug_name,
        strategy=strategy,
        intrinsic_solubility_mg_mL=intrinsic_solubility_mg_mL,
        enhanced_solubility_mg_mL=enhanced_sol,
        enhancement_fold=fold,
        predicted_bioavailability_pct=ba_pct,
        physical_stability_score=_STABILITY_SCORES[strategy],
        manufacturing_complexity=_MFG_COMPLEXITY[strategy],
        notes=_NOTES[strategy],
    )


def compare_strategies(
    drug_name: str,
    intrinsic_solubility_mg_mL: float,
    logp: float,
    pka: float,
    mw_da: float,
) -> list:
    """Run all 6 strategies and return list sorted by enhancement_fold descending.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    intrinsic_solubility_mg_mL:
        Aqueous intrinsic solubility of the neutral form (mg/mL).
    logp:
        Octanol-water partition coefficient.
    pka:
        Most relevant pKa value.
    mw_da:
        Molecular weight in Daltons.

    Returns
    -------
    list of SolubilityEnhancementResult sorted by enhancement_fold descending.
    """
    results = []
    for strat in _VALID_STRATEGIES:
        result = predict_solubility_enhancement(
            drug_name=drug_name,
            intrinsic_solubility_mg_mL=intrinsic_solubility_mg_mL,
            logp=logp,
            pka=pka,
            mw_da=mw_da,
            strategy=strat,
        )
        results.append(result)
    results.sort(key=lambda r: r.enhancement_fold, reverse=True)
    return results


__all__ = [
    "SolubilityEnhancementResult",
    "predict_solubility_enhancement",
    "compare_strategies",
]
