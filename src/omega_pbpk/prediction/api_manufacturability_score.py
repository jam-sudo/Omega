"""
Phase 202 — API Manufacturability Score
=========================================
Predict Active Pharmaceutical Ingredient (API) manufacturability
based on physical properties.

Pure Python (stdlib: math, dataclasses) — no numpy, no scipy.
"""

from __future__ import annotations

from dataclasses import dataclass

# Valid categories
_VALID_HYGROSCOPICITY = {"non", "slight", "moderate", "high"}
_VALID_CRYSTALLINITY = {"crystalline", "semi_crystalline", "amorphous"}


@dataclass(frozen=True)
class ManufacturabilityResult:
    """Result of API manufacturability scoring."""

    drug_name: str
    mw_Da: float
    mp_C: float
    logP: float
    mw_score: float
    mp_score: float
    solubility_score: float
    hygroscopicity_score: float
    crystallinity_score: float
    total_score: float
    manufacturability_class: str
    top_issue: str
    notes: str


def _score_mw(mw_Da: float) -> float:
    """MW score: 0-20. Optimal 200-500 Da."""
    if mw_Da <= 0:
        return 0.0
    if 200.0 <= mw_Da <= 500.0:
        # Full score in optimal range
        return 20.0
    elif mw_Da < 200.0:
        # Penalty for too small (uncommon but loses points linearly)
        deficit = 200.0 - mw_Da
        return max(0.0, 20.0 - deficit * 0.1)
    else:
        # mw_Da > 500: penalty
        excess = mw_Da - 500.0
        return max(0.0, 20.0 - excess * 0.04)


def _score_mp(mp_C: float) -> float:
    """Melting point score: 0-20. Optimal 130-200°C."""
    if 130.0 <= mp_C <= 200.0:
        return 20.0
    elif mp_C < 130.0:
        # Too low → handling issues (sticky, difficult to process)
        deficit = 130.0 - mp_C
        return max(0.0, 20.0 - deficit * 0.4)
    else:
        # mp_C > 200: too high → processing/milling issues
        excess = mp_C - 200.0
        return max(0.0, 20.0 - excess * 0.4)


def _score_solubility(solubility_mg_mL: float) -> float:
    """Solubility score: discrete bands."""
    if solubility_mg_mL >= 1.0:
        return 20.0
    elif solubility_mg_mL >= 0.1:
        return 15.0
    elif solubility_mg_mL >= 0.01:
        return 8.0
    else:
        return 2.0


def _score_hygroscopicity(hygroscopicity: str) -> float:
    """Hygroscopicity score: discrete mapping."""
    mapping = {
        "non": 20.0,
        "slight": 16.0,
        "moderate": 10.0,
        "high": 4.0,
    }
    return mapping[hygroscopicity]


def _score_crystallinity(crystallinity: str) -> float:
    """Crystallinity score: discrete mapping."""
    mapping = {
        "crystalline": 20.0,
        "semi_crystalline": 14.0,
        "amorphous": 6.0,
    }
    return mapping[crystallinity]


def _classify(total_score: float) -> str:
    if total_score >= 80.0:
        return "excellent"
    elif total_score >= 60.0:
        return "good"
    else:
        return "challenging"


def score_manufacturability(
    drug_name: str,
    mw_Da: float,
    mp_C: float,
    logP: float,
    solubility_mg_mL: float,
    hygroscopicity: str,
    crystallinity: str,
) -> ManufacturabilityResult:
    """
    Score API manufacturability based on physical properties.

    Parameters
    ----------
    drug_name : str
    mw_Da : float              Molecular weight (Da), must be > 0
    mp_C : float               Melting point (°C)
    logP : float               Octanol-water partition coefficient
    solubility_mg_mL : float   Aqueous solubility (mg/mL)
    hygroscopicity : str       One of: "non", "slight", "moderate", "high"
    crystallinity : str        One of: "crystalline", "semi_crystalline", "amorphous"

    Returns
    -------
    ManufacturabilityResult
    """
    if mw_Da <= 0:
        raise ValueError(f"mw_Da must be > 0, got {mw_Da}")
    if hygroscopicity not in _VALID_HYGROSCOPICITY:
        raise ValueError(
            f"hygroscopicity must be one of {_VALID_HYGROSCOPICITY}, got '{hygroscopicity}'"
        )
    if crystallinity not in _VALID_CRYSTALLINITY:
        raise ValueError(
            f"crystallinity must be one of {_VALID_CRYSTALLINITY}, got '{crystallinity}'"
        )

    mw_score = _score_mw(mw_Da)
    mp_score = _score_mp(mp_C)
    sol_score = _score_solubility(solubility_mg_mL)
    hygro_score = _score_hygroscopicity(hygroscopicity)
    cryst_score = _score_crystallinity(crystallinity)

    total_score = mw_score + mp_score + sol_score + hygro_score + cryst_score
    # Clamp to [0, 100]
    total_score = max(0.0, min(100.0, total_score))

    manufacturability_class = _classify(total_score)

    # Identify top issue (lowest scoring component)
    component_scores = {
        "molecular_weight": mw_score,
        "melting_point": mp_score,
        "solubility": sol_score,
        "hygroscopicity": hygro_score,
        "crystallinity": cryst_score,
    }
    top_issue = min(component_scores, key=lambda k: component_scores[k])

    notes = (
        f"mw_score={mw_score:.1f}, mp_score={mp_score:.1f}, "
        f"sol_score={sol_score:.1f}, hygro_score={hygro_score:.1f}, "
        f"cryst_score={cryst_score:.1f}"
    )

    return ManufacturabilityResult(
        drug_name=drug_name,
        mw_Da=mw_Da,
        mp_C=mp_C,
        logP=logP,
        mw_score=mw_score,
        mp_score=mp_score,
        solubility_score=sol_score,
        hygroscopicity_score=hygro_score,
        crystallinity_score=cryst_score,
        total_score=total_score,
        manufacturability_class=manufacturability_class,
        top_issue=top_issue,
        notes=notes,
    )


def screen_api_candidates(candidates: list) -> list:
    """
    Score multiple API candidates and return sorted by total_score descending.

    Each candidate should be a dict with keys matching score_manufacturability parameters.

    Parameters
    ----------
    candidates : list of dict

    Returns
    -------
    list of ManufacturabilityResult, sorted by total_score descending
    """
    results = []
    for c in candidates:
        result = score_manufacturability(
            drug_name=c["drug_name"],
            mw_Da=c["mw_Da"],
            mp_C=c["mp_C"],
            logP=c["logP"],
            solubility_mg_mL=c["solubility_mg_mL"],
            hygroscopicity=c["hygroscopicity"],
            crystallinity=c["crystallinity"],
        )
        results.append(result)

    results.sort(key=lambda r: r.total_score, reverse=True)
    return results
