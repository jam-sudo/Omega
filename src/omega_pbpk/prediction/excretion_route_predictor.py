"""Excretion route predictor: renal vs. biliary vs. mixed.

Rule-based scoring from drug physicochemical properties (MW, logP, pKa,
ionization type, fu_plasma) to predict the primary elimination route.
"""

from __future__ import annotations

from dataclasses import dataclass

_VALID_IONIZATION_TYPES = frozenset({"acid", "base", "neutral"})


@dataclass(frozen=True)
class ExcretionRouteResult:
    """Result from excretion route prediction."""

    drug_name: str
    mw_Da: float
    logP: float
    ionization_type: str
    fu_plasma: float
    renal_score: float
    biliary_score: float
    primary_route: str  # "renal", "biliary", or "mixed"
    fe_renal_estimated: float  # fraction excreted renally
    fe_biliary_estimated: float  # = 1 - fe_renal
    ckd_sensitivity: str  # "high" if renal_score > 60, "low" otherwise
    hepatic_sensitivity: str  # "high" if biliary_score > 60, "low" otherwise
    notes: str


def _validate_inputs(
    mw_Da: float,
    logP: float,
    ionization_type: str,
    fu_plasma: float,
) -> None:
    if mw_Da <= 0:
        raise ValueError(f"mw_Da must be > 0, got {mw_Da}")
    if not (-5.0 <= logP <= 10.0):
        raise ValueError(f"logP must be in [-5, 10], got {logP}")
    if ionization_type not in _VALID_IONIZATION_TYPES:
        raise ValueError(
            f"ionization_type must be one of {sorted(_VALID_IONIZATION_TYPES)}, "
            f"got '{ionization_type}'"
        )
    if not (0 < fu_plasma <= 1.0):
        raise ValueError(f"fu_plasma must be in (0, 1], got {fu_plasma}")


def predict_excretion_route(
    drug_name: str,
    mw_Da: float,
    logP: float,
    ionization_type: str = "neutral",
    fu_plasma: float = 0.5,
) -> ExcretionRouteResult:
    """Predict primary excretion route using rule-based scoring.

    Scoring rules:

    Renal score (0-100):
    - MW < 300: +40; 300-500: +20; > 500: +0
    - logP < 1: +30; 1-3: +15; > 3: +0
    - fu_plasma > 0.3: +20
    - base drug: +10 (tubular secretion)

    Biliary score (0-100):
    - MW > 500: +40; 300-500: +20; < 300: +0
    - logP > 3: +30; < 1: +0
    - acid drug: +10 (bile acid-like)
    - fu_plasma < 0.1: +10

    Primary route: "renal" if renal_score > biliary_score + 20,
                   "biliary" if biliary_score > renal_score + 20,
                   else "mixed".

    Parameters
    ----------
    drug_name:
        Name of the drug compound.
    mw_Da:
        Molecular weight in Daltons.
    logP:
        Octanol-water partition coefficient.
    ionization_type:
        "acid", "base", or "neutral".
    fu_plasma:
        Fraction unbound in plasma (0, 1].

    Returns
    -------
    ExcretionRouteResult
    """
    _validate_inputs(mw_Da, logP, ionization_type, fu_plasma)

    # --- Renal score ---
    renal_score = 0.0

    # MW contribution
    if mw_Da < 300:
        renal_score += 40.0
    elif mw_Da <= 500:
        renal_score += 20.0
    # > 500: +0

    # logP contribution
    if logP < 1.0:
        renal_score += 30.0
    elif logP <= 3.0:
        renal_score += 15.0
    # > 3: +0

    # fu_plasma
    if fu_plasma > 0.3:
        renal_score += 20.0

    # ionization (tubular secretion bonus for bases)
    if ionization_type == "base":
        renal_score += 10.0

    # --- Biliary score ---
    biliary_score = 0.0

    # MW contribution
    if mw_Da > 500:
        biliary_score += 40.0
    elif mw_Da >= 300:
        biliary_score += 20.0
    # < 300: +0

    # logP contribution
    if logP > 3.0:
        biliary_score += 30.0
    # logP < 1: +0; 1-3: no explicit rule (0 points)

    # ionization (acid bonus: bile acid-like)
    if ionization_type == "acid":
        biliary_score += 10.0

    # fu_plasma (highly bound → biliary)
    if fu_plasma < 0.1:
        biliary_score += 10.0

    # --- Primary route determination ---
    if renal_score > biliary_score + 20.0:
        primary_route = "renal"
    elif biliary_score > renal_score + 20.0:
        primary_route = "biliary"
    else:
        primary_route = "mixed"

    # --- Fractional estimates ---
    total = renal_score + biliary_score
    if total > 0:
        fe_renal = renal_score / total
        fe_biliary = biliary_score / total
    else:
        fe_renal = 0.5
        fe_biliary = 0.5

    # --- Sensitivity classifications ---
    ckd_sensitivity = "high" if renal_score > 60 else "low"
    hepatic_sensitivity = "high" if biliary_score > 60 else "low"

    notes = (
        f"renal_score={renal_score:.0f}; biliary_score={biliary_score:.0f}; "
        f"primary_route={primary_route}; "
        f"CKD sensitivity={ckd_sensitivity}; hepatic sensitivity={hepatic_sensitivity}"
    )

    return ExcretionRouteResult(
        drug_name=drug_name,
        mw_Da=mw_Da,
        logP=logP,
        ionization_type=ionization_type,
        fu_plasma=fu_plasma,
        renal_score=renal_score,
        biliary_score=biliary_score,
        primary_route=primary_route,
        fe_renal_estimated=fe_renal,
        fe_biliary_estimated=fe_biliary,
        ckd_sensitivity=ckd_sensitivity,
        hepatic_sensitivity=hepatic_sensitivity,
        notes=notes,
    )


def screen_drugs(drugs: list[dict]) -> list[ExcretionRouteResult]:
    """Screen multiple drugs and return sorted by fe_renal_estimated descending.

    Each dict should contain keys: drug_name, mw_Da, logP, ionization_type, fu_plasma.
    Optional keys default to predict_excretion_route defaults.

    Parameters
    ----------
    drugs:
        List of drug property dicts.

    Returns
    -------
    list of ExcretionRouteResult sorted by fe_renal_estimated descending.
    """
    results = []
    for drug in drugs:
        result = predict_excretion_route(
            drug_name=drug["drug_name"],
            mw_Da=drug["mw_Da"],
            logP=drug["logP"],
            ionization_type=drug.get("ionization_type", "neutral"),
            fu_plasma=drug.get("fu_plasma", 0.5),
        )
        results.append(result)
    return sorted(results, key=lambda r: r.fe_renal_estimated, reverse=True)
