"""Drug metabolism pathway prediction from physicochemical descriptors.

Predicts primary and secondary metabolic pathways, metabolite types,
CYP inhibition/induction risk, gut-wall metabolism, and oral bioavailability
using a score-based rule system.

References
----------
- Wienkers & Heath, Nat Rev Drug Discov 2005;4:825-833 — CYP substrate rules
- Obach 1999 (Drug Metab Dispos) — intrinsic clearance scaling
- Houston & Carlile 1997 (Drug Metab Rev) — well-stirred hepatic model
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "MetabolismPredictionResult",
    "predict_metabolism",
    "predict_metabolite_library",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_QH_ML_PER_MIN_PER_KG = 21.0  # hepatic blood flow mL/min/kg (well-stirred model)

_VALID_PATHWAYS = frozenset(["CYP3A4", "CYP2D6", "CYP2C9", "CYP2C19", "UGT", "non_CYP"])
_VALID_RISK = frozenset(["low", "moderate", "high"])
_CYP_PATHWAYS = frozenset(["CYP3A4", "CYP2D6", "CYP2C9", "CYP2C19"])

_METABOLITE_MAP: dict[str, str] = {
    "CYP3A4": "hydroxylated, N-dealkylated",
    "CYP2D6": "hydroxylated, N-dealkylated",
    "CYP2C9": "hydroxylated, N-dealkylated",
    "CYP2C19": "hydroxylated, N-dealkylated",
    "UGT": "glucuronide conjugate",
    "non_CYP": "carboxylic acid",
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MetabolismPredictionResult:
    """Predicted metabolic fate of a drug compound."""

    drug_name: str
    primary_pathway: str
    secondary_pathway: str
    fm_primary: float
    fm_secondary: float
    predicted_metabolites: str
    cyp_inhibition_risk: str
    cyp_induction_risk: str
    gut_wall_metabolism: bool
    hepatic_first_pass_fraction: float
    intrinsic_cl_mL_per_min_per_kg: float
    bioavailability_fraction: float
    ddi_victim_risk: str
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _pathway_scores(
    logp: float,
    mw_da: float,
    pka: float,
    num_aromatic_rings: int,
    num_rotatable_bonds: int,
    has_hydroxyl: bool,
    has_amine: bool,
    has_ester: bool,
    has_amide: bool,
) -> dict[str, int]:
    """Compute integer scores for each metabolic pathway."""
    scores: dict[str, int] = {
        "CYP3A4": 0,
        "CYP2D6": 0,
        "CYP2C9": 0,
        "CYP2C19": 0,
        "UGT": 0,
        "non_CYP": 0,
    }

    # CYP3A4
    if mw_da > 400:
        scores["CYP3A4"] += 2
    if logp > 3:
        scores["CYP3A4"] += 1
    if num_aromatic_rings >= 2:
        scores["CYP3A4"] += 1

    # CYP2D6
    if has_amine and pka > 8:
        scores["CYP2D6"] += 3
    if mw_da < 400:
        scores["CYP2D6"] += 1

    # CYP2C9
    if pka < 5 and has_hydroxyl:
        scores["CYP2C9"] += 3
    if mw_da < 400:
        scores["CYP2C9"] += 1

    # CYP2C19
    if has_amine and pka < 8:
        scores["CYP2C19"] += 2
    if num_rotatable_bonds > 5:
        scores["CYP2C19"] += 1

    # UGT
    if has_hydroxyl and logp < 2:
        scores["UGT"] += 3
    if has_amide:
        scores["UGT"] += 2

    # non_CYP (esterases)
    if has_ester:
        scores["non_CYP"] += 2

    return scores


def _select_pathways(scores: dict[str, int]) -> tuple[str, str]:
    """Select primary and secondary pathways by score (tiebreak: CYP3A4)."""
    # Sort: first by score descending, then by pathway name to guarantee tiebreak order
    # Tiebreak rule: CYP3A4 wins — place it first among equals
    pathway_order = ["CYP3A4", "CYP2D6", "CYP2C9", "CYP2C19", "UGT", "non_CYP"]

    ranked = sorted(
        pathway_order,
        key=lambda p: (-scores[p], pathway_order.index(p)),
    )

    primary = ranked[0]
    secondary_candidate = ranked[1]
    secondary = secondary_candidate if scores[secondary_candidate] >= 2 else ""
    return primary, secondary


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def predict_metabolism(
    drug_name: str,
    logp: float,
    mw_da: float,
    pka: float,
    num_aromatic_rings: int,
    num_rotatable_bonds: int,
    has_hydroxyl: bool,
    has_amine: bool,
    has_ester: bool,
    has_amide: bool,
) -> MetabolismPredictionResult:
    """Predict dominant metabolic pathways for a drug using physicochemical rules.

    Parameters
    ----------
    drug_name:
        Identifier for the compound.
    logp:
        Octanol-water partition coefficient.
    mw_da:
        Molecular weight (Daltons).
    pka:
        Most basic or acidic pKa (context-dependent for scoring).
    num_aromatic_rings:
        Count of aromatic ring systems.
    num_rotatable_bonds:
        Count of rotatable bonds.
    has_hydroxyl:
        True if a hydroxyl (-OH) group is present.
    has_amine:
        True if an amine (-NH2 or secondary/tertiary amine) is present.
    has_ester:
        True if an ester group is present.
    has_amide:
        True if an amide group is present.

    Returns
    -------
    MetabolismPredictionResult
    """
    scores = _pathway_scores(
        logp=logp,
        mw_da=mw_da,
        pka=pka,
        num_aromatic_rings=num_aromatic_rings,
        num_rotatable_bonds=num_rotatable_bonds,
        has_hydroxyl=has_hydroxyl,
        has_amine=has_amine,
        has_ester=has_ester,
        has_amide=has_amide,
    )

    primary, secondary = _select_pathways(scores)
    primary_score = scores[primary]
    secondary_score = scores[secondary] if secondary else 0

    # Fraction metabolized
    fm_primary = _clamp(0.4 + 0.1 * primary_score, 0.4, 0.9)
    fm_secondary = _clamp(0.1 + 0.05 * secondary_score, 0.05, 0.5) if secondary else 0.0

    # Predicted metabolites
    predicted_metabolites = _METABOLITE_MAP[primary]

    # CYP inhibition risk
    if logp > 4 and pka > 6:
        cyp_inhibition_risk = "high"
    elif logp > 2:
        cyp_inhibition_risk = "moderate"
    else:
        cyp_inhibition_risk = "low"

    # CYP induction risk
    if mw_da > 500 and logp > 4:
        cyp_induction_risk = "high"
    elif logp > 3:
        cyp_induction_risk = "moderate"
    else:
        cyp_induction_risk = "low"

    # Gut-wall metabolism
    gut_wall_metabolism = primary == "CYP3A4" and logp > 2

    # Intrinsic clearance (mL/min/kg)
    intrinsic_cl = _clamp(5.0 + 10.0 * logp, 1.0, 100.0)

    # Hepatic first-pass using well-stirred model: FH = CLint / (CLint + QH)
    hepatic_first_pass_fraction = intrinsic_cl / (intrinsic_cl + _QH_ML_PER_MIN_PER_KG)
    hepatic_first_pass_fraction = _clamp(hepatic_first_pass_fraction, 0.0, 1.0)

    # Oral bioavailability: F = fa * (1 - FG) * (1 - FH)
    fa = 0.8
    fg = 0.2 if gut_wall_metabolism else 0.0
    bioavailability_fraction = fa * (1.0 - fg) * (1.0 - hepatic_first_pass_fraction)
    bioavailability_fraction = _clamp(bioavailability_fraction, 0.0, 1.0)

    # DDI victim risk
    if fm_primary > 0.7 and primary in _CYP_PATHWAYS:
        ddi_victim_risk = "high"
    elif fm_primary > 0.5:
        ddi_victim_risk = "moderate"
    else:
        ddi_victim_risk = "low"

    notes = (
        f"primary={primary}(score={primary_score}); "
        f"secondary={secondary or 'none'}(score={secondary_score}); "
        f"CLint={intrinsic_cl:.1f} mL/min/kg; "
        f"FH={hepatic_first_pass_fraction:.3f}; F={bioavailability_fraction:.3f}"
    )

    return MetabolismPredictionResult(
        drug_name=drug_name,
        primary_pathway=primary,
        secondary_pathway=secondary,
        fm_primary=fm_primary,
        fm_secondary=fm_secondary,
        predicted_metabolites=predicted_metabolites,
        cyp_inhibition_risk=cyp_inhibition_risk,
        cyp_induction_risk=cyp_induction_risk,
        gut_wall_metabolism=gut_wall_metabolism,
        hepatic_first_pass_fraction=hepatic_first_pass_fraction,
        intrinsic_cl_mL_per_min_per_kg=intrinsic_cl,
        bioavailability_fraction=bioavailability_fraction,
        ddi_victim_risk=ddi_victim_risk,
        notes=notes,
    )


def predict_metabolite_library(
    drug_names_data: list[dict],
) -> list[MetabolismPredictionResult]:
    """Run metabolism prediction for a list of compounds.

    Parameters
    ----------
    drug_names_data:
        List of dicts, each containing all keyword arguments for
        ``predict_metabolism`` (drug_name, logp, mw_da, pka,
        num_aromatic_rings, num_rotatable_bonds, has_hydroxyl,
        has_amine, has_ester, has_amide).

    Returns
    -------
    list[MetabolismPredictionResult]
        One result per input compound, in input order.
    """
    results = []
    for entry in drug_names_data:
        result = predict_metabolism(
            drug_name=entry["drug_name"],
            logp=entry["logp"],
            mw_da=entry["mw_da"],
            pka=entry["pka"],
            num_aromatic_rings=entry["num_aromatic_rings"],
            num_rotatable_bonds=entry["num_rotatable_bonds"],
            has_hydroxyl=entry["has_hydroxyl"],
            has_amine=entry["has_amine"],
            has_ester=entry["has_ester"],
            has_amide=entry["has_amide"],
        )
        results.append(result)
    return results
