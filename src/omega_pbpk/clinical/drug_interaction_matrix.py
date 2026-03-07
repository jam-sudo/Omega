"""Drug-drug interaction matrix for polypharmacy — Phase 489.

Generates a comprehensive DDI matrix for a polypharmacy regimen by computing
pairwise AUCRs for every perpetrator-victim pair using static inhibition
theory (FDA 2020 DDI Guidance).

AUCR model
----------
For a victim drug V metabolised across multiple CYP isoforms with fraction
metabolised fm_i by isoform i, and a set of perpetrators P that inhibit
isoform i with Ki_P,i (uM), the combined AUCR is:

    AUCR_V = 1 / sum_i( fm_i / R_i )

where
    R_i = product over all perpetrators P of (1 + Cmax_P / Ki_P,i)

Drugs that are marked as inducers reduce R by 50 % (approximate net effect)
for those CYP isoforms they induce.

References
----------
- FDA. In Vitro Drug Interaction Studies — DDI Guidance for Industry. 2020.
- Rowland M & Matin SB. JPKE. 1973;1(2):123-136.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Interaction classification thresholds (FDA 2020)
# ---------------------------------------------------------------------------

_CLASSIFY_NONE = 1.25
_CLASSIFY_WEAK = 2.0
_CLASSIFY_MODERATE = 5.0

# Approximate fold-change in fm when an inducer is present (reduces R by ~50%)
_INDUCTION_R_FACTOR = 0.5


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class InteractionMatrixResult:
    """Full DDI matrix result for a polypharmacy regimen.

    Attributes
    ----------
    drug_names : Names of the drugs (same order as matrix axes).
    aucr_matrix : Perpetrator (row) × victim (column) AUCR values.
                  Element [i][j] is the AUCR of drug j when drug i is the
                  sole perpetrator (self-interaction = 1.0 on diagonal).
    interaction_classes : Same shape as aucr_matrix; string classification.
    max_aucr : Maximum off-diagonal AUCR observed.
    min_aucr : Minimum off-diagonal AUCR observed.
    n_strong : Number of strong interactions (AUCR ≥ 5).
    n_moderate : Number of moderate interactions (2 ≤ AUCR < 5).
    high_risk_pairs : Human-readable list of pairs with strong DDI.
    notes : Free-text summary.
    """

    drug_names: list[str] = field(default_factory=list)
    aucr_matrix: list[list[float]] = field(default_factory=list)
    interaction_classes: list[list[str]] = field(default_factory=list)
    max_aucr: float = 1.0
    min_aucr: float = 1.0
    n_strong: int = 0
    n_moderate: int = 0
    high_risk_pairs: list[str] = field(default_factory=list)
    notes: str = ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_interaction(aucr: float) -> str:
    """Classify an AUCR value according to FDA DDI thresholds.

    Parameters
    ----------
    aucr:
        Area-under-the-curve ratio (victim AUC with perpetrator / alone).

    Returns
    -------
    str
        'none'     — AUCR < 1.25
        'weak'     — 1.25 ≤ AUCR < 2.0
        'moderate' — 2.0 ≤ AUCR < 5.0
        'strong'   — AUCR ≥ 5.0
    """
    if aucr < _CLASSIFY_NONE:
        return "none"
    if aucr < _CLASSIFY_WEAK:
        return "weak"
    if aucr < _CLASSIFY_MODERATE:
        return "moderate"
    return "strong"


def _validate_drug(drug: dict[str, Any], index: int) -> None:
    """Raise ValueError for a malformed drug entry."""
    required = {"name", "cyp_substrates", "cyp_inhibitors", "cyp_inducers", "c_max_uM", "fm"}
    missing = required - set(drug.keys())
    if missing:
        raise ValueError(
            f"Drug at index {index} is missing required keys: {missing}"
        )
    if not isinstance(drug["name"], str) or not drug["name"].strip():
        raise ValueError(f"Drug at index {index} has an empty or non-string 'name'.")
    if drug["c_max_uM"] < 0:
        raise ValueError(
            f"Drug '{drug['name']}': c_max_uM must be ≥ 0, got {drug['c_max_uM']}."
        )
    if drug["fm"] and not math.isclose(sum(drug["fm"].values()), 1.0, abs_tol=0.05):
        raise ValueError(
            f"Drug '{drug['name']}': fm values should sum to ~1.0, "
            f"got {sum(drug['fm'].values()):.3f}."
        )


def _compute_pairwise_aucr(
    victim: dict[str, Any],
    perpetrator: dict[str, Any],
) -> float:
    """Return AUCR of *victim* in the presence of *perpetrator* alone.

    Returns 1.0 when no shared CYP pathways or victim has no fm entries.
    """
    fm: dict[str, float] = victim.get("fm", {})
    if not fm:
        return 1.0

    p_inhibitors: dict[str, float] = perpetrator.get("cyp_inhibitors", {})
    p_inducers: list[str] = perpetrator.get("cyp_inducers", [])
    c_max_p: float = perpetrator.get("c_max_uM", 0.0)

    denom = 0.0
    for cyp, fm_i in fm.items():
        r_i = 1.0
        # Inhibition: R_i = 1 + Cmax_P / Ki
        ki = p_inhibitors.get(cyp)
        if ki is not None and ki > 0 and c_max_p > 0:
            r_i *= 1.0 + c_max_p / ki
        # Induction: reduce R by 50 % (net effect approximation)
        if cyp in p_inducers:
            r_i *= _INDUCTION_R_FACTOR
        # Guard: R_i must stay ≥ 1.0 after combining effects; induction
        # alone (no inhibition) keeps r_i = 0.5, which is valid (inducer).
        denom += fm_i / r_i

    if denom <= 0:
        return 1.0

    return 1.0 / denom


def build_interaction_matrix(drugs: list[dict[str, Any]]) -> InteractionMatrixResult:
    """Build a pairwise DDI matrix for all drugs in the regimen.

    Parameters
    ----------
    drugs:
        List of drug dictionaries.  Each must contain:
        - ``name`` (str)
        - ``cyp_substrates`` (list[str]) — CYP isoforms for which the drug is
          a substrate.
        - ``cyp_inhibitors`` (dict[str, float]) — mapping of CYP isoform →
          Ki (uM).  Empty dict if not an inhibitor.
        - ``cyp_inducers`` (list[str]) — CYP isoforms induced by the drug.
        - ``c_max_uM`` (float) — unbound Cmax at therapeutic dose (uM).
        - ``fm`` (dict[str, float]) — fraction metabolised by each CYP; values
          must sum to ~1.0 if non-empty.

    Returns
    -------
    InteractionMatrixResult
    """
    if not drugs:
        raise ValueError("drugs list must not be empty.")

    for idx, drug in enumerate(drugs):
        _validate_drug(drug, idx)

    n = len(drugs)
    names = [d["name"] for d in drugs]

    # Build n×n AUCR matrix (row = perpetrator, col = victim)
    aucr_matrix: list[list[float]] = [[1.0] * n for _ in range(n)]
    class_matrix: list[list[str]] = [["none"] * n for _ in range(n)]

    off_diag_aucrs: list[float] = []
    n_strong = 0
    n_moderate = 0
    high_risk_pairs: list[str] = []

    for i, perp in enumerate(drugs):
        for j, victim in enumerate(drugs):
            if i == j:
                aucr_matrix[i][j] = 1.0
                class_matrix[i][j] = "none"
                continue

            aucr = _compute_pairwise_aucr(victim, perp)
            aucr_matrix[i][j] = round(aucr, 4)
            cls = classify_interaction(aucr)
            class_matrix[i][j] = cls
            off_diag_aucrs.append(aucr)

            if cls == "strong":
                n_strong += 1
                pair_label = f"{perp['name']} → {victim['name']}"
                if pair_label not in high_risk_pairs:
                    high_risk_pairs.append(pair_label)
            elif cls == "moderate":
                n_moderate += 1

    max_aucr = max(off_diag_aucrs) if off_diag_aucrs else 1.0
    min_aucr = min(off_diag_aucrs) if off_diag_aucrs else 1.0

    notes_parts = [f"{n} drugs evaluated; {n * (n - 1)} directional pairs assessed."]
    if n_strong:
        notes_parts.append(f"{n_strong} strong DDI(s) detected.")
    if n_moderate:
        notes_parts.append(f"{n_moderate} moderate DDI(s) detected.")
    if not n_strong and not n_moderate:
        notes_parts.append("No clinically significant DDIs detected.")

    return InteractionMatrixResult(
        drug_names=names,
        aucr_matrix=aucr_matrix,
        interaction_classes=class_matrix,
        max_aucr=round(max_aucr, 4),
        min_aucr=round(min_aucr, 4),
        n_strong=n_strong,
        n_moderate=n_moderate,
        high_risk_pairs=high_risk_pairs,
        notes=" ".join(notes_parts),
    )


def identify_high_risk_combinations(
    drugs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return all drug pairs with a strong DDI (AUCR ≥ 5).

    Parameters
    ----------
    drugs:
        Same format as :func:`build_interaction_matrix`.

    Returns
    -------
    list[dict]
        Each element: ``{'perpetrator': str, 'victim': str, 'aucr': float,
        'classification': str}``.
    """
    if not drugs:
        raise ValueError("drugs list must not be empty.")

    for idx, drug in enumerate(drugs):
        _validate_drug(drug, idx)

    high_risk: list[dict[str, Any]] = []
    for i, perp in enumerate(drugs):
        for j, victim in enumerate(drugs):
            if i == j:
                continue
            aucr = _compute_pairwise_aucr(victim, perp)
            if classify_interaction(aucr) == "strong":
                high_risk.append(
                    {
                        "perpetrator": perp["name"],
                        "victim": victim["name"],
                        "aucr": round(aucr, 4),
                        "classification": "strong",
                    }
                )
    return high_risk


def calculate_net_exposure_change(
    drug_name: str,
    all_drugs: list[dict[str, Any]],
) -> float:
    """Calculate net AUCR for *drug_name* considering all other perpetrators.

    All other drugs in ``all_drugs`` act simultaneously as perpetrators.
    The net effect is modelled by combining the R_i contributions from every
    perpetrator for each CYP isoform:

        R_i_combined = product over perpetrators P of (1 + Cmax_P / Ki_P,i)

    Then:

        AUCR_net = 1 / sum_i( fm_i / R_i_combined )

    Parameters
    ----------
    drug_name:
        Name of the victim drug (must appear in ``all_drugs``).
    all_drugs:
        Full drug list (same format as :func:`build_interaction_matrix`).

    Returns
    -------
    float
        Net AUCR for the victim drug.
    """
    if not all_drugs:
        raise ValueError("all_drugs list must not be empty.")

    for idx, drug in enumerate(all_drugs):
        _validate_drug(drug, idx)

    victim: dict[str, Any] | None = None
    perpetrators: list[dict[str, Any]] = []
    for drug in all_drugs:
        if drug["name"] == drug_name:
            victim = drug
        else:
            perpetrators.append(drug)

    if victim is None:
        raise ValueError(
            f"Drug '{drug_name}' not found in all_drugs. "
            f"Available names: {[d['name'] for d in all_drugs]}."
        )

    fm: dict[str, float] = victim.get("fm", {})
    if not fm or not perpetrators:
        return 1.0

    denom = 0.0
    for cyp, fm_i in fm.items():
        r_i = 1.0
        for perp in perpetrators:
            p_inhibitors = perp.get("cyp_inhibitors", {})
            p_inducers = perp.get("cyp_inducers", [])
            c_max_p = perp.get("c_max_uM", 0.0)

            ki = p_inhibitors.get(cyp)
            if ki is not None and ki > 0 and c_max_p > 0:
                r_i *= 1.0 + c_max_p / ki
            if cyp in p_inducers:
                r_i *= _INDUCTION_R_FACTOR

        denom += fm_i / r_i

    if denom <= 0:
        return 1.0

    return round(1.0 / denom, 4)
