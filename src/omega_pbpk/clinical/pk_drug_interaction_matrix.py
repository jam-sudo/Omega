"""Phase 203 — CYP-based drug-drug interaction matrix for polypharmacy risk."""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# CYP isoform constants
# ---------------------------------------------------------------------------

SUPPORTED_CYPS = ("CYP3A4", "CYP2D6", "CYP2C9", "CYP2C19", "CYP1A2")

# Severity scores
SCORE_INDUCER_SUBSTRATE = 1  # minor
SCORE_INHIBITOR_SUBSTRATE = 2  # moderate
SCORE_STRONG_INHIBITOR_SUBSTRATE = 3  # major


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DDIMatrixResult:
    """Result of build_interaction_matrix."""

    drug_names: tuple
    interaction_matrix: tuple  # tuple of tuples of int (n x n)
    flagged_pairs: tuple  # tuple of (name_a, name_b, score, cyp_list)
    max_severity: int
    total_interactions: int
    notes: str


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _normalize_drug(drug: dict) -> dict:
    """Return a copy of drug dict with normalised CYP lists (uppercase strings)."""
    return {
        "name": str(drug.get("name", "Unknown")),
        "cyp_inhibitor": [c.upper() for c in drug.get("cyp_inhibitor", [])],
        "cyp_substrate": [c.upper() for c in drug.get("cyp_substrate", [])],
        "cyp_inducer": [c.upper() for c in drug.get("cyp_inducer", [])],
        "strong_inhibitor": bool(drug.get("strong_inhibitor", False)),
    }


def _pair_score(drug_a: dict, drug_b: dict) -> tuple[int, list[str]]:
    """
    Calculate interaction score where drug_a is the perpetrator and drug_b is
    the victim (substrate-carrying drug).

    Returns (score, list_of_interacting_cyps).
    """
    inh_a = set(drug_a["cyp_inhibitor"])
    ind_a = set(drug_a["cyp_inducer"])
    sub_b = set(drug_b["cyp_substrate"])

    score = 0
    interacting = []

    for cyp in SUPPORTED_CYPS:
        if cyp in sub_b:
            if cyp in inh_a:
                if drug_a["strong_inhibitor"]:
                    score += SCORE_STRONG_INHIBITOR_SUBSTRATE
                else:
                    score += SCORE_INHIBITOR_SUBSTRATE
                interacting.append(cyp)
            elif cyp in ind_a:
                score += SCORE_INDUCER_SUBSTRATE
                interacting.append(cyp)

    return score, interacting


def _score_pair(drug_a: dict, drug_b: dict) -> int:
    """
    Bidirectional interaction score between drug_a and drug_b.

    Considers both directions: A inhibits/induces CYPs that B is substrate of,
    and vice versa.
    """
    score_ab, _ = _pair_score(drug_a, drug_b)
    score_ba, _ = _pair_score(drug_b, drug_a)
    return score_ab + score_ba


def _pair_cyps(drug_a: dict, drug_b: dict) -> list[str]:
    """Collect all CYPs involved in the interaction between a and b."""
    _, cyps_ab = _pair_score(drug_a, drug_b)
    _, cyps_ba = _pair_score(drug_b, drug_a)
    seen: list[str] = []
    for c in cyps_ab + cyps_ba:
        if c not in seen:
            seen.append(c)
    return seen


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_interaction_matrix(drugs: list[dict]) -> DDIMatrixResult:
    """
    Build an n×n CYP-based DDI interaction matrix for a medication list.

    Parameters
    ----------
    drugs:
        List of dicts with keys:
          - "name" (str)
          - "cyp_inhibitor" (list[str])
          - "cyp_substrate" (list[str])
          - "cyp_inducer" (list[str])
          - "strong_inhibitor" (bool, optional)

    Returns
    -------
    DDIMatrixResult
    """
    if not drugs:
        raise ValueError("drugs list must not be empty")

    normed = [_normalize_drug(d) for d in drugs]
    n = len(normed)
    names = tuple(d["name"] for d in normed)

    # Build n x n matrix
    matrix: list[list[int]] = [[0] * n for _ in range(n)]

    flagged: list[tuple] = []
    total = 0
    max_sev = 0

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # Score where drug_i is perpetrator, drug_j is victim
            score_ij, cyps_ij = _pair_score(normed[i], normed[j])
            matrix[i][j] = score_ij
            if score_ij > 0:
                total += 1
                if score_ij > max_sev:
                    max_sev = score_ij

    # Symmetric flagging: flag each unordered pair once
    flagged_set: set[tuple[int, int]] = set()
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            combined = matrix[i][j] + matrix[j][i]
            if combined > 0:
                pair_key = (min(i, j), max(i, j))
                if pair_key not in flagged_set:
                    flagged_set.add(pair_key)
                    cyps = _pair_cyps(normed[i], normed[j])
                    flagged.append((names[i], names[j], combined, tuple(cyps)))

    # Recompute total_interactions as number of directed non-zero pairs
    total = sum(1 for i in range(n) for j in range(n) if i != j and matrix[i][j] > 0)

    # max_severity over directed scores
    all_scores = [matrix[i][j] for i in range(n) for j in range(n) if i != j]
    max_sev = max(all_scores) if all_scores else 0

    matrix_tuple = tuple(tuple(row) for row in matrix)

    notes = (
        f"Analyzed {n} drugs across {len(SUPPORTED_CYPS)} CYP isoforms. "
        f"{len(flagged)} interacting pairs detected."
    )

    return DDIMatrixResult(
        drug_names=names,
        interaction_matrix=matrix_tuple,
        flagged_pairs=tuple(flagged),
        max_severity=max_sev,
        total_interactions=total,
        notes=notes,
    )


def screen_polypharmacy_risk(drugs: list[dict]) -> dict:
    """
    Screen a medication list for polypharmacy DDI risk.

    Returns
    -------
    dict with keys:
        total_interactions (int)
        major_interactions (int)
        moderate_interactions (int)
        high_risk_drugs (list[str])
    """
    if not drugs:
        raise ValueError("drugs list must not be empty")

    result = build_interaction_matrix(drugs)
    n = len(drugs)
    names = result.drug_names
    matrix = result.interaction_matrix

    major = sum(
        1
        for i in range(n)
        for j in range(n)
        if i != j and matrix[i][j] >= SCORE_STRONG_INHIBITOR_SUBSTRATE
    )
    moderate = sum(
        1
        for i in range(n)
        for j in range(n)
        if i != j and matrix[i][j] == SCORE_INHIBITOR_SUBSTRATE
    )

    # Count interactions per drug (as perpetrator or victim)
    drug_interaction_count: dict[str, int] = {name: 0 for name in names}
    for i in range(n):
        for j in range(n):
            if i != j and matrix[i][j] > 0:
                drug_interaction_count[names[i]] += 1
                drug_interaction_count[names[j]] += 1

    high_risk = [name for name, count in drug_interaction_count.items() if count >= 2]

    return {
        "total_interactions": result.total_interactions,
        "major_interactions": major,
        "moderate_interactions": moderate,
        "high_risk_drugs": high_risk,
    }
