"""Metabolic DDI network analysis — Phase 469.

Analyzes drug-drug interactions at the metabolic level for multi-drug regimens.
Provides pairwise AUCR matrices, perpetrator/victim identification, and
overall regimen DDI risk classification.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "DDIMatrixResult",
    "build_ddi_matrix",
    "identify_perpetrators",
    "identify_victims",
    "calculate_regimen_ddi_risk",
]

# ---------------------------------------------------------------------------
# AUCR thresholds (FDA 2020 DDI guidance)
# ---------------------------------------------------------------------------

_AUCR_HIGH_THRESHOLD = 5.0  # strong inhibition
_AUCR_MODERATE_THRESHOLD = 2.0  # moderate inhibition
_AUCR_INDUCTION_THRESHOLD = 0.8  # induction (< 0.8 = meaningful reduction)

# Induction magnitude: 50% reduction in AUC (typical strong inducer)
_INDUCTION_FOLD_REDUCTION = 0.5


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class DDIMatrixResult:
    """Result from pairwise DDI matrix analysis.

    aucr_matrix[i][j] = AUCR of drug j when drug i is co-administered.
    A value > 1 indicates inhibition; < 1 indicates induction; 1.0 means
    no interaction.  Diagonal is always 1.0 (drug does not interact with
    itself in the regimen sense).
    """

    drug_names: list[str]
    aucr_matrix: list[list[float]]
    max_aucr: float
    n_high_risk: int
    n_moderate_risk: int
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_drug(drug: dict, index: int) -> None:
    """Raise ValueError for missing mandatory keys."""
    required = {"name", "cyp_inhibited", "cyp_induced", "fm_cyp", "ki_uM", "dose_mg", "c_max_uM"}
    missing = required - set(drug.keys())
    if missing:
        raise ValueError(f"Drug at index {index} is missing keys: {sorted(missing)}")
    if not isinstance(drug["cyp_inhibited"], list):
        raise ValueError(f"Drug '{drug['name']}' cyp_inhibited must be a list.")
    if not isinstance(drug["cyp_induced"], list):
        raise ValueError(f"Drug '{drug['name']}' cyp_induced must be a list.")
    if not isinstance(drug["fm_cyp"], dict):
        raise ValueError(f"Drug '{drug['name']}' fm_cyp must be a dict.")
    if not isinstance(drug["ki_uM"], dict):
        raise ValueError(f"Drug '{drug['name']}' ki_uM must be a dict.")
    if drug["dose_mg"] < 0:
        raise ValueError(f"Drug '{drug['name']}' dose_mg must be non-negative.")
    if drug["c_max_uM"] < 0:
        raise ValueError(f"Drug '{drug['name']}' c_max_uM must be non-negative.")


def _calculate_pair_aucr(perpetrator: dict, victim: dict) -> float:
    """Calculate AUCR for victim PK when perpetrator is co-administered.

    Uses simplified static model:
      AUCR_inhibition = 1 + (C_max_perpetrator / Ki_perpetrator_for_cyp)
      weighted by fm_cyp of victim.

    For induction: AUCR is reduced by _INDUCTION_FOLD_REDUCTION per
    induced CYP that contributes to victim metabolism.

    Returns 1.0 when no shared CYP pathways exist.
    """
    c_max = perpetrator["c_max_uM"]
    ki_map = perpetrator["ki_uM"]  # CYP -> Ki in uM
    cyp_inhibited = set(perpetrator["cyp_inhibited"])
    cyp_induced = set(perpetrator["cyp_induced"])
    fm_victim = victim["fm_cyp"]  # CYP -> fraction metabolized

    if not fm_victim:
        return 1.0

    total_fm = sum(fm_victim.values())
    if total_fm <= 0.0:
        return 1.0

    # Normalize fm fractions
    fm_norm = {cyp: fm / total_fm for cyp, fm in fm_victim.items()}

    # Build composite AUCR across all CYP pathways
    # For each CYP the victim is metabolised by:
    #   - If perpetrator inhibits it: fold_change = 1 + C_max/Ki
    #   - If perpetrator induces it: fold_change = induction reduction
    #   - Otherwise: fold_change = 1.0  (no effect)
    # Composite AUCR ≈ 1 / sum(fm_i / fold_change_i)

    denominator = 0.0
    for cyp, fm_i in fm_norm.items():
        if cyp in cyp_inhibited and c_max > 0:
            ki = ki_map.get(cyp, None)
            if ki is not None and ki > 0:
                # AUCR for this pathway = 1 + C_max/Ki
                # Pathway-level AUCR contribution to denominator
                fold_change = 1.0 + c_max / ki
                denominator += fm_i / fold_change
            else:
                # Inhibited but no Ki provided — assume moderate inhibition
                denominator += fm_i / 2.0
        elif cyp in cyp_induced:
            # Induction reduces clearance pathway — higher clearance → lower AUC
            # fold_change < 1 for victim AUC → treat as increased clearance
            denominator += fm_i / _INDUCTION_FOLD_REDUCTION
        else:
            denominator += fm_i  # no change

    if denominator <= 0.0:
        return 1.0

    aucr = 1.0 / denominator
    return round(aucr, 4)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_ddi_matrix(drugs: list[dict]) -> DDIMatrixResult:
    """Build pairwise AUCR matrix for a list of drugs.

    Parameters
    ----------
    drugs:
        List of drug dicts.  Each must have keys: name, cyp_inhibited,
        cyp_induced, fm_cyp, ki_uM, dose_mg, c_max_uM.

    Returns
    -------
    DDIMatrixResult
        n×n AUCR matrix where aucr_matrix[i][j] is the AUCR of drug j
        when drug i acts as perpetrator.
    """
    if not isinstance(drugs, list):
        raise TypeError("drugs must be a list of dicts.")
    if len(drugs) == 0:
        raise ValueError("drugs list must not be empty.")

    for idx, drug in enumerate(drugs):
        _validate_drug(drug, idx)

    n = len(drugs)
    names = [d["name"] for d in drugs]

    matrix: list[list[float]] = []
    for i in range(n):
        row: list[float] = []
        for j in range(n):
            if i == j:
                row.append(1.0)
            else:
                aucr = _calculate_pair_aucr(drugs[i], drugs[j])
                row.append(aucr)
        matrix.append(row)

    # Collect off-diagonal AUCR values for summary statistics
    off_diag = [matrix[i][j] for i in range(n) for j in range(n) if i != j]

    max_aucr = max(off_diag) if off_diag else 1.0
    n_high = sum(1 for v in off_diag if v > _AUCR_HIGH_THRESHOLD)
    n_mod = sum(1 for v in off_diag if _AUCR_MODERATE_THRESHOLD <= v <= _AUCR_HIGH_THRESHOLD)

    notes_parts: list[str] = [f"{n} drugs, {n * (n - 1)} directed pairs."]
    if n_high:
        notes_parts.append(f"{n_high} high-risk pairs (AUCR > 5).")
    if n_mod:
        notes_parts.append(f"{n_mod} moderate-risk pairs (AUCR 2–5).")
    notes = " ".join(notes_parts)

    return DDIMatrixResult(
        drug_names=names,
        aucr_matrix=matrix,
        max_aucr=max_aucr,
        n_high_risk=n_high,
        n_moderate_risk=n_mod,
        notes=notes,
    )


def identify_perpetrators(drugs: list[dict]) -> list[dict]:
    """Identify perpetrator drugs sorted by maximum inhibition potential.

    A perpetrator is a drug that can inhibit or induce the metabolism of
    other drugs in the regimen.  Sorted descending by max AUCR caused.

    Parameters
    ----------
    drugs:
        List of drug dicts (same schema as build_ddi_matrix).

    Returns
    -------
    list[dict]
        Each entry has: name, max_aucr_caused, n_victims_inhibited,
        n_victims_induced, cyp_inhibited, cyp_induced.
    """
    if not isinstance(drugs, list):
        raise TypeError("drugs must be a list of dicts.")
    if len(drugs) == 0:
        raise ValueError("drugs list must not be empty.")

    for idx, drug in enumerate(drugs):
        _validate_drug(drug, idx)

    result: list[dict] = []
    for i, perp in enumerate(drugs):
        max_aucr_caused = 1.0
        n_inhibited = 0
        n_induced = 0

        for j, victim in enumerate(drugs):
            if i == j:
                continue
            aucr = _calculate_pair_aucr(perp, victim)
            if aucr > max_aucr_caused:
                max_aucr_caused = aucr
            if aucr > _AUCR_MODERATE_THRESHOLD:
                n_inhibited += 1
            elif aucr < _AUCR_INDUCTION_THRESHOLD:
                n_induced += 1

        result.append(
            {
                "name": perp["name"],
                "max_aucr_caused": round(max_aucr_caused, 4),
                "n_victims_inhibited": n_inhibited,
                "n_victims_induced": n_induced,
                "cyp_inhibited": perp["cyp_inhibited"],
                "cyp_induced": perp["cyp_induced"],
            }
        )

    result.sort(key=lambda x: x["max_aucr_caused"], reverse=True)
    return result


def identify_victims(drugs: list[dict]) -> list[dict]:
    """Identify victim drugs sorted by maximum AUCR vulnerability.

    A victim is a drug whose exposure is altered by co-administered drugs.
    Sorted descending by max AUCR experienced.

    Parameters
    ----------
    drugs:
        List of drug dicts (same schema as build_ddi_matrix).

    Returns
    -------
    list[dict]
        Each entry has: name, max_aucr_experienced, min_aucr_experienced,
        n_inhibitors, n_inducers, dominant_cyp.
    """
    if not isinstance(drugs, list):
        raise TypeError("drugs must be a list of dicts.")
    if len(drugs) == 0:
        raise ValueError("drugs list must not be empty.")

    for idx, drug in enumerate(drugs):
        _validate_drug(drug, idx)

    result: list[dict] = []
    for j, victim in enumerate(drugs):
        max_aucr = 1.0
        min_aucr = 1.0
        n_inhibitors = 0
        n_inducers = 0

        for i, perp in enumerate(drugs):
            if i == j:
                continue
            aucr = _calculate_pair_aucr(perp, victim)
            if aucr > max_aucr:
                max_aucr = aucr
            if aucr < min_aucr:
                min_aucr = aucr
            if aucr > _AUCR_MODERATE_THRESHOLD:
                n_inhibitors += 1
            elif aucr < _AUCR_INDUCTION_THRESHOLD:
                n_inducers += 1

        # Find dominant CYP (highest fm)
        fm = victim["fm_cyp"]
        dominant_cyp = max(fm, key=lambda k: fm[k]) if fm else "unknown"

        result.append(
            {
                "name": victim["name"],
                "max_aucr_experienced": round(max_aucr, 4),
                "min_aucr_experienced": round(min_aucr, 4),
                "n_inhibitors": n_inhibitors,
                "n_inducers": n_inducers,
                "dominant_cyp": dominant_cyp,
            }
        )

    result.sort(key=lambda x: x["max_aucr_experienced"], reverse=True)
    return result


def calculate_regimen_ddi_risk(drugs: list[dict]) -> dict:
    """Calculate overall DDI risk for a multi-drug regimen.

    Parameters
    ----------
    drugs:
        List of drug dicts.

    Returns
    -------
    dict
        Keys: overall_risk ('low'/'moderate'/'high'/'critical'),
        n_pairs, high_risk_pairs, moderate_risk_pairs, low_risk_pairs,
        max_aucr, summary.
    """
    if not isinstance(drugs, list):
        raise TypeError("drugs must be a list of dicts.")
    if len(drugs) == 0:
        raise ValueError("drugs list must not be empty.")

    for idx, drug in enumerate(drugs):
        _validate_drug(drug, idx)

    n = len(drugs)
    if n == 1:
        return {
            "overall_risk": "low",
            "n_pairs": 0,
            "high_risk_pairs": [],
            "moderate_risk_pairs": [],
            "low_risk_pairs": [],
            "max_aucr": 1.0,
            "summary": "Single drug — no DDI pairs.",
        }

    high_risk_pairs: list[dict] = []
    moderate_risk_pairs: list[dict] = []
    low_risk_pairs: list[dict] = []
    max_aucr = 1.0

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            aucr = _calculate_pair_aucr(drugs[i], drugs[j])
            pair_info = {
                "perpetrator": drugs[i]["name"],
                "victim": drugs[j]["name"],
                "aucr": round(aucr, 4),
            }
            if aucr > max_aucr:
                max_aucr = aucr
            if aucr > _AUCR_HIGH_THRESHOLD:
                high_risk_pairs.append(pair_info)
            elif aucr > _AUCR_MODERATE_THRESHOLD:
                moderate_risk_pairs.append(pair_info)
            else:
                low_risk_pairs.append(pair_info)

    n_pairs = n * (n - 1)

    if high_risk_pairs:
        if len(high_risk_pairs) > n_pairs // 2 or max_aucr > 20.0:
            overall_risk = "critical"
        else:
            overall_risk = "high"
    elif moderate_risk_pairs:
        overall_risk = "moderate"
    else:
        overall_risk = "low"

    summary = (
        f"{n_pairs} directed pairs: {len(high_risk_pairs)} high-risk, "
        f"{len(moderate_risk_pairs)} moderate-risk, "
        f"{len(low_risk_pairs)} low-risk. "
        f"Max AUCR: {round(max_aucr, 2)}."
    )

    return {
        "overall_risk": overall_risk,
        "n_pairs": n_pairs,
        "high_risk_pairs": high_risk_pairs,
        "moderate_risk_pairs": moderate_risk_pairs,
        "low_risk_pairs": low_risk_pairs,
        "max_aucr": round(max_aucr, 4),
        "summary": summary,
    }
