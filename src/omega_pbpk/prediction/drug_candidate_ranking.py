"""Phase 1134 — Drug Candidate Ranking.

Multi-criteria drug candidate ranking for lead optimization using a
composite scoring system across potency, selectivity, ADME, safety,
and developability dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

_REQUIRED_KEYS = frozenset(
    [
        "drug_name",
        "ic50_nM",
        "selectivity_ratio",
        "f_oral_pct",
        "t_half_h",
        "herg_ic50_uM",
        "ames_positive",
        "logP",
        "mw_Da",
    ]
)


@dataclass(frozen=True)
class CandidateRankingResult:
    drug_name: str
    ic50_nM: float
    selectivity_ratio: float
    f_oral_pct: float
    t_half_h: float
    herg_ic50_uM: float
    ames_positive: bool
    logP: float
    mw_Da: float
    potency_score: float
    selectivity_score: float
    adme_score: float
    safety_score: float
    developability_score: float
    composite_score: float
    rank: int
    go_no_go: str
    notes: str


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _potency_score(ic50_nM: float) -> float:
    if ic50_nM < 1:
        return 20.0
    if ic50_nM < 10:
        return 17.0
    if ic50_nM < 100:
        return 13.0
    if ic50_nM < 1000:
        return 8.0
    return 3.0


def _selectivity_score(selectivity_ratio: float) -> float:
    if selectivity_ratio > 1000:
        return 20.0
    if selectivity_ratio > 100:
        return 15.0
    if selectivity_ratio > 10:
        return 10.0
    if selectivity_ratio > 1:
        return 5.0
    return 0.0


def _bioavailability_score(f_oral_pct: float) -> float:
    if f_oral_pct > 70:
        return 20.0
    if f_oral_pct > 40:
        return 15.0
    if f_oral_pct > 20:
        return 10.0
    if f_oral_pct > 5:
        return 5.0
    return 0.0


def _half_life_score(t_half_h: float) -> float:
    if 6 <= t_half_h <= 24:
        return 20.0
    if 2 <= t_half_h < 6:
        return 15.0
    if 24 < t_half_h <= 72:
        return 10.0
    # < 2h or > 72h
    return 5.0


def _adme_score(f_oral_pct: float, t_half_h: float) -> float:
    return (_bioavailability_score(f_oral_pct) + _half_life_score(t_half_h)) / 2.0


def _safety_score(herg_ic50_uM: float, ames_positive: bool) -> float:
    if herg_ic50_uM > 30:
        herg = 20.0
    elif herg_ic50_uM > 10:
        herg = 15.0
    elif herg_ic50_uM > 1:
        herg = 10.0
    elif herg_ic50_uM > 0.1:
        herg = 5.0
    else:
        herg = 0.0
    penalty = 10.0 if ames_positive else 0.0
    return max(0.0, herg - penalty)


def _developability_score(logP: float, mw_Da: float) -> float:
    violations = 0
    if mw_Da >= 500:
        violations += 1
    if logP >= 5:
        violations += 1
    if violations == 0:
        return 20.0
    if violations == 1:
        return 12.0
    return 5.0


def _go_no_go(composite: float) -> str:
    if composite >= 60:
        return "go"
    if composite >= 40:
        return "maybe"
    return "no_go"


def _build_notes(
    potency: float,
    selectivity: float,
    adme: float,
    safety: float,
    dev: float,
    ames_positive: bool,
) -> str:
    parts = []
    if potency >= 17:
        parts.append("high potency")
    elif potency <= 3:
        parts.append("low potency")
    if selectivity >= 15:
        parts.append("good selectivity")
    elif selectivity == 0:
        parts.append("poor selectivity")
    if ames_positive:
        parts.append("Ames positive (genotoxicity concern)")
    if safety == 0:
        parts.append("significant hERG risk")
    if adme >= 17:
        parts.append("favorable ADME")
    if dev == 5:
        parts.append("Lipinski violations")
    return "; ".join(parts) if parts else "standard profile"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_candidate(cand: dict) -> None:
    missing = _REQUIRED_KEYS - cand.keys()
    if missing:
        raise ValueError(f"Missing required keys: {missing}")
    if cand["ic50_nM"] <= 0:
        raise ValueError("ic50_nM must be > 0")
    if cand["selectivity_ratio"] <= 0:
        raise ValueError("selectivity_ratio must be > 0")
    if cand["herg_ic50_uM"] <= 0:
        raise ValueError("herg_ic50_uM must be > 0")
    if cand["mw_Da"] <= 0:
        raise ValueError("mw_Da must be > 0")


# ---------------------------------------------------------------------------
# Scoring entry point
# ---------------------------------------------------------------------------


def _score_candidate(cand: dict) -> tuple[float, float, float, float, float, float]:
    """Return (potency, selectivity, adme, safety, dev, composite)."""
    pot = _potency_score(cand["ic50_nM"])
    sel = _selectivity_score(cand["selectivity_ratio"])
    adme = _adme_score(cand["f_oral_pct"], cand["t_half_h"])
    saf = _safety_score(cand["herg_ic50_uM"], cand["ames_positive"])
    dev = _developability_score(cand["logP"], cand["mw_Da"])
    composite = pot + sel + adme + saf + dev
    return pot, sel, adme, saf, dev, composite


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def rank_drug_candidates(candidates: list[dict]) -> list[CandidateRankingResult]:
    """Rank drug candidates by composite score (highest first).

    Parameters
    ----------
    candidates:
        List of candidate dicts, each containing the required keys.

    Returns
    -------
    List of CandidateRankingResult sorted by composite_score descending.
    Raises ValueError for empty list or missing/invalid fields.
    """
    if not candidates:
        raise ValueError("candidates list must not be empty")

    # Validate all
    for cand in candidates:
        _validate_candidate(cand)

    # Score all
    scored: list[tuple[float, dict]] = []
    for cand in candidates:
        pot, sel, adme, saf, dev, composite = _score_candidate(cand)
        scored.append((composite, pot, sel, adme, saf, dev, cand))

    # Sort descending by composite
    scored.sort(key=lambda x: x[0], reverse=True)

    results: list[CandidateRankingResult] = []
    for rank_idx, (composite, pot, sel, adme, saf, dev, cand) in enumerate(scored, 1):
        notes = _build_notes(pot, sel, adme, saf, dev, cand["ames_positive"])
        results.append(
            CandidateRankingResult(
                drug_name=cand["drug_name"],
                ic50_nM=float(cand["ic50_nM"]),
                selectivity_ratio=float(cand["selectivity_ratio"]),
                f_oral_pct=float(cand["f_oral_pct"]),
                t_half_h=float(cand["t_half_h"]),
                herg_ic50_uM=float(cand["herg_ic50_uM"]),
                ames_positive=bool(cand["ames_positive"]),
                logP=float(cand["logP"]),
                mw_Da=float(cand["mw_Da"]),
                potency_score=pot,
                selectivity_score=sel,
                adme_score=adme,
                safety_score=saf,
                developability_score=dev,
                composite_score=composite,
                rank=rank_idx,
                go_no_go=_go_no_go(composite),
                notes=notes,
            )
        )
    return results


def compare_two_candidates(candidate_a: dict, candidate_b: dict) -> dict:
    """Compare two drug candidates dimension by dimension.

    Returns
    -------
    dict with keys: winner (drug_name), margin (score diff), comparison (per-dim).
    """
    _validate_candidate(candidate_a)
    _validate_candidate(candidate_b)

    pot_a, sel_a, adme_a, saf_a, dev_a, comp_a = _score_candidate(candidate_a)
    pot_b, sel_b, adme_b, saf_b, dev_b, comp_b = _score_candidate(candidate_b)

    winner = candidate_a["drug_name"] if comp_a >= comp_b else candidate_b["drug_name"]
    margin = abs(comp_a - comp_b)

    comparison = {
        "potency": {"A": pot_a, "B": pot_b},
        "selectivity": {"A": sel_a, "B": sel_b},
        "adme": {"A": adme_a, "B": adme_b},
        "safety": {"A": saf_a, "B": saf_b},
        "developability": {"A": dev_a, "B": dev_b},
        "composite": {"A": comp_a, "B": comp_b},
    }

    return {
        "winner": winner,
        "margin": margin,
        "comparison": comparison,
    }
