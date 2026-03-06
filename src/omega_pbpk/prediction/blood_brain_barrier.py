"""BBB permeability prediction from physicochemical descriptors (Phase 201).

Clark 2003 log(PS) model with CNS MPO scoring (Wager 2010).
No ODE dynamics — purely property-based prediction.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["BBBPredictionResult", "predict_bbb_permeability", "screen_cns_candidates"]


@dataclass(frozen=True)
class BBBPredictionResult:
    """Predicted BBB permeability and CNS drug-likeness."""

    compound_name: str
    smiles: str
    mw: float
    logP: float
    psa: float
    n_hbd: int
    log_ps: float           # log10(PS), PS = permeability-surface area product (cm/s)
    ps_cm_per_s: float      # permeability-surface area product (cm/s)
    bbb_score: float        # 0–100, higher = better CNS penetration
    penetration_class: str  # "excellent", "good", "moderate", "poor", "none"
    cns_mpo_score: float    # CNS MPO score (0–6, Pfizer criteria)
    pgp_substrate_risk: bool  # P-gp efflux risk flag
    lipinski_cns: bool      # MW<400, logP 1–5, PSA<60, HBD<3
    notes: str


def _compute_log_ps(logP: float, psa: float, n_hbd: int) -> float:
    """Clark 2003 log(PS) model."""
    raw = 0.152 * logP - 0.0148 * psa - 0.139 * n_hbd - 0.508
    return float(max(-6.0, min(0.0, raw)))


def _compute_bbb_score(logP: float, psa: float, mw: float, n_hbd: int) -> float:
    """Heuristic BBB score 0–100."""
    score = 100.0
    score -= max(0.0, psa - 60.0) * 0.5      # PSA penalty
    score -= max(0.0, mw - 400.0) * 0.1      # MW penalty
    score -= max(0.0, n_hbd - 2) * 10.0      # HBD penalty
    score -= max(0.0, logP - 5.0) * 5.0      # excess logP penalty
    score += min(logP, 3.0) * 3.0            # moderate logP bonus
    return float(max(0.0, min(100.0, score)))


def _penetration_class(bbb_score: float) -> str:
    if bbb_score >= 80.0:
        return "excellent"
    if bbb_score >= 60.0:
        return "good"
    if bbb_score >= 40.0:
        return "moderate"
    if bbb_score >= 20.0:
        return "poor"
    return "none"


def _cns_mpo(logP: float, psa: float, mw: float, n_hbd: int, pka: float) -> float:
    """Simplified Wager 2010 CNS MPO score (0–6)."""
    score = 0
    if logP <= 3.0:
        score += 1
    if logP >= 1.0:
        score += 1
    if psa <= 90.0:
        score += 1
    if mw <= 360.0:
        score += 1
    if n_hbd <= 1:
        score += 1
    if pka <= 8.0:
        score += 1
    return float(score)


def predict_bbb_permeability(
    compound_name: str,
    smiles: str,
    mw: float,
    logP: float,
    psa: float = 60.0,
    n_hbd: int = 1,
    n_hba: int = 3,
    pka: float = 7.0,
) -> BBBPredictionResult:
    """Predict BBB permeability from physicochemical descriptors.

    Parameters
    ----------
    compound_name : str
        Human-readable compound name.
    smiles : str
        SMILES string (used for P-gp risk heuristic and validation).
    mw : float
        Molecular weight (g/mol).
    logP : float
        Lipophilicity (log P octanol/water).
    psa : float
        Polar surface area (Å²).
    n_hbd : int
        Number of hydrogen bond donors.
    n_hba : int
        Number of hydrogen bond acceptors.
    pka : float
        Most basic pKa (used for CNS MPO criterion 6).

    Returns
    -------
    BBBPredictionResult
    """
    if mw <= 0:
        raise ValueError(f"mw must be positive, got {mw}")
    if not smiles or not smiles.strip():
        raise ValueError("smiles must be a non-empty string")

    log_ps = _compute_log_ps(logP, psa, n_hbd)
    ps_cm_per_s = 10.0**log_ps

    bbb_score = _compute_bbb_score(logP, psa, mw, n_hbd)
    pen_class = _penetration_class(bbb_score)
    mpo = _cns_mpo(logP, psa, mw, n_hbd, pka)

    # P-gp substrate risk: large MW, basic nitrogen present, lipophilic
    pgp_risk = bool((mw > 400) and ("N" in smiles) and (logP > 2.0))

    # CNS Lipinski subset: MW<400, logP 1–5, PSA<60, HBD<3
    lipinski_cns = bool((mw < 400) and (1.0 <= logP <= 5.0) and (psa < 60.0) and (n_hbd < 3))

    notes_parts: list[str] = []
    if psa > 120.0:
        notes_parts.append("Very high PSA; CNS penetration unlikely")
    if mw > 500.0:
        notes_parts.append("MW exceeds CNS drug-like threshold")
    if pgp_risk:
        notes_parts.append("P-gp efflux risk detected")
    if mpo >= 4.0:
        notes_parts.append(f"CNS MPO {mpo:.0f}/6 — favorable")
    notes = "; ".join(notes_parts) if notes_parts else "Within acceptable CNS parameter space"

    return BBBPredictionResult(
        compound_name=compound_name,
        smiles=smiles,
        mw=mw,
        logP=logP,
        psa=psa,
        n_hbd=n_hbd,
        log_ps=log_ps,
        ps_cm_per_s=ps_cm_per_s,
        bbb_score=bbb_score,
        penetration_class=pen_class,
        cns_mpo_score=mpo,
        pgp_substrate_risk=pgp_risk,
        lipinski_cns=lipinski_cns,
        notes=notes,
    )


def screen_cns_candidates(compounds: list[dict]) -> list[BBBPredictionResult]:
    """Screen a list of compounds for CNS penetration potential.

    Parameters
    ----------
    compounds : list[dict]
        Each dict must have keys: "name", "smiles", "mw", "logP".
        Optional keys: "psa", "n_hbd", "n_hba", "pka".

    Returns
    -------
    list[BBBPredictionResult]
        Sorted by bbb_score descending (best CNS candidates first).
    """
    results: list[BBBPredictionResult] = []
    for c in compounds:
        result = predict_bbb_permeability(
            compound_name=c["name"],
            smiles=c["smiles"],
            mw=c["mw"],
            logP=c["logP"],
            psa=c.get("psa", 60.0),
            n_hbd=c.get("n_hbd", 1),
            n_hba=c.get("n_hba", 3),
            pka=c.get("pka", 7.0),
        )
        results.append(result)
    results.sort(key=lambda r: r.bbb_score, reverse=True)
    return results
