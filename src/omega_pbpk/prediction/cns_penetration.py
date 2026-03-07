"""CNS drug penetration predictor using physicochemical descriptors (Phase 292).

Implements the CNS MPO (Multiparameter Optimization) score from Wager et al. 2010
(J. Med. Chem. 53:5359-5374) and estimates blood-brain barrier penetration.

Reference:
    Wager TT et al. (2010) Moving beyond rules: the development of a central nervous
    system multiparameter optimization (MPO) design tool to enhance the alignment of
    drug properties and CNS penetration. ACS Chem Neurosci 1:435-449.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = [
    "BBBPenetrationResult",
    "compute_mpo_cns",
    "predict_bbb_penetration",
    "batch_cns_screen",
]


@dataclass(frozen=True)
class BBBPenetrationResult:
    """Result of CNS penetration prediction."""

    logP: float
    logD74: float
    mw: float
    hbd: int
    pka: float
    tpsa: float
    mpo_score: float  # 0-6, sum of desirability functions
    kp_uu_brain: float  # estimated unbound brain/plasma ratio
    bbb_class: str  # 'high' / 'moderate' / 'low'
    cns_exposure: float  # Kp_uu * fu_plasma
    component_scores: dict[str, float] = field(default_factory=dict)
    notes: str = ""


def _logP_desirability(logP: float) -> float:
    """logP desirability: 1 if <=3, linear decrease 3-5, 0 if >5."""
    if logP <= 3.0:
        return 1.0
    if logP >= 5.0:
        return 0.0
    return (5.0 - logP) / 2.0


def _logD74_desirability(logD74: float) -> float:
    """logD74 desirability: 1 if <=2, linear decrease 2-4, 0 if >4."""
    if logD74 <= 2.0:
        return 1.0
    if logD74 >= 4.0:
        return 0.0
    return (4.0 - logD74) / 2.0


def _mw_desirability(mw: float) -> float:
    """MW desirability: 1 if <=360, linear decrease 360-500, 0 if >500."""
    if mw <= 360.0:
        return 1.0
    if mw >= 500.0:
        return 0.0
    return (500.0 - mw) / 140.0


def _hbd_desirability(hbd: int) -> float:
    """HBD desirability: 1 if <=0, linear decrease 0-3, 0 if >3."""
    if hbd <= 0:
        return 1.0
    if hbd >= 3:
        return 0.0
    return (3.0 - hbd) / 3.0


def _pka_desirability(pka: float) -> float:
    """pKa desirability: 1 if pKa in [8, 10], 0 otherwise (step function)."""
    if 8.0 <= pka <= 10.0:
        return 1.0
    return 0.0


def _tpsa_desirability(tpsa: float) -> float:
    """TPSA desirability: 1 if <=40, linear decrease 40-90, 0 if >90."""
    if tpsa <= 40.0:
        return 1.0
    if tpsa >= 90.0:
        return 0.0
    return (90.0 - tpsa) / 50.0


def compute_mpo_cns(
    logP: float,
    logD74: float,
    mw: float,
    hbd: int,
    pka: float,
    tpsa: float,
) -> float:
    """Compute CNS MPO score (Wager et al. 2010).

    Six desirability functions, each in [0, 1], summed to give a total score
    in [0, 6]. Higher scores indicate better CNS drug-likeness.

    Parameters
    ----------
    logP : float
        Calculated lipophilicity (log P octanol/water).
    logD74 : float
        Distribution coefficient at pH 7.4.
    mw : float
        Molecular weight (g/mol), must be > 0.
    hbd : int
        Number of hydrogen bond donors (>= 0).
    pka : float
        Most basic pKa of the compound.
    tpsa : float
        Topological polar surface area (A^2), must be >= 0.

    Returns
    -------
    float
        CNS MPO score in [0, 6].
    """
    if mw <= 0:
        raise ValueError(f"mw must be > 0, got {mw}")
    if hbd < 0:
        raise ValueError(f"hbd must be >= 0, got {hbd}")
    if tpsa < 0:
        raise ValueError(f"tpsa must be >= 0, got {tpsa}")

    d_logP = _logP_desirability(logP)
    d_logD = _logD74_desirability(logD74)
    d_mw = _mw_desirability(mw)
    d_hbd = _hbd_desirability(hbd)
    d_pka = _pka_desirability(pka)
    d_tpsa = _tpsa_desirability(tpsa)

    return float(d_logP + d_logD + d_mw + d_hbd + d_pka + d_tpsa)


def _sigmoid(x: float) -> float:
    """Logistic sigmoid function."""
    return 1.0 / (1.0 + math.exp(-x))


def predict_bbb_penetration(
    logP: float,
    logD74: float,
    mw: float,
    hbd: int,
    pka: float,
    tpsa: float,
    fu_plasma: float = 0.1,
) -> BBBPenetrationResult:
    """Predict BBB penetration from physicochemical descriptors.

    Parameters
    ----------
    logP : float
        Calculated lipophilicity.
    logD74 : float
        Distribution coefficient at pH 7.4.
    mw : float
        Molecular weight (g/mol), must be > 0.
    hbd : int
        Number of hydrogen bond donors (>= 0).
    pka : float
        Most basic pKa.
    tpsa : float
        Topological polar surface area (A^2), must be >= 0.
    fu_plasma : float
        Unbound fraction in plasma (0 < fu_plasma <= 1). Default 0.1.

    Returns
    -------
    BBBPenetrationResult
    """
    if mw <= 0:
        raise ValueError(f"mw must be > 0, got {mw}")
    if hbd < 0:
        raise ValueError(f"hbd must be >= 0, got {hbd}")
    if tpsa < 0:
        raise ValueError(f"tpsa must be >= 0, got {tpsa}")
    if not (0 < fu_plasma <= 1.0):
        raise ValueError(f"fu_plasma must be in (0, 1], got {fu_plasma}")

    mpo_score = compute_mpo_cns(logP, logD74, mw, hbd, pka, tpsa)

    # Kp_uu estimated via sigmoid centered at MPO=3
    kp_uu_brain = _sigmoid(mpo_score - 3.0)

    # BBB penetration class
    if mpo_score >= 4.0:
        bbb_class = "high"
    elif mpo_score >= 2.5:
        bbb_class = "moderate"
    else:
        bbb_class = "low"

    cns_exposure = kp_uu_brain * fu_plasma

    # Component scores dictionary
    component_scores = {
        "logP": _logP_desirability(logP),
        "logD74": _logD74_desirability(logD74),
        "mw": _mw_desirability(mw),
        "hbd": _hbd_desirability(hbd),
        "pka": _pka_desirability(pka),
        "tpsa": _tpsa_desirability(tpsa),
    }

    # Build notes
    notes_parts: list[str] = []
    if mpo_score >= 4.0:
        notes_parts.append(f"MPO={mpo_score:.2f}/6 — good CNS drug-likeness")
    elif mpo_score >= 2.5:
        notes_parts.append(f"MPO={mpo_score:.2f}/6 — moderate CNS penetration")
    else:
        notes_parts.append(f"MPO={mpo_score:.2f}/6 — poor CNS penetration expected")
    if tpsa > 90.0:
        notes_parts.append("High TPSA limits passive diffusion")
    if mw > 500.0:
        notes_parts.append("High MW reduces CNS penetration")
    if hbd > 3:
        notes_parts.append("High HBD count unfavorable for BBB")
    notes = "; ".join(notes_parts)

    return BBBPenetrationResult(
        logP=logP,
        logD74=logD74,
        mw=mw,
        hbd=hbd,
        pka=pka,
        tpsa=tpsa,
        mpo_score=mpo_score,
        kp_uu_brain=kp_uu_brain,
        bbb_class=bbb_class,
        cns_exposure=cns_exposure,
        component_scores=component_scores,
        notes=notes,
    )


def batch_cns_screen(compounds: list[dict]) -> list[BBBPenetrationResult]:
    """Screen multiple compounds for CNS penetration.

    Parameters
    ----------
    compounds : list[dict]
        Each dict must contain:
        - "logP" (float)
        - "logD74" (float)
        - "mw" (float)
        - "hbd" (int)
        - "pka" (float)
        - "tpsa" (float)
        Optional:
        - "fu_plasma" (float, default 0.1)

    Returns
    -------
    list[BBBPenetrationResult]
        Sorted by mpo_score descending (best CNS candidates first).
    """
    results: list[BBBPenetrationResult] = []
    for c in compounds:
        result = predict_bbb_penetration(
            logP=c["logP"],
            logD74=c["logD74"],
            mw=c["mw"],
            hbd=c["hbd"],
            pka=c["pka"],
            tpsa=c["tpsa"],
            fu_plasma=c.get("fu_plasma", 0.1),
        )
        results.append(result)

    results.sort(key=lambda r: r.mpo_score, reverse=True)
    return results
