"""Scaffold hopping scorer — predicts likelihood that a structural modification
retains activity (Phase 806).

Scoring is based on physicochemical property similarity, bioisostere type,
and known active scaffold information.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ScaffoldHoppingResult:
    """Result of scaffold hopping analysis."""

    original_compound: str
    analog_compound: str
    mw_delta: float
    logp_delta: float
    psa_delta: float
    hbd_delta: int
    hba_delta: int
    tanimoto_similarity: float  # simplified structural similarity (0-1)
    property_similarity: float  # similarity in ADME properties
    bioisostere_score: float  # 0-100: likelihood of retaining activity via bioisostere
    scaffold_hop_feasibility: str  # "high", "moderate", "low"
    predicted_activity_retention_pct: float  # % chance activity is retained
    adme_improvement: str  # "improved", "neutral", "worsened"
    notes: str


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value to [lo, hi]."""
    return max(lo, min(hi, value))


def score_scaffold_hop(
    original_compound: str,
    analog_compound: str,
    # Original properties
    orig_mw: float,
    orig_logp: float,
    orig_psa: float,
    orig_hbd: int,
    orig_hba: int,
    # Analog properties
    analog_mw: float,
    analog_logp: float,
    analog_psa: float,
    analog_hbd: int,
    analog_hba: int,
    # Bioisostere information
    bioisostere_type: str = "none",
    known_active_scaffold: bool = False,
) -> ScaffoldHoppingResult:
    """Score the feasibility of a scaffold hop from original to analog compound.

    Parameters
    ----------
    original_compound : str
        Name or identifier of the original compound.
    analog_compound : str
        Name or identifier of the analog compound.
    orig_mw, orig_logp, orig_psa : float
        Physicochemical properties of the original compound.
    orig_hbd, orig_hba : int
        H-bond donors and acceptors of the original.
    analog_mw, analog_logp, analog_psa : float
        Physicochemical properties of the analog compound.
    analog_hbd, analog_hba : int
        H-bond donors and acceptors of the analog.
    bioisostere_type : str
        Type of bioisosteric replacement:
        "ring_replacement", "group_replacement", "chain_extension", or "none".
    known_active_scaffold : bool
        Whether the analog contains a known active scaffold.

    Returns
    -------
    ScaffoldHoppingResult
        Scoring result with feasibility classification.
    """
    if orig_mw <= 0:
        raise ValueError("orig_mw must be > 0")
    if analog_mw <= 0:
        raise ValueError("analog_mw must be > 0")
    if not original_compound or not isinstance(original_compound, str):
        raise ValueError("original_compound must be a non-empty string")
    if not analog_compound or not isinstance(analog_compound, str):
        raise ValueError("analog_compound must be a non-empty string")

    valid_bioisostere_types = {"ring_replacement", "group_replacement", "chain_extension", "none"}
    if bioisostere_type not in valid_bioisostere_types:
        raise ValueError(
            f"bioisostere_type must be one of {valid_bioisostere_types}, got '{bioisostere_type}'"
        )

    # Property deltas
    mw_delta = analog_mw - orig_mw
    logp_delta = analog_logp - orig_logp
    psa_delta = analog_psa - orig_psa
    hbd_delta = analog_hbd - orig_hbd
    hba_delta = analog_hba - orig_hba

    # Property similarity: exp(-sqrt(sum of squared normalized deltas))
    norm_mw = mw_delta / 100.0
    norm_logp = logp_delta
    norm_psa = psa_delta / 30.0
    norm_hbd = float(hbd_delta)
    sq_sum = norm_mw**2 + norm_logp**2 + norm_psa**2 + norm_hbd**2
    property_similarity = _clamp(math.exp(-math.sqrt(sq_sum)), 0.0, 1.0)

    # Tanimoto (physicochemical): 1 / (1 + |dMW/300| + |dLogP| + |dPSA/50|)
    tanimoto_similarity = 1.0 / (
        1.0 + abs(mw_delta / 300.0) + abs(logp_delta) + abs(psa_delta / 50.0)
    )
    tanimoto_similarity = _clamp(tanimoto_similarity, 0.0, 1.0)

    # Bioisostere bonus
    bioisostere_bonuses = {
        "ring_replacement": 20.0,
        "group_replacement": 15.0,
        "chain_extension": 10.0,
        "none": 0.0,
    }
    bioisostere_bonus = bioisostere_bonuses[bioisostere_type]

    # Known active scaffold bonus
    scaffold_bonus = 15.0 if known_active_scaffold else 0.0

    # Bioisostere score for reporting
    bioisostere_score = _clamp(bioisostere_bonus + scaffold_bonus, 0.0, 100.0)

    # Activity retention: property_similarity * 50 + bioisostere_bonus + scaffold_bonus
    activity_retention = _clamp(
        property_similarity * 50.0 + bioisostere_bonus + scaffold_bonus,
        0.0,
        100.0,
    )

    # ADME improvement: if psa_delta > 0 (higher PSA, typically better solubility)
    # or logp_delta < 0 (lower logP, better aqueous solubility) → "improved"
    if psa_delta > 0 or logp_delta < 0:
        adme_improvement = "improved"
    elif psa_delta == 0 and logp_delta == 0:
        adme_improvement = "neutral"
    else:
        adme_improvement = "worsened"

    # Scaffold hop feasibility
    if activity_retention > 60.0:
        scaffold_hop_feasibility = "high"
    elif activity_retention >= 40.0:
        scaffold_hop_feasibility = "moderate"
    else:
        scaffold_hop_feasibility = "low"

    # Notes
    notes_parts = []
    if bioisostere_type != "none":
        notes_parts.append(f"Bioisostere: {bioisostere_type} (+{bioisostere_bonus:.0f} pts)")
    if known_active_scaffold:
        notes_parts.append("Known active scaffold (+15 pts)")
    if property_similarity < 0.3:
        notes_parts.append("Low property similarity — significant structural change")
    if abs(mw_delta) > 200:
        notes_parts.append(f"Large MW change ({mw_delta:+.0f} Da)")
    notes = "; ".join(notes_parts) if notes_parts else "Standard scaffold hop analysis"

    return ScaffoldHoppingResult(
        original_compound=original_compound,
        analog_compound=analog_compound,
        mw_delta=mw_delta,
        logp_delta=logp_delta,
        psa_delta=psa_delta,
        hbd_delta=hbd_delta,
        hba_delta=hba_delta,
        tanimoto_similarity=tanimoto_similarity,
        property_similarity=property_similarity,
        bioisostere_score=bioisostere_score,
        scaffold_hop_feasibility=scaffold_hop_feasibility,
        predicted_activity_retention_pct=activity_retention,
        adme_improvement=adme_improvement,
        notes=notes,
    )


def screen_analogs(
    original_compound: str,
    original_props: dict,
    analogs: list[dict],
) -> list[ScaffoldHoppingResult]:
    """Screen a list of analogs for scaffold hopping potential.

    Parameters
    ----------
    original_compound : str
        Name or identifier of the reference compound.
    original_props : dict
        Must contain keys: mw, logp, psa, hbd, hba.
    analogs : list[dict]
        Each analog dict must contain: name, mw, logp, psa, hbd, hba.
        Optional: bioisostere_type, known_active_scaffold.

    Returns
    -------
    list[ScaffoldHoppingResult]
        Sorted by predicted_activity_retention_pct descending.
    """
    if not original_compound or not isinstance(original_compound, str):
        raise ValueError("original_compound must be a non-empty string")
    if not analogs:
        raise ValueError("analogs list must not be empty")

    required_orig_keys = {"mw", "logp", "psa", "hbd", "hba"}
    missing = required_orig_keys - set(original_props.keys())
    if missing:
        raise ValueError(f"original_props missing keys: {missing}")

    results = []
    for analog in analogs:
        name = analog.get("name", "unknown")
        result = score_scaffold_hop(
            original_compound=original_compound,
            analog_compound=name,
            orig_mw=float(original_props["mw"]),
            orig_logp=float(original_props["logp"]),
            orig_psa=float(original_props["psa"]),
            orig_hbd=int(original_props["hbd"]),
            orig_hba=int(original_props["hba"]),
            analog_mw=float(analog.get("mw", original_props["mw"])),
            analog_logp=float(analog.get("logp", original_props["logp"])),
            analog_psa=float(analog.get("psa", original_props["psa"])),
            analog_hbd=int(analog.get("hbd", original_props["hbd"])),
            analog_hba=int(analog.get("hba", original_props["hba"])),
            bioisostere_type=analog.get("bioisostere_type", "none"),
            known_active_scaffold=bool(analog.get("known_active_scaffold", False)),
        )
        results.append(result)

    # Sort by predicted_activity_retention_pct descending
    results.sort(key=lambda r: r.predicted_activity_retention_pct, reverse=True)
    return results


__all__ = ["ScaffoldHoppingResult", "score_scaffold_hop", "screen_analogs"]
