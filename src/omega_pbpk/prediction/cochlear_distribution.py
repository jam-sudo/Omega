"""Phase 915 — Drug Cochlear Distribution.

Predict drug distribution in cochlear fluids (perilymph, endolymph) —
relevant for ototoxicity (aminoglycosides, cisplatin) and intratympanic
drug delivery.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CochlearDistributionResult:
    """Result of cochlear drug distribution prediction."""

    drug_name: str
    logp: float
    mw_Da: float
    route: str
    perilymph_plasma_ratio: float
    predicted_perilymph_conc_ug_mL: float
    plasma_conc_input_mg_L: float
    blb_permeability_class: str
    ototoxicity_risk_score: float
    ototoxicity_risk_level: str
    hair_cell_accumulation_factor: float
    notes: str


def predict_cochlear_distribution(
    drug_name: str,
    logp: float = 1.0,
    mw_Da: float = 500.0,
    plasma_conc_mg_L: float = 1.0,
    route: str = "systemic",
    has_met_channel_entry: bool = False,
    fu_plasma: float = 0.5,
) -> CochlearDistributionResult:
    """Predict cochlear fluid drug distribution.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    logp:
        Partition coefficient (log scale).
    mw_Da:
        Molecular weight in Daltons.
    plasma_conc_mg_L:
        Plasma drug concentration (mg/L).
    route:
        Delivery route — "systemic" or "intratympanic".
    has_met_channel_entry:
        True if drug enters hair cells via MET channels (aminoglycoside-type).
    fu_plasma:
        Unbound fraction in plasma.

    Returns
    -------
    CochlearDistributionResult
    """
    if mw_Da <= 0:
        raise ValueError("mw_Da must be > 0")
    if plasma_conc_mg_L <= 0:
        raise ValueError("plasma_conc_mg_L must be > 0")
    if route not in {"systemic", "intratympanic"}:
        raise ValueError("route must be 'systemic' or 'intratympanic'")

    # Blood-labyrinth barrier permeability classification
    blb_score = logp - mw_Da / 400.0
    if blb_score > 0.5:
        blb_permeability_class = "high"
        base_ratio = 0.3
    elif blb_score > -1.0:
        blb_permeability_class = "moderate"
        base_ratio = 0.05
    else:
        blb_permeability_class = "low"
        base_ratio = 0.005

    perilymph_plasma_ratio = base_ratio

    # Intratympanic delivery bypasses BLB
    if route == "intratympanic":
        perilymph_plasma_ratio *= 10.0

    # MET channel entry increases cochlear uptake
    if has_met_channel_entry:
        perilymph_plasma_ratio *= 2.0

    # Clamp ratio
    perilymph_plasma_ratio = max(0.001, min(5.0, perilymph_plasma_ratio))

    # Predicted perilymph concentration (ug/mL = mg/L * 1000 * fu * ratio)
    predicted_perilymph_conc_ug_mL = perilymph_plasma_ratio * plasma_conc_mg_L * fu_plasma * 1000.0

    # Hair cell accumulation factor
    if has_met_channel_entry:
        hair_cell_accumulation_factor = 50.0
    else:
        hair_cell_accumulation_factor = 1.0 + logp * 2.0

    hair_cell_accumulation_factor = max(1.0, min(100.0, hair_cell_accumulation_factor))

    # Ototoxicity risk score
    mw_logp_penalty = 20.0 if (mw_Da > 400 and logp < 0) else 0.0
    ototoxicity_risk_score = min(
        100.0,
        perilymph_plasma_ratio * 100.0 + hair_cell_accumulation_factor * 0.5 + mw_logp_penalty,
    )

    # Risk level
    if ototoxicity_risk_score > 60:
        ototoxicity_risk_level = "high"
    elif ototoxicity_risk_score > 30:
        ototoxicity_risk_level = "moderate"
    else:
        ototoxicity_risk_level = "low"

    # Notes — priority: high risk > MET channel > default
    if ototoxicity_risk_level == "high":
        notes = "High ototoxicity risk — monitor audiometry. Limit duration and dose."
    elif has_met_channel_entry:
        notes = "MET channel entry: aminoglycoside-type ototoxicity mechanism detected"
    else:
        notes = "Low cochlear accumulation"

    return CochlearDistributionResult(
        drug_name=drug_name,
        logp=logp,
        mw_Da=mw_Da,
        route=route,
        perilymph_plasma_ratio=perilymph_plasma_ratio,
        predicted_perilymph_conc_ug_mL=predicted_perilymph_conc_ug_mL,
        plasma_conc_input_mg_L=plasma_conc_mg_L,
        blb_permeability_class=blb_permeability_class,
        ototoxicity_risk_score=ototoxicity_risk_score,
        ototoxicity_risk_level=ototoxicity_risk_level,
        hair_cell_accumulation_factor=hair_cell_accumulation_factor,
        notes=notes,
    )


def screen_ototoxic_drugs(candidates: list[dict]) -> list[CochlearDistributionResult]:
    """Screen a list of drug candidates for ototoxicity risk.

    Parameters
    ----------
    candidates:
        List of dicts with keys: "name", "logp", "mw_Da",
        and optionally "has_met_channel_entry" (bool).

    Returns
    -------
    List of CochlearDistributionResult sorted by ototoxicity_risk_score descending.
    """
    results: list[CochlearDistributionResult] = []
    for c in candidates:
        result = predict_cochlear_distribution(
            drug_name=c["name"],
            logp=c.get("logp", 1.0),
            mw_Da=c.get("mw_Da", 500.0),
            has_met_channel_entry=c.get("has_met_channel_entry", False),
        )
        results.append(result)

    results.sort(key=lambda r: r.ototoxicity_risk_score, reverse=True)
    return results
