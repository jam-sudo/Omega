"""Phase 255: Drug membrane permeability classifier.

Classifies drug membrane permeability and predicts absorption class
based on Caco-2 and PAMPA data using empirical rule-based models.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PermeabilityClassificationResult:
    """Result of permeability classification for a single compound."""

    drug_name: str
    mw: float
    logp: float
    psa_A2: float
    hbd_count: int
    papp_caco2_cm_s: float
    papp_pampa_cm_s: float
    permeability_class_caco2: str  # "high", "moderate", "low"
    permeability_class_pampa: str  # "high", "moderate", "low"
    absorption_class: str  # "class1", "class2", "class3"
    efflux_risk: bool
    passive_permeability_score: float  # 0-100
    bcs_permeability_class: str  # "high" or "low"
    notes: str


def _classify_caco2(papp: float) -> str:
    """Classify Caco-2 permeability based on Papp thresholds."""
    if papp > 10e-6:
        return "high"
    elif papp >= 1e-6:
        return "moderate"
    else:
        return "low"


def _classify_pampa(papp: float) -> str:
    """Classify PAMPA permeability based on Pe thresholds."""
    if papp > 5e-6:
        return "high"
    elif papp >= 0.5e-6:
        return "moderate"
    else:
        return "low"


def _determine_absorption_class(caco2_class: str, pampa_class: str) -> str:
    """Determine overall absorption class from both assay classifications."""
    if caco2_class == "high" and pampa_class == "high":
        return "class1"
    elif caco2_class == "low" and pampa_class == "low":
        return "class3"
    else:
        return "class2"


def _determine_bcs_permeability(caco2_class: str, pampa_class: str) -> str:
    """BCS permeability class: high if both assays >= moderate, low otherwise."""
    high_or_moderate = {"high", "moderate"}
    if caco2_class in high_or_moderate and pampa_class in high_or_moderate:
        return "high"
    return "low"


def classify_permeability(
    drug_name: str,
    mw: float,
    logp: float,
    psa_A2: float,
    hbd_count: int = 2,
) -> PermeabilityClassificationResult:
    """Classify membrane permeability and predict absorption class.

    Parameters
    ----------
    drug_name : str
        Name of the compound.
    mw : float
        Molecular weight (g/mol). Must be > 0.
    logp : float
        Octanol-water partition coefficient. Must be in (-5, 10).
    psa_A2 : float
        Polar surface area (Angstrom^2). Must be >= 0.
    hbd_count : int
        Hydrogen bond donor count. Must be >= 0.

    Returns
    -------
    PermeabilityClassificationResult
    """
    if mw <= 0:
        raise ValueError(f"mw must be > 0, got {mw}")
    if not (-5 < logp < 10):
        raise ValueError(f"logp must be in (-5, 10), got {logp}")
    if psa_A2 < 0:
        raise ValueError(f"psa_A2 must be >= 0, got {psa_A2}")
    if hbd_count < 0:
        raise ValueError(f"hbd_count must be >= 0, got {hbd_count}")

    # Empirical Caco-2 model
    log_papp_caco2 = (
        0.5 * logp
        - 0.01 * psa_A2
        - 0.002 * mw
        + 0.1 * (5 - hbd_count)
        - 5.5
    )
    papp_caco2 = 10.0 ** log_papp_caco2  # cm/s

    # PAMPA typically higher (no active transport)
    papp_pampa = papp_caco2 * 1.5  # cm/s

    caco2_class = _classify_caco2(papp_caco2)
    pampa_class = _classify_pampa(papp_pampa)
    absorption_class = _determine_absorption_class(caco2_class, pampa_class)
    bcs_perm = _determine_bcs_permeability(caco2_class, pampa_class)

    # Efflux risk: likely P-gp / BCRP substrate if low logP, high PSA, not too large
    efflux_risk = bool(logp < 2 and psa_A2 > 60 and mw < 500)

    # Passive permeability score 0-100
    raw_score = 50.0 + 10.0 * logp - 0.2 * psa_A2 - 0.05 * (mw - 300)
    passive_permeability_score = min(100.0, max(0.0, raw_score))

    # Build notes
    note_parts = []
    if caco2_class == "high":
        note_parts.append("Excellent Caco-2 permeability predicted.")
    elif caco2_class == "moderate":
        note_parts.append("Moderate Caco-2 permeability predicted.")
    else:
        note_parts.append("Low Caco-2 permeability; absorption may be limited.")

    if efflux_risk:
        note_parts.append("Efflux transporter risk (low logP, high PSA).")
    if psa_A2 > 140:
        note_parts.append("High PSA (>140 A2) likely limits passive permeability.")
    if mw > 500:
        note_parts.append("MW >500 Da may reduce membrane permeability.")

    notes = " ".join(note_parts) if note_parts else "Permeability classification complete."

    return PermeabilityClassificationResult(
        drug_name=drug_name,
        mw=mw,
        logp=logp,
        psa_A2=psa_A2,
        hbd_count=hbd_count,
        papp_caco2_cm_s=papp_caco2,
        papp_pampa_cm_s=papp_pampa,
        permeability_class_caco2=caco2_class,
        permeability_class_pampa=pampa_class,
        absorption_class=absorption_class,
        efflux_risk=efflux_risk,
        passive_permeability_score=passive_permeability_score,
        bcs_permeability_class=bcs_perm,
        notes=notes,
    )


def screen_compound_series(
    compounds: list[dict],
) -> list[PermeabilityClassificationResult]:
    """Screen a series of compounds and rank by passive permeability score.

    Parameters
    ----------
    compounds : list of dict
        Each dict must have keys: "name", "mw", "logp", "psa_A2".
        Optional key: "hbd_count" (default 2).

    Returns
    -------
    list of PermeabilityClassificationResult, sorted descending by
    passive_permeability_score.
    """
    results = []
    for cpd in compounds:
        result = classify_permeability(
            drug_name=cpd["name"],
            mw=cpd["mw"],
            logp=cpd["logp"],
            psa_A2=cpd["psa_A2"],
            hbd_count=cpd.get("hbd_count", 2),
        )
        results.append(result)

    results.sort(key=lambda r: r.passive_permeability_score, reverse=True)
    return results
