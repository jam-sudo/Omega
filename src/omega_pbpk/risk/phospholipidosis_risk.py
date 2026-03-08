"""Phase 498 — Drug Induced Phospholipidosis Risk.

Predicts risk of drug-induced phospholipidosis (DIPL) from physicochemical properties.
DIPL: drug accumulates in lysosomes, displaces phospholipids -> foamy macrophages.
Risk factors: cationic amphiphilic drugs (CADs), high logP, pKa > 7, ring structures.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PhospholipidosisRiskResult:
    """Result of phospholipidosis risk prediction."""

    drug_name: str
    logP: float
    pKa: float
    mw_Da: float
    ring_count: int
    tertiary_amine: bool
    positive_charge: bool
    risk_score: float
    risk_class: str
    cad_pattern: bool
    lysosomal_trapping_factor: float
    preclinical_flag: bool
    mitigation_strategy: str
    notes: str


def predict_phospholipidosis_risk(
    drug_name: str,
    logP: float,
    pKa: float,
    mw_Da: float,
    ring_count: int = 2,
    tertiary_amine: bool = False,
    positive_charge: bool = False,
) -> PhospholipidosisRiskResult:
    """Predict phospholipidosis risk from physicochemical properties.

    Parameters
    ----------
    drug_name:
        Name of the compound.
    logP:
        Lipophilicity.
    pKa:
        Most basic pKa.
    mw_Da:
        Molecular weight in Daltons.
    ring_count:
        Number of ring structures.
    tertiary_amine:
        Whether the drug contains a tertiary amine.
    positive_charge:
        Whether the drug carries a positive charge at physiological pH.

    Returns
    -------
    PhospholipidosisRiskResult
    """
    # Validation
    if mw_Da <= 0:
        raise ValueError("mw_Da must be > 0")
    if ring_count < 0:
        raise ValueError("ring_count must be >= 0")

    # Risk scoring
    score = 0.0

    # pKa contribution
    if pKa > 8.5:
        score += 30.0
    elif pKa >= 7.0:
        score += 20.0
    elif pKa >= 5.0:
        score += 5.0

    # logP contribution
    if logP > 4.0:
        score += 25.0
    elif logP >= 2.0:
        score += 15.0
    elif logP < 0.0:
        score -= 5.0

    # MW contribution
    if mw_Da > 450.0:
        score += 15.0  # +10 base + 5 extra
    elif mw_Da > 300.0:
        score += 10.0

    # Amphiphilic structure
    if ring_count >= 2 and positive_charge:
        score += 20.0

    # Tertiary amine
    if tertiary_amine:
        score += 15.0

    # Clamp 0-100
    score = max(0.0, min(100.0, score))

    # Risk classification
    if score < 30.0:
        risk_class = "low"
    elif score <= 60.0:
        risk_class = "moderate"
    else:
        risk_class = "high"

    # CAD pattern: cationic amphiphilic drug
    cad_pattern = pKa > 7.0 and logP > 2.0 and ring_count >= 1

    # Lysosomal trapping factor
    if pKa > 7.0:
        raw_ltf = 10.0 ** (pKa - 7.0)
        lysosomal_trapping_factor = max(1.0, min(1000.0, raw_ltf))
    else:
        lysosomal_trapping_factor = 1.0

    preclinical_flag = score > 50.0

    if risk_class == "low":
        mitigation_strategy = "none"
    elif risk_class == "moderate":
        mitigation_strategy = "in_vitro_testing"
    else:
        mitigation_strategy = "avoid_or_redesign"

    notes = (
        f"risk_score={score:.1f} ({risk_class}); "
        f"CAD={cad_pattern}; "
        f"lysosomal_trapping={lysosomal_trapping_factor:.2f}x; "
        f"preclinical_flag={preclinical_flag}"
    )

    return PhospholipidosisRiskResult(
        drug_name=drug_name,
        logP=logP,
        pKa=pKa,
        mw_Da=mw_Da,
        ring_count=ring_count,
        tertiary_amine=tertiary_amine,
        positive_charge=positive_charge,
        risk_score=score,
        risk_class=risk_class,
        cad_pattern=cad_pattern,
        lysosomal_trapping_factor=lysosomal_trapping_factor,
        preclinical_flag=preclinical_flag,
        mitigation_strategy=mitigation_strategy,
        notes=notes,
    )


def screen_phospholipidosis(compounds: list) -> list:
    """Screen multiple compounds for phospholipidosis risk.

    Parameters
    ----------
    compounds:
        List of dicts with keys matching predict_phospholipidosis_risk parameters.

    Returns
    -------
    List of PhospholipidosisRiskResult sorted by risk_score descending.
    """
    results = []
    for c in compounds:
        result = predict_phospholipidosis_risk(
            drug_name=c["drug_name"],
            logP=c["logP"],
            pKa=c["pKa"],
            mw_Da=c["mw_Da"],
            ring_count=c.get("ring_count", 2),
            tertiary_amine=c.get("tertiary_amine", False),
            positive_charge=c.get("positive_charge", False),
        )
        results.append(result)
    results.sort(key=lambda r: r.risk_score, reverse=True)
    return results
