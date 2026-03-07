"""Phase 352 — Drug-Induced Phospholipidosis (DIPL) Predictor.

Predicts drug-induced phospholipidosis risk from physicochemical properties
using cationic amphiphilic drug (CAD) scoring criteria.
"""

from __future__ import annotations

from dataclasses import dataclass

_VALID_DRUG_TYPES = {"acid", "base", "neutral", "zwitterion"}


@dataclass
class PhospholipidosisResult:
    """Result of phospholipidosis risk prediction."""

    drug_name: str
    logP: float
    pKa: float
    mw_Da: float
    cad_score: float
    dipl_risk_score: float
    dipl_risk_class: str
    lysosomal_accumulation_index: float
    is_cad_drug: bool
    structural_alerts: list
    mitigation_note: str
    notes: str


def _compute_cad_score(
    logP: float,
    pKa: float,
    drug_type: str,
    mw_Da: float,
    n_rings: int,
    n_positive_charges: int,
    has_long_alkyl_chain: bool,
) -> tuple[float, list[str]]:
    """Compute raw CAD score and list of structural alerts.

    Returns:
        Tuple of (raw_score, alerts).
    """
    alerts: list[str] = []
    score = 0.0

    # logP contribution (max 25)
    logp_contrib = min(25.0, max(0.0, logP * 5.0))
    if logp_contrib > 0:
        score += logp_contrib
        alerts.append(f"Lipophilic (logP={logP:.2f}, contribution={logp_contrib:.1f})")

    # Basic pKa (20 if pKa > 7.4 and basic/zwitterion)
    if pKa > 7.4 and drug_type in ("base", "zwitterion"):
        score += 20.0
        alerts.append(f"Basic nitrogen (pKa={pKa:.2f} > 7.4, drug_type={drug_type})")

    # Molecular weight
    if mw_Da > 300:
        score += 10.0
        alerts.append(f"High MW ({mw_Da:.1f} Da > 300)")

    # Aromatic rings (max 15)
    ring_contrib = min(15.0, n_rings * 5.0)
    if ring_contrib > 0:
        score += ring_contrib
        alerts.append(f"Aromatic rings (n={n_rings}, contribution={ring_contrib:.1f})")

    # Positive charges (max 15)
    charge_contrib = min(15.0, n_positive_charges * 10.0)
    if charge_contrib > 0:
        score += charge_contrib
        alerts.append(
            f"Positive charges (n={n_positive_charges}, contribution={charge_contrib:.1f})"
        )

    # Long alkyl chain
    if has_long_alkyl_chain:
        score += 15.0
        alerts.append("Long alkyl chain (>= 6 carbons)")

    return score, alerts


def _dipl_risk_class(dipl_risk_score: float) -> str:
    """Classify DIPL risk from normalized score."""
    if dipl_risk_score < 25.0:
        return "low"
    if dipl_risk_score <= 60.0:
        return "moderate"
    return "high"


def _lysosomal_accumulation_index(
    lysosomal_trapping_pct: float,
    logP: float,
) -> float:
    """Compute lysosomal accumulation index (LAI), clamped to [0, 200]."""
    if logP > 0:
        lai = lysosomal_trapping_pct * (1.0 + logP / 10.0)
    else:
        lai = lysosomal_trapping_pct
    return max(0.0, min(200.0, lai))


def _mitigation_note(dipl_risk_class: str, is_cad: bool) -> str:
    """Generate mitigation note based on risk class."""
    if dipl_risk_class == "high":
        return (
            "High DIPL risk. Consider structural modification to reduce basicity or lipophilicity. "
            "Monitor lysosomal phospholipid levels in preclinical studies (2-week rat study). "
            "Include lamellar body count as histopathology endpoint."
        )
    if dipl_risk_class == "moderate":
        return (
            "Moderate DIPL risk (CAD drug). "
            "Conduct in vitro phospholipidosis assay (NR8383 or cell-based). "
            "Monitor lysosomal markers in toxicology studies."
        )
    if is_cad:
        return (
            "Low DIPL risk score, but drug exhibits some CAD characteristics. "
            "Routine monitoring recommended."
        )
    return "Low DIPL risk. No specific phospholipidosis monitoring required."


def predict_phospholipidosis(
    drug_name: str,
    logP: float,
    pKa: float,
    drug_type: str,
    mw_Da: float,
    n_rings: int,
    n_positive_charges: int,
    has_long_alkyl_chain: bool,
    lysosomal_trapping_pct: float,
) -> PhospholipidosisResult:
    """Predict drug-induced phospholipidosis (DIPL) risk.

    Args:
        drug_name: Name of the drug compound.
        logP: Partition coefficient; must be in [-5, 10].
        pKa: Acid-base dissociation constant.
        drug_type: Ionization type ("acid", "base", "neutral", "zwitterion").
        mw_Da: Molecular weight in Daltons (must be > 0).
        n_rings: Number of aromatic rings (must be >= 0).
        n_positive_charges: Number of positive charges at physiological pH (must be >= 0).
        has_long_alkyl_chain: True if drug contains an alkyl chain of >= 6 carbons.
        lysosomal_trapping_pct: Estimated percentage of drug trapped in lysosomes [0, 100].

    Returns:
        PhospholipidosisResult with DIPL risk assessment.

    Raises:
        ValueError: On invalid input parameters.
    """
    if not (-5.0 <= logP <= 10.0):
        raise ValueError(f"logP must be in [-5, 10]; got {logP}")
    if mw_Da <= 0:
        raise ValueError(f"mw_Da must be > 0; got {mw_Da}")
    if n_rings < 0:
        raise ValueError(f"n_rings must be >= 0; got {n_rings}")
    if n_positive_charges < 0:
        raise ValueError(f"n_positive_charges must be >= 0; got {n_positive_charges}")
    if not (0.0 <= lysosomal_trapping_pct <= 100.0):
        raise ValueError(
            f"lysosomal_trapping_pct must be in [0, 100]; got {lysosomal_trapping_pct}"
        )
    if drug_type not in _VALID_DRUG_TYPES:
        raise ValueError(f"drug_type must be one of {sorted(_VALID_DRUG_TYPES)}; got {drug_type!r}")

    raw_score, alerts = _compute_cad_score(
        logP=logP,
        pKa=pKa,
        drug_type=drug_type,
        mw_Da=mw_Da,
        n_rings=n_rings,
        n_positive_charges=n_positive_charges,
        has_long_alkyl_chain=has_long_alkyl_chain,
    )

    dipl_score = min(100.0, raw_score)
    risk_class = _dipl_risk_class(dipl_score)
    is_cad = dipl_score > 25.0

    lai = _lysosomal_accumulation_index(lysosomal_trapping_pct, logP)

    mitigation = _mitigation_note(risk_class, is_cad)

    notes = (
        f"Raw CAD score: {raw_score:.1f}; "
        f"DIPL risk score: {dipl_score:.1f}; "
        f"Risk class: {risk_class}; "
        f"LAI: {lai:.2f}; "
        f"Structural alerts: {len(alerts)}"
    )

    return PhospholipidosisResult(
        drug_name=drug_name,
        logP=logP,
        pKa=pKa,
        mw_Da=mw_Da,
        cad_score=round(raw_score, 4),
        dipl_risk_score=round(dipl_score, 4),
        dipl_risk_class=risk_class,
        lysosomal_accumulation_index=round(lai, 4),
        is_cad_drug=is_cad,
        structural_alerts=alerts,
        mitigation_note=mitigation,
        notes=notes,
    )
