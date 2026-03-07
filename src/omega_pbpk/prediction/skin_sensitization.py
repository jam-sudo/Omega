"""Phase 1036 — Drug Skin Sensitization Prediction."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SkinSensitizationResult:
    drug_name: str
    protein_reactivity_score: float  # 0-100
    skin_penetration_score: float  # 0-100
    metabolism_activation_score: float  # 0-100
    overall_sensitization_score: float  # 0-100 (weighted)
    sensitization_class: str  # "non-sensitizer", "weak", "moderate", "strong", "extreme"
    EC3_estimate_pct: float  # estimated EC3 in LLNA assay
    therapeutic_risk: str  # "low", "moderate", "high"
    structural_alerts: list = field(default_factory=list)
    notes: str = ""


_VALID_CLASSES = {"non-sensitizer", "weak", "moderate", "strong", "extreme"}
_VALID_RISKS = {"low", "moderate", "high"}


def predict_skin_sensitization(
    drug_name: str,
    mw_Da: float,
    logp: float,
    has_electrophile: bool = False,
    has_snh: bool = False,
    has_quinone: bool = False,
    has_epoxide: bool = False,
    has_isocyanate: bool = False,
    topical_use: bool = False,
    concentration_pct: float = 1.0,
) -> SkinSensitizationResult:
    """Predict skin sensitization potential based on structural/physicochemical features."""
    # Validation
    if mw_Da <= 0:
        raise ValueError(f"mw_Da must be > 0, got {mw_Da}")
    if concentration_pct <= 0:
        raise ValueError(f"concentration_pct must be > 0, got {concentration_pct}")
    if not (-5.0 <= logp <= 10.0):
        raise ValueError(f"logp must be in [-5, 10], got {logp}")

    # protein_reactivity_score
    prs = 0.0
    if has_electrophile:
        prs += 40.0
    if has_snh:
        prs += 20.0
    if has_quinone:
        prs += 35.0
    if has_epoxide:
        prs += 45.0
    if has_isocyanate:
        prs += 60.0
    prs = max(0.0, min(100.0, prs))

    # skin_penetration_score
    if 1.0 <= logp <= 4.0:
        sps = 70.0 + 10.0 * (logp - 1.0) / 3.0
    elif logp < 1.0:
        if logp >= -1.0:
            sps = 70.0 * (logp + 1.0) / 2.0
        else:
            sps = 10.0
    else:  # logp > 4.0
        sps = 80.0 - 20.0 * (logp - 4.0) / 2.0
        sps = max(10.0, min(80.0, sps))

    # MW penalty
    mw_factor = max(0.3, 1.0 - (mw_Da - 300.0) * 0.001)
    sps = sps * mw_factor
    sps = max(0.0, min(100.0, sps))

    # metabolism_activation_score
    mas = 10.0
    if has_quinone:
        mas += 30.0
    if has_snh:
        mas += 15.0
    if topical_use:
        mas += 10.0
    mas = max(0.0, min(100.0, mas))

    # overall_sensitization_score
    oss = prs * 0.5 + sps * 0.3 + mas * 0.2
    oss = max(0.0, min(100.0, oss))

    # sensitization_class
    if oss < 10.0:
        sensitization_class = "non-sensitizer"
    elif oss < 30.0:
        sensitization_class = "weak"
    elif oss < 55.0:
        sensitization_class = "moderate"
    elif oss < 75.0:
        sensitization_class = "strong"
    else:
        sensitization_class = "extreme"

    # EC3_estimate_pct
    ec3 = 100.0 / (1.0 + oss)
    ec3 = max(0.1, min(100.0, ec3))

    # therapeutic_risk
    if sensitization_class in ("non-sensitizer", "weak"):
        therapeutic_risk = "low"
    elif sensitization_class == "moderate":
        therapeutic_risk = "moderate"
    else:
        therapeutic_risk = "high"

    # structural_alerts
    alerts: list = []
    if has_electrophile:
        alerts.append("Michael acceptor / electrophile")
    if has_epoxide:
        alerts.append("Epoxide")
    if has_quinone:
        alerts.append("Quinone / pre-quinone")
    if has_isocyanate:
        alerts.append("Isocyanate")
    if has_snh:
        alerts.append("Reactive S-H or N-H")
    if topical_use:
        alerts.append("Topical application (enhanced exposure)")
    if not alerts:
        alerts = ["No structural alerts identified"]

    # notes
    notes = (
        f"{drug_name}: overall sensitization score {oss:.1f}/100 ({sensitization_class}). "
        f"Protein reactivity {prs:.1f}, skin penetration {sps:.1f}, "
        f"metabolism activation {mas:.1f}. "
        f"Estimated EC3 {ec3:.2f}%. "
        f"Therapeutic risk: {therapeutic_risk}."
    )

    return SkinSensitizationResult(
        drug_name=drug_name,
        protein_reactivity_score=prs,
        skin_penetration_score=sps,
        metabolism_activation_score=mas,
        overall_sensitization_score=oss,
        sensitization_class=sensitization_class,
        EC3_estimate_pct=ec3,
        therapeutic_risk=therapeutic_risk,
        structural_alerts=alerts,
        notes=notes,
    )
