"""Skin sensitization and contact allergy risk prediction (Phase 790).

Implements ECHA/LLNA-based potency classification and GHS labelling
using structural alert scoring and physicochemical modifiers.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "SkinSensitizationResultP790",
    "predict_skin_sensitization",
    "screen_sensitization",
]


@dataclass(frozen=True)
class SkinSensitizationResultP790:
    """Result of skin sensitization risk prediction (Phase 790 API)."""

    compound_name: str
    has_michael_acceptor: bool
    has_acyl_transfer: bool
    has_sn2_electrophile: bool
    has_radical_mechanism: bool
    n_structural_alerts: int
    dna_reactivity_score: float  # proxy for LLNA EC3 reactivity
    protein_binding_score: float
    potency_class: str  # ECHA: "extreme", "strong", "moderate", "weak", "non-sensitizer"
    llna_ec3_estimate_pct: float  # estimated LLNA EC3 (% conc)
    ghs_classification: str  # "H317", "H317 Cat.1A", "H317 Cat.1B", or "no classification"
    recommendations: str
    notes: str


_ALERT_SCORES = {
    "michael_acceptor": 40,
    "acyl_transfer": 50,
    "sn2_electrophile": 30,
    "radical_mechanism": 20,
}

_POTENCY_CLASSES = {"extreme", "strong", "moderate", "weak", "non-sensitizer"}


def _logp_modifier(logp: float) -> float:
    """Penetration modifier based on logP."""
    if 1.0 <= logp <= 4.0:
        return 10.0  # lipophilic: enhanced penetration
    elif logp < 0.0:
        return -15.0  # very hydrophilic: reduced penetration
    return 0.0


def _ec3_from_score(score: float) -> float:
    """Estimate LLNA EC3 (%) from protein binding score.

    score > 60  → EC3 < 1%     → strong–extreme
    40–60       → EC3 1–3%     → moderate–strong
    20–40       → EC3 3–10%    → weak–moderate
    < 20        → EC3 > 10%    → non-sensitizer–weak
    """
    if score > 60.0:
        # Map 60–100 → EC3 in [0.01, 1.0] (log-linear)
        # score=60 → EC3=1.0; score=100 → EC3=0.01
        t = (score - 60.0) / 40.0  # 0 at score=60, 1 at score=100
        return max(0.01, 1.0 * (0.01**t))
    elif score >= 40.0:
        # 40–60 → EC3 1–3%
        t = (score - 40.0) / 20.0  # 0 at score=40, 1 at score=60
        return 3.0 - t * 2.0  # 3 at t=0, 1 at t=1
    elif score >= 20.0:
        # 20–40 → EC3 3–10%
        t = (score - 20.0) / 20.0
        return 10.0 - t * 7.0  # 10 at t=0, 3 at t=1
    else:
        # < 20 → > 10%
        # score=0 → EC3>100 (non-sensitizer); score=20 → EC3=10%
        if score <= 0.0:
            return 200.0  # above 100 → non-sensitizer
        t = score / 20.0
        return 110.0 - t * 100.0  # 110 at t=0, 10 at t=1


def _potency_from_ec3(ec3_pct: float) -> str:
    """ECHA potency classification from LLNA EC3 (%).

    extreme:       EC3 < 0.1%
    strong:        0.1% <= EC3 < 1%
    moderate:      1% <= EC3 <= 10%
    weak:          10% < EC3 <= 100%
    non-sensitizer: EC3 > 100%
    """
    if ec3_pct < 0.1:
        return "extreme"
    elif ec3_pct < 1.0:
        return "strong"
    elif ec3_pct <= 10.0:
        return "moderate"
    elif ec3_pct < 100.0:
        return "weak"
    else:
        return "non-sensitizer"


def _ghs_from_potency(potency_class: str) -> str:
    """GHS classification from ECHA potency class."""
    if potency_class in ("extreme", "strong"):
        return "H317 Cat.1A"
    elif potency_class in ("moderate", "weak"):
        return "H317 Cat.1B"
    else:
        return "no classification"


def predict_skin_sensitization(
    compound_name: str,
    logp: float = 2.0,
    mw_da: float = 300.0,
    has_michael_acceptor: bool = False,
    has_acyl_transfer: bool = False,
    has_sn2_electrophile: bool = False,
    has_radical_mechanism: bool = False,
    psa_A2: float = 60.0,
    skin_penetration_factor: float = 1.0,
) -> SkinSensitizationResultP790:
    """Predict skin sensitization risk using structural alerts and physicochemical properties.

    Scoring:
    - michael_acceptor: +40; acyl_transfer: +50; sn2_electrophile: +30; radical: +20
    - logP 1–4: +10 (enhanced penetration); logP < 0: −15 (reduced penetration)
    - protein_binding_score = base_score * skin_penetration_factor

    Args:
        compound_name: Name of the compound.
        logp: Octanol-water partition coefficient.
        mw_da: Molecular weight (Da). Must be > 0.
        has_michael_acceptor: Alpha,beta-unsaturated carbonyl present.
        has_acyl_transfer: Acid chloride/anhydride present.
        has_sn2_electrophile: Alkyl halide/epoxide present.
        has_radical_mechanism: Peroxide/photoreactive group present.
        psa_A2: Polar surface area (Å²).
        skin_penetration_factor: Relative dermal absorption multiplier.

    Returns:
        SkinSensitizationResultP790.
    """
    if mw_da <= 0:
        raise ValueError(f"mw_da must be > 0, got {mw_da}")

    # Count structural alerts
    n_alerts = sum(
        [
            has_michael_acceptor,
            has_acyl_transfer,
            has_sn2_electrophile,
            has_radical_mechanism,
        ]
    )

    # Base score from alerts
    base_score = 0.0
    if has_michael_acceptor:
        base_score += _ALERT_SCORES["michael_acceptor"]
    if has_acyl_transfer:
        base_score += _ALERT_SCORES["acyl_transfer"]
    if has_sn2_electrophile:
        base_score += _ALERT_SCORES["sn2_electrophile"]
    if has_radical_mechanism:
        base_score += _ALERT_SCORES["radical_mechanism"]

    # Modify by logP — only when structural alerts are present
    if base_score > 0.0:
        base_score += _logp_modifier(logp)
    base_score = max(0.0, base_score)

    # DNA reactivity score (proxy: same as base score normalised to 0-100)
    dna_reactivity_score = min(100.0, base_score)

    # Protein binding score = modified by skin penetration
    protein_binding_score = min(100.0, base_score * skin_penetration_factor)
    protein_binding_score = max(0.0, protein_binding_score)

    # LLNA EC3 estimate
    llna_ec3_pct = _ec3_from_score(protein_binding_score)

    # Potency and GHS
    potency_class = _potency_from_ec3(llna_ec3_pct)
    ghs = _ghs_from_potency(potency_class)

    # Recommendations
    rec_parts = []
    if n_alerts == 0:
        rec_parts.append("No structural alerts detected; standard safety assessment.")
    else:
        rec_parts.append(f"{n_alerts} structural alert(s) detected.")
        if potency_class in ("extreme", "strong"):
            rec_parts.append("Recommend LLNA or h-CLAT testing; consider reformulation.")
        elif potency_class in ("moderate", "weak"):
            rec_parts.append("Recommend QSAr confirmation and patch-test monitoring.")
        else:
            rec_parts.append("Low sensitization potential; standard monitoring.")

    if mw_da > 500:
        rec_parts.append("High MW may limit dermal penetration.")
    if psa_A2 > 120:
        rec_parts.append("High PSA may reduce skin permeation.")

    recommendations = " ".join(rec_parts)

    # Notes
    notes_parts = [
        f"logP={logp:.2f}, MW={mw_da:.1f} Da, PSA={psa_A2:.1f} A2",
        f"base_score={base_score:.1f}",
        f"protein_binding_score={protein_binding_score:.2f}",
        f"LLNA EC3 estimate={llna_ec3_pct:.2f}%",
    ]
    notes = "; ".join(notes_parts)

    return SkinSensitizationResultP790(
        compound_name=compound_name,
        has_michael_acceptor=has_michael_acceptor,
        has_acyl_transfer=has_acyl_transfer,
        has_sn2_electrophile=has_sn2_electrophile,
        has_radical_mechanism=has_radical_mechanism,
        n_structural_alerts=n_alerts,
        dna_reactivity_score=round(dna_reactivity_score, 4),
        protein_binding_score=round(protein_binding_score, 4),
        potency_class=potency_class,
        llna_ec3_estimate_pct=round(llna_ec3_pct, 4),
        ghs_classification=ghs,
        recommendations=recommendations,
        notes=notes,
    )


def screen_sensitization(
    compounds: list[dict],
) -> list[SkinSensitizationResultP790]:
    """Screen a list of compounds for skin sensitization risk.

    Each dict must contain 'compound_name' and may contain any of the
    keyword arguments accepted by predict_skin_sensitization().

    Returns list sorted by llna_ec3_estimate_pct ascending (most potent first).
    """
    results = []
    for comp in compounds:
        name = comp["compound_name"]
        result = predict_skin_sensitization(
            compound_name=name,
            logp=comp.get("logp", 2.0),
            mw_da=comp.get("mw_da", 300.0),
            has_michael_acceptor=comp.get("has_michael_acceptor", False),
            has_acyl_transfer=comp.get("has_acyl_transfer", False),
            has_sn2_electrophile=comp.get("has_sn2_electrophile", False),
            has_radical_mechanism=comp.get("has_radical_mechanism", False),
            psa_A2=comp.get("psa_A2", 60.0),
            skin_penetration_factor=comp.get("skin_penetration_factor", 1.0),
        )
        results.append(result)

    # Sort by llna_ec3_estimate_pct ascending (most potent sensitizers first)
    results.sort(key=lambda r: r.llna_ec3_estimate_pct)
    return results
