"""Phase 986 — Drug efflux transporter substrate/inhibitor prediction.

Predicts efflux transporter substrate and inhibitor probabilities for 6
transporters: P-gp (MDR1), BCRP (ABCG2), MRP2, MRP4, MATE1, MATE2K.
"""

from __future__ import annotations

from dataclasses import dataclass

_VALID_IONIZATION_TYPES = {"neutral", "acid", "base", "zwitterion"}

_TRANSPORTER_NAMES = ["P-gp", "BCRP", "MRP2", "MATE"]


@dataclass(frozen=True)
class EffluxTransporterResult:
    """Result of efflux transporter prediction."""

    drug_name: str
    logp: float
    mw: float
    pka: float
    pgp_substrate_prob: float
    pgp_inhibitor_prob: float
    bcrp_substrate_prob: float
    bcrp_inhibitor_prob: float
    mrp2_substrate_prob: float
    mate_substrate_prob: float
    dominant_efflux_transporter: str
    bbb_efflux_risk: str
    oral_efflux_risk: str
    notes: str


def predict_efflux_transporters(
    drug_name: str,
    logp: float,
    mw: float,
    pka: float = 7.0,
    ionization_type: str = "neutral",
    hbd: int = 2,
    hba: int = 4,
) -> EffluxTransporterResult:
    """Predict drug efflux transporter substrate/inhibitor probabilities.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    logp:
        Octanol-water partition coefficient (log10 scale).
    mw:
        Molecular weight (g/mol).
    pka:
        pKa of the drug (ionizable group).
    ionization_type:
        "neutral", "acid", "base", or "zwitterion".
    hbd:
        Number of hydrogen bond donors.
    hba:
        Number of hydrogen bond acceptors.

    Returns
    -------
    EffluxTransporterResult
    """
    if ionization_type not in _VALID_IONIZATION_TYPES:
        raise ValueError(
            f"ionization_type must be one of {_VALID_IONIZATION_TYPES}, got '{ionization_type}'"
        )

    # --- P-gp substrate ---
    pgp_sub = 0.2
    if mw > 400:
        pgp_sub += 0.3
    if 1 <= logp <= 4:
        pgp_sub += 0.2
    if hbd > 2:
        pgp_sub += 0.1
    if hba > 4:
        pgp_sub += 0.1
    pgp_substrate_prob = min(0.95, pgp_sub)

    # --- P-gp inhibitor ---
    pgp_inh = 0.1
    if logp > 3:
        pgp_inh += 0.3
    if mw > 500:
        pgp_inh += 0.2
    if ionization_type == "base":
        pgp_inh += 0.2
    pgp_inhibitor_prob = min(0.95, pgp_inh)

    # --- BCRP substrate ---
    bcrp_sub = 0.15
    if ionization_type == "acid" and 3 <= pka <= 7:
        bcrp_sub += 0.3
    if mw > 300:
        bcrp_sub += 0.2
    if logp < 1:
        bcrp_sub += 0.15
    bcrp_substrate_prob = min(0.95, bcrp_sub)

    # --- BCRP inhibitor ---
    bcrp_inh = 0.1
    if logp > 3:
        bcrp_inh += 0.3
    if mw > 400:
        bcrp_inh += 0.2
    if ionization_type == "base":
        bcrp_inh += 0.15
    bcrp_inhibitor_prob = min(0.95, bcrp_inh)

    # --- MRP2 substrate ---
    mrp2_sub = 0.1
    if ionization_type == "acid":
        mrp2_sub += 0.3
    if mw > 450:
        mrp2_sub += 0.2
    if hba > 5:
        mrp2_sub += 0.15
    mrp2_substrate_prob = min(0.95, mrp2_sub)

    # --- MATE substrate ---
    mate_sub = 0.1
    if ionization_type == "base" and pka > 7:
        mate_sub += 0.3
    if mw < 400:
        mate_sub += 0.2
    if logp < 2:
        mate_sub += 0.15
    mate_substrate_prob = min(0.95, mate_sub)

    # --- Dominant efflux transporter ---
    probs = {
        "P-gp": pgp_substrate_prob,
        "BCRP": bcrp_substrate_prob,
        "MRP2": mrp2_substrate_prob,
        "MATE": mate_substrate_prob,
    }
    dominant_efflux_transporter = max(probs, key=lambda k: probs[k])

    # --- BBB efflux risk ---
    bbb_max = max(pgp_substrate_prob, bcrp_substrate_prob)
    if bbb_max > 0.6:
        bbb_efflux_risk = "high"
    elif bbb_max > 0.4:
        bbb_efflux_risk = "moderate"
    else:
        bbb_efflux_risk = "low"

    # --- Oral efflux risk ---
    oral_max = max(pgp_substrate_prob, bcrp_substrate_prob, mrp2_substrate_prob)
    if oral_max > 0.6:
        oral_efflux_risk = "high"
    elif oral_max > 0.35:
        oral_efflux_risk = "moderate"
    else:
        oral_efflux_risk = "low"

    # --- Notes ---
    notes_parts = []
    if pgp_substrate_prob > 0.6:
        notes_parts.append("High P-gp substrate probability; CNS penetration may be limited.")
    if bcrp_substrate_prob > 0.6:
        notes_parts.append("High BCRP substrate probability.")
    if mate_substrate_prob > 0.5:
        notes_parts.append("MATE transporter involvement; renal secretion likely.")
    if not notes_parts:
        notes_parts.append("Low efflux transporter risk predicted.")
    notes = " ".join(notes_parts)

    return EffluxTransporterResult(
        drug_name=drug_name,
        logp=logp,
        mw=mw,
        pka=pka,
        pgp_substrate_prob=pgp_substrate_prob,
        pgp_inhibitor_prob=pgp_inhibitor_prob,
        bcrp_substrate_prob=bcrp_substrate_prob,
        bcrp_inhibitor_prob=bcrp_inhibitor_prob,
        mrp2_substrate_prob=mrp2_substrate_prob,
        mate_substrate_prob=mate_substrate_prob,
        dominant_efflux_transporter=dominant_efflux_transporter,
        bbb_efflux_risk=bbb_efflux_risk,
        oral_efflux_risk=oral_efflux_risk,
        notes=notes,
    )


def screen_efflux_transporters(drugs: list) -> list:
    """Screen a list of drug parameter dicts and return sorted results.

    Each dict in `drugs` must have keys: drug_name, logp, mw, and optionally
    pka, ionization_type, hbd, hba.

    Returns a list of EffluxTransporterResult sorted by pgp_substrate_prob
    descending.
    """
    results = []
    for drug in drugs:
        result = predict_efflux_transporters(
            drug_name=drug["drug_name"],
            logp=drug["logp"],
            mw=drug["mw"],
            pka=drug.get("pka", 7.0),
            ionization_type=drug.get("ionization_type", "neutral"),
            hbd=drug.get("hbd", 2),
            hba=drug.get("hba", 4),
        )
        results.append(result)
    results.sort(key=lambda r: r.pgp_substrate_prob, reverse=True)
    return results


__all__ = [
    "EffluxTransporterResult",
    "predict_efflux_transporters",
    "screen_efflux_transporters",
]
