"""Phase 943: Drug metabolism pathway prediction based on structural features.

Predicts CYP enzyme contributions and metabolic liabilities using rule-based scoring.
No numpy/scipy — pure Python stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "DrugMetabolismPredictionResult",
    "predict_metabolism",
    "screen_metabolic_liability",
]


@dataclass(frozen=True)
class DrugMetabolismPredictionResult:
    """Result of drug metabolism pathway prediction (Phase 943)."""

    drug_name: str
    logp: float
    pka: float
    ionization_type: str
    mw: float
    cyp3a4_fraction: float
    cyp2d6_fraction: float
    cyp2c9_fraction: float
    cyp2c19_fraction: float
    cyp1a2_fraction: float
    primary_enzyme: str
    phase_i_likely: bool
    phase_ii_likely: bool
    high_cyp_variability_risk: bool
    metabolic_liability: str
    notes: str


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _score_cyp3a4(logp: float, pka: float, mw: float, ionization_type: str) -> float:
    score = 0.5
    if 2.0 <= logp <= 5.0:
        score += 0.2
    if 300.0 <= mw <= 600.0:
        score += 0.1
    if ionization_type == "base" and 7.0 <= pka <= 10.0:
        score += 0.1
    return _clamp01(score)


def _score_cyp2d6(logp: float, pka: float, mw: float, ionization_type: str) -> float:
    score = 0.1
    if ionization_type == "base" and pka > 7.0:
        score += 0.4
    if mw < 400.0:
        score += 0.15
    if 1.0 <= logp <= 4.0:
        score += 0.1
    return _clamp01(score)


def _score_cyp2c9(logp: float, pka: float, mw: float, ionization_type: str) -> float:
    score = 0.1
    if ionization_type == "acid" and 3.0 <= pka <= 7.0:
        score += 0.4
    if 2.0 <= logp <= 5.0:
        score += 0.15
    if mw < 450.0:
        score += 0.1
    return _clamp01(score)


def _score_cyp2c19(logp: float, pka: float, mw: float, ionization_type: str) -> float:
    score = 0.1
    # neutral or weak base (pKa 5-9)
    if ionization_type == "neutral":
        score += 0.3
    elif ionization_type == "base" and 5.0 <= pka <= 9.0:
        score += 0.3
    if mw < 500.0:
        score += 0.1
    if 1.0 <= logp <= 4.0:
        score += 0.1
    return _clamp01(score)


def _score_cyp1a2(logp: float, pka: float, mw: float, ionization_type: str) -> float:
    score = 0.05
    # PSA proxy: PSA ≈ MW * 0.12 * (1 - 0.08*logP) + 5
    psa_est = mw * 0.12 * (1.0 - 0.08 * logp) + 5.0
    if psa_est < 50.0 and mw < 400.0:
        score += 0.4
    if 0.0 <= logp <= 3.0:
        score += 0.15
    return _clamp01(score)


def predict_metabolism(
    drug_name: str,
    logp: float,
    pka: float,
    ionization_type: str,
    mw: float,
) -> DrugMetabolismPredictionResult:
    """Predict metabolic pathways and CYP enzyme contributions for a drug.

    Parameters
    ----------
    drug_name:
        Name of the drug compound.
    logp:
        Lipophilicity (log partition coefficient).
    pka:
        Ionization pKa.
    ionization_type:
        One of "acid", "base", or "neutral".
    mw:
        Molecular weight (Da).

    Returns
    -------
    DrugMetabolismPredictionResult
    """
    if mw <= 0:
        raise ValueError("mw must be > 0")
    if ionization_type not in {"acid", "base", "neutral"}:
        raise ValueError(
            f"ionization_type must be 'acid', 'base', or 'neutral', got '{ionization_type}'"
        )

    raw_3a4 = _score_cyp3a4(logp, pka, mw, ionization_type)
    raw_2d6 = _score_cyp2d6(logp, pka, mw, ionization_type)
    raw_2c9 = _score_cyp2c9(logp, pka, mw, ionization_type)
    raw_2c19 = _score_cyp2c19(logp, pka, mw, ionization_type)
    raw_1a2 = _score_cyp1a2(logp, pka, mw, ionization_type)

    total = raw_3a4 + raw_2d6 + raw_2c9 + raw_2c19 + raw_1a2
    if total == 0.0:
        total = 1.0

    f3a4 = raw_3a4 / total
    f2d6 = raw_2d6 / total
    f2c9 = raw_2c9 / total
    f2c19 = raw_2c19 / total
    f1a2 = raw_1a2 / total

    enzyme_fractions = {
        "CYP3A4": f3a4,
        "CYP2D6": f2d6,
        "CYP2C9": f2c9,
        "CYP2C19": f2c19,
        "CYP1A2": f1a2,
    }
    primary_enzyme = max(enzyme_fractions, key=lambda k: enzyme_fractions[k])
    primary_fraction = enzyme_fractions[primary_enzyme]

    phase_i_likely = logp > 1.0
    phase_ii_likely = logp < 0.0 or mw > 500.0 or (ionization_type == "neutral" and mw < 300.0)

    high_cyp_variability_risk = f2d6 > 0.3 or f2c19 > 0.3

    if primary_fraction > 0.7:
        metabolic_liability = "high"
    elif primary_fraction < 0.3:
        metabolic_liability = "low"
    else:
        metabolic_liability = "moderate"

    notes_parts = [
        f"Primary enzyme: {primary_enzyme} ({primary_fraction:.1%}).",
    ]
    if high_cyp_variability_risk:
        notes_parts.append("High variability risk due to polymorphic enzyme involvement.")
    if phase_i_likely:
        notes_parts.append("Phase I oxidation likely.")
    if phase_ii_likely:
        notes_parts.append("Phase II conjugation likely.")
    notes = " ".join(notes_parts)

    return DrugMetabolismPredictionResult(
        drug_name=drug_name,
        logp=logp,
        pka=pka,
        ionization_type=ionization_type,
        mw=mw,
        cyp3a4_fraction=f3a4,
        cyp2d6_fraction=f2d6,
        cyp2c9_fraction=f2c9,
        cyp2c19_fraction=f2c19,
        cyp1a2_fraction=f1a2,
        primary_enzyme=primary_enzyme,
        phase_i_likely=phase_i_likely,
        phase_ii_likely=phase_ii_likely,
        high_cyp_variability_risk=high_cyp_variability_risk,
        metabolic_liability=metabolic_liability,
        notes=notes,
    )


def screen_metabolic_liability(
    compounds: list[dict],
) -> list[DrugMetabolismPredictionResult]:
    """Screen a list of compounds for metabolic liability.

    Parameters
    ----------
    compounds:
        List of dicts with keys: drug_name, logp, pka, ionization_type, mw.

    Returns
    -------
    List of DrugMetabolismPredictionResult sorted by primary enzyme fraction descending.
    """
    results = []
    for c in compounds:
        result = predict_metabolism(
            drug_name=c["drug_name"],
            logp=c["logp"],
            pka=c["pka"],
            ionization_type=c["ionization_type"],
            mw=c["mw"],
        )
        results.append(result)

    def _primary_fraction(r: DrugMetabolismPredictionResult) -> float:
        fractions = {
            "CYP3A4": r.cyp3a4_fraction,
            "CYP2D6": r.cyp2d6_fraction,
            "CYP2C9": r.cyp2c9_fraction,
            "CYP2C19": r.cyp2c19_fraction,
            "CYP1A2": r.cyp1a2_fraction,
        }
        return fractions[r.primary_enzyme]

    results.sort(key=_primary_fraction, reverse=True)
    return results
