"""Phase 491 — Ocular Bioavailability Predictor.

Rule-based predictor for topical ophthalmic drug bioavailability.
Factors: logP (corneal permeation optimum 2-3), MW (<500 Da), pKa (unionized form penetrates).
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class OcularBioavailabilityResult:
    """Result of ocular bioavailability prediction."""

    drug_name: str
    logP: float
    mw_Da: float
    pKa: float
    ionization_type: str
    kp_cornea_cm_per_h: float
    f_unionized: float
    kp_effective_cm_per_h: float
    f_ocular_aqueous: float
    f_ocular_vitreous: float
    aqueous_penetration_class: str
    dosing_recommendation: str
    notes: str


_VALID_IONIZATION_TYPES = {"acid", "base", "neutral"}
_TEAR_FILM_PH = 7.4


def _compute_f_unionized(pKa: float, ionization_type: str) -> float:
    """Compute fraction unionized at tear film pH 7.4 using Henderson-Hasselbalch."""
    if ionization_type == "neutral":
        return 1.0
    delta = _TEAR_FILM_PH - pKa
    if ionization_type == "acid":
        # HA <-> H+ + A-; fraction unionized = 1 / (1 + 10^(pH - pKa))
        return 1.0 / (1.0 + math.pow(10.0, delta))
    # base: BH+ <-> B + H+; fraction unionized (free base) = 1 / (1 + 10^(pKa - pH))
    return 1.0 / (1.0 + math.pow(10.0, -delta))


def _aqueous_penetration_class(kp_eff: float) -> str:
    """Classify aqueous penetration from effective corneal permeability."""
    if kp_eff < 0.001:
        return "poor"
    if kp_eff <= 0.01:
        return "moderate"
    return "good"


def _dosing_recommendation(penetration_class: str) -> str:
    """Return dosing frequency based on penetration class."""
    mapping = {"poor": "TID", "moderate": "BID", "good": "QD"}
    return mapping[penetration_class]


def predict_ocular_bioavailability(
    drug_name: str,
    logP: float,
    mw_Da: float,
    pKa: float = 7.4,
    ionization_type: str = "neutral",
) -> OcularBioavailabilityResult:
    """Predict topical ophthalmic drug bioavailability to ocular tissues.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    logP:
        Octanol-water partition coefficient (log scale).
    mw_Da:
        Molecular weight in Daltons.
    pKa:
        Acid/base dissociation constant. Used with ionization_type to compute f_unionized.
    ionization_type:
        One of "acid", "base", "neutral".

    Returns
    -------
    OcularBioavailabilityResult
    """
    if mw_Da <= 0:
        raise ValueError(f"mw_Da must be > 0, got {mw_Da}")
    if ionization_type not in _VALID_IONIZATION_TYPES:
        raise ValueError(
            f"ionization_type must be one of {_VALID_IONIZATION_TYPES}, got {ionization_type!r}"
        )

    # Corneal permeability: Gaussian shape around logP=2.5, MW penalty
    log_kp_cornea = -1.5 - 0.5 * (logP - 2.5) ** 2 / 4.0 - 0.003 * mw_Da
    kp_cornea = math.pow(10.0, log_kp_cornea)

    f_unionized = _compute_f_unionized(pKa, ionization_type)

    kp_effective = kp_cornea * f_unionized
    # Clamp to [1e-8, 0.1]
    kp_effective = max(1e-8, min(0.1, kp_effective))

    f_ocular_aqueous = min(0.05, kp_effective * 10.0)
    f_ocular_vitreous = kp_effective * 0.01

    pen_class = _aqueous_penetration_class(kp_effective)
    dosing = _dosing_recommendation(pen_class)

    notes_parts = []
    if logP < 1.0 or logP > 4.0:
        notes_parts.append(f"logP={logP:.1f} is outside optimal range (1-4)")
    if mw_Da > 500:
        notes_parts.append(f"MW={mw_Da:.0f} Da may limit corneal penetration")
    if f_unionized < 0.1:
        notes_parts.append(
            f"Drug is mostly ionized (f_unionized={f_unionized:.3f}); consider pH adjustment"
        )
    if not notes_parts:
        notes_parts.append("Physicochemical properties within acceptable range")
    notes = "; ".join(notes_parts)

    return OcularBioavailabilityResult(
        drug_name=drug_name,
        logP=logP,
        mw_Da=mw_Da,
        pKa=pKa,
        ionization_type=ionization_type,
        kp_cornea_cm_per_h=kp_cornea,
        f_unionized=f_unionized,
        kp_effective_cm_per_h=kp_effective,
        f_ocular_aqueous=f_ocular_aqueous,
        f_ocular_vitreous=f_ocular_vitreous,
        aqueous_penetration_class=pen_class,
        dosing_recommendation=dosing,
        notes=notes,
    )


def screen_ophthalmic_candidates(
    compounds: list,
) -> list:
    """Screen a list of compound parameter dicts and rank by kp_effective_cm_per_h.

    Each dict must have keys: drug_name, logP, mw_Da.
    Optional keys: pKa (default 7.4), ionization_type (default "neutral").

    Returns list of OcularBioavailabilityResult sorted descending by kp_effective_cm_per_h.
    """
    results = []
    for compound in compounds:
        result = predict_ocular_bioavailability(
            drug_name=compound["drug_name"],
            logP=compound["logP"],
            mw_Da=compound["mw_Da"],
            pKa=compound.get("pKa", 7.4),
            ionization_type=compound.get("ionization_type", "neutral"),
        )
        results.append(result)
    results.sort(key=lambda r: r.kp_effective_cm_per_h, reverse=True)
    return results
