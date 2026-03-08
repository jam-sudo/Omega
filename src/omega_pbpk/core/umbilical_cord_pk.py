"""Phase 1108 — Drug Umbilical Cord Blood Levels.

Predicts drug concentrations in umbilical cord blood at delivery based on
maternal plasma concentration, placental transfer, and fetal distribution.
Relevant for neonatal exposure assessment.
"""

from __future__ import annotations

from dataclasses import dataclass

# Valid ionization types
_VALID_IONIZATION = {"acid", "base", "neutral"}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class UmbilicalCordResult:
    """Result from umbilical cord blood level prediction."""

    drug_name: str
    mw_Da: float
    logP: float
    ionization_type: str
    fu_maternal: float
    c_maternal_mg_L: float
    placental_transfer_fraction: float
    c_fetal_blood_mg_L: float
    umbilical_to_maternal_ratio: float
    neonatal_exposure_class: str
    safety_concern: bool
    recommended_monitoring: str
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_inputs(
    mw_Da: float,
    logP: float,
    ionization_type: str,
    fu_maternal: float,
    c_maternal_mg_L: float,
) -> None:
    if mw_Da <= 0:
        raise ValueError(f"mw_Da must be > 0; got {mw_Da}")
    if not (-5 <= logP <= 10):
        raise ValueError(f"logP must be in [-5, 10]; got {logP}")
    if ionization_type not in _VALID_IONIZATION:
        raise ValueError(
            f"ionization_type must be one of {_VALID_IONIZATION}; got {ionization_type!r}"
        )
    if not (0 < fu_maternal <= 1):
        raise ValueError(f"fu_maternal must be in (0, 1]; got {fu_maternal}")
    if c_maternal_mg_L < 0:
        raise ValueError(f"c_maternal_mg_L must be >= 0; got {c_maternal_mg_L}")


def _placental_transfer_fraction(
    mw_Da: float,
    logP: float,
    fu_maternal: float,
) -> float:
    """Calculate placental transfer fraction based on molecular properties.

    Rules:
      MW < 500 and logP in [-1, 3] and not highly bound → 0.8
      MW < 500 and logP outside [-1, 3]               → 0.5
      MW 500–1000                                       → 0.3
      MW > 1000 (biologics)                             → 0.05

    Then multiply by (1 – protein_binding_correction):
      protein_binding_correction = 0.5 * (1 – fu_maternal)
    """
    # Base fraction from MW and logP
    if mw_Da < 500:
        if -1.0 <= logP <= 3.0:
            base = 0.8
        else:
            base = 0.5
    elif mw_Da <= 1000:
        base = 0.3
    else:
        base = 0.05

    # Protein-binding correction: highly bound → less transfer
    pbc = 0.5 * (1.0 - fu_maternal)
    fraction = base * (1.0 - pbc)

    # Clamp to [0, 1]
    return max(0.0, min(1.0, fraction))


def _fetal_plasma_factor(ionization_type: str) -> float:
    """Ion trapping factor for fetal plasma (pH 7.35 vs maternal 7.40).

    Basic drugs: slight ion trapping in fetal blood → 1.1
    Acid drugs:  slight ion trapping in maternal blood → 0.9
    Neutral:     1.0
    """
    if ionization_type == "base":
        return 1.1
    if ionization_type == "acid":
        return 0.9
    return 1.0


def _exposure_class(ratio: float) -> str:
    if ratio < 0.1:
        return "minimal"
    if ratio < 0.3:
        return "low"
    if ratio < 0.7:
        return "moderate"
    return "high"


def _monitoring_recommendation(
    ratio: float,
    safety_concern: bool,
    drug_name: str,
) -> str:
    if ratio < 0.1:
        return f"Minimal neonatal exposure to {drug_name}; routine monitoring adequate."
    if ratio < 0.3:
        return (
            f"Low neonatal exposure to {drug_name}; "
            "monitor neonate for known drug effects during first 24–48 h."
        )
    if not safety_concern:
        return (
            f"Moderate neonatal exposure to {drug_name}; "
            "clinical monitoring for 48–72 h post-delivery recommended."
        )
    return (
        f"High neonatal exposure to {drug_name} (U/M ratio > 0.5); "
        "close monitoring in NICU setting advised; consider pharmacological "
        "assessment of neonate."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def predict_cord_blood_level(
    drug_name: str,
    mw_Da: float,
    logP: float,
    ionization_type: str,
    fu_maternal: float = 0.3,
    c_maternal_mg_L: float = 1.0,
    pka: float = 7.0,  # reserved for future pH-partitioning refinement
) -> UmbilicalCordResult:
    """Predict umbilical cord blood drug concentration at delivery.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    mw_Da:
        Molecular weight (Da).
    logP:
        Octanol–water partition coefficient.
    ionization_type:
        "acid", "base", or "neutral".
    fu_maternal:
        Unbound fraction in maternal plasma.
    c_maternal_mg_L:
        Maternal plasma concentration at delivery (mg/L).
    pka:
        pKa of drug (currently informational; reserved for extended model).

    Returns
    -------
    UmbilicalCordResult
    """
    _validate_inputs(mw_Da, logP, ionization_type, fu_maternal, c_maternal_mg_L)

    ptf = _placental_transfer_fraction(mw_Da, logP, fu_maternal)
    fpf = _fetal_plasma_factor(ionization_type)

    c_fetal = c_maternal_mg_L * ptf * fpf
    ratio = c_fetal / c_maternal_mg_L if c_maternal_mg_L > 0 else 0.0

    # Clamp ratio to non-negative (edge case: c_maternal = 0 handled above)
    ratio = max(0.0, ratio)

    exp_class = _exposure_class(ratio)
    concern = ratio > 0.5
    monitoring = _monitoring_recommendation(ratio, concern, drug_name)

    notes_parts = [
        f"MW={mw_Da:.1f} Da, logP={logP:.2f}, {ionization_type}.",
        f"Placental transfer fraction={ptf:.3f}.",
        f"Fetal plasma factor={fpf:.2f} (ion trapping).",
        f"U/M ratio={ratio:.3f} → {exp_class} neonatal exposure.",
    ]
    if mw_Da > 1000:
        notes_parts.append("High MW biologic — minimal placental transfer expected.")

    return UmbilicalCordResult(
        drug_name=drug_name,
        mw_Da=mw_Da,
        logP=logP,
        ionization_type=ionization_type,
        fu_maternal=fu_maternal,
        c_maternal_mg_L=c_maternal_mg_L,
        placental_transfer_fraction=ptf,
        c_fetal_blood_mg_L=c_fetal,
        umbilical_to_maternal_ratio=ratio,
        neonatal_exposure_class=exp_class,
        safety_concern=concern,
        recommended_monitoring=monitoring,
        notes=" ".join(notes_parts),
    )


def screen_neonatal_exposure(
    drugs: list[dict],
) -> list[UmbilicalCordResult]:
    """Screen a list of drugs for neonatal exposure risk.

    Each entry is a dict with keys matching the arguments of
    `predict_cord_blood_level`.  Required: "drug_name", "mw_Da", "logP",
    "ionization_type".  Optional: "fu_maternal", "c_maternal_mg_L", "pka".

    Returns results sorted by umbilical_to_maternal_ratio descending
    (highest fetal exposure first).
    """
    results: list[UmbilicalCordResult] = []
    for drug in drugs:
        result = predict_cord_blood_level(
            drug_name=drug["drug_name"],
            mw_Da=drug["mw_Da"],
            logP=drug["logP"],
            ionization_type=drug["ionization_type"],
            fu_maternal=drug.get("fu_maternal", 0.3),
            c_maternal_mg_L=drug.get("c_maternal_mg_L", 1.0),
            pka=drug.get("pka", 7.0),
        )
        results.append(result)

    results.sort(key=lambda r: r.umbilical_to_maternal_ratio, reverse=True)
    return results
