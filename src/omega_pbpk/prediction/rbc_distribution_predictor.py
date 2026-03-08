"""Drug distribution into red blood cells (RBC) predictor.

Predicts drug partitioning into RBC using blood-to-plasma ratio (B/P ratio).
Relates to plasma protein binding, logP, and ionization at physiologic pH.
"""

from __future__ import annotations

from dataclasses import dataclass

_VALID_IONIZATION_TYPES = {"acid", "base", "neutral"}


@dataclass(frozen=True)
class RBCDistributionResult:
    drug_name: str
    logp: float
    pka: float
    ionization_type: str
    fu_plasma: float
    hematocrit: float
    fu_rbc_estimate: float
    bp_ratio: float
    distribution_class: str
    notes: str


def _validate_inputs(
    logp: float,
    pka: float,
    ionization_type: str,
    fu_plasma: float,
    hematocrit: float,
) -> None:
    if not (-5 < logp < 10):
        raise ValueError(f"logp must be in (-5, 10), got {logp}")
    if not (0 < pka <= 14):
        raise ValueError(f"pka must be in (0, 14], got {pka}")
    if ionization_type not in _VALID_IONIZATION_TYPES:
        raise ValueError(
            f"ionization_type must be one of {_VALID_IONIZATION_TYPES}, got {ionization_type!r}"
        )
    if not (0 < fu_plasma <= 1):
        raise ValueError(f"fu_plasma must be in (0, 1], got {fu_plasma}")
    if not (0 < hematocrit < 1):
        raise ValueError(f"hematocrit must be in (0, 1), got {hematocrit}")


def _fu_rbc_from_logp(logp: float) -> float:
    """Estimate unbound fraction in RBC from logP."""
    if logp > 2:
        return 0.2  # lipophilic — extensive RBC binding
    elif logp >= 0:
        return 0.5
    else:
        return 0.9  # hydrophilic — low RBC binding


def predict_rbc_distribution(
    drug_name: str,
    logp: float,
    pka: float,
    ionization_type: str,
    fu_plasma: float,
    hematocrit: float = 0.45,
) -> RBCDistributionResult:
    """Predict drug distribution into red blood cells.

    Parameters
    ----------
    drug_name:
        Name of the drug compound.
    logp:
        Octanol-water partition coefficient, must be in (-5, 10).
    pka:
        Acid dissociation constant, must be in (0, 14].
    ionization_type:
        One of "acid", "base", or "neutral".
    fu_plasma:
        Unbound fraction in plasma, must be in (0, 1].
    hematocrit:
        Fraction of blood volume occupied by RBC, default 0.45.

    Returns
    -------
    RBCDistributionResult
    """
    _validate_inputs(logp, pka, ionization_type, fu_plasma, hematocrit)

    fu_rbc = _fu_rbc_from_logp(logp)

    # Standard B/P ratio formula
    bp_raw = (1.0 - hematocrit) + hematocrit * (fu_plasma / fu_rbc)

    # Correction for basic drugs (ion trapping in slightly acidic RBC)
    if ionization_type == "base" and pka > 7.4:
        # fi = ionized fraction at pH 7.4 for base
        fi = 1.0 / (1.0 + 10 ** (7.4 - pka))
        bp_raw *= 1.0 + 0.3 * fi * (pka - 7.4) / 3.0

    bp_ratio = max(0.3, min(10.0, bp_raw))

    # Distribution class
    if bp_ratio > 2.0:
        distribution_class = "extensive_rbc"
    elif bp_ratio >= 1.0:
        distribution_class = "moderate_rbc"
    else:
        distribution_class = "low_rbc"

    notes_parts: list[str] = []
    if fu_rbc < 0.3:
        notes_parts.append("extensive RBC binding due to high lipophilicity")
    if ionization_type == "base" and pka > 7.4:
        notes_parts.append("basic drug may be ion-trapped in RBC")
    if bp_ratio > 2.0:
        notes_parts.append("plasma CL may underestimate whole-blood CL")
    if bp_ratio < 0.6:
        notes_parts.append("low RBC partitioning; plasma measurements reliable")
    if not notes_parts:
        notes_parts.append("moderate RBC distribution")
    notes = "; ".join(notes_parts)

    return RBCDistributionResult(
        drug_name=drug_name,
        logp=logp,
        pka=pka,
        ionization_type=ionization_type,
        fu_plasma=fu_plasma,
        hematocrit=hematocrit,
        fu_rbc_estimate=fu_rbc,
        bp_ratio=bp_ratio,
        distribution_class=distribution_class,
        notes=notes,
    )


def compare_ionization_effects(
    drug_name: str,
    logp: float,
    pka: float,
    fu_plasma: float,
) -> list[RBCDistributionResult]:
    """Compare B/P ratio across acid, base, and neutral ionization types.

    Parameters
    ----------
    drug_name:
        Name of the drug compound.
    logp:
        Octanol-water partition coefficient.
    pka:
        Acid dissociation constant.
    fu_plasma:
        Unbound fraction in plasma.

    Returns
    -------
    List of three RBCDistributionResult (acid, base, neutral) sorted by
    bp_ratio descending.
    """
    results: list[RBCDistributionResult] = []
    for ion_type in ("acid", "base", "neutral"):
        result = predict_rbc_distribution(
            drug_name=drug_name,
            logp=logp,
            pka=pka,
            ionization_type=ion_type,
            fu_plasma=fu_plasma,
        )
        results.append(result)

    results.sort(key=lambda r: r.bp_ratio, reverse=True)
    return results
