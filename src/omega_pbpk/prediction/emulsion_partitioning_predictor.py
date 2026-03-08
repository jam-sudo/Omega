"""Phase 311 — Emulsion Drug Partitioning Predictor.

Predicts drug partitioning between oil and water phases in pharmaceutical emulsions,
determining predominant phase residence and effects on drug release/bioavailability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

VALID_EMULSION_TYPES = {"o/w", "w/o"}
VALID_IONIZATION_TYPES = {"acid", "base", "neutral"}


@dataclass(frozen=True)
class EmulsionPartitioningResult:
    """Result of emulsion drug partitioning prediction."""

    drug_name: str
    logp: float
    pka: float
    ionization_type: str
    oil_fraction: float
    emulsion_type: str
    ph_aqueous: float
    log_d_at_ph: float
    partition_coefficient_d: float
    oil_phase_fraction_drug: float
    water_phase_fraction_drug: float
    bioavailability_factor: float
    release_rate_relative: str
    stability_risk: str
    notes: str


def _validate_inputs(
    logp: float,
    pka: float,
    oil_fraction: float,
    emulsion_type: str,
    ph_aqueous: float,
    ionization_type: str,
) -> None:
    """Validate input parameters, raising ValueError on invalid inputs."""
    if not (-5 < logp < 10):
        raise ValueError(f"logp must be in (-5, 10), got {logp}")
    if not (0 < pka <= 14):
        raise ValueError(f"pka must be in (0, 14], got {pka}")
    if not (0 < oil_fraction < 1):
        raise ValueError(f"oil_fraction must be in (0, 1), got {oil_fraction}")
    if emulsion_type not in VALID_EMULSION_TYPES:
        raise ValueError(
            f"emulsion_type must be one of {VALID_EMULSION_TYPES}, got {emulsion_type!r}"
        )
    if not (0 < ph_aqueous < 14):
        raise ValueError(f"ph_aqueous must be in (0, 14), got {ph_aqueous}")
    if ionization_type not in VALID_IONIZATION_TYPES:
        raise ValueError(
            f"ionization_type must be one of {VALID_IONIZATION_TYPES}, got {ionization_type!r}"
        )


def _compute_log_d(logp: float, pka: float, ionization_type: str, ph_aqueous: float) -> float:
    """Compute logD at the given aqueous pH."""
    if ionization_type == "acid":
        log_d = logp - math.log10(1.0 + 10.0 ** (ph_aqueous - pka))
    elif ionization_type == "base":
        log_d = logp - math.log10(1.0 + 10.0 ** (pka - ph_aqueous))
    else:
        log_d = logp
    return log_d


def predict_emulsion_partitioning(
    drug_name: str,
    logp: float,
    pka: float,
    ionization_type: str,
    oil_fraction: float,
    emulsion_type: str,
    ph_aqueous: float,
) -> EmulsionPartitioningResult:
    """Predict drug partitioning between oil and water phases in a pharmaceutical emulsion.

    Parameters
    ----------
    drug_name:
        Identifier for the drug compound.
    logp:
        Octanol-water partition coefficient (log scale). Must be in (-5, 10).
    pka:
        Acid dissociation constant. Must be in (0, 14].
    ionization_type:
        One of "acid", "base", or "neutral".
    oil_fraction:
        Volume fraction of the oil phase (0 < oil_fraction < 1).
    emulsion_type:
        Either "o/w" (oil-in-water) or "w/o" (water-in-oil).
    ph_aqueous:
        pH of the aqueous phase. Must be in (0, 14).

    Returns
    -------
    EmulsionPartitioningResult
        Dataclass with partitioning fractions, bioavailability estimate, and risk flags.
    """
    _validate_inputs(logp, pka, oil_fraction, emulsion_type, ph_aqueous, ionization_type)

    log_d = _compute_log_d(logp, pka, ionization_type, ph_aqueous)
    d = 10.0**log_d  # oil/water partition coefficient at pH

    water_fraction = 1.0 - oil_fraction

    # Drug fraction in each phase
    f_oil = d * oil_fraction / (d * oil_fraction + water_fraction)
    f_water = 1.0 - f_oil

    # Bioavailability factor depends on emulsion type
    if emulsion_type == "o/w":
        # Aqueous phase is continuous; drug in water is immediately bioavailable
        bioavailability_factor = f_water
    else:
        # w/o: oil is continuous; drug must transfer from oil to aqueous for absorption
        bioavailability_factor = f_oil * 0.7

    # Release rate based on partition coefficient D
    if d > 100:
        release_rate_relative = "slow"
    elif d > 10:
        release_rate_relative = "moderate"
    else:
        release_rate_relative = "fast"

    # Stability risk: mostly in oil → creaming risk
    stability_risk = "high" if f_oil > 0.9 else "low"

    # Build notes
    notes_parts: list[str] = []
    notes_parts.append(f"logD at pH {ph_aqueous:.1f} = {log_d:.2f}")
    notes_parts.append(f"D = {d:.2f}")
    if stability_risk == "high":
        notes_parts.append("Drug predominantly in oil phase; creaming/coalescence risk elevated.")
    if emulsion_type == "w/o" and f_oil > 0.5:
        notes_parts.append(
            "w/o emulsion with high oil-phase drug; absorption may require phase transfer."
        )
    notes = " ".join(notes_parts)

    return EmulsionPartitioningResult(
        drug_name=drug_name,
        logp=logp,
        pka=pka,
        ionization_type=ionization_type,
        oil_fraction=oil_fraction,
        emulsion_type=emulsion_type,
        ph_aqueous=ph_aqueous,
        log_d_at_ph=log_d,
        partition_coefficient_d=d,
        oil_phase_fraction_drug=f_oil,
        water_phase_fraction_drug=f_water,
        bioavailability_factor=bioavailability_factor,
        release_rate_relative=release_rate_relative,
        stability_risk=stability_risk,
        notes=notes,
    )


def screen_oil_fractions(
    drug_name: str,
    logp: float,
    pka: float,
    ionization_type: str,
    oil_fractions_list: list[float],
) -> list[EmulsionPartitioningResult]:
    """Screen multiple oil fractions and return results sorted by oil_phase_fraction_drug descending.

    Uses o/w emulsion type and physiological pH 7.4 for screening.

    Parameters
    ----------
    drug_name:
        Identifier for the drug compound.
    logp:
        Octanol-water partition coefficient (log scale).
    pka:
        Acid dissociation constant.
    ionization_type:
        One of "acid", "base", or "neutral".
    oil_fractions_list:
        List of oil volume fractions to evaluate (each must be in (0, 1)).

    Returns
    -------
    list[EmulsionPartitioningResult]
        Results sorted descending by oil_phase_fraction_drug.
    """
    results: list[EmulsionPartitioningResult] = []
    for oil_frac in oil_fractions_list:
        result = predict_emulsion_partitioning(
            drug_name=drug_name,
            logp=logp,
            pka=pka,
            ionization_type=ionization_type,
            oil_fraction=oil_frac,
            emulsion_type="o/w",
            ph_aqueous=7.4,
        )
        results.append(result)

    results.sort(key=lambda r: r.oil_phase_fraction_drug, reverse=True)
    return results
