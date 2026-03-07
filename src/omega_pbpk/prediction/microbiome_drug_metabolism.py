"""Phase 1132 — Gut microbiome-mediated drug metabolism prediction.

Models reduction, hydrolysis, and conjugate cleavage by gut bacteria.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_VALID_IONIZATION = {"acid", "base", "neutral"}


@dataclass(frozen=True)
class MicrobiomeMetabolismResult:
    drug_name: str
    logP: float
    ionization_type: str
    has_azo_bond: bool
    has_nitro_group: bool
    has_glucuronide: bool
    residence_time_h: float
    colonic_pH: float
    total_metabolic_rate_per_h: float
    metabolic_fraction: float
    metabolized_pct: float
    primary_pathway: str
    reabsorption_fraction: float
    bioavailability_impact: str
    notes: str


def predict_microbiome_metabolism(
    drug_name: str,
    logP: float,
    ionization_type: str,
    has_azo_bond: bool,
    has_nitro_group: bool,
    has_glucuronide: bool,
    residence_time_h: float,
    colonic_pH: float,
) -> MicrobiomeMetabolismResult:
    """Predict gut microbiome-mediated drug metabolism.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    logP:
        Octanol-water partition coefficient ([-5, 10]).
    ionization_type:
        One of "acid", "base", or "neutral".
    has_azo_bond:
        Whether the drug has an azo (N=N) bond.
    has_nitro_group:
        Whether the drug has a nitro group.
    has_glucuronide:
        Whether the drug has a glucuronide conjugate.
    residence_time_h:
        Colonic residence time in hours (0, 72].
    colonic_pH:
        Colonic luminal pH (5.0-8.0).

    Returns
    -------
    MicrobiomeMetabolismResult
    """
    if not (-5 <= logP <= 10):
        raise ValueError(f"logP must be in [-5, 10], got {logP}")
    if ionization_type not in _VALID_IONIZATION:
        raise ValueError(
            f"ionization_type must be one of {_VALID_IONIZATION}, got {ionization_type!r}"
        )
    if not (0 < residence_time_h <= 72):
        raise ValueError(f"residence_time_h must be in (0, 72], got {residence_time_h}")
    if not (5.0 <= colonic_pH <= 8.0):
        raise ValueError(f"colonic_pH must be in [5.0, 8.0], got {colonic_pH}")

    pathway_rates: dict[str, float] = {}

    # Azoreduction
    if has_azo_bond:
        ph_factor = max(0.0, (8.0 - colonic_pH) / 3.0)
        pathway_rates["azoreduction"] = 0.15 * ph_factor

    # Nitroreduction
    if has_nitro_group:
        pathway_rates["nitroreduction"] = 0.08

    # Glucuronide cleavage
    if has_glucuronide:
        pathway_rates["glucuronide_cleavage"] = 0.20

    # Passive fermentation
    passive_rate = 0.01 * max(0.0, 3.0 - logP)
    pathway_rates["passive_fermentation"] = passive_rate

    total_rate = sum(pathway_rates.values())

    metabolic_fraction = 1.0 - math.exp(-total_rate * residence_time_h)
    metabolic_fraction = max(0.0, min(1.0, metabolic_fraction))
    metabolized_pct = metabolic_fraction * 100.0

    # Primary pathway
    primary_pathway = max(pathway_rates, key=lambda k: pathway_rates[k])

    # Reabsorption fraction
    reabsorption_fraction = 0.3 if ionization_type == "base" else 0.1

    # Bioavailability impact
    if metabolized_pct > 50.0:
        bioavailability_impact = "high_reduction"
    elif metabolized_pct >= 20.0:
        bioavailability_impact = "moderate"
    else:
        bioavailability_impact = "low"

    pathway_summary = ", ".join(f"{k}={v:.4f}/h" for k, v in pathway_rates.items())
    notes = (
        f"Pathways: {pathway_summary}; "
        f"total_rate={total_rate:.4f}/h; "
        f"metabolized={metabolized_pct:.1f}%; "
        f"reabsorption={reabsorption_fraction:.2f}"
    )

    return MicrobiomeMetabolismResult(
        drug_name=drug_name,
        logP=logP,
        ionization_type=ionization_type,
        has_azo_bond=has_azo_bond,
        has_nitro_group=has_nitro_group,
        has_glucuronide=has_glucuronide,
        residence_time_h=residence_time_h,
        colonic_pH=colonic_pH,
        total_metabolic_rate_per_h=total_rate,
        metabolic_fraction=metabolic_fraction,
        metabolized_pct=metabolized_pct,
        primary_pathway=primary_pathway,
        reabsorption_fraction=reabsorption_fraction,
        bioavailability_impact=bioavailability_impact,
        notes=notes,
    )


def rank_microbiome_substrates(drugs: list[dict]) -> list[MicrobiomeMetabolismResult]:
    """Rank a list of drugs by metabolic fraction (descending).

    Each dict must have keys matching the parameters of
    ``predict_microbiome_metabolism``.
    """
    results = []
    for d in drugs:
        result = predict_microbiome_metabolism(
            drug_name=d["drug_name"],
            logP=d["logP"],
            ionization_type=d["ionization_type"],
            has_azo_bond=d["has_azo_bond"],
            has_nitro_group=d["has_nitro_group"],
            has_glucuronide=d["has_glucuronide"],
            residence_time_h=d["residence_time_h"],
            colonic_pH=d["colonic_pH"],
        )
        results.append(result)
    results.sort(key=lambda r: r.metabolic_fraction, reverse=True)
    return results
