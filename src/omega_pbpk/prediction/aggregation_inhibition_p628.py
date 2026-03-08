"""Phase 628 — Drug Protein Aggregation Inhibition.

Predicts risk of drug-induced protein aggregation inhibition, relevant for
biologic stability and formulation development.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AggregationInhibitionResult:
    """Result of aggregation inhibition prediction."""

    drug_name: str
    target_protein: str
    drug_mw_Da: float
    drug_logP: float
    inhibition_mechanism: str
    ki_uM: float
    inhibition_pct_at_1uM: float
    inhibition_pct_at_10uM: float
    aggregation_rate_reduction: float
    preferred_excipient: str
    formulation_ph_optimum: float
    stability_improvement_pct: float
    notes: str


def predict_aggregation_inhibition(
    drug_name: str,
    target_protein: str,
    drug_mw_Da: float,
    drug_logP: float,
    drug_charge: str,
    drug_pka: float = 7.0,
    concentration_uM: float = 10.0,
) -> AggregationInhibitionResult:
    """Predict drug-induced protein aggregation inhibition.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    target_protein:
        Name of the target protein.
    drug_mw_Da:
        Molecular weight of the drug in Daltons.
    drug_logP:
        Octanol-water partition coefficient (log units).
    drug_charge:
        Charge state: "positive", "negative", "neutral", or "zwitterionic".
    drug_pka:
        pKa of ionizable group (default 7.0).
    concentration_uM:
        Drug concentration for inhibition calculation (uM, default 10.0).

    Returns
    -------
    AggregationInhibitionResult
    """
    # Validation
    if drug_mw_Da <= 0:
        raise ValueError("drug_mw_Da must be > 0")
    if drug_charge not in ("positive", "negative", "neutral", "zwitterionic"):
        raise ValueError('drug_charge must be "positive", "negative", "neutral", or "zwitterionic"')
    if concentration_uM <= 0:
        raise ValueError("concentration_uM must be > 0")

    # Determine inhibition mechanism
    n_electrostatic = 1 if drug_charge == "positive" else 0
    n_hydrophobic = 1 if drug_logP > 2 else 0
    n_steric = 1 if (drug_mw_Da > 500 and drug_charge == "neutral") else 0
    n_mechanisms = n_electrostatic + n_hydrophobic + n_steric

    if n_mechanisms == 0:
        inhibition_mechanism = "none"
    elif n_mechanisms >= 2:
        inhibition_mechanism = "mixed"
    elif drug_charge == "positive":
        inhibition_mechanism = "electrostatic"
    elif drug_logP > 2:
        inhibition_mechanism = "hydrophobic"
    else:
        inhibition_mechanism = "steric"

    # Ki estimation
    ki_uM = 50.0
    if inhibition_mechanism == "electrostatic":
        ki_uM /= 3.0
    elif inhibition_mechanism == "hydrophobic":
        ki_uM /= 1.0 + drug_logP
    elif inhibition_mechanism == "steric":
        ki_uM /= 1.5
    elif inhibition_mechanism == "mixed":
        ki_uM /= 5.0
    # "none" keeps ki_uM = 50.0

    ki_uM = max(0.1, min(500.0, ki_uM))

    # Inhibition percentages
    inhibition_pct_at_1uM = 1.0 / (1.0 + ki_uM) * 100.0
    inhibition_pct_at_10uM = 10.0 / (10.0 + ki_uM) * 100.0

    # Aggregation rate reduction
    raw_rate_reduction = 1.0 / (1.0 - inhibition_pct_at_10uM / 100.0 * 0.8)
    aggregation_rate_reduction = max(1.0, min(100.0, raw_rate_reduction))

    # Preferred excipient
    if inhibition_mechanism in ("electrostatic", "mixed"):
        preferred_excipient = "arginine"
    elif inhibition_mechanism == "hydrophobic":
        preferred_excipient = "polysorbate 80"
    elif inhibition_mechanism == "steric":
        preferred_excipient = "PEG 4000"
    else:
        preferred_excipient = "sucrose"

    # Formulation pH optimum
    if drug_charge == "positive":
        formulation_ph_optimum = 5.0
    elif drug_charge == "negative":
        formulation_ph_optimum = 7.4
    else:
        formulation_ph_optimum = 6.5

    # Stability improvement
    raw_stability = inhibition_pct_at_10uM * 0.7
    stability_improvement_pct = max(0.0, min(100.0, raw_stability))

    notes = (
        f"{drug_name} vs {target_protein}: MW={drug_mw_Da:.0f} Da, "
        f"logP={drug_logP:.2f}, charge={drug_charge}, "
        f"mechanism={inhibition_mechanism}, Ki={ki_uM:.2f} uM, "
        f"excipient={preferred_excipient}, pH_opt={formulation_ph_optimum:.1f}"
    )

    return AggregationInhibitionResult(
        drug_name=drug_name,
        target_protein=target_protein,
        drug_mw_Da=drug_mw_Da,
        drug_logP=drug_logP,
        inhibition_mechanism=inhibition_mechanism,
        ki_uM=ki_uM,
        inhibition_pct_at_1uM=inhibition_pct_at_1uM,
        inhibition_pct_at_10uM=inhibition_pct_at_10uM,
        aggregation_rate_reduction=aggregation_rate_reduction,
        preferred_excipient=preferred_excipient,
        formulation_ph_optimum=formulation_ph_optimum,
        stability_improvement_pct=stability_improvement_pct,
        notes=notes,
    )


def screen_aggregation_inhibitors(compounds: list) -> list:
    """Screen multiple compounds for aggregation inhibition.

    Parameters
    ----------
    compounds:
        List of dicts, each with keys: drug_name, target_protein, drug_mw_Da,
        drug_logP, drug_charge, and optionally drug_pka, concentration_uM.

    Returns
    -------
    list of AggregationInhibitionResult sorted by stability_improvement_pct descending.
    """
    results = []
    for comp in compounds:
        result = predict_aggregation_inhibition(
            drug_name=comp["drug_name"],
            target_protein=comp["target_protein"],
            drug_mw_Da=comp["drug_mw_Da"],
            drug_logP=comp["drug_logP"],
            drug_charge=comp["drug_charge"],
            drug_pka=comp.get("drug_pka", 7.0),
            concentration_uM=comp.get("concentration_uM", 10.0),
        )
        results.append(result)

    results.sort(key=lambda r: r.stability_improvement_pct, reverse=True)
    return results
