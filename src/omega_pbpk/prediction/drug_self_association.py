"""Drug self-association and aggregation prediction (Phase 1044).

Predicts drug self-association (micelle formation, aggregation) in solution
that can affect PK/bioavailability. CMC (critical micelle concentration) is
driven by logP (amphiphilicity), MW, ionization state, and ionic strength.
"""

from __future__ import annotations

from dataclasses import dataclass

_VALID_IONIZATION_TYPES = frozenset({"neutral", "acid", "base"})


@dataclass(frozen=True)
class SelfAssociationResult:
    """Result of drug self-association prediction (Phase 1044)."""

    drug_name: str
    concentration_mg_mL: float
    cmc_mg_mL: float
    fraction_aggregated: float
    fraction_monomer: float
    effective_free_fraction: float
    aggregate_size_nm: float
    aggregation_risk: str
    pk_impact: str
    solubility_reduction_fold: float
    notes: str


def predict_self_association(
    drug_name: str,
    mw_Da: float,
    logp: float,
    concentration_mg_mL: float,
    pka: float = 7.0,
    ionization_type: str = "neutral",
    ph: float = 7.4,
    ionic_strength_mM: float = 150.0,
) -> SelfAssociationResult:
    """Predict drug self-association and micelle formation.

    Uses a physicochemical CMC model based on logP, MW, ionization, and
    ionic strength to estimate fraction of drug in aggregates and the
    resulting PK impact.

    Args:
        drug_name: Drug identifier.
        mw_Da: Molecular weight in Da (must be > 0).
        logp: Octanol-water partition coefficient.
        concentration_mg_mL: Drug concentration in solution in mg/mL (> 0).
        pka: Acid/base pKa value [0, 14].
        ionization_type: "neutral", "acid", or "base".
        ph: Solution pH.
        ionic_strength_mM: Ionic strength in mM (>= 0).

    Returns:
        SelfAssociationResult with CMC, aggregation fractions, and risk.

    Raises:
        ValueError: If inputs are out of valid range.
    """
    # Validation
    if mw_Da <= 0.0:
        raise ValueError(f"mw_Da must be > 0, got {mw_Da}")
    if concentration_mg_mL <= 0.0:
        raise ValueError(f"concentration_mg_mL must be > 0, got {concentration_mg_mL}")
    if ionic_strength_mM < 0.0:
        raise ValueError(f"ionic_strength_mM must be >= 0, got {ionic_strength_mM}")
    if ionization_type not in _VALID_IONIZATION_TYPES:
        raise ValueError(
            f"ionization_type must be one of {sorted(_VALID_IONIZATION_TYPES)}, "
            f"got {ionization_type!r}"
        )
    if not (0.0 <= pka <= 14.0):
        raise ValueError(f"pka must be in [0, 14], got {pka}")

    # Base CMC from logP
    cmc_0 = 10.0 ** (3.0 - 0.5 * logp)
    cmc_0 = max(0.01, min(100.0, cmc_0))

    # Ionized fraction
    if ionization_type == "acid":
        ionized_fraction = 1.0 / (1.0 + 10.0 ** (pka - ph))
    elif ionization_type == "base":
        ionized_fraction = 1.0 / (1.0 + 10.0 ** (ph - pka))
    else:
        ionized_fraction = 0.0

    # Ionization raises CMC (charged drugs harder to form micelles)
    cmc_factor = 1.0 + 2.0 * ionized_fraction
    cmc = cmc_0 * cmc_factor

    # Ionic strength screening reduces repulsion → lowers CMC for charged drugs
    is_factor = max(0.5, 1.0 - ionized_fraction * ionic_strength_mM / 500.0)
    cmc *= is_factor

    # Clamp final CMC
    cmc = max(0.01, min(200.0, cmc))

    # Aggregation fractions
    conc = concentration_mg_mL
    if conc <= cmc:
        fraction_aggregated = 0.0
        fraction_monomer = 1.0
        effective_free_fraction = 1.0
    else:
        fraction_aggregated = (conc - cmc) / conc
        fraction_monomer = 1.0 - fraction_aggregated
        effective_free_fraction = cmc / conc

    # Aggregate size
    if fraction_aggregated < 1e-9:
        aggregate_size_nm = 0.3  # monomer size
    else:
        _raw = 0.3 * (conc / cmc) ** 0.3 * 50.0
        aggregate_size_nm = max(10.0, _raw)
    aggregate_size_nm = max(0.3, min(500.0, aggregate_size_nm))

    # Aggregation risk
    if fraction_aggregated < 0.05:
        aggregation_risk = "none"
    elif fraction_aggregated < 0.2:
        aggregation_risk = "low"
    elif fraction_aggregated < 0.5:
        aggregation_risk = "moderate"
    else:
        aggregation_risk = "high"

    # PK impact
    if fraction_aggregated < 0.1:
        pk_impact = "negligible"
    elif fraction_aggregated < 0.3:
        pk_impact = "minor"
    else:
        pk_impact = "significant"

    # Solubility reduction fold
    if effective_free_fraction > 0.0:
        solubility_reduction_fold = 1.0 / effective_free_fraction
    else:
        solubility_reduction_fold = 1.0
    solubility_reduction_fold = max(1.0, solubility_reduction_fold)

    # Notes
    notes = (
        f"CMC = {cmc:.3f} mg/mL; {fraction_aggregated * 100:.1f}% aggregated; "
        f"free fraction = {effective_free_fraction:.3f}; "
        f"risk: {aggregation_risk}; PK impact: {pk_impact}."
    )

    return SelfAssociationResult(
        drug_name=drug_name,
        concentration_mg_mL=conc,
        cmc_mg_mL=cmc,
        fraction_aggregated=fraction_aggregated,
        fraction_monomer=fraction_monomer,
        effective_free_fraction=effective_free_fraction,
        aggregate_size_nm=aggregate_size_nm,
        aggregation_risk=aggregation_risk,
        pk_impact=pk_impact,
        solubility_reduction_fold=solubility_reduction_fold,
        notes=notes,
    )


def screen_concentrations(
    drug_name: str,
    mw_Da: float,
    logp: float,
    concentrations_mg_mL: list,
    pka: float = 7.0,
    ionization_type: str = "neutral",
) -> list:
    """Screen a drug across multiple concentrations for self-association.

    Args:
        drug_name: Drug identifier.
        mw_Da: Molecular weight in Da.
        logp: Octanol-water partition coefficient.
        concentrations_mg_mL: List of concentrations to screen (mg/mL).
        pka: Acid/base pKa value.
        ionization_type: "neutral", "acid", or "base".

    Returns:
        List of SelfAssociationResult in the same order as input concentrations.
    """
    return [
        predict_self_association(
            drug_name=drug_name,
            mw_Da=mw_Da,
            logp=logp,
            concentration_mg_mL=c,
            pka=pka,
            ionization_type=ionization_type,
        )
        for c in concentrations_mg_mL
    ]
