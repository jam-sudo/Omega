"""Environmental fate and ecotoxicology risk assessment for pharmaceutical compounds."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "EnvironmentalFateResult",
    "assess_environmental_fate",
    "screen_environmental_risk",
    "estimate_aquatic_toxicity_lc50",
]


@dataclass(frozen=True)
class EnvironmentalFateResult:
    """Result of environmental fate and risk assessment."""

    compound_name: str
    logP: float
    mw: float
    bcf: float
    log_bcf: float
    biodegradation_t_half_days: float
    water_solubility_mg_per_L: float
    is_persistent: bool
    is_bioaccumulative: bool
    is_pbt: bool
    environmental_risk_class: str
    erq_estimate: float
    notes: str


def _estimate_biodegradation_t_half(logP: float, mw: float) -> float:
    """Estimate biodegradation half-life in days based on logP and MW."""
    if logP < 2.0 and mw < 300.0:
        return 2.0  # Readily biodegradable
    elif 2.0 <= logP <= 4.0:
        return 30.0  # Slowly biodegradable
    else:
        return 180.0  # Persistent (logP > 4 or mw > 500)


def _estimate_water_solubility(logP: float, mw: float) -> float:
    """Estimate water solubility in mg/L using simplified GSE equation."""
    # log_S (mol/L) = 0.5 - 0.89 * logP - 0.012 * mw
    log_s_mol_per_L = 0.5 - 0.89 * logP - 0.012 * mw
    # Convert mol/L to mg/L: multiply by MW (g/mol → mg/g = 1000 × g/mol)
    s_mg_per_L = max(0.001, (10.0**log_s_mol_per_L) * mw * 1000.0)
    return s_mg_per_L


def assess_environmental_fate(
    compound_name: str,
    logP: float,
    mw: float,
    pka_acid: float | None = None,
    pka_base: float | None = None,
    biodegradation_half_life_days: float | None = None,
    water_solubility_mg_per_L: float | None = None,
) -> EnvironmentalFateResult:
    """Assess environmental fate and ecotoxicology risk of a pharmaceutical compound.

    Args:
        compound_name: Name of the compound
        logP: Octanol-water partition coefficient
        mw: Molecular weight (g/mol), must be > 0
        pka_acid: Acidic pKa (optional)
        pka_base: Basic pKa (optional)
        biodegradation_half_life_days: Measured biodegradation t½ (optional)
        water_solubility_mg_per_L: Measured water solubility in mg/L (optional)

    Returns:
        EnvironmentalFateResult with risk classification

    Raises:
        ValueError: If mw <= 0
    """
    if mw <= 0.0:
        raise ValueError(f"mw must be > 0, got {mw}")

    # 1. Bioaccumulation factor (BCF) — Arnot & Gobas 2003 simplified
    log_bcf_raw = 0.85 * logP - 0.70
    log_bcf = max(0.0, min(6.0, log_bcf_raw))
    bcf = 10.0**log_bcf

    # 2. Biodegradation half-life
    if biodegradation_half_life_days is not None:
        t_half = biodegradation_half_life_days
    else:
        t_half = _estimate_biodegradation_t_half(logP, mw)

    # 3. Water solubility
    if water_solubility_mg_per_L is not None:
        solubility = water_solubility_mg_per_L
    else:
        solubility = _estimate_water_solubility(logP, mw)

    # 4. PBT screening
    is_persistent = t_half > 60.0
    is_bioaccumulative = bcf > 2000.0
    is_toxic_placeholder = logP > 5.0
    is_pbt = is_persistent and is_bioaccumulative and is_toxic_placeholder

    # 5. Environmental risk quotient (ERQ)
    # Simplified placeholder: ERQ = 1000 / (BCF * 10)
    erq_estimate = 1000.0 / (bcf * 10.0)

    if erq_estimate < 1.0:
        env_risk_class = "high"
    elif erq_estimate <= 10.0:
        env_risk_class = "medium"
    else:
        env_risk_class = "low"

    # Build notes
    note_parts = [
        f"log_BCF={log_bcf:.2f}",
        f"BCF={bcf:.1f}",
        f"t½={t_half:.0f}d",
        f"S={solubility:.2f}mg/L",
        f"ERQ={erq_estimate:.2f}",
    ]
    if is_pbt:
        note_parts.append("PBT compound")
    if is_persistent:
        note_parts.append("persistent")
    if is_bioaccumulative:
        note_parts.append("bioaccumulative")

    notes = "; ".join(note_parts)

    return EnvironmentalFateResult(
        compound_name=compound_name,
        logP=logP,
        mw=mw,
        bcf=bcf,
        log_bcf=log_bcf,
        biodegradation_t_half_days=t_half,
        water_solubility_mg_per_L=solubility,
        is_persistent=is_persistent,
        is_bioaccumulative=is_bioaccumulative,
        is_pbt=is_pbt,
        environmental_risk_class=env_risk_class,
        erq_estimate=erq_estimate,
        notes=notes,
    )


def screen_environmental_risk(compounds: list[dict]) -> list[EnvironmentalFateResult]:
    """Screen a list of compounds for environmental risk.

    Each dict must have keys: compound_name, logP, mw.
    Optional keys: pka_acid, pka_base, biodegradation_half_life_days, water_solubility_mg_per_L.

    Returns:
        List of EnvironmentalFateResult sorted by BCF descending.
    """
    results = []
    for cpd in compounds:
        result = assess_environmental_fate(
            compound_name=cpd["compound_name"],
            logP=cpd["logP"],
            mw=cpd["mw"],
            pka_acid=cpd.get("pka_acid"),
            pka_base=cpd.get("pka_base"),
            biodegradation_half_life_days=cpd.get("biodegradation_half_life_days"),
            water_solubility_mg_per_L=cpd.get("water_solubility_mg_per_L"),
        )
        results.append(result)

    results.sort(key=lambda r: r.bcf, reverse=True)
    return results


def estimate_aquatic_toxicity_lc50(logP: float, mw: float) -> float:
    """Estimate fish LC50 (mg/L) using simplified baseline QSAR for narcosis.

    Formula: log_LC50 = -0.87 * logP + 2.0
    LC50 = max(0.001, 10^log_LC50)

    Returns:
        LC50 in mg/L (lower value = more toxic)
    """
    log_lc50 = -0.87 * logP + 2.0
    lc50 = max(0.001, 10.0**log_lc50)
    return lc50
