"""Phase 1068 — Drug Aggregation Risk Profiler.

Predicts aggregation risk for proteins/biologics during manufacturing and storage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_VALID_MOLECULE_TYPES = {"mAb", "protein", "peptide", "ADC", "fusion"}

# Base rates (multiplied by 1e-4 to get /h)
_BASE_RATES: dict[str, float] = {
    "mAb": 0.5,
    "protein": 1.0,
    "peptide": 2.0,
    "ADC": 1.5,
    "fusion": 0.8,
}


@dataclass(frozen=True)
class AggregationRiskResult:
    """Result of aggregation risk prediction."""

    molecule_name: str
    molecule_type: str
    aggregation_rate_per_h: float  # first-order aggregation rate (/h)
    monomer_fraction_at_1y: float  # fraction remaining as monomer after 1 year (~8760h)
    monomer_fraction_at_2y: float  # fraction remaining as monomer after 2 years
    aggregation_risk_score: float  # 0-100
    risk_category: str  # "low" (<25), "moderate" (25-50), "high" (50-75), "critical" (>75)
    primary_aggregation_mechanism: (
        str  # "hydrophobic", "electrostatic", "stress_induced", "chemical"
    )
    recommended_ph: float  # pH that minimizes aggregation
    recommended_excipients: tuple  # (str, ...) — up to 3 stabilizers
    notes: str


def predict_aggregation_risk(
    molecule_name: str,
    mw_Da: float,
    pi: float,  # isoelectric point (pI)
    hydrophobicity_score: float,  # 0-10 (10 = very hydrophobic surface)
    molecule_type: str = "mAb",  # "mAb", "protein", "peptide", "ADC", "fusion"
    formulation_ph: float = 7.0,
    ionic_strength_mM: float = 150.0,
    protein_conc_mg_mL: float = 10.0,
    has_disulfide: bool = True,
    has_glycosylation: bool = False,
    storage_temp_C: float = 4.0,
) -> AggregationRiskResult:
    """Predict aggregation risk for a biologic molecule.

    Args:
        molecule_name: Name of the molecule.
        mw_Da: Molecular weight in Daltons (must be > 0).
        pi: Isoelectric point (pI), must be in [3, 11].
        hydrophobicity_score: Surface hydrophobicity score, 0-10.
        molecule_type: One of mAb, protein, peptide, ADC, fusion.
        formulation_ph: pH of formulation buffer.
        ionic_strength_mM: Ionic strength in mM.
        protein_conc_mg_mL: Protein concentration in mg/mL.
        has_disulfide: Whether the molecule has disulfide bonds.
        has_glycosylation: Whether the molecule is glycosylated.
        storage_temp_C: Storage temperature in Celsius, must be in [-20, 60].

    Returns:
        AggregationRiskResult with computed aggregation parameters.

    Raises:
        ValueError: If any parameter is outside valid range.
    """
    # Validation
    if mw_Da <= 0:
        raise ValueError(f"mw_Da must be > 0, got {mw_Da}")
    if pi < 3 or pi > 11:
        raise ValueError(f"pi must be in [3, 11], got {pi}")
    if hydrophobicity_score < 0 or hydrophobicity_score > 10:
        raise ValueError(f"hydrophobicity_score must be in [0, 10], got {hydrophobicity_score}")
    if protein_conc_mg_mL <= 0:
        raise ValueError(f"protein_conc_mg_mL must be > 0, got {protein_conc_mg_mL}")
    if molecule_type not in _VALID_MOLECULE_TYPES:
        raise ValueError(
            f"molecule_type must be one of {_VALID_MOLECULE_TYPES}, got {molecule_type!r}"
        )
    if storage_temp_C < -20 or storage_temp_C > 60:
        raise ValueError(f"storage_temp_C must be in [-20, 60], got {storage_temp_C}")

    # Base rate from molecule type (dimensionless, to be multiplied by 1e-4)
    base_rate_raw = _BASE_RATES[molecule_type]

    # pH-induced aggregation (near pI → less electrostatic repulsion → more aggregation)
    ph_distance = abs(formulation_ph - pi)
    if ph_distance < 1:
        ph_factor = 3.0
    elif ph_distance < 2:
        ph_factor = 1.5
    else:
        ph_factor = 1.0

    # Hydrophobic aggregation
    hydro_factor = 1.0 + hydrophobicity_score * 0.3

    # Concentration dependence
    conc_factor = 1.0 + protein_conc_mg_mL * 0.02

    # Ionic strength effect (optimal ~100-200 mM; extremes worse)
    if ionic_strength_mM < 50 or ionic_strength_mM > 500:
        is_factor = 1.5
    else:
        is_factor = 1.0

    # Temperature effect (Arrhenius, reference at 4°C)
    temp_factor = math.exp(0.05 * (storage_temp_C - 4.0))
    temp_factor = max(0.1, temp_factor)

    # Stabilizer factors
    k_factor = 1.0
    if has_disulfide:
        k_factor *= 0.8  # reduces chemical aggregation
    if has_glycosylation:
        k_factor *= 0.7  # glycans reduce hydrophobic aggregation

    # Final aggregation rate (/h)
    aggregation_rate = (
        base_rate_raw
        * ph_factor
        * hydro_factor
        * conc_factor
        * is_factor
        * temp_factor
        * k_factor
        * 1e-4  # convert to /h
    )

    # Monomer fraction at 1 and 2 years
    t_1y = 8760.0  # hours in 1 year
    t_2y = 17520.0  # hours in 2 years

    monomer_at_1y = math.exp(-aggregation_rate * t_1y)
    monomer_at_1y = max(0.01, min(1.0, monomer_at_1y))

    monomer_at_2y = math.exp(-aggregation_rate * t_2y)
    monomer_at_2y = max(0.01, min(1.0, monomer_at_2y))

    # Aggregation risk score (0-100)
    aggregation_risk_score = (1.0 - monomer_at_1y) * 100.0
    aggregation_risk_score = max(0.0, min(100.0, aggregation_risk_score))

    # Risk category
    if aggregation_risk_score < 25:
        risk_category = "low"
    elif aggregation_risk_score < 50:
        risk_category = "moderate"
    elif aggregation_risk_score < 75:
        risk_category = "high"
    else:
        risk_category = "critical"

    # Primary aggregation mechanism
    if ph_factor > 2 and ph_distance < 1:
        primary_mechanism = "electrostatic"
    elif hydrophobicity_score > 6:
        primary_mechanism = "hydrophobic"
    elif storage_temp_C > 25:
        primary_mechanism = "stress_induced"
    else:
        primary_mechanism = "chemical"

    # Recommended pH (2 units from pI, clamped to [5.0, 8.5])
    if pi < 7:
        rec_ph = pi + 2.0
    else:
        rec_ph = pi - 2.0
    rec_ph = max(5.0, min(8.5, rec_ph))

    # Recommended excipients based on mechanism
    if primary_mechanism == "hydrophobic":
        rec_excipients = ("polysorbate_80", "sucrose", "arginine")
    elif primary_mechanism == "electrostatic":
        rec_excipients = ("NaCl", "arginine", "histidine")
    elif primary_mechanism == "chemical":
        rec_excipients = ("EDTA", "methionine", "ascorbic_acid")
    else:  # stress_induced
        rec_excipients = ("sucrose", "trehalose", "polysorbate_20")

    # Notes
    note_parts = [
        f"molecule_type={molecule_type}",
        f"ph_factor={ph_factor:.1f}",
        f"hydro_factor={hydro_factor:.2f}",
        f"risk_score={aggregation_risk_score:.1f}",
        f"mechanism={primary_mechanism}",
        f"monomer_1y={monomer_at_1y:.3f}",
    ]
    if has_glycosylation:
        note_parts.append("glycosylation stabilizes molecule")
    if has_disulfide:
        note_parts.append("disulfide bonds reduce chemical aggregation")
    notes = "; ".join(note_parts)

    return AggregationRiskResult(
        molecule_name=molecule_name,
        molecule_type=molecule_type,
        aggregation_rate_per_h=aggregation_rate,
        monomer_fraction_at_1y=monomer_at_1y,
        monomer_fraction_at_2y=monomer_at_2y,
        aggregation_risk_score=aggregation_risk_score,
        risk_category=risk_category,
        primary_aggregation_mechanism=primary_mechanism,
        recommended_ph=rec_ph,
        recommended_excipients=rec_excipients,
        notes=notes,
    )
