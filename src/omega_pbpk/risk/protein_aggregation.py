"""Protein aggregation and amyloid prediction for biologic drug formulations.

Predicts aggregation risk for protein/biologic drugs (mAbs, peptides)
to guide formulation development.

References
----------
Chennamsetty 2009, Wang 2010 (AAPS J), ICH Q8R2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "ProteinAggregationResult",
    "assess_protein_aggregation",
    "optimize_formulation",
]

_EXCIPIENT_CREDITS: dict[str, float] = {
    "sucrose": -10,
    "trehalose": -10,
    "sorbitol": -8,
    "polysorbate80": -5,
    "polysorbate20": -5,
    "arginine": -8,
    "histidine": -5,
}


@dataclass(frozen=True)
class ProteinAggregationResult:
    """Protein aggregation assessment result."""

    protein_name: str
    mw_kDa: float
    ph: float
    temperature_C: float
    concentration_mg_mL: float
    ionic_strength_mM: float
    excipients: tuple[str, ...]
    aggregation_propensity_score: float
    aggregation_risk: str
    thermal_stability_proxy: float
    solubility_proxy_mg_mL: float
    iep_proxy: float
    recommended_ph_range: tuple[float, float]
    shelf_life_months_at_4C: float
    shelf_life_months_at_25C: float
    formulation_recommendation: str
    notes: str


def assess_protein_aggregation(
    protein_name: str,
    mw_kDa: float,
    ph: float = 7.0,
    temperature_C: float = 25.0,
    ionic_strength_mM: float = 150.0,
    concentration_mg_mL: float = 10.0,
    excipients: list[str] | None = None,
) -> ProteinAggregationResult:
    """Assess protein aggregation risk for a biologic formulation.

    Parameters
    ----------
    protein_name : Name of the protein drug.
    mw_kDa : Molecular weight in kDa (must be > 0).
    ph : Formulation pH (must be in (0, 14)).
    temperature_C : Temperature in Celsius.
    ionic_strength_mM : Ionic strength in mM.
    concentration_mg_mL : Protein concentration (must be > 0).
    excipients : List of excipient names.

    Returns
    -------
    ProteinAggregationResult

    Raises
    ------
    ValueError
        If any parameter is out of valid range.
    """
    if mw_kDa <= 0:
        raise ValueError("mw_kDa must be > 0")
    if concentration_mg_mL <= 0:
        raise ValueError("concentration_mg_mL must be > 0")
    if ph <= 0 or ph >= 14:
        raise ValueError("ph must be in (0, 14)")

    if excipients is None:
        excipients = []
    exc_tuple = tuple(excipients)

    # Thermal stability proxy
    tm_proxy = 70 - 0.5 * concentration_mg_mL - 0.1 * abs(ph - 7)

    # Solubility proxy
    sol_proxy = max(5, 200 - 2 * concentration_mg_mL - 5 * abs(ionic_strength_mM / 150 - 1))

    # Aggregation propensity score
    base = 20 + 2 * concentration_mg_mL + 5 * abs(ph - 7.0) + 0.1 * abs(ionic_strength_mM - 150)
    base += max(0, (temperature_C - 25) * 2)
    for exc in excipients:
        base += _EXCIPIENT_CREDITS.get(exc.lower(), 0)
    score = max(0, min(100, base))

    # Risk classification
    if score < 30:
        risk = "low"
    elif score < 60:
        risk = "moderate"
    else:
        risk = "high"

    # Isoelectric point proxy
    iep_proxy = 7.0 + 0.1 * (mw_kDa - 50) / 10

    # Recommended pH range
    rec_ph = (iep_proxy - 2, iep_proxy - 0.5)

    # Shelf life proxies (Arrhenius-based)
    t_ref = max(1, 60 - score * 0.5)
    t_25c = max(0.5, t_ref * math.exp(-0.1))

    # Formulation recommendation
    recs: list[str] = []
    exc_lower = [e.lower() for e in excipients]
    if risk in ("moderate", "high") and "sucrose" not in exc_lower:
        recs.append("Add cryoprotectant (sucrose/trehalose)")
    if abs(ph - iep_proxy) < 1:
        recs.append("Adjust pH away from isoelectric point")
    if concentration_mg_mL > 50:
        recs.append("Consider dilution or viscosity-reducing agents")
    if not recs:
        recs.append("Current formulation appears acceptable")
    recommendation = "; ".join(recs)

    notes = (
        f"Aggregation assessment for {protein_name}: "
        f"score={score:.1f}, risk={risk}, Tm_proxy={tm_proxy:.1f}C."
    )

    return ProteinAggregationResult(
        protein_name=protein_name,
        mw_kDa=mw_kDa,
        ph=ph,
        temperature_C=temperature_C,
        concentration_mg_mL=concentration_mg_mL,
        ionic_strength_mM=ionic_strength_mM,
        excipients=exc_tuple,
        aggregation_propensity_score=score,
        aggregation_risk=risk,
        thermal_stability_proxy=tm_proxy,
        solubility_proxy_mg_mL=sol_proxy,
        iep_proxy=iep_proxy,
        recommended_ph_range=rec_ph,
        shelf_life_months_at_4C=t_ref,
        shelf_life_months_at_25C=t_25c,
        formulation_recommendation=recommendation,
        notes=notes,
    )


def optimize_formulation(
    protein_name: str,
    mw_kDa: float,
    concentration_mg_mL: float,
    temperature_C: float = 25.0,
    ionic_strength_mM: float = 150.0,
    excipients: list[str] | None = None,
) -> list[ProteinAggregationResult]:
    """Screen formulations across a pH range and return sorted by score ascending.

    Parameters
    ----------
    protein_name : Name of the protein drug.
    mw_kDa : Molecular weight in kDa.
    concentration_mg_mL : Protein concentration.
    temperature_C : Temperature in Celsius.
    ionic_strength_mM : Ionic strength in mM.
    excipients : List of excipient names.

    Returns
    -------
    list[ProteinAggregationResult]
        One result per pH (5.0, 5.5, 6.0, 6.5, 7.0, 7.4), sorted by score ascending.
    """
    ph_values = [5.0, 5.5, 6.0, 6.5, 7.0, 7.4]
    results = [
        assess_protein_aggregation(
            protein_name=protein_name,
            mw_kDa=mw_kDa,
            ph=p,
            temperature_C=temperature_C,
            ionic_strength_mM=ionic_strength_mM,
            concentration_mg_mL=concentration_mg_mL,
            excipients=excipients,
        )
        for p in ph_values
    ]
    return sorted(results, key=lambda r: r.aggregation_propensity_score)
