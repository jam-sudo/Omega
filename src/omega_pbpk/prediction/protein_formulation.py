"""Protein/mAb drug formulation optimization — Phase 507."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExcipientCompatibilityResult:
    """Result of excipient compatibility analysis for a protein drug."""

    protein_name: str
    ph: float
    net_charge: float
    recommended_stabilizers: tuple
    recommended_buffers: tuple
    surfactant: str
    tonicity_agent: str
    compatibility_score: float
    notes: str


@dataclass
class FormulationRecommendationResult:
    """Comprehensive formulation recommendation for a protein drug."""

    protein_name: str
    route: str
    target_concentration_mg_mL: float
    buffer_system: dict
    excipients: list
    stabilizers: list
    surfactant: str
    tonicity_agent: str
    viscosity_strategy: str
    storage_conditions: str
    overall_score: float
    risks: list
    notes: str


def _net_charge(ph: float, pi: float) -> float:
    """Estimate net charge: positive if pH < pI, negative if pH > pI."""
    return pi - ph


def _select_buffers(target_ph: float) -> list[str]:
    """Select appropriate buffers for target pH."""
    if target_ph < 4.0:
        return ["citrate", "glycine"]
    elif target_ph <= 6.0:
        return ["histidine", "acetate"]
    elif target_ph <= 8.0:
        return ["phosphate", "histidine"]
    else:
        return ["Tris", "borate"]


def predict_excipient_compatibility(
    protein_name: str,
    isoelectric_point: float,
    mw_kDa: float,
    ph: float = 7.4,
    ionic_strength_mM: float = 150.0,
) -> ExcipientCompatibilityResult:
    """Predict optimal excipient compatibility for a protein drug formulation.

    Parameters
    ----------
    protein_name:
        Name identifier for the protein/mAb.
    isoelectric_point:
        Isoelectric point (pI) of the protein (pH units).
    mw_kDa:
        Molecular weight in kDa.
    ph:
        Target formulation pH.
    ionic_strength_mM:
        Ionic strength of formulation buffer in mM.

    Returns
    -------
    ExcipientCompatibilityResult
        Excipient recommendations and compatibility score.
    """
    if mw_kDa <= 0:
        raise ValueError("mw_kDa must be positive.")
    if isoelectric_point <= 0:
        raise ValueError("isoelectric_point must be positive.")
    if not (0.0 < ph < 14.0):
        raise ValueError("ph must be between 0 and 14.")

    net_charge = _net_charge(ph, isoelectric_point)
    charge_distance = abs(net_charge)

    # Buffer selection
    recommended_buffers = tuple(_select_buffers(ph))

    # Stabilizer selection
    stabilizers = []
    if charge_distance < 1.0:
        # Near pI: high aggregation risk
        stabilizers.append("sucrose")
        stabilizers.append("polysorbate 80")
    elif charge_distance < 2.0:
        stabilizers.append("sucrose")
    if mw_kDa > 100:
        if "sucrose" not in stabilizers:
            stabilizers.append("sucrose")
        stabilizers.append("trehalose")
    if not stabilizers:
        stabilizers.append("sucrose")

    # Surfactant
    surfactant = "polysorbate 80" if charge_distance < 2.0 else "polysorbate 20"

    # Tonicity agent
    tonicity_agent = "sodium chloride" if ionic_strength_mM >= 100 else "mannitol"

    # Compatibility score: penalize near-pI, reward electrostatic repulsion
    score = 50.0
    if charge_distance >= 3.0:
        score += 30.0
    elif charge_distance >= 2.0:
        score += 20.0
    elif charge_distance >= 1.0:
        score += 5.0
    else:
        score -= 20.0

    # Penalize extreme pH
    if ph < 4.0 or ph > 9.0:
        score -= 15.0

    # High MW: mild penalty for complexity
    if mw_kDa > 150:
        score -= 5.0

    # Ionic strength adjustment
    if 100 <= ionic_strength_mM <= 200:
        score += 5.0

    score = max(0.0, min(100.0, score))

    # Notes
    notes_parts = []
    if charge_distance < 1.0:
        notes_parts.append(
            f"pH {ph:.1f} is near pI {isoelectric_point:.1f}: high aggregation risk."
        )
    else:
        notes_parts.append(
            f"pH {ph:.1f} provides charge separation of {charge_distance:.1f} pH units from pI."
        )
    if mw_kDa > 100:
        notes_parts.append("High MW protein: cryoprotection recommended.")
    notes = " ".join(notes_parts)

    return ExcipientCompatibilityResult(
        protein_name=protein_name,
        ph=ph,
        net_charge=net_charge,
        recommended_stabilizers=tuple(stabilizers),
        recommended_buffers=recommended_buffers,
        surfactant=surfactant,
        tonicity_agent=tonicity_agent,
        compatibility_score=score,
        notes=notes,
    )


def optimize_buffer_system(
    target_ph: float,
    protein_pi: float,
    stability_range: tuple = (3.0, 9.0),
) -> dict:
    """Optimize buffer system selection for a target pH and protein pI.

    Parameters
    ----------
    target_ph:
        Target formulation pH.
    protein_pi:
        Protein isoelectric point.
    stability_range:
        Tuple of (min_pH, max_pH) for protein stability.

    Returns
    -------
    dict
        Buffer system recommendation with primary buffer, pH range,
        and concentration guidance.
    """
    if not (0.0 < target_ph < 14.0):
        raise ValueError("target_ph must be between 0 and 14.")
    if protein_pi <= 0:
        raise ValueError("protein_pi must be positive.")

    min_stable, max_stable = stability_range
    if target_ph < min_stable or target_ph > max_stable:
        warning = f"Target pH {target_ph:.1f} is outside protein stability range {stability_range}."
    else:
        warning = None

    buffers = _select_buffers(target_ph)
    primary_buffer = buffers[0]

    # pKa reference values for common buffers
    pka_map = {
        "histidine": 6.0,
        "acetate": 4.76,
        "phosphate": 7.2,
        "citrate": 3.1,
        "Tris": 8.1,
        "glycine": 2.35,
        "borate": 9.24,
    }

    pka = pka_map.get(primary_buffer, target_ph)
    buffering_capacity = "good" if abs(target_ph - pka) <= 1.0 else "moderate"

    charge_distance = abs(target_ph - protein_pi)
    if charge_distance < 1.0:
        ph_recommendation = (
            f"Consider shifting pH away from pI ({protein_pi:.1f}) to reduce aggregation risk."
        )
    else:
        ph_recommendation = (
            f"pH {target_ph:.1f} provides adequate charge separation from pI ({protein_pi:.1f})."
        )

    result = {
        "primary_buffer": primary_buffer,
        "alternative_buffers": buffers[1:] if len(buffers) > 1 else [],
        "recommended_concentration_mM": 10 if primary_buffer == "phosphate" else 20,
        "target_ph": target_ph,
        "pka_primary": pka,
        "buffering_capacity": buffering_capacity,
        "ph_recommendation": ph_recommendation,
    }
    if warning:
        result["warning"] = warning

    return result


def predict_aggregation_propensity(
    mw_kDa: float,
    isoelectric_point: float,
    ph: float = 7.4,
    temperature_C: float = 25.0,
    concentration_mg_mL: float = 1.0,
) -> float:
    """Predict aggregation propensity score (0-1) for a protein.

    Parameters
    ----------
    mw_kDa:
        Molecular weight in kDa.
    isoelectric_point:
        Protein pI.
    ph:
        Formulation pH.
    temperature_C:
        Storage/formulation temperature in Celsius.
    concentration_mg_mL:
        Protein concentration in mg/mL.

    Returns
    -------
    float
        Aggregation propensity score from 0 (low) to 1 (high).
    """
    if mw_kDa <= 0:
        raise ValueError("mw_kDa must be positive.")
    if isoelectric_point <= 0:
        raise ValueError("isoelectric_point must be positive.")

    charge_distance = abs(ph - isoelectric_point)

    # Base score from charge distance: near pI → high aggregation
    if charge_distance < 0.5:
        charge_score = 0.9
    elif charge_distance < 1.0:
        charge_score = 0.75
    elif charge_distance < 2.0:
        charge_score = 0.5
    elif charge_distance < 3.0:
        charge_score = 0.3
    else:
        charge_score = 0.15

    # MW contribution: larger proteins aggregate more easily
    mw_score = min(0.3, mw_kDa / 1000.0)

    # Temperature contribution: elevated temp increases aggregation
    if temperature_C <= 4.0:
        temp_score = 0.0
    elif temperature_C <= 25.0:
        temp_score = 0.05
    elif temperature_C <= 37.0:
        temp_score = 0.15
    else:
        temp_score = 0.25

    # Concentration: higher conc increases self-association
    if concentration_mg_mL <= 1.0:
        conc_score = 0.0
    elif concentration_mg_mL <= 10.0:
        conc_score = 0.05
    elif concentration_mg_mL <= 50.0:
        conc_score = 0.1
    else:
        conc_score = 0.2

    total = charge_score * 0.6 + mw_score + temp_score + conc_score
    return max(0.0, min(1.0, total))


def formulation_recommendation(
    protein_name: str,
    mw_kDa: float,
    pi: float,
    target_concentration_mg_mL: float,
    route: str = "iv",
) -> FormulationRecommendationResult:
    """Generate comprehensive formulation recommendation for a protein drug.

    Parameters
    ----------
    protein_name:
        Name of the protein/mAb.
    mw_kDa:
        Molecular weight in kDa.
    pi:
        Isoelectric point.
    target_concentration_mg_mL:
        Target protein concentration.
    route:
        Administration route ('iv', 'sc', 'im').

    Returns
    -------
    FormulationRecommendationResult
        Comprehensive formulation recommendation.
    """
    if mw_kDa <= 0:
        raise ValueError("mw_kDa must be positive.")
    if pi <= 0:
        raise ValueError("pi must be positive.")
    if target_concentration_mg_mL <= 0:
        raise ValueError("target_concentration_mg_mL must be positive.")

    route = route.lower()
    valid_routes = ("iv", "sc", "im")
    if route not in valid_routes:
        raise ValueError(f"route must be one of {valid_routes}.")

    # Determine target pH: for basic proteins (pI > 8) use slightly acidic pH
    if pi > 8.0:
        target_ph = 5.5
    elif pi < 5.0:
        target_ph = 7.0
    else:
        target_ph = 6.0

    buffer_system = optimize_buffer_system(target_ph, pi)
    compat = predict_excipient_compatibility(protein_name, pi, mw_kDa, ph=target_ph)

    excipients = list(compat.recommended_stabilizers) + [compat.tonicity_agent]
    stabilizers = list(compat.recommended_stabilizers)

    # Viscosity strategy for SC high-concentration formulations
    if route == "sc" and target_concentration_mg_mL > 50:
        viscosity_strategy = (
            "High-concentration SC formulation: consider L-arginine or NaCl to reduce viscosity; "
            "evaluate co-formulation with hyaluronidase; target ≤20 cP at injection."
        )
    elif route == "sc":
        viscosity_strategy = (
            "SC route: monitor viscosity at target concentration; "
            "consider arginine hydrochloride if viscosity exceeds 10 cP."
        )
    else:
        viscosity_strategy = (
            "IV route: viscosity generally not limiting; standard dilution in NS or D5W."
        )

    # Storage conditions
    if route == "sc" and target_concentration_mg_mL > 50:
        storage_conditions = "2-8°C, protected from light; avoid freeze-thaw cycles."
    else:
        storage_conditions = "2-8°C, protected from light; single freeze-thaw cycle acceptable."

    # Risks
    risks = []
    agg_score = predict_aggregation_propensity(
        mw_kDa, pi, ph=target_ph, concentration_mg_mL=target_concentration_mg_mL
    )
    if agg_score > 0.6:
        risks.append("High aggregation propensity — consider additional stabilization.")
    if target_concentration_mg_mL > 100:
        risks.append("Very high concentration: opalescence and viscosity may be limiting.")
    if route == "sc" and target_concentration_mg_mL < 50:
        risks.append("SC route with low concentration may require large injection volume (>2 mL).")

    # Overall score
    overall_score = max(0.0, min(100.0, compat.compatibility_score - agg_score * 20))

    notes = (
        f"Protein MW {mw_kDa:.0f} kDa, pI {pi:.1f}, target pH {target_ph:.1f}. "
        f"Aggregation propensity: {agg_score:.2f}."
    )

    return FormulationRecommendationResult(
        protein_name=protein_name,
        route=route,
        target_concentration_mg_mL=target_concentration_mg_mL,
        buffer_system=buffer_system,
        excipients=excipients,
        stabilizers=stabilizers,
        surfactant=compat.surfactant,
        tonicity_agent=compat.tonicity_agent,
        viscosity_strategy=viscosity_strategy,
        storage_conditions=storage_conditions,
        overall_score=overall_score,
        risks=risks,
        notes=notes,
    )
