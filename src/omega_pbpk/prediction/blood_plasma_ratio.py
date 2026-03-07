"""Phase 738 — Blood-to-Plasma Ratio Predictor.

Predicts the blood-to-plasma concentration ratio (Rb = Cb/Cp) for a drug.
Rb is critical for correctly scaling IV pharmacokinetic parameters:
    CL_blood = CL_plasma / Rb

Based on Austin et al. 2016 and Hinderling 1997 empirical relationships.
RBC partitioning is driven by lipophilicity, charge at pH 7.4, and
plasma protein binding.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BloodPlasmaRatioResult:
    """Result of blood-to-plasma ratio prediction."""

    compound_name: str
    logp: float
    charge_at_ph74: float
    fu_plasma: float
    rb: float
    cl_blood_L_per_h: float | None
    rbc_affinity: str
    impact_on_vd: str
    notes: str


def predict_blood_plasma_ratio(
    compound_name: str,
    logp: float = 2.0,
    charge_at_ph74: float = 0.0,
    fu_plasma: float = 0.5,
    cl_plasma_L_per_h: float | None = None,
) -> BloodPlasmaRatioResult:
    """Predict the blood-to-plasma concentration ratio (Rb).

    Uses empirical relationships based on lipophilicity, ionization state,
    and plasma protein binding (Austin et al. 2016; Hinderling 1997).

    Args:
        compound_name: Name of the compound.
        logp: Octanol-water partition coefficient (log scale). Higher logP
            generally increases RBC partitioning.
        charge_at_ph74: Net charge at physiological pH 7.4.
            Positive values indicate basic/cationic drugs (higher RBC affinity).
            Negative values indicate acidic/anionic drugs (excluded from RBC).
            Zero indicates neutral compound.
        fu_plasma: Unbound fraction in plasma (0 to 1). High plasma protein
            binding (low fu_plasma) keeps drug in plasma, reducing Rb.
        cl_plasma_L_per_h: Plasma clearance (L/h). If provided, blood clearance
            is computed as CL_blood = CL_plasma / Rb.

    Returns:
        BloodPlasmaRatioResult with Rb, optional CL_blood, and interpretation.
    """
    # Base estimate: hematocrit effect gives ~55% of drug in RBC water fraction
    # logP scaling: each unit of logP adds modest RBC partitioning
    rb_base = 0.55 + logp * 0.05

    # Charge effect at physiological pH
    if charge_at_ph74 > 0:
        # Cationic drugs bind RBC membrane phospholipids → higher Rb
        rb_base *= 1.2
    elif charge_at_ph74 < 0:
        # Anionic drugs are excluded from negatively charged RBC membrane
        rb_base *= 0.85

    # Plasma protein binding: highly bound drugs stay in plasma with protein
    if fu_plasma < 0.1:
        rb_base *= 0.7

    # Clamp to physiologically plausible range
    rb = max(0.4, min(3.0, rb_base))

    # Compute blood clearance if plasma clearance provided
    cl_blood: float | None
    if cl_plasma_L_per_h is not None:
        cl_blood = cl_plasma_L_per_h / rb
    else:
        cl_blood = None

    # RBC affinity classification
    if rb < 0.8:
        rbc_affinity = "low"
    elif rb <= 1.2:
        rbc_affinity = "neutral"
    else:
        rbc_affinity = "high"

    # Impact on volume of distribution estimation
    if rb < 1.0:
        impact_on_vd = "may underestimate Vd"
    else:
        impact_on_vd = "accurate"

    # Build notes
    notes_parts: list[str] = []
    if logp > 3:
        notes_parts.append("High logP promotes RBC partitioning")
    if charge_at_ph74 > 0:
        notes_parts.append("Cationic character enhances RBC membrane binding")
    elif charge_at_ph74 < 0:
        notes_parts.append("Anionic character leads to RBC exclusion")
    if fu_plasma < 0.1:
        notes_parts.append("High plasma protein binding reduces blood partitioning")
    if rb < 1.0:
        notes_parts.append(
            "Rb < 1: drug excluded from RBC; using plasma concentration "
            "overestimates blood concentration"
        )
    elif rb > 1.2:
        notes_parts.append("Rb > 1.2: significant RBC partitioning; blood CL lower than plasma CL")
    else:
        notes_parts.append("Rb near unity: plasma and blood concentrations are similar")

    notes = "; ".join(notes_parts) if notes_parts else "Standard blood-to-plasma partitioning"

    return BloodPlasmaRatioResult(
        compound_name=compound_name,
        logp=logp,
        charge_at_ph74=charge_at_ph74,
        fu_plasma=fu_plasma,
        rb=rb,
        cl_blood_L_per_h=cl_blood,
        rbc_affinity=rbc_affinity,
        impact_on_vd=impact_on_vd,
        notes=notes,
    )


def screen_blood_plasma_ratio(
    compounds: list[dict],
) -> list[BloodPlasmaRatioResult]:
    """Screen a list of compounds for blood-to-plasma ratio.

    Args:
        compounds: List of dicts with keys matching predict_blood_plasma_ratio
            parameters (compound_name required; others optional).

    Returns:
        List of BloodPlasmaRatioResult sorted by rb descending
        (highest RBC affinity first).
    """
    results: list[BloodPlasmaRatioResult] = []
    for comp in compounds:
        result = predict_blood_plasma_ratio(
            compound_name=comp["compound_name"],
            logp=comp.get("logp", 2.0),
            charge_at_ph74=comp.get("charge_at_ph74", 0.0),
            fu_plasma=comp.get("fu_plasma", 0.5),
            cl_plasma_L_per_h=comp.get("cl_plasma_L_per_h", None),
        )
        results.append(result)

    results.sort(key=lambda r: r.rb, reverse=True)
    return results
