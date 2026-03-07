"""Phase 520 — IV Precipitation Risk Assessment.

Predicts the risk of drug precipitation in IV formulations when two drugs
are mixed, using Henderson-Hasselbalch solubility calculations.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IVCompatibilityResult:
    """Result of an IV drug compatibility assessment."""

    drug_a_name: str
    drug_b_name: str
    sol_a_mg_mL: float
    sol_b_mg_mL: float
    precipitates_a: bool
    precipitates_b: bool
    risk: str
    recommendation: str
    notes: str


def calculate_solubility_at_ph(
    pka: float,
    logP: float,
    s_intrinsic_mg_mL: float,
    drug_type: str,
    ph: float,
) -> float:
    """Calculate drug solubility at a given pH using Henderson-Hasselbalch.

    Args:
        pka: Drug pKa value.
        logP: Octanol-water partition coefficient (log scale).
        s_intrinsic_mg_mL: Intrinsic (unionised) solubility in mg/mL (>0).
        drug_type: 'weak_acid', 'weak_base', or 'neutral'.
        ph: Target pH value.

    Returns:
        Solubility at the given pH in mg/mL.

    Raises:
        ValueError: If s_intrinsic_mg_mL <= 0 or drug_type is invalid.
    """
    if s_intrinsic_mg_mL <= 0:
        raise ValueError("s_intrinsic_mg_mL must be positive")
    drug_type = drug_type.lower()
    if drug_type not in ("weak_acid", "weak_base", "neutral"):
        raise ValueError("drug_type must be 'weak_acid', 'weak_base', or 'neutral'")

    if drug_type == "weak_acid":
        solubility = s_intrinsic_mg_mL * (1.0 + 10.0 ** (ph - pka))
    elif drug_type == "weak_base":
        solubility = s_intrinsic_mg_mL * (1.0 + 10.0 ** (pka - ph))
    else:  # neutral
        solubility = s_intrinsic_mg_mL

    return solubility


def assess_iv_compatibility(
    drug_a_name: str,
    drug_b_name: str,
    conc_a_mg_mL: float,
    conc_b_mg_mL: float,
    ph_mix: float,
    pka_a: float,
    pka_b: float,
    logP_a: float,
    logP_b: float,
    s_intrinsic_a: float,
    s_intrinsic_b: float,
    drug_type_a: str = "weak_acid",
    drug_type_b: str = "weak_base",
) -> IVCompatibilityResult:
    """Assess IV compatibility between two drugs at a mixed pH.

    Args:
        drug_a_name: Name of drug A.
        drug_b_name: Name of drug B.
        conc_a_mg_mL: Concentration of drug A in mg/mL (>=0).
        conc_b_mg_mL: Concentration of drug B in mg/mL (>=0).
        ph_mix: pH of the mixed solution.
        pka_a: pKa of drug A.
        pka_b: pKa of drug B.
        logP_a: logP of drug A.
        logP_b: logP of drug B.
        s_intrinsic_a: Intrinsic solubility of drug A in mg/mL.
        s_intrinsic_b: Intrinsic solubility of drug B in mg/mL.
        drug_type_a: Type of drug A ('weak_acid', 'weak_base', or 'neutral').
        drug_type_b: Type of drug B ('weak_acid', 'weak_base', or 'neutral').

    Returns:
        IVCompatibilityResult with risk assessment.

    Raises:
        ValueError: If concentrations are negative or solubilities invalid.
    """
    if conc_a_mg_mL < 0:
        raise ValueError("conc_a_mg_mL must be non-negative")
    if conc_b_mg_mL < 0:
        raise ValueError("conc_b_mg_mL must be non-negative")

    sol_a = calculate_solubility_at_ph(pka_a, logP_a, s_intrinsic_a, drug_type_a, ph_mix)
    sol_b = calculate_solubility_at_ph(pka_b, logP_b, s_intrinsic_b, drug_type_b, ph_mix)

    precipitates_a = conc_a_mg_mL > sol_a
    precipitates_b = conc_b_mg_mL > sol_b

    risk = "high" if (precipitates_a or precipitates_b) else "low"

    precipitating = []
    if precipitates_a:
        precipitating.append(drug_a_name)
    if precipitates_b:
        precipitating.append(drug_b_name)

    if risk == "high":
        recommendation = (
            f"Do NOT mix {drug_a_name} and {drug_b_name} at pH {ph_mix:.1f}. "
            f"Precipitation predicted for: {', '.join(precipitating)}."
        )
        notes = (
            f"Solubility of {drug_a_name} at pH {ph_mix:.1f}: {sol_a:.3f} mg/mL "
            f"(conc: {conc_a_mg_mL} mg/mL); "
            f"solubility of {drug_b_name}: {sol_b:.3f} mg/mL "
            f"(conc: {conc_b_mg_mL} mg/mL). "
            "Use separate IV lines or reformulate at appropriate pH."
        )
    else:
        recommendation = (
            f"{drug_a_name} and {drug_b_name} are predicted compatible at pH {ph_mix:.1f}."
        )
        notes = (
            f"Solubility of {drug_a_name}: {sol_a:.3f} mg/mL (conc: {conc_a_mg_mL} mg/mL); "
            f"solubility of {drug_b_name}: {sol_b:.3f} mg/mL (conc: {conc_b_mg_mL} mg/mL). "
            "Verify compatibility experimentally before clinical use."
        )

    return IVCompatibilityResult(
        drug_a_name=drug_a_name,
        drug_b_name=drug_b_name,
        sol_a_mg_mL=sol_a,
        sol_b_mg_mL=sol_b,
        precipitates_a=precipitates_a,
        precipitates_b=precipitates_b,
        risk=risk,
        recommendation=recommendation,
        notes=notes,
    )


def ph_stability_window(
    pka: float,
    drug_type: str,
    s_intrinsic_mg_mL: float,
    target_conc_mg_mL: float,
    ph_range: tuple = (4.0, 9.0),
    n_points: int = 50,
) -> dict:
    """Compute solubility across a pH range and identify stable pH window.

    Args:
        pka: Drug pKa.
        drug_type: 'weak_acid', 'weak_base', or 'neutral'.
        s_intrinsic_mg_mL: Intrinsic solubility in mg/mL.
        target_conc_mg_mL: Target drug concentration in mg/mL.
        ph_range: Tuple of (min_pH, max_pH) to scan.
        n_points: Number of pH points to evaluate.

    Returns:
        Dict with ph_values, solubility_mg_mL, stable_ph_min, stable_ph_max.

    Raises:
        ValueError: If parameters are invalid.
    """
    if s_intrinsic_mg_mL <= 0:
        raise ValueError("s_intrinsic_mg_mL must be positive")
    if target_conc_mg_mL <= 0:
        raise ValueError("target_conc_mg_mL must be positive")
    if n_points < 2:
        raise ValueError("n_points must be at least 2")
    if ph_range[0] >= ph_range[1]:
        raise ValueError("ph_range[0] must be less than ph_range[1]")

    ph_min, ph_max = ph_range
    step = (ph_max - ph_min) / (n_points - 1)

    ph_values = [ph_min + i * step for i in range(n_points)]
    solubility_values = [
        calculate_solubility_at_ph(pka, 0.0, s_intrinsic_mg_mL, drug_type, ph) for ph in ph_values
    ]

    stable_ph_min = None
    stable_ph_max = None

    for ph, sol in zip(ph_values, solubility_values):
        if sol >= target_conc_mg_mL:
            if stable_ph_min is None:
                stable_ph_min = ph
            stable_ph_max = ph

    return {
        "ph_values": ph_values,
        "solubility_mg_mL": solubility_values,
        "stable_ph_min": stable_ph_min,
        "stable_ph_max": stable_ph_max,
    }
