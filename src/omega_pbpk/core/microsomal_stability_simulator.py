"""Microsomal stability assay simulator.

Simulates in vitro microsomal stability assays and scales CLint to in vivo
hepatic clearance using the well-stirred liver model.

References:
    - Obach RS. Drug Metab Dispos. 1999;27(11):1350-9.
    - Houston JB. Biochem Pharmacol. 1994;47(9):1469-79.
    - Davies B, Morris T. Pharm Res. 1993;10(7):1093-5.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Species physiological constants
# ---------------------------------------------------------------------------

# Microsomal protein per gram of liver (mg/g liver)
_PROTEIN_PER_G_LIVER: dict[str, float] = {
    "human": 52.5,
    "rat": 52.5,
    "mouse": 45.0,
    "dog": 40.0,
}

# Liver weight per body weight (g liver / kg BW)
_G_LIVER_PER_KG_BW: dict[str, float] = {
    "human": 21.4,
    "rat": 40.0,
    "mouse": 88.0,
    "dog": 31.0,
}

# Hepatic blood flow (mL/min/kg BW)
_Q_H_ML_PER_MIN_PER_KG: dict[str, float] = {
    "human": 20.7,
    "rat": 70.0,
    "mouse": 90.0,
    "dog": 30.0,
}

_VALID_SPECIES = frozenset({"human", "rat", "mouse", "dog"})

# Standard incubation volume (uL)
_INCUBATION_VOL_UL: float = 1000.0


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MicrosomalStabilityResult:
    """Result from a simulated microsomal stability assay.

    Attributes:
        drug_name: Identifier for the compound.
        species: Species used ("human", "rat", "mouse", "dog").
        microsomal_protein_mg_per_mL: Protein concentration in assay (mg/mL).
        substrate_conc_uM: Initial substrate concentration (uM).
        clint_uL_per_min_per_mg: In vitro CLint per mg microsomal protein.
        clint_mL_per_min_per_kg: Scaled in vivo CLint (mL/min/kg BW).
        cl_hepatic_mL_per_min_per_kg: Hepatic CL via well-stirred model.
        fu_mic: Unbound fraction in microsomes (0–1].
        t_half_microsomal_min: Apparent half-life in the assay (min).
        metabolic_stability: "high" (t1/2>30 min), "moderate" (>10 min), "low".
        predicted_clearance_class: "low", "moderate", or "high" hepatic CL.
        scaling_factor: mg_protein/g_liver * g_liver/kg_BW.
        notes: Textual summary notes.
    """

    drug_name: str
    species: str
    microsomal_protein_mg_per_mL: float
    substrate_conc_uM: float
    clint_uL_per_min_per_mg: float
    clint_mL_per_min_per_kg: float
    cl_hepatic_mL_per_min_per_kg: float
    fu_mic: float
    t_half_microsomal_min: float
    metabolic_stability: str
    predicted_clearance_class: str
    scaling_factor: float
    notes: str


# ---------------------------------------------------------------------------
# Core simulation function
# ---------------------------------------------------------------------------


def simulate_microsomal_stability(
    drug_name: str,
    t_half_microsomal_min: float,
    microsomal_protein_mg_per_mL: float = 0.5,
    species: str = "human",
    fu_mic: float = 1.0,
    substrate_conc_uM: float = 1.0,
) -> MicrosomalStabilityResult:
    """Simulate a microsomal stability assay and scale CLint to in vivo.

    Args:
        drug_name: Compound identifier.
        t_half_microsomal_min: Observed half-life in microsomal incubation (min).
        microsomal_protein_mg_per_mL: Microsomal protein concentration (mg/mL).
        species: One of "human", "rat", "mouse", "dog".
        fu_mic: Unbound fraction in microsomes (0, 1].
        substrate_conc_uM: Initial substrate concentration (uM).

    Returns:
        MicrosomalStabilityResult with all derived PK parameters.

    Raises:
        ValueError: If t_half_microsomal_min <= 0, microsomal_protein <= 0,
                    fu_mic not in (0, 1], or species is unknown.
    """
    if t_half_microsomal_min <= 0:
        raise ValueError(f"t_half_microsomal_min must be > 0, got {t_half_microsomal_min}")
    if microsomal_protein_mg_per_mL <= 0:
        raise ValueError(
            f"microsomal_protein_mg_per_mL must be > 0, got {microsomal_protein_mg_per_mL}"
        )
    if fu_mic <= 0 or fu_mic > 1.0:
        raise ValueError(f"fu_mic must be in (0, 1], got {fu_mic}")
    if species not in _VALID_SPECIES:
        raise ValueError(f"species must be one of {sorted(_VALID_SPECIES)}, got {species!r}")

    # In vitro CLint (uL/min/mg protein)
    # CLint = 0.693 / t_half * (incubation_vol_uL / protein_mg_in_incubation)
    # protein_mg_in_incubation = protein_conc * incubation_vol / 1000
    # (protein_conc in mg/mL, vol in uL -> divide by 1000 for mL)
    protein_mg_in_incubation = microsomal_protein_mg_per_mL * (_INCUBATION_VOL_UL / 1000.0)
    clint_invitro = (0.693 / t_half_microsomal_min) * (
        _INCUBATION_VOL_UL / protein_mg_in_incubation
    )

    # Scaling factor: mg_protein/g_liver * g_liver/kg_BW
    protein_per_g = _PROTEIN_PER_G_LIVER[species]
    g_liver_per_kg = _G_LIVER_PER_KG_BW[species]
    scaling_factor = protein_per_g * g_liver_per_kg

    # In vivo CLint (mL/min/kg) = CLint_invitro (uL/min/mg) * scaling_factor (mg/kg) * fu_mic / 1000
    clint_invivo = clint_invitro * scaling_factor * fu_mic / 1000.0

    # Hepatic CL via well-stirred model
    q_h = _Q_H_ML_PER_MIN_PER_KG[species]

    # Simplified plasma protein binding assumption based on CLint magnitude
    fu_p = 0.1 if clint_invivo > 10.0 else 0.5

    cl_hepatic = q_h * fu_p * clint_invivo / (q_h + fu_p * clint_invivo)

    # Metabolic stability classification
    if t_half_microsomal_min > 30.0:
        metabolic_stability = "high"
    elif t_half_microsomal_min > 10.0:
        metabolic_stability = "moderate"
    else:
        metabolic_stability = "low"

    # Hepatic clearance class relative to Q_H
    if cl_hepatic < 0.3 * q_h:
        clearance_class = "low"
    elif cl_hepatic > 0.7 * q_h:
        clearance_class = "high"
    else:
        clearance_class = "moderate"

    notes = (
        f"Species: {species}; "
        f"CLint_invitro={clint_invitro:.2f} uL/min/mg; "
        f"scaling_factor={scaling_factor:.1f} mg/kg; "
        f"CLint_invivo={clint_invivo:.2f} mL/min/kg; "
        f"Q_H={q_h:.1f} mL/min/kg; "
        f"fu_p (assumed)={fu_p}; "
        f"CL_H={cl_hepatic:.2f} mL/min/kg"
    )

    return MicrosomalStabilityResult(
        drug_name=drug_name,
        species=species,
        microsomal_protein_mg_per_mL=microsomal_protein_mg_per_mL,
        substrate_conc_uM=substrate_conc_uM,
        clint_uL_per_min_per_mg=clint_invitro,
        clint_mL_per_min_per_kg=clint_invivo,
        cl_hepatic_mL_per_min_per_kg=cl_hepatic,
        fu_mic=fu_mic,
        t_half_microsomal_min=t_half_microsomal_min,
        metabolic_stability=metabolic_stability,
        predicted_clearance_class=clearance_class,
        scaling_factor=scaling_factor,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Multi-species comparison
# ---------------------------------------------------------------------------


def compare_species(
    drug_name: str,
    t_half_microsomal_min: float,
    fu_mic: float = 1.0,
) -> list[MicrosomalStabilityResult]:
    """Run the microsomal stability simulation for all four species.

    Args:
        drug_name: Compound identifier.
        t_half_microsomal_min: Observed microsomal half-life (min).
        fu_mic: Unbound fraction in microsomes (0, 1].

    Returns:
        List of MicrosomalStabilityResult for human, rat, mouse, dog,
        sorted by cl_hepatic_mL_per_min_per_kg descending.
    """
    results = [
        simulate_microsomal_stability(
            drug_name=drug_name,
            t_half_microsomal_min=t_half_microsomal_min,
            species=sp,
            fu_mic=fu_mic,
        )
        for sp in ("human", "rat", "mouse", "dog")
    ]
    results.sort(key=lambda r: r.cl_hepatic_mL_per_min_per_kg, reverse=True)
    return results
