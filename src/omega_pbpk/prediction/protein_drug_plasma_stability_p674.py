"""Phase 674 - Protein Drug Stability in Plasma."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ProteinPlasmaStabilityResult:
    drug_name: str
    mw_kDa: float
    is_pegylated: bool
    protease_susceptibility: float
    t_half_plasma_min: float
    degradation_rate_per_min: float
    intact_fraction_at_1h_pct: float
    intact_fraction_at_4h_pct: float
    dominant_protease: str
    stability_class: str
    half_life_enhancement_vs_native: float
    formulation_strategy: str
    notes: str


def predict_protein_plasma_stability(
    drug_name: str,
    mw_kDa: float,
    n_peptide_bonds: int,
    is_pegylated: bool = False,
    is_cyclic: bool = False,
    has_disulfide: bool = False,
    has_d_amino_acids: bool = False,
    plasma_volume_mL: float = 5.0,
) -> ProteinPlasmaStabilityResult:
    if mw_kDa <= 0:
        raise ValueError("mw_kDa must be > 0")
    if n_peptide_bonds <= 0:
        raise ValueError("n_peptide_bonds must be > 0")

    # Base degradation rate
    k_base = n_peptide_bonds * 0.001 / 60.0
    k_base = max(1e-5, min(0.1, k_base))

    # MW correction
    mw_factor = 1.0 / (1.0 + mw_kDa / 50.0)
    k_base_native = k_base * mw_factor

    k_deg = k_base_native

    # Structural protection factors
    if is_pegylated:
        k_deg *= 0.1
    if is_cyclic:
        k_deg *= 0.3
    if has_disulfide:
        k_deg *= 0.7
    if has_d_amino_acids:
        k_deg *= 0.05

    k_deg = max(1e-6, k_deg)

    t_half_plasma_min = 0.693 / k_deg
    protease_susceptibility = min(10.0, k_deg * 100)
    intact_fraction_at_1h_pct = math.exp(-k_deg * 60) * 100
    intact_fraction_at_4h_pct = math.exp(-k_deg * 240) * 100

    # Dominant protease
    if n_peptide_bonds < 5:
        dominant_protease = "aminopeptidase"
    elif mw_kDa > 50:
        dominant_protease = "plasmin"
    elif is_cyclic:
        dominant_protease = "none"
    else:
        dominant_protease = "endoprotease"

    # Stability class
    if t_half_plasma_min >= 240:
        stability_class = "high"
    elif t_half_plasma_min >= 60:
        stability_class = "medium"
    else:
        stability_class = "low"

    # Half-life enhancement vs native
    half_life_enhancement = k_base_native / k_deg
    half_life_enhancement = max(1.0, min(1000.0, half_life_enhancement))

    # Formulation strategy
    if stability_class == "low":
        formulation_strategy = "PEGylation or D-amino acid substitution recommended"
    elif stability_class == "medium":
        formulation_strategy = "consider cyclization or disulfide engineering"
    else:
        formulation_strategy = "stable; standard formulation"

    return ProteinPlasmaStabilityResult(
        drug_name=drug_name,
        mw_kDa=mw_kDa,
        is_pegylated=is_pegylated,
        protease_susceptibility=protease_susceptibility,
        t_half_plasma_min=t_half_plasma_min,
        degradation_rate_per_min=k_deg,
        intact_fraction_at_1h_pct=intact_fraction_at_1h_pct,
        intact_fraction_at_4h_pct=intact_fraction_at_4h_pct,
        dominant_protease=dominant_protease,
        stability_class=stability_class,
        half_life_enhancement_vs_native=half_life_enhancement,
        formulation_strategy=formulation_strategy,
        notes=f"k_base={k_base:.6f}, mw_factor={mw_factor:.4f}, k_deg={k_deg:.6f}",
    )
