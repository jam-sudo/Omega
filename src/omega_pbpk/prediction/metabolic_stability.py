"""Metabolic stability prediction in liver microsomes and hepatocytes (Phase 442).

Empirical physicochemical descriptor-based model for predicting CLint and t½
in human liver microsomes (HLM).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass (frozen — no mutable fields)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MicrosomalStabilityResult:
    """Result of microsomal stability prediction.

    Attributes
    ----------
    drug_name : str
        Name of the drug compound.
    mw : float
        Molecular weight (g/mol).
    logP : float
        Lipophilicity.
    n_aromatic_rings : int
        Number of aromatic rings.
    n_rot_bonds : int
        Number of rotatable bonds.
    clint_uL_per_min_per_mg : float
        Intrinsic clearance (µL/min/mg microsomal protein).
    t_half_microsomal_min : float
        Microsomal half-life (minutes) at 45 mg protein/mL incubation.
    stability_class : str
        "stable" (t½ > 60 min), "moderate" (30–60 min), or "unstable" (< 30 min).
    predicted_f_unmetabolized : float
        Fraction unmetabolized at steady state (0–1).
    notes : str
        Descriptor values and correction factors used.
    """

    drug_name: str
    mw: float
    logP: float
    n_aromatic_rings: int
    n_rot_bonds: int
    clint_uL_per_min_per_mg: float
    t_half_microsomal_min: float
    stability_class: str
    predicted_f_unmetabolized: float
    notes: str


# ---------------------------------------------------------------------------
# Model implementation
# ---------------------------------------------------------------------------

_PROTEIN_CONC_MG_PER_ML = 45.0  # mg microsomal protein / mL incubation


def _base_clint(logP: float) -> float:
    """Base CLint = 15 * exp(0.3 * logP) µL/min/mg protein."""
    return 15.0 * math.exp(0.3 * logP)


def _aromatic_ring_correction(n_aromatic_rings: int) -> float:
    """More aromatic rings → more CYP oxidation sites: factor = 1 + 0.15 * n_rings."""
    return 1.0 + 0.15 * n_aromatic_rings


def _flexibility_correction(n_rot_bonds: int) -> float:
    """Rigid molecules (fewer rot bonds) are more metabolically stable.

    factor = 1 - 0.05 * min(n_rot_bonds, 10)
    """
    return 1.0 - 0.05 * min(n_rot_bonds, 10)


def _mw_correction(mw: float) -> float:
    """Large MW molecules are less efficiently metabolized.

    factor = max(0.3, 1 - (mw - 400) / 1000)
    """
    return max(0.3, 1.0 - (mw - 400.0) / 1000.0)


def _stability_class(t_half_min: float) -> str:
    """Classify microsomal stability based on half-life."""
    if t_half_min > 60.0:
        return "stable"
    if t_half_min >= 30.0:
        return "moderate"
    return "unstable"


def predict_microsomal_stability(
    drug_name: str,
    mw: float,
    logP: float,
    n_aromatic_rings: int,
    n_rot_bonds: int,
    molecular_formula_atoms: dict[str, int] | None = None,
) -> MicrosomalStabilityResult:
    """Predict microsomal stability using an empirical physicochemical model.

    Parameters
    ----------
    drug_name : str
        Name of the drug compound.
    mw : float
        Molecular weight (g/mol). Must be > 0.
    logP : float
        Lipophilicity (log octanol/water partition coefficient).
    n_aromatic_rings : int
        Number of aromatic rings. Must be >= 0.
    n_rot_bonds : int
        Number of rotatable bonds. Must be >= 0.
    molecular_formula_atoms : dict[str, int] or None
        Atomic composition dict e.g. {"C": 20, "H": 25, "N": 2, "O": 3}.
        Currently stored for traceability; reserved for future heteroatom corrections.

    Returns
    -------
    MicrosomalStabilityResult
        Predicted CLint, t½, stability class, and fraction unmetabolized.
    """
    # --- Input validation ---
    if not drug_name or not drug_name.strip():
        raise ValueError("drug_name must be a non-empty string")
    if mw <= 0:
        raise ValueError(f"mw must be > 0, got {mw}")
    if n_aromatic_rings < 0:
        raise ValueError(f"n_aromatic_rings must be >= 0, got {n_aromatic_rings}")
    if n_rot_bonds < 0:
        raise ValueError(f"n_rot_bonds must be >= 0, got {n_rot_bonds}")

    if molecular_formula_atoms is None:
        molecular_formula_atoms = {}

    # Validate atom counts if provided
    for elem, count in molecular_formula_atoms.items():
        if not isinstance(count, int) or count < 0:
            raise ValueError(
                f"molecular_formula_atoms['{elem}'] must be a non-negative integer, got {count}"
            )

    # --- Empirical model ---
    clint_base = _base_clint(logP)
    f_aromatic = _aromatic_ring_correction(n_aromatic_rings)
    f_flex = _flexibility_correction(n_rot_bonds)
    f_mw = _mw_correction(mw)

    clint = clint_base * f_aromatic * f_flex * f_mw
    clint = max(clint, 0.001)  # floor to avoid division issues

    # t½ in microsomes: t½ = 0.693 / (CLint * protein_conc / 1000)
    # CLint in µL/min/mg, protein_conc in mg/mL → k_deg in min⁻¹
    k_deg_per_min = clint * _PROTEIN_CONC_MG_PER_ML / 1000.0
    t_half_min = 0.693 / k_deg_per_min

    stab_class = _stability_class(t_half_min)

    # Fraction unmetabolized at steady state (approximately exp(-CL * tau / Vd))
    # Simple surrogate: f_unmet = exp(-k_deg_per_min * 60) for a 1h incubation
    f_unmet = math.exp(-k_deg_per_min * 60.0)
    f_unmet = float(max(0.0, min(1.0, f_unmet)))

    # Build notes
    atom_str = (
        ", ".join(f"{k}:{v}" for k, v in sorted(molecular_formula_atoms.items()))
        if molecular_formula_atoms
        else "not provided"
    )
    notes = (
        f"CLint_base={clint_base:.2f}; f_aromatic={f_aromatic:.3f}; "
        f"f_flex={f_flex:.3f}; f_mw={f_mw:.3f}; atoms=[{atom_str}]"
    )

    return MicrosomalStabilityResult(
        drug_name=drug_name,
        mw=mw,
        logP=logP,
        n_aromatic_rings=n_aromatic_rings,
        n_rot_bonds=n_rot_bonds,
        clint_uL_per_min_per_mg=float(clint),
        t_half_microsomal_min=float(t_half_min),
        stability_class=stab_class,
        predicted_f_unmetabolized=f_unmet,
        notes=notes,
    )


__all__ = [
    "MicrosomalStabilityResult",
    "predict_microsomal_stability",
]
