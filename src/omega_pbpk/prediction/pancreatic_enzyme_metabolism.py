"""Phase 528 — Pancreatic Enzyme Drug Metabolism.

Predict drug metabolism/degradation by pancreatic enzymes in the GI tract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PancreaticEnzymeMetabolismResult:
    """Result of pancreatic enzyme metabolism prediction."""

    compound_name: str
    mw_Da: float
    is_peptide: bool
    drug_class: str
    ph_labile: bool
    enzyme_activity_multiplier: float
    k_deg_per_h: float
    fraction_remaining_gi: float
    fa_gi: float
    stability_class: str  # "stable", "moderate", "labile"
    gi_protection_needed: bool
    recommended_formulation: str  # "standard", "enteric_coated", "non_oral"
    notes: str


def _compute_k_deg(
    mw_Da: float,
    is_peptide: bool,
    drug_class: str,
    ph_labile: bool,
) -> float:
    """Compute GI lumen degradation rate constant (per hour)."""
    k_deg = 0.0

    # Base degradation rate
    if is_peptide or drug_class in {"peptide", "protein"}:
        # Peptide/protein: susceptible to pancreatic proteases
        peptide_bond_count = mw_Da / 110.0  # average amino acid MW = 110 Da
        k_deg = 0.3 * (1.0 + math.log10(max(1.0, peptide_bond_count)))
    else:
        # Small molecule: generally resistant
        k_deg = 0.01

    # Ester susceptibility
    if drug_class == "ester":
        k_deg += 0.2

    # pH lability
    if ph_labile:
        k_deg += 0.1

    return k_deg


def _stability_class(fa_gi: float) -> str:
    """Classify GI stability."""
    if fa_gi > 0.8:
        return "stable"
    elif fa_gi >= 0.5:
        return "moderate"
    else:
        return "labile"


def _recommended_formulation(stability: str) -> str:
    """Map stability class to recommended formulation."""
    if stability == "stable":
        return "standard"
    elif stability == "moderate":
        return "enteric_coated"
    else:
        return "non_oral"


def predict_pancreatic_enzyme_metabolism(
    compound_name: str,
    mw_Da: float,
    is_peptide: bool = False,
    drug_class: str = "small_molecule",
    ph_labile: bool = False,
    enzyme_activity_multiplier: float = 1.0,
) -> PancreaticEnzymeMetabolismResult:
    """Predict GI degradation by pancreatic enzymes.

    Parameters
    ----------
    compound_name : str
        Name of the compound.
    mw_Da : float
        Molecular weight in Daltons.
    is_peptide : bool
        True if the drug is a peptide or protein.
    drug_class : str
        One of "small_molecule", "ester", "peptide", "protein".
    ph_labile : bool
        True if drug is susceptible to acid-mediated degradation.
    enzyme_activity_multiplier : float
        Relative pancreatic enzyme activity (1.0=normal, 0.1=exocrine insufficiency).

    Returns
    -------
    PancreaticEnzymeMetabolismResult
    """
    # --- Validation ---
    if mw_Da <= 0:
        raise ValueError(f"mw_Da must be positive. Got {mw_Da}.")
    if not (0 < enzyme_activity_multiplier <= 2.0):
        raise ValueError(
            f"enzyme_activity_multiplier must be in (0, 2]. Got {enzyme_activity_multiplier}."
        )
    valid_classes = {"small_molecule", "ester", "peptide", "protein"}
    if drug_class not in valid_classes:
        raise ValueError(f"drug_class must be one of {valid_classes}. Got '{drug_class}'.")

    # --- Degradation rate ---
    k_deg = _compute_k_deg(mw_Da, is_peptide, drug_class, ph_labile)

    # --- GI transit time ---
    transit_h = 4.0  # small intestine transit

    # --- Fraction remaining ---
    fraction_remaining_gi = math.exp(-k_deg * transit_h)

    # --- fa_gi: fraction absorbed (adjusted for enzyme activity) ---
    # fraction degraded = (1 - exp(-k_deg * 4)) * enzyme_activity_multiplier
    degraded_fraction = (1.0 - fraction_remaining_gi) * enzyme_activity_multiplier
    # Clamp to [0, 1]
    fa_gi = max(0.0, min(1.0, 1.0 - degraded_fraction))

    # --- Classification ---
    stability = _stability_class(fa_gi)
    protection_needed = stability in {"moderate", "labile"}
    formulation = _recommended_formulation(stability)

    # --- Notes ---
    peptide_bond_count = (
        mw_Da / 110.0 if (is_peptide or drug_class in {"peptide", "protein"}) else 0
    )
    notes_parts = [
        f"k_deg={k_deg:.4f}/h",
        f"fraction_remaining={fraction_remaining_gi:.3f}",
        f"fa_gi={fa_gi:.3f}",
        f"stability={stability}",
    ]
    if is_peptide or drug_class in {"peptide", "protein"}:
        notes_parts.append(f"est_peptide_bonds={peptide_bond_count:.1f}")
    if enzyme_activity_multiplier < 1.0:
        notes_parts.append(f"enzyme_insufficiency (multiplier={enzyme_activity_multiplier})")
    notes = "; ".join(notes_parts)

    return PancreaticEnzymeMetabolismResult(
        compound_name=compound_name,
        mw_Da=mw_Da,
        is_peptide=is_peptide,
        drug_class=drug_class,
        ph_labile=ph_labile,
        enzyme_activity_multiplier=enzyme_activity_multiplier,
        k_deg_per_h=k_deg,
        fraction_remaining_gi=fraction_remaining_gi,
        fa_gi=fa_gi,
        stability_class=stability,
        gi_protection_needed=protection_needed,
        recommended_formulation=formulation,
        notes=notes,
    )


def compare_enzyme_conditions(
    compound_name: str,
    mw_Da: float,
    is_peptide: bool,
    activity_levels: list,
) -> list:
    """Compare metabolism across different enzyme activity levels.

    Parameters
    ----------
    compound_name : str
        Drug name.
    mw_Da : float
        Molecular weight in Da.
    is_peptide : bool
        Whether the drug is a peptide.
    activity_levels : list of float
        Enzyme activity multipliers to compare.

    Returns
    -------
    list of PancreaticEnzymeMetabolismResult
        Results sorted by fa_gi descending (highest bioavailability first).
    """
    drug_class = "peptide" if is_peptide else "small_molecule"
    results = [
        predict_pancreatic_enzyme_metabolism(
            compound_name=compound_name,
            mw_Da=mw_Da,
            is_peptide=is_peptide,
            drug_class=drug_class,
            enzyme_activity_multiplier=level,
        )
        for level in activity_levels
    ]
    # Sort by fa_gi descending (best bioavailability first)
    results.sort(key=lambda r: r.fa_gi, reverse=True)
    return results
