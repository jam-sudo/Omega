"""
Phase 1138 — Peptide drug stability prediction.

Predicts proteolytic degradation half-life and aggregation risk
based on structural and sequence properties.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class PeptideStabilityResult:
    drug_name: str
    n_residues: int
    hydrophobic_pct: float
    is_cyclic: bool
    is_pegylated: bool
    d_aa_fraction: float
    n_methylated_bonds: int
    plasma_half_life_min: float
    plasma_half_life_h: float
    aggregation_propensity: float
    stability_class: str
    aggregation_risk: str
    recommended_modifications: list
    notes: str


def _stability_class(t_half_min: float) -> str:
    if t_half_min < 30:
        return "unstable"
    elif t_half_min <= 240:
        return "moderate"
    return "stable"


def _aggregation_risk(propensity: float) -> str:
    if propensity < 30:
        return "low"
    elif propensity <= 60:
        return "moderate"
    return "high"


def _recommended_mods(
    t_half_min: float,
    aggregation_propensity: float,
    is_cyclic: bool,
    is_pegylated: bool,
    d_aa_fraction: float,
    n_methylated_bonds: int,
) -> list:
    recs: list = []
    if t_half_min < 60:
        if not is_cyclic:
            recs.append("Consider cyclization (5× half-life increase)")
        if d_aa_fraction < 0.25:
            recs.append("Incorporate D-amino acids for protease resistance")
        if n_methylated_bonds == 0:
            recs.append("N-methylation of amide bonds to improve stability")
    if aggregation_propensity > 30 and not is_pegylated:
        recs.append("PEGylation to reduce aggregation propensity")
    if not recs:
        recs.append("Current modifications appear adequate")
    return recs


def predict_peptide_stability(
    drug_name: str,
    n_residues: int,
    hydrophobic_pct: float,
    n_beta_residues: int,
    abs_charge: float,
    d_aa_fraction: float = 0.0,
    n_methylated_bonds: int = 0,
    is_cyclic: bool = False,
    is_pegylated: bool = False,
) -> PeptideStabilityResult:
    """Predict proteolytic stability and aggregation risk for a peptide.

    Parameters
    ----------
    drug_name:
        Identifier for the peptide drug.
    n_residues:
        Total number of amino acid residues.
    hydrophobic_pct:
        Percentage of hydrophobic residues (0–100).
    n_beta_residues:
        Number of beta-sheet forming residues.
    abs_charge:
        Absolute net charge at physiological pH.
    d_aa_fraction:
        Fraction of residues that are D-amino acids (0–1).
    n_methylated_bonds:
        Number of N-methylated amide bonds.
    is_cyclic:
        Whether the peptide is head-to-tail cyclized.
    is_pegylated:
        Whether the peptide carries a PEG chain.
    """
    # --- Validation ---
    if not isinstance(n_residues, int) or n_residues <= 0:
        raise ValueError("n_residues must be a positive integer")
    if not (0 <= hydrophobic_pct <= 100):
        raise ValueError("hydrophobic_pct must be in [0, 100]")
    if not (0 <= d_aa_fraction <= 1):
        raise ValueError("d_aa_fraction must be in [0, 1]")
    if n_methylated_bonds < 0:
        raise ValueError("n_methylated_bonds must be >= 0")
    if not isinstance(n_beta_residues, int) or not (0 <= n_beta_residues <= n_residues):
        raise ValueError("n_beta_residues must be in [0, n_residues]")
    if abs_charge < 0:
        raise ValueError("abs_charge must be >= 0")

    # --- Proteolytic half-life ---
    t_half_base = 120.0  # minutes

    # N-methylation modifier
    if n_methylated_bonds > 0:
        t_half_base *= 3.0

    # Cyclization modifier
    if is_cyclic:
        t_half_base *= 5.0

    # D-amino acid modifier
    t_half_base *= 1.0 + 4.0 * d_aa_fraction

    # PEGylation modifier
    if is_pegylated:
        t_half_base *= 2.0

    # Length factor: exp(-0.05*(n-5)), clamped to [0.1, 2.0]
    length_factor = math.exp(-0.05 * (n_residues - 5))
    length_factor = max(0.1, min(2.0, length_factor))
    t_half_min = t_half_base * length_factor

    t_half_h = t_half_min / 60.0

    # --- Aggregation propensity ---
    agg = 0.0

    if hydrophobic_pct > 40:
        agg += 35

    beta_fraction = n_beta_residues / n_residues if n_residues > 0 else 0.0
    if beta_fraction > 0.3:
        agg += 25

    if abs_charge < 2:
        agg += 20

    # PEGylation protection: +15 if base agg > 30 and not PEGylated
    if not is_pegylated and agg > 30:
        agg += 15

    agg = min(100.0, agg)

    stab_cls = _stability_class(t_half_min)
    agg_risk = _aggregation_risk(agg)
    recs = _recommended_mods(
        t_half_min, agg, is_cyclic, is_pegylated, d_aa_fraction, n_methylated_bonds
    )

    notes = (
        f"Length factor={length_factor:.3f}; "
        f"Beta fraction={beta_fraction:.2f}; "
        f"D-AA modifier={1.0 + 4.0 * d_aa_fraction:.2f}x"
    )

    return PeptideStabilityResult(
        drug_name=drug_name,
        n_residues=n_residues,
        hydrophobic_pct=hydrophobic_pct,
        is_cyclic=is_cyclic,
        is_pegylated=is_pegylated,
        d_aa_fraction=d_aa_fraction,
        n_methylated_bonds=n_methylated_bonds,
        plasma_half_life_min=t_half_min,
        plasma_half_life_h=t_half_h,
        aggregation_propensity=agg,
        stability_class=stab_cls,
        aggregation_risk=agg_risk,
        recommended_modifications=recs,
        notes=notes,
    )


def compare_modifications(
    drug_name: str,
    n_residues: int,
    hydrophobic_pct: float,
    n_beta_residues: int,
    abs_charge: float,
) -> list:
    """Return 4 PeptideStabilityResult objects for common modification strategies.

    Variants returned (in order):
    1. Linear/unmodified baseline
    2. Cyclic
    3. PEGylated
    4. D-amino acid enriched (d_aa_fraction=0.5)
    """
    baseline = predict_peptide_stability(
        drug_name=f"{drug_name}_linear",
        n_residues=n_residues,
        hydrophobic_pct=hydrophobic_pct,
        n_beta_residues=n_beta_residues,
        abs_charge=abs_charge,
    )
    cyclic = predict_peptide_stability(
        drug_name=f"{drug_name}_cyclic",
        n_residues=n_residues,
        hydrophobic_pct=hydrophobic_pct,
        n_beta_residues=n_beta_residues,
        abs_charge=abs_charge,
        is_cyclic=True,
    )
    pegylated = predict_peptide_stability(
        drug_name=f"{drug_name}_peg",
        n_residues=n_residues,
        hydrophobic_pct=hydrophobic_pct,
        n_beta_residues=n_beta_residues,
        abs_charge=abs_charge,
        is_pegylated=True,
    )
    d_aa = predict_peptide_stability(
        drug_name=f"{drug_name}_daa",
        n_residues=n_residues,
        hydrophobic_pct=hydrophobic_pct,
        n_beta_residues=n_beta_residues,
        abs_charge=abs_charge,
        d_aa_fraction=0.5,
    )
    return [baseline, cyclic, pegylated, d_aa]
