"""Synthetic accessibility and drug-likeness prediction.

Predicts whether a molecule satisfies common drug-likeness filters:
- Lipinski Rule of Five (RO5)
- Veber rules (oral bioavailability)
- Egan model (BBB/GI absorption)
- Lead-likeness criteria

Also estimates synthetic accessibility score (SA score) and a simplified
QED (Quantitative Estimate of Drug-likeness).

References:
    - Lipinski CA et al., Adv Drug Deliv Rev. 2001;46(1-3):3-26
    - Veber DF et al., J Med Chem. 2002;45(12):2615-23
    - Egan WJ et al., J Med Chem. 2000;43(21):3867-77
    - Bickerton GR et al., Nat Chem. 2012;4(2):90-8 (QED)
    - Ertl P & Schuffenhauer A, J Cheminform. 2009;1:8 (SA score)
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DrugLikenessResult:
    """Result of drug-likeness and synthetic accessibility prediction.

    Attributes
    ----------
    drug_name : Drug or compound identifier.
    mw : Molecular weight (Da).
    logP : Octanol-water partition coefficient.
    psa : Polar surface area (A^2).
    n_hbd : Number of hydrogen bond donors.
    n_hba : Number of hydrogen bond acceptors.
    n_rot_bonds : Number of rotatable bonds.
    n_rings : Total ring count.
    n_stereocenters : Number of stereocenters.
    lipinski_pass : Passes Lipinski RO5 (<=1 violation allowed by default).
    veber_pass : Passes Veber oral bioavailability criteria.
    egan_pass : Passes Egan absorption model criteria.
    lead_like : Satisfies lead-likeness criteria.
    n_ro5_violations : Number of Lipinski RO5 violations.
    bioavailability_score : Score 2-6 based on filter violations.
    sa_score : Synthetic accessibility score (1=easy, 10=hard).
    qed_score : Quantitative estimate of drug-likeness (0-1).
    overall_drug_likeness : 'poor', 'acceptable', 'good', or 'excellent'.
    notes : Summary text.
    """

    drug_name: str
    mw: float
    logP: float
    psa: float
    n_hbd: int
    n_hba: int
    n_rot_bonds: int
    n_rings: int
    n_stereocenters: int
    lipinski_pass: bool
    veber_pass: bool
    egan_pass: bool
    lead_like: bool
    n_ro5_violations: int
    bioavailability_score: int
    sa_score: float
    qed_score: float
    overall_drug_likeness: str
    notes: str


def _norm_score(value: float, optimal: float, width: float) -> float:
    """Gaussian-like normalised property score for QED calculation."""
    return math.exp(-((value - optimal) ** 2) / (2.0 * width ** 2))


def _calc_qed(
    mw: float,
    logP: float,
    psa: float,
    n_hbd: int,
    n_hba: int,
    n_rot_bonds: int,
) -> float:
    """Simplified QED: geometric mean of normalised property scores.

    Optimal reference values are taken from the Bickerton 2012 paper.
    """
    scores = [
        _norm_score(mw, 330.0, 120.0),
        _norm_score(logP, 2.5, 2.0),
        _norm_score(psa, 80.0, 50.0),
        _norm_score(n_hbd, 1.0, 1.5),
        _norm_score(n_hba, 4.0, 2.5),
        _norm_score(n_rot_bonds, 4.0, 3.0),
    ]
    # Geometric mean
    product = 1.0
    for s in scores:
        product *= max(1e-9, s)
    return product ** (1.0 / len(scores))


def predict_drug_likeness(
    drug_name: str,
    mw: float,
    logP: float,
    psa: float,
    n_hbd: int,
    n_hba: int,
    n_rot_bonds: int,
    n_rings: int,
    n_stereocenters: int,
    n_violations_allowed: int = 1,
) -> DrugLikenessResult:
    """Predict drug-likeness using multiple physicochemical filter rules.

    Parameters
    ----------
    drug_name:
        Name or identifier of the compound.
    mw:
        Molecular weight in Da. Must be > 0.
    logP:
        Calculated logP (octanol-water partition coefficient).
    psa:
        Polar surface area in A^2. Must be >= 0.
    n_hbd:
        Number of hydrogen bond donors. Must be >= 0.
    n_hba:
        Number of hydrogen bond acceptors. Must be >= 0.
    n_rot_bonds:
        Number of rotatable bonds. Must be >= 0.
    n_rings:
        Total ring count. Must be >= 0.
    n_stereocenters:
        Number of stereocenters. Must be >= 0.
    n_violations_allowed:
        Number of RO5 violations allowed for Lipinski pass (default 1).
        Must be >= 0.

    Returns
    -------
    DrugLikenessResult
        Frozen dataclass with filter results, scores, and recommendation.

    Raises
    ------
    ValueError
        If any parameter is outside acceptable range.
    """
    # --- Input validation ---
    if not drug_name or not drug_name.strip():
        raise ValueError("drug_name must be a non-empty string")
    if mw <= 0:
        raise ValueError(f"mw must be > 0, got {mw}")
    if psa < 0:
        raise ValueError(f"psa must be >= 0, got {psa}")
    if n_hbd < 0:
        raise ValueError(f"n_hbd must be >= 0, got {n_hbd}")
    if n_hba < 0:
        raise ValueError(f"n_hba must be >= 0, got {n_hba}")
    if n_rot_bonds < 0:
        raise ValueError(f"n_rot_bonds must be >= 0, got {n_rot_bonds}")
    if n_rings < 0:
        raise ValueError(f"n_rings must be >= 0, got {n_rings}")
    if n_stereocenters < 0:
        raise ValueError(f"n_stereocenters must be >= 0, got {n_stereocenters}")
    if n_violations_allowed < 0:
        raise ValueError(f"n_violations_allowed must be >= 0, got {n_violations_allowed}")

    # --- Lipinski Rule of Five ---
    ro5_violations = 0
    if mw > 500:
        ro5_violations += 1
    if logP > 5:
        ro5_violations += 1
    if n_hbd > 5:
        ro5_violations += 1
    if n_hba > 10:
        ro5_violations += 1

    lipinski_pass = ro5_violations <= n_violations_allowed

    # --- Veber rules (oral bioavailability) ---
    veber_pass = psa <= 140 and n_rot_bonds <= 10

    # --- Egan model ---
    egan_pass = psa <= 132 and logP <= 5.88

    # --- Lead-likeness ---
    lead_like = (mw <= 350) and (logP <= 3.5) and (n_hbd <= 3) and (n_hba <= 7)

    # --- Bioavailability score (based on total violations across all rules) ---
    total_violations = ro5_violations
    if not veber_pass:
        total_violations += 1
    if not egan_pass:
        total_violations += 1

    bioavailability_score_map = {0: 6, 1: 5, 2: 4, 3: 3}
    bioavailability_score = bioavailability_score_map.get(total_violations, 2)

    # --- Synthetic Accessibility Score ---
    # SA score ≈ 1 + 0.5*stereocenters + 0.3*rings, clamped to [1, 10]
    sa_raw = 1.0 + 0.5 * n_stereocenters + 0.3 * n_rings
    sa_score = min(10.0, max(1.0, sa_raw))

    # --- QED (simplified geometric mean) ---
    qed_score = _calc_qed(mw, logP, psa, n_hbd, n_hba, n_rot_bonds)
    qed_score = min(1.0, max(0.0, qed_score))

    # --- Overall drug-likeness classification ---
    likeness_points = 0
    if lipinski_pass:
        likeness_points += 2
    if veber_pass:
        likeness_points += 1
    if egan_pass:
        likeness_points += 1
    if lead_like:
        likeness_points += 1
    if sa_score <= 4:
        likeness_points += 1
    if qed_score >= 0.5:
        likeness_points += 1

    if likeness_points >= 6:
        overall = "excellent"
    elif likeness_points >= 4:
        overall = "good"
    elif likeness_points >= 2:
        overall = "acceptable"
    else:
        overall = "poor"

    # --- Notes ---
    note_parts: list[str] = []
    note_parts.append(f"RO5 violations: {ro5_violations}")
    if lipinski_pass:
        note_parts.append("Lipinski: PASS")
    else:
        note_parts.append(f"Lipinski: FAIL ({ro5_violations} violations)")
    if veber_pass:
        note_parts.append("Veber: PASS")
    else:
        note_parts.append(f"Veber: FAIL (PSA={psa:.1f}, rotBonds={n_rot_bonds})")
    if egan_pass:
        note_parts.append("Egan: PASS")
    else:
        note_parts.append(f"Egan: FAIL (PSA={psa:.1f}, logP={logP:.2f})")
    note_parts.append(f"SA score: {sa_score:.1f}/10")
    note_parts.append(f"QED: {qed_score:.3f}")
    notes = "; ".join(note_parts)

    return DrugLikenessResult(
        drug_name=drug_name,
        mw=mw,
        logP=logP,
        psa=psa,
        n_hbd=n_hbd,
        n_hba=n_hba,
        n_rot_bonds=n_rot_bonds,
        n_rings=n_rings,
        n_stereocenters=n_stereocenters,
        lipinski_pass=lipinski_pass,
        veber_pass=veber_pass,
        egan_pass=egan_pass,
        lead_like=lead_like,
        n_ro5_violations=ro5_violations,
        bioavailability_score=bioavailability_score,
        sa_score=sa_score,
        qed_score=qed_score,
        overall_drug_likeness=overall,
        notes=notes,
    )


def fragment_analysis(
    drug_name: str,
    n_aromatic_rings: int,
    n_aliphatic_rings: int,
    n_heteroatoms: int,
    mw: float,
    logP: float,
) -> dict:
    """Perform fragment-level analysis of a compound.

    Classifies the compound based on ring complexity, heteroatom content,
    and physicochemical properties relative to fragment-likeness criteria.

    Parameters
    ----------
    drug_name:
        Name or identifier of the compound.
    n_aromatic_rings:
        Number of aromatic rings. Must be >= 0.
    n_aliphatic_rings:
        Number of aliphatic rings. Must be >= 0.
    n_heteroatoms:
        Number of heteroatoms (N, O, S, halogens, etc.). Must be >= 0.
    mw:
        Molecular weight in Da. Must be > 0.
    logP:
        Calculated logP value.

    Returns
    -------
    dict with keys:
        - drug_name (str)
        - ring_count_class (str): 'no_rings', 'monocyclic', 'bicyclic', 'polycyclic'
        - heteroatom_fraction (float): fraction of heavy atoms that are heteroatoms
        - fragment_likeness (str): 'fragment-like', 'lead-like', 'drug-like', 'complex'
        - n_aromatic_rings (int)
        - n_aliphatic_rings (int)
        - n_heteroatoms (int)
        - total_rings (int)
        - mw (float)
        - logP (float)

    Raises
    ------
    ValueError
        If any parameter is out of valid range.
    """
    # --- Input validation ---
    if not drug_name or not drug_name.strip():
        raise ValueError("drug_name must be a non-empty string")
    if n_aromatic_rings < 0:
        raise ValueError(f"n_aromatic_rings must be >= 0, got {n_aromatic_rings}")
    if n_aliphatic_rings < 0:
        raise ValueError(f"n_aliphatic_rings must be >= 0, got {n_aliphatic_rings}")
    if n_heteroatoms < 0:
        raise ValueError(f"n_heteroatoms must be >= 0, got {n_heteroatoms}")
    if mw <= 0:
        raise ValueError(f"mw must be > 0, got {mw}")

    total_rings = n_aromatic_rings + n_aliphatic_rings

    # --- Ring count class ---
    if total_rings == 0:
        ring_count_class = "no_rings"
    elif total_rings == 1:
        ring_count_class = "monocyclic"
    elif total_rings == 2:
        ring_count_class = "bicyclic"
    else:
        ring_count_class = "polycyclic"

    # --- Heteroatom fraction ---
    # Estimate heavy atom count from MW (avg heavy atom MW ~13 Da)
    est_heavy_atoms = max(1, round(mw / 13.0))
    heteroatom_fraction = min(1.0, n_heteroatoms / est_heavy_atoms)

    # --- Fragment-likeness classification ---
    # Fragment-like: MW<=300, logP<=3, total_rings<=2
    # Lead-like: MW<=350, total_rings<=3
    # Drug-like: MW<=500
    # Complex: MW>500 or many rings
    if mw <= 300 and logP <= 3.0 and total_rings <= 2:
        fragment_likeness = "fragment-like"
    elif mw <= 350 and total_rings <= 3:
        fragment_likeness = "lead-like"
    elif mw <= 500:
        fragment_likeness = "drug-like"
    else:
        fragment_likeness = "complex"

    return {
        "drug_name": drug_name,
        "ring_count_class": ring_count_class,
        "heteroatom_fraction": round(heteroatom_fraction, 4),
        "fragment_likeness": fragment_likeness,
        "n_aromatic_rings": n_aromatic_rings,
        "n_aliphatic_rings": n_aliphatic_rings,
        "n_heteroatoms": n_heteroatoms,
        "total_rings": total_rings,
        "mw": mw,
        "logP": logP,
    }
