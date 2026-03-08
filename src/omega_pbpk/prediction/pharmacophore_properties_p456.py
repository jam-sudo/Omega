"""
Phase 456 — Pharmacophore Properties Predictor
Rule-based pharmacophore feature prediction from molecular descriptors.
Pure Python, no external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PharmacophoreResult:
    """Result of pharmacophore property prediction."""

    drug_name: str
    mw: float
    logp: float
    hbd: int
    hba: int
    rotatable_bonds: int
    psa: float
    charge: int
    hydrophobic_centers: int
    aromatic_rings_approx: int
    hbond_donor_features: int
    hbond_acceptor_features: int
    charged_features: int
    total_features: int
    drug_likeness_score: float
    drug_likeness_class: str
    lipinski_violations: int
    veber_compliant: bool
    notes: str


def predict_pharmacophore(
    drug_name: str,
    mw: float,
    logp: float,
    hbd: int,
    hba: int,
    rotatable_bonds: int = 5,
    psa: float = 80.0,
    charge: int = 0,
) -> PharmacophoreResult:
    """Predict pharmacophore features and drug-likeness from molecular descriptors.

    Args:
        drug_name: Name/identifier of the drug.
        mw: Molecular weight (Da). Must be > 0.
        logp: Lipophilicity (octanol/water). Must be in [-5, 10].
        hbd: Number of hydrogen bond donors. Must be >= 0.
        hba: Number of hydrogen bond acceptors. Must be >= 0.
        rotatable_bonds: Number of rotatable bonds (default 5). Must be >= 0.
        psa: Polar surface area (A^2, default 80). Must be >= 0.
        charge: Formal charge of the molecule (integer, default 0).

    Returns:
        PharmacophoreResult with pharmacophore features and drug-likeness metrics.
    """
    # --- Validation ---
    if mw <= 0:
        raise ValueError("mw must be > 0")
    if not (-5 <= logp <= 10):
        raise ValueError("logp must be in [-5, 10]")
    if hbd < 0:
        raise ValueError("hbd must be >= 0")
    if hba < 0:
        raise ValueError("hba must be >= 0")
    if rotatable_bonds < 0:
        raise ValueError("rotatable_bonds must be >= 0")
    if psa < 0:
        raise ValueError("psa must be >= 0")

    # --- Pharmacophore feature estimation ---
    # Hydrophobic centers: logP-based, clamped [0, 10]
    hydrophobic_centers = max(0, min(10, int(logp * 1.5)))

    # Aromatic rings approximation from MW
    aromatic_rings_approx = max(0, min(5, int(mw / 120.0 - 0.5)))

    hbond_donor_features = hbd
    hbond_acceptor_features = hba
    charged_features = 1 if abs(charge) >= 1 else 0

    total_features = (
        hydrophobic_centers
        + aromatic_rings_approx
        + hbond_donor_features
        + hbond_acceptor_features
        + charged_features
    )

    # --- Lipinski Rule of Five violations ---
    lipinski_violations = sum(
        [
            mw > 500,
            logp > 5,
            hbd > 5,
            hba > 10,
        ]
    )

    # --- Veber compliance ---
    veber_compliant = rotatable_bonds <= 10 and psa <= 140

    # --- Drug-likeness score [0, 100] ---
    score = 100.0
    score -= 10.0 * lipinski_violations
    if not veber_compliant:
        score -= 10.0
    drug_likeness_score = max(0.0, min(100.0, score))

    # --- Drug-likeness classification ---
    # Order matters: fragment_like > lead_like > drug_like > beyond_rule_of_5
    if mw < 300 and logp <= 3:
        drug_likeness_class = "fragment_like"
    elif mw <= 350 and logp <= 3:
        drug_likeness_class = "lead_like"
    elif lipinski_violations == 0 and veber_compliant:
        drug_likeness_class = "drug_like"
    else:
        drug_likeness_class = "beyond_rule_of_5"

    notes = (
        f"lipinski_violations={lipinski_violations}, "
        f"veber_compliant={veber_compliant}, "
        f"total_features={total_features}"
    )

    return PharmacophoreResult(
        drug_name=drug_name,
        mw=mw,
        logp=logp,
        hbd=hbd,
        hba=hba,
        rotatable_bonds=rotatable_bonds,
        psa=psa,
        charge=charge,
        hydrophobic_centers=hydrophobic_centers,
        aromatic_rings_approx=aromatic_rings_approx,
        hbond_donor_features=hbond_donor_features,
        hbond_acceptor_features=hbond_acceptor_features,
        charged_features=charged_features,
        total_features=total_features,
        drug_likeness_score=drug_likeness_score,
        drug_likeness_class=drug_likeness_class,
        lipinski_violations=lipinski_violations,
        veber_compliant=veber_compliant,
        notes=notes,
    )


def screen_pharmacophore(drugs: list[dict]) -> list[PharmacophoreResult]:
    """Screen a list of drugs and return results sorted by drug_likeness_score descending.

    Args:
        drugs: List of dicts with keys matching predict_pharmacophore parameters.
               Required keys: drug_name, mw, logp, hbd, hba
               Optional keys: rotatable_bonds, psa, charge

    Returns:
        List of PharmacophoreResult sorted by drug_likeness_score (highest first).
    """
    results = []
    for drug in drugs:
        result = predict_pharmacophore(
            drug_name=drug["drug_name"],
            mw=drug["mw"],
            logp=drug["logp"],
            hbd=drug["hbd"],
            hba=drug["hba"],
            rotatable_bonds=drug.get("rotatable_bonds", 5),
            psa=drug.get("psa", 80.0),
            charge=drug.get("charge", 0),
        )
        results.append(result)
    results.sort(key=lambda r: r.drug_likeness_score, reverse=True)
    return results
