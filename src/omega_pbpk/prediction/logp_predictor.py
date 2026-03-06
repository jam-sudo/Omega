"""logP prediction from atom counts and physicochemical fragments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LogPResult:
    drug_name: str
    logP_predicted: float
    method: str
    confidence: float
    classification: str  # hydrophilic (<0) / drug-like (0-5) / lipophilic (>5)
    water_solubility_class: str  # high / moderate / low / insoluble
    bbb_penetration: bool  # blood-brain barrier prediction


# Atomic contributions (simplified Ghose-Crippen / Wildman-Crippen approach)
# Carbon types and heteroatom contributions
_ATOM_CONTRIB = {
    "C_aromatic": 0.13,
    "C_aliphatic": 0.20,
    "C_halide_attached": 0.10,
    "N_amine": -1.03,
    "N_amide": -1.03,
    "N_aromatic": -0.57,
    "O_hydroxyl": -0.67,
    "O_ether": -0.61,
    "O_carbonyl": -0.55,
    "S_thiol": 0.15,
    "S_thioether": 0.35,
    "F": 0.14,
    "Cl": 0.60,
    "Br": 0.87,
    "I": 1.12,
    "P": -0.18,
}


def predict_logp(
    drug_name: str,
    n_carbons: int,
    n_nitrogens: int = 0,
    n_oxygens: int = 0,
    n_halogens: int = 0,
    n_aromatic_rings: int = 0,
    n_rotatable_bonds: int = 0,
    MW: float | None = None,
    PSA: float = 60.0,
    halogen_type: str = "Cl",
    n_hbd: int = 0,
    n_hba: int = 0,
) -> LogPResult:
    """Predict logP from atom counts using Wildman-Crippen fragment approach.

    Simplified model based on fragment contributions.
    Typical accuracy: ±1 logP unit.

    Parameters
    ----------
    n_carbons : int
        Total carbon count.
    n_aromatic_rings : int
        Number of aromatic rings (affects C aromaticity).
    n_hbd : int
        Hydrogen bond donors.
    n_hba : int
        Hydrogen bond acceptors.
    """
    # Estimate aromatic vs aliphatic carbons
    n_aromatic_c = min(n_aromatic_rings * 6, n_carbons)
    n_aliphatic_c = n_carbons - n_aromatic_c

    logP = (
        n_aliphatic_c * _ATOM_CONTRIB["C_aliphatic"]
        + n_aromatic_c * _ATOM_CONTRIB["C_aromatic"]
        + n_nitrogens * _ATOM_CONTRIB["N_amine"]
        + n_oxygens * _ATOM_CONTRIB["O_hydroxyl"]
        + n_halogens * _ATOM_CONTRIB.get(halogen_type, _ATOM_CONTRIB["Cl"])
    )

    # Corrections
    # PSA correction: high PSA reduces logP
    logP -= 0.005 * max(PSA - 60.0, 0.0)

    # HBD correction
    logP -= 0.15 * n_hbd

    # Rotatable bond correction (increases flexibility, slight hydrophilicity)
    logP -= 0.02 * n_rotatable_bonds

    # MW correction
    if MW is not None:
        logP += 0.002 * (MW - 300.0)

    # Classification
    if logP < 0:
        classification = "hydrophilic"
    elif logP <= 5:
        classification = "drug-like"
    else:
        classification = "lipophilic"

    # Water solubility class (simplified Yalkowsky)
    if logP < 1:
        sol_class = "high"
    elif logP < 3:
        sol_class = "moderate"
    elif logP < 5:
        sol_class = "low"
    else:
        sol_class = "insoluble"

    # BBB: logP 1-4, MW<400, PSA<90, n_hbd<=3
    mw_ok = MW is None or MW < 400.0
    bbb = (1.0 <= logP <= 4.5) and mw_ok and (PSA < 90.0) and (n_hbd <= 3)

    # Confidence based on input completeness
    confidence = 0.65
    if MW is not None:
        confidence += 0.05
    if n_hbd > 0 or n_hba > 0:
        confidence += 0.05
    confidence = min(confidence, 0.80)

    return LogPResult(
        drug_name=drug_name,
        logP_predicted=float(logP),
        method="Wildman-Crippen fragment",
        confidence=confidence,
        classification=classification,
        water_solubility_class=sol_class,
        bbb_penetration=bbb,
    )


def screen_logp(compounds: list[dict]) -> list[LogPResult]:
    """Predict logP for a list of compounds.

    Each dict: {drug_name, n_carbons, n_nitrogens, n_oxygens, ...}
    """
    results = []
    for c in compounds:
        r = predict_logp(
            drug_name=c.get("drug_name", "unknown"),
            n_carbons=c.get("n_carbons", 20),
            n_nitrogens=c.get("n_nitrogens", 0),
            n_oxygens=c.get("n_oxygens", 0),
            n_halogens=c.get("n_halogens", 0),
            n_aromatic_rings=c.get("n_aromatic_rings", 0),
            n_rotatable_bonds=c.get("n_rotatable_bonds", 0),
            MW=c.get("MW"),
            PSA=c.get("PSA", 60.0),
            n_hbd=c.get("n_hbd", 0),
            n_hba=c.get("n_hba", 0),
        )
        results.append(r)
    return results


__all__ = [
    "LogPResult",
    "predict_logp",
    "screen_logp",
]
