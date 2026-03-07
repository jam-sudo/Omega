"""pKa prediction from SMILES-based structural features.

Rule-based pKa estimation using SMILES pattern matching for common
functional groups (carboxylic acids, phenols, amines, sulfonamides, etc.).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PKaPredictionResult:
    """Result of pKa prediction from SMILES."""

    smiles: str
    molecule_type: str
    detected_group: str | None
    pka_predicted: float | None
    ionization_at_pH7: float
    solubility_impact: str  # "improved", "reduced", "neutral"
    notes: str


# Allowed molecule types
_VALID_MOLECULE_TYPES = {"acid", "base", "neutral", "amphoteric"}

# Functional group pKa base values
_GROUP_PKA = {
    "carboxylic_acid": 4.5,
    "phenol": 9.5,
    "amine_primary": 10.0,
    "amine_secondary": 9.5,
    "sulfonamide": 10.5,
    "imidazole": 6.8,
    "pyridine": 5.2,
}


def _estimate_mw_from_smiles(smiles: str) -> float:
    """Rough MW estimate from atom counts in SMILES string."""
    s = smiles
    n_c = s.count("C") + s.count("c")
    n_n = s.count("N") + s.count("n")
    n_o = s.count("O") + s.count("o")
    n_s = s.count("S") + s.count("s")
    n_cl = s.upper().count("CL")
    n_br = s.upper().count("BR")
    mw = 12 * n_c + 14 * n_n + 16 * n_o + 32 * n_s + 35.5 * n_cl + 80 * n_br + 2.0
    return max(mw, 50.0)


def _detect_functional_group(smiles: str) -> tuple[str | None, float | None]:
    """Detect the primary ionizable functional group from SMILES patterns.

    Returns (group_name, base_pka) or (None, None) if no pattern matched.
    """
    s = smiles.lower()

    # Carboxylic acid: C(=O)O or COOH patterns
    if "c(=o)o" in s or "c(o)=o" in s or "oc(=o)" in s or "cooh" in s or "c(=o)oh" in s:
        return "carboxylic_acid", _GROUP_PKA["carboxylic_acid"]

    # Sulfonamide: S(=O)(=O)N
    if "s(=o)(=o)n" in s or "s(=o)(=o)[nh" in s:
        return "sulfonamide", _GROUP_PKA["sulfonamide"]

    # Imidazole ring: c1cnc or n1cnc patterns
    if "c1cnc" in s or "n1cnc" in s or "cnc" in s and "n" in s:
        # Check specifically for imidazole (not just any CNc pattern)
        if "c1cnc" in s or "n1cnc" in s or "[nH]1ccnc1" in smiles or "cnch" in s:
            return "imidazole", _GROUP_PKA["imidazole"]

    # Pyridine ring: n1cccc or c1ccnc patterns
    if "n1cccc" in s or "c1ccnc" in s or "c1ccccn1" in s or "c1ccncc1" in s:
        return "pyridine", _GROUP_PKA["pyridine"]

    # Phenol: aromatic C-OH (oc1 or c1o followed by ring closure)
    if ("oc1" in s or "c1o" in s) and ("c" in s):
        # Distinguish from aliphatic OH by checking for ring indicators
        if any(ch.isdigit() for ch in s):
            return "phenol", _GROUP_PKA["phenol"]

    # Primary amine: CN or NC patterns (aliphatic)
    if "cn" in s or "nc" in s:
        # Check for secondary vs primary: look for N with 2 C neighbors
        if "n(" in s or "n[" in s:
            return "amine_secondary", _GROUP_PKA["amine_secondary"]
        return "amine_primary", _GROUP_PKA["amine_primary"]

    return None, None


def ionization_at_pH(pka: float, molecule_type: str, pH: float) -> float:
    """Calculate fraction ionized using Henderson-Hasselbalch equation.

    Parameters
    ----------
    pka : float
        The pKa value of the ionizable group.
    molecule_type : str
        "acid" or "base" (determines ionization direction).
    pH : float
        The pH at which to calculate ionization fraction.

    Returns
    -------
    float
        Fraction ionized (0.0 to 1.0).
    """
    if pka <= 0:
        raise ValueError(f"pka must be positive, got {pka}")
    if pH < 0 or pH > 14:
        raise ValueError(f"pH must be between 0 and 14, got {pH}")
    if molecule_type not in _VALID_MOLECULE_TYPES:
        raise ValueError(
            f"molecule_type must be one of {_VALID_MOLECULE_TYPES}, got {molecule_type!r}"
        )

    delta = pH - pka
    if molecule_type in ("acid", "neutral"):
        # Acid: HA ⇌ H⁺ + A⁻; fraction ionized = 1/(1 + 10^(pKa-pH))
        fraction = 1.0 / (1.0 + math.pow(10.0, pka - pH))
    elif molecule_type == "base":
        # Base (protonated): BH⁺ ⇌ H⁺ + B; fraction protonated = 1/(1 + 10^(pH-pKa))
        fraction = 1.0 / (1.0 + math.pow(10.0, delta))
    else:
        # Amphoteric: average of acid and base ionization
        frac_acid = 1.0 / (1.0 + math.pow(10.0, pka - pH))
        frac_base = 1.0 / (1.0 + math.pow(10.0, delta))
        fraction = (frac_acid + frac_base) / 2.0

    return float(max(0.0, min(1.0, fraction)))


def predict_pka(smiles: str, molecule_type: str) -> PKaPredictionResult:
    """Predict pKa from SMILES structural features using rule-based approach.

    Parameters
    ----------
    smiles : str
        SMILES string of the molecule.
    molecule_type : str
        One of "acid", "base", "neutral", "amphoteric".

    Returns
    -------
    PKaPredictionResult
        Predicted pKa, detected functional group, and ionization data.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if not smiles or not smiles.strip():
        raise ValueError("smiles must be a non-empty string")
    if molecule_type not in _VALID_MOLECULE_TYPES:
        raise ValueError(
            f"molecule_type must be one of {_VALID_MOLECULE_TYPES}, got {molecule_type!r}"
        )

    smiles = smiles.strip()

    # Detect functional group
    detected_group, base_pka = _detect_functional_group(smiles)

    # Apply MW-based correction to carboxylic acid pKa
    pka_predicted: float | None = None
    if base_pka is not None:
        mw = _estimate_mw_from_smiles(smiles)
        if detected_group == "carboxylic_acid":
            # Electron-withdrawing groups (higher MW, more heteroatoms) lower pKa slightly
            mw_effect = -0.001 * max(mw - 200.0, 0.0)
            pka_predicted = float(base_pka + mw_effect)
        else:
            pka_predicted = float(base_pka)

    # Calculate ionization at pH 7
    if pka_predicted is not None:
        ionization_7 = ionization_at_pH(pka_predicted, molecule_type, 7.0)
    else:
        ionization_7 = 0.0

    # Determine solubility impact
    # Ionized species are generally more water-soluble
    if pka_predicted is not None and ionization_7 > 0.5:
        sol_impact = "improved"
    elif pka_predicted is not None and ionization_7 < 0.1:
        sol_impact = "reduced"
    else:
        sol_impact = "neutral"

    # Build notes
    if detected_group is not None:
        notes = (
            f"Detected {detected_group} group; pKa {pka_predicted:.2f}; "
            f"{ionization_7 * 100:.1f}% ionized at pH 7.4 (physiological)."
        )
    else:
        notes = (
            "No common ionizable group detected in SMILES. "
            "pKa prediction requires explicit functional group patterns."
        )

    return PKaPredictionResult(
        smiles=smiles,
        molecule_type=molecule_type,
        detected_group=detected_group,
        pka_predicted=pka_predicted,
        ionization_at_pH7=ionization_7,
        solubility_impact=sol_impact,
        notes=notes,
    )


__all__ = [
    "PKaPredictionResult",
    "predict_pka",
    "ionization_at_pH",
]
