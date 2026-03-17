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
    "enol_lactone": 5.0,
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

    # Enol-lactone (e.g. 4-hydroxycoumarin, warfarin): aromatic OH + ring C=O
    # Distinguished from phenol by the presence of a ring-closing C=O bond (cN=o pattern),
    # which indicates a lactone or coumarin carbonyl that phenols lack.
    if ("oc1" in s or "c1o" in s or "c(o)" in s) and any(ch.isdigit() for ch in s):
        if any(f"c{d}=o" in s for d in "123456789"):
            return "enol_lactone", _GROUP_PKA["enol_lactone"]

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
    # Phase 646 additions
    "PKaPrediction",
    "predict_pka_v2",
    "classify_ionization",
    "screen_pka",
]


# ===========================================================================
# Phase 646 — pKa Predictor (fragment-based QSPR, new API)
# ===========================================================================


@dataclass(frozen=True)
class PKaPrediction:
    """Comprehensive pKa prediction result (Phase 646)."""

    compound_name: str
    smiles_like_formula: str
    n_acidic_groups: int
    n_basic_groups: int
    predicted_pka_acids: list
    predicted_pka_bases: list
    physiological_charge: float
    ionization_at_ph74: float
    dominant_ionization_form: str
    permeability_impact: str
    renal_clearance_impact: str
    notes: str


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _cooh_pka(logP: float) -> float:
    return 4.2 - logP * 0.05


def _phenol_pka(logP: float) -> float:
    return _clip(9.5 - logP * 0.1, 8.0, 11.0)


def _primary_amine_pka(logP: float) -> float:
    return _clip(10.5 - logP * 0.2, 8.0, 12.0)


def _secondary_amine_pka(logP: float) -> float:
    return _clip(9.5 - logP * 0.15, 7.0, 11.0)


def _tertiary_amine_pka(logP: float) -> float:
    return _clip(8.5 - logP * 0.1, 6.0, 10.0)


def _pyridine_pka(logP: float) -> float:
    return _clip(4.5 + logP * 0.1, 2.0, 7.0)


def _imidazole_pka(logP: float) -> float:
    return 6.8  # constant, minor logP effect handled in notes


def _acid_charge_contribution(pka: float, ph: float = 7.4) -> float:
    """Fraction ionized for acid × (-1) charge."""
    return -(1.0 / (1.0 + 10.0 ** (pka - ph)))


def _base_charge_contribution(pka: float, ph: float = 7.4) -> float:
    """Fraction ionized (protonated) for base × (+1) charge."""
    return 1.0 / (1.0 + 10.0 ** (ph - pka))


def predict_pka_v2(
    compound_name: str,
    n_cooh_groups: int = 0,
    n_phenol_groups: int = 0,
    n_primary_amine: int = 0,
    n_secondary_amine: int = 0,
    n_tertiary_amine: int = 0,
    n_pyridine_n: int = 0,
    n_imidazole: int = 0,
    logP: float = 1.0,
    mw: float = 300.0,
    temperature_c: float = 37.0,
) -> PKaPrediction:
    """Predict pKa values from molecular descriptors using fragment QSPR.

    Parameters
    ----------
    compound_name:
        Name or identifier of the compound.
    n_cooh_groups:
        Number of carboxylic acid groups.
    n_phenol_groups:
        Number of phenolic OH groups.
    n_primary_amine:
        Number of primary amine groups (-NH2).
    n_secondary_amine:
        Number of secondary amine groups (-NH-).
    n_tertiary_amine:
        Number of tertiary amine groups (-N<).
    n_pyridine_n:
        Number of pyridine nitrogen atoms.
    n_imidazole:
        Number of imidazole groups.
    logP:
        Octanol-water partition coefficient.
    mw:
        Molecular weight in Da.
    temperature_c:
        Temperature in Celsius (physiological default 37.0).
    """
    # Validation
    for name, val in [
        ("n_cooh_groups", n_cooh_groups),
        ("n_phenol_groups", n_phenol_groups),
        ("n_primary_amine", n_primary_amine),
        ("n_secondary_amine", n_secondary_amine),
        ("n_tertiary_amine", n_tertiary_amine),
        ("n_pyridine_n", n_pyridine_n),
        ("n_imidazole", n_imidazole),
    ]:
        if val < 0:
            raise ValueError(f"{name} must be >= 0, got {val}")
    if mw <= 0:
        raise ValueError(f"mw must be > 0, got {mw}")
    if not (20.0 <= temperature_c <= 45.0):
        raise ValueError(f"temperature_c must be in [20, 45], got {temperature_c}")

    # Build acid pKa list
    acid_pkas: list[float] = []
    for _ in range(n_cooh_groups):
        acid_pkas.append(_cooh_pka(logP))
    for _ in range(n_phenol_groups):
        acid_pkas.append(_phenol_pka(logP))
    acid_pkas.sort(reverse=True)  # highest pKa first (weakest acid first)

    # Build base pKa list
    base_pkas: list[float] = []
    for _ in range(n_primary_amine):
        base_pkas.append(_primary_amine_pka(logP))
    for _ in range(n_secondary_amine):
        base_pkas.append(_secondary_amine_pka(logP))
    for _ in range(n_tertiary_amine):
        base_pkas.append(_tertiary_amine_pka(logP))
    for _ in range(n_pyridine_n):
        base_pkas.append(_pyridine_pka(logP))
    for _ in range(n_imidazole):
        base_pkas.append(_imidazole_pka(logP))
    base_pkas.sort(reverse=True)  # highest pKa first (strongest base first)

    n_acidic = n_cooh_groups + n_phenol_groups
    n_basic = n_primary_amine + n_secondary_amine + n_tertiary_amine + n_pyridine_n + n_imidazole

    # Physiological charge at pH 7.4
    ph = 7.4
    total_charge = 0.0
    for pka in acid_pkas:
        total_charge += _acid_charge_contribution(pka, ph)
    for pka in base_pkas:
        total_charge += _base_charge_contribution(pka, ph)

    # Fraction ionized: sum of absolute charge contributions
    abs_charge = abs(total_charge)
    total_groups = n_acidic + n_basic
    ionization_at_ph74 = abs_charge / total_groups if total_groups > 0 else 0.0
    ionization_at_ph74 = min(1.0, ionization_at_ph74)

    # Dominant form
    has_acid = n_acidic > 0
    has_base = n_basic > 0
    if abs(total_charge) < 0.2:
        dominant_form = "neutral"
    elif has_acid and has_base:
        dominant_form = "zwitterion"
    elif total_charge < -0.2:
        dominant_form = "anion"
    else:
        dominant_form = "cation"

    # Permeability impact
    if abs(total_charge) < 0.2:
        permeability_impact = "high"
    elif abs(total_charge) < 0.8:
        permeability_impact = "moderate"
    else:
        permeability_impact = "low"

    # Renal clearance impact: pH-dependent if any pKa in 5-8 range
    all_pkas = acid_pkas + base_pkas
    if any(5.0 <= pka <= 8.0 for pka in all_pkas):
        renal_clearance_impact = "pH-dependent"
    else:
        renal_clearance_impact = "minimal"

    formula_desc = f"acids:{n_acidic} bases:{n_basic} logP:{logP:.1f} MW:{mw:.0f}"

    notes_parts = [f"Compound: {compound_name}."]
    if acid_pkas:
        notes_parts.append(f"Acid pKas: {[round(v, 2) for v in acid_pkas]}.")
    if base_pkas:
        notes_parts.append(f"Base pKas: {[round(v, 2) for v in base_pkas]}.")
    notes_parts.append(f"Net charge at pH 7.4: {total_charge:.2f}. Form: {dominant_form}.")
    if not acid_pkas and not base_pkas:
        notes_parts.append("No ionizable groups detected; compound is neutral.")
    notes = " ".join(notes_parts)

    return PKaPrediction(
        compound_name=compound_name,
        smiles_like_formula=formula_desc,
        n_acidic_groups=n_acidic,
        n_basic_groups=n_basic,
        predicted_pka_acids=acid_pkas,
        predicted_pka_bases=base_pkas,
        physiological_charge=total_charge,
        ionization_at_ph74=ionization_at_ph74,
        dominant_ionization_form=dominant_form,
        permeability_impact=permeability_impact,
        renal_clearance_impact=renal_clearance_impact,
        notes=notes,
    )


def classify_ionization(pka_values: list, ph: float = 7.4) -> dict:
    """Return fraction ionized, dominant form, and charge at given pH.

    Parameters
    ----------
    pka_values:
        List of dicts with keys 'pka' (float) and 'type' ('acid' or 'base').
    ph:
        pH at which to evaluate ionization.
    """
    total_charge = 0.0
    n_acid = 0
    n_base = 0
    fractions = []

    for item in pka_values:
        pka = float(item["pka"])
        group_type = item.get("type", "acid")
        if group_type == "acid":
            frac = 1.0 / (1.0 + 10.0 ** (pka - ph))
            total_charge += -frac
            n_acid += 1
            fractions.append(frac)
        else:
            frac = 1.0 / (1.0 + 10.0 ** (ph - pka))
            total_charge += frac
            n_base += 1
            fractions.append(frac)

    total_groups = n_acid + n_base
    fraction_ionized = sum(fractions) / total_groups if total_groups > 0 else 0.0

    has_acid = n_acid > 0
    has_base = n_base > 0
    if abs(total_charge) < 0.2:
        dominant_form = "neutral"
    elif has_acid and has_base:
        dominant_form = "zwitterion"
    elif total_charge < -0.2:
        dominant_form = "anion"
    else:
        dominant_form = "cation"

    return {
        "fraction_ionized": min(1.0, fraction_ionized),
        "dominant_form": dominant_form,
        "charge": total_charge,
        "ph": ph,
    }


def screen_pka(compounds: list) -> list:
    """Screen multiple compounds, return list sorted by physiological_charge ascending.

    Parameters
    ----------
    compounds:
        List of dicts, each containing keyword arguments for predict_pka_v2.
    """
    results = []
    for compound in compounds:
        result = predict_pka_v2(**compound)
        results.append(result)
    results.sort(key=lambda r: r.physiological_charge)
    return results
