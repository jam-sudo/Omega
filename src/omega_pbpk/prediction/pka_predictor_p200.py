"""Phase 200 — pKa prediction from molecular descriptors (pka_predictor_p200.py).

Predicts pKa values using a rule-based approach driven by structural flags
(has_carboxyl, has_amine, has_phenol, has_imidazole) and physicochemical
descriptors (mw_Da, logP, smiles_fragments).

Multiple ionizable groups are supported — each yields one pKa value.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PKaResult:
    """Result of pKa prediction (Phase 200).

    Uses tuples (not lists) so frozen=True is valid.
    """

    drug_name: str
    pka_values: tuple  # tuple of floats
    ionization_types: tuple  # tuple of str, parallel to pka_values
    dominant_ionization: str
    mw_Da: float
    logP: float
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _ewg_count(smiles_fragments: list) -> int:
    """Count electron-withdrawing group indicators in fragment list."""
    ewg_keywords = {"F", "Cl", "Br", "NO2", "CF3", "CN", "SO2", "C(F)(F)F", "N+"}
    count = 0
    for frag in smiles_fragments:
        if frag in ewg_keywords:
            count += 1
    return count


def _predict_carboxyl_pka(mw_Da: float, logP: float, ewg_n: int) -> float:
    """Predict pKa for a carboxylic acid group."""
    base = 4.2
    # Electron-withdrawing groups lower pKa
    ewg_correction = -0.5 * min(ewg_n, 3)  # cap at -1.5
    # High MW fragment density (EWG-rich) → stronger acid
    mw_correction = -0.1 * max(0.0, (mw_Da - 300.0) / 100.0)
    mw_correction = max(-1.0, mw_correction)  # cap at -1.0
    # lipophilic → slightly higher pKa
    logp_correction = 0.3 if logP > 2.0 else 0.0
    pka = base + ewg_correction + mw_correction + logp_correction
    return _clip(pka, 2.0, 6.0)


def _predict_amine_pka(logP: float, ewg_n: int, aromatic: bool = False) -> float:
    """Predict pKa for an amine group."""
    if aromatic:
        base = 4.5
        ewg_correction = -1.0 * min(ewg_n, 2)
    else:
        base = 10.5
        ewg_correction = -1.5 * min(ewg_n, 1) + -0.5 * max(0, ewg_n - 1)
    pka = base + ewg_correction
    if aromatic:
        return _clip(pka, 2.0, 6.5)
    return _clip(pka, 7.0, 12.0)


def _predict_phenol_pka(logP: float, ewg_n: int) -> float:
    """Predict pKa for a phenol group."""
    base = 9.8
    ewg_correction = -0.4 * min(ewg_n, 2)
    return _clip(base + ewg_correction, 7.5, 11.5)


def _predict_imidazole_pka(logP: float) -> float:
    """Predict pKa for an imidazole group."""
    return 6.0  # nearly constant


def _henderson_hasselbalch_ionized(pka: float, ion_type: str, ph: float) -> float:
    """Return fraction ionized (0–1) using Henderson-Hasselbalch."""
    if ion_type == "acid":
        # HA → H⁺ + A⁻; fraction ionized = 1/(1 + 10^(pKa-pH))
        return 1.0 / (1.0 + 10.0 ** (pka - ph))
    else:
        # BH⁺ → B + H⁺; fraction ionized (protonated) = 1/(1 + 10^(pH-pKa))
        return 1.0 / (1.0 + 10.0 ** (ph - pka))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def predict_pka(
    drug_name: str,
    smiles_fragments: list,
    mw_Da: float,
    logP: float,
    has_carboxyl: bool = False,
    has_amine: bool = False,
    has_phenol: bool = False,
    has_imidazole: bool = False,
    aromatic_amine: bool = False,
) -> PKaResult:
    """Predict pKa values from molecular descriptors.

    Parameters
    ----------
    drug_name:
        Identifier for the compound.
    smiles_fragments:
        List of SMILES fragment strings used to detect EWGs.
    mw_Da:
        Molecular weight in Daltons.
    logP:
        Octanol-water partition coefficient.
    has_carboxyl:
        True if molecule contains a carboxylic acid group.
    has_amine:
        True if molecule contains an amine group.
    has_phenol:
        True if molecule contains a phenol group.
    has_imidazole:
        True if molecule contains an imidazole ring.
    aromatic_amine:
        True if the amine is aromatic (lowers pKa to ~4.5 range).

    Returns
    -------
    PKaResult
    """
    if mw_Da <= 0:
        raise ValueError(f"mw_Da must be > 0, got {mw_Da}")

    ewg_n = _ewg_count(smiles_fragments)

    pka_values: list = []
    ionization_types: list = []

    if has_carboxyl:
        pka = _predict_carboxyl_pka(mw_Da, logP, ewg_n)
        pka_values.append(pka)
        ionization_types.append("acid")

    if has_phenol:
        pka = _predict_phenol_pka(logP, ewg_n)
        pka_values.append(pka)
        ionization_types.append("acid")

    if has_amine:
        pka = _predict_amine_pka(logP, ewg_n, aromatic=aromatic_amine)
        pka_values.append(pka)
        ionization_types.append("base")

    if has_imidazole:
        pka = _predict_imidazole_pka(logP)
        pka_values.append(pka)
        ionization_types.append("base")

    # Determine dominant ionization
    if not pka_values:
        dominant = "non-ionizable"
        notes = "No ionizable groups detected; compound is non-ionizable."
    else:
        # Evaluate ionization at physiological pH 7.4 to find dominant group
        ph_physiological = 7.4
        max_ionized = -1.0
        dominant_idx = 0
        for i, (pka, ion_type) in enumerate(zip(pka_values, ionization_types)):
            frac = _henderson_hasselbalch_ionized(pka, ion_type, ph_physiological)
            if frac > max_ionized:
                max_ionized = frac
                dominant_idx = i
        dominant = ionization_types[dominant_idx]
        notes = (
            f"Predicted {len(pka_values)} ionizable group(s). "
            f"logP={logP:.2f}, MW={mw_Da:.1f} Da, EWG count={ewg_n}. "
            f"Dominant ionization at pH 7.4: {dominant}."
        )

    return PKaResult(
        drug_name=drug_name,
        pka_values=tuple(pka_values),
        ionization_types=tuple(ionization_types),
        dominant_ionization=dominant,
        mw_Da=mw_Da,
        logP=logP,
        notes=notes,
    )


def classify_ionization_at_ph(
    drug_name: str,
    pka_values: tuple,
    ionization_types: tuple,
    ph: float,
) -> dict:
    """Classify ionization state of a drug at a given pH.

    Parameters
    ----------
    drug_name:
        Compound identifier.
    pka_values:
        Tuple of pKa values (parallel to ionization_types).
    ionization_types:
        Tuple of "acid" or "base" strings (parallel to pka_values).
    ph:
        Target pH value (must be in [0, 14]).

    Returns
    -------
    dict with keys:
        ionized_fraction (float): overall fraction ionized in [0, 1]
        charge_state (float): net charge at this pH
        predominant_form (str): "neutral", "cation", "anion", "zwitterion"
    """
    if not (0.0 <= ph <= 14.0):
        raise ValueError(f"ph must be in [0, 14], got {ph}")

    if not pka_values:
        return {
            "ionized_fraction": 0.0,
            "charge_state": 0.0,
            "predominant_form": "neutral",
        }

    fractions = []
    net_charge = 0.0
    n_acid = 0
    n_base = 0

    for pka, ion_type in zip(pka_values, ionization_types):
        frac = _henderson_hasselbalch_ionized(float(pka), ion_type, ph)
        fractions.append(frac)
        if ion_type == "acid":
            net_charge += -frac  # ionized acid carries -1 charge
            n_acid += 1
        else:
            net_charge += frac  # ionized base carries +1 charge
            n_base += 1

    ionized_fraction = sum(fractions) / len(fractions) if fractions else 0.0
    ionized_fraction = _clip(ionized_fraction, 0.0, 1.0)

    has_acid = n_acid > 0
    has_base = n_base > 0
    if abs(net_charge) < 0.2:
        predominant_form = "neutral"
    elif has_acid and has_base:
        predominant_form = "zwitterion"
    elif net_charge < -0.2:
        predominant_form = "anion"
    else:
        predominant_form = "cation"

    return {
        "ionized_fraction": ionized_fraction,
        "charge_state": net_charge,
        "predominant_form": predominant_form,
    }


__all__ = [
    "PKaResult",
    "predict_pka",
    "classify_ionization_at_ph",
]
