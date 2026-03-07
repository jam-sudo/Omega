"""Phase 559 — logD (distribution coefficient) prediction at physiological pH.

Pure Python, no numpy/scipy.
"""

from __future__ import annotations

import math

_VALID_DRUG_TYPES = {"acid", "base", "neutral", "zwitterion"}


def _validate_drug_type(drug_type: str) -> None:
    if drug_type not in _VALID_DRUG_TYPES:
        raise ValueError(
            f"Invalid drug_type '{drug_type}'. Must be one of: {sorted(_VALID_DRUG_TYPES)}"
        )


def logd_from_logp_pka(logP: float, pka: float, pH: float, drug_type: str) -> float:
    """Predict logD at a given pH from logP and pKa.

    For acid:      logD = logP - log10(1 + 10^(pH - pKa))
    For base:      logD = logP - log10(1 + 10^(pKa - pH))
    For neutral:   logD = logP
    For zwitterion: average of acid and base corrections.
    """
    _validate_drug_type(drug_type)

    if drug_type == "neutral":
        return logP

    if drug_type == "acid":
        return logP - math.log10(1.0 + 10.0 ** (pH - pka))

    if drug_type == "base":
        return logP - math.log10(1.0 + 10.0 ** (pka - pH))

    # zwitterion: average of acid and base corrections
    logd_acid = logP - math.log10(1.0 + 10.0 ** (pH - pka))
    logd_base = logP - math.log10(1.0 + 10.0 ** (pka - pH))
    return (logd_acid + logd_base) / 2.0


def ionized_fraction(pka: float, pH: float, drug_type: str) -> float:
    """Fraction ionized at a given pH.

    For acid:      fi = 1 / (1 + 10^(pKa - pH))
    For base:      fi = 1 / (1 + 10^(pH - pKa))
    For neutral:   fi = 0.0
    For zwitterion: average of acid and base fi.

    Returns value clamped to [0, 1].
    """
    _validate_drug_type(drug_type)

    if drug_type == "neutral":
        return 0.0

    if drug_type == "acid":
        fi = 1.0 / (1.0 + 10.0 ** (pka - pH))
        return max(0.0, min(1.0, fi))

    if drug_type == "base":
        fi = 1.0 / (1.0 + 10.0 ** (pH - pka))
        return max(0.0, min(1.0, fi))

    # zwitterion
    fi_acid = 1.0 / (1.0 + 10.0 ** (pka - pH))
    fi_base = 1.0 / (1.0 + 10.0 ** (pH - pka))
    return max(0.0, min(1.0, (fi_acid + fi_base) / 2.0))


def logd_at_multiple_ph(
    logP: float,
    pka: float,
    drug_type: str,
    ph_values: list[float] | None = None,
) -> list[dict]:
    """Compute logD and ionized_fraction at multiple pH values.

    Default pH values represent GI compartments and plasma: [1.2, 4.5, 6.8, 7.4].

    Returns list of dicts with keys: pH, logD, ionized_fraction.
    """
    _validate_drug_type(drug_type)

    if ph_values is None:
        ph_values = [1.2, 4.5, 6.8, 7.4]

    results = []
    for ph in ph_values:
        logd = logd_from_logp_pka(logP, pka, ph, drug_type)
        fi = ionized_fraction(pka, ph, drug_type)
        results.append({"pH": ph, "logD": logd, "ionized_fraction": fi})

    return results


def predict_membrane_permeability(logD_74: float) -> dict:
    """Estimate membrane permeability from logD at pH 7.4.

    Empirical: Papp (cm/s) = 10^(0.8 * logD_74 - 5.7)

    Classification:
      high     : Papp > 1e-5 cm/s
      moderate : 1e-6 <= Papp <= 1e-5 cm/s
      low      : Papp < 1e-6 cm/s

    Returns: logD_74, papp_cm_s, permeability_class.
    """
    papp = 10.0 ** (0.8 * logD_74 - 5.7)

    if papp > 1e-5:
        perm_class = "high"
    elif papp >= 1e-6:
        perm_class = "moderate"
    else:
        perm_class = "low"

    return {
        "logD_74": logD_74,
        "papp_cm_s": papp,
        "permeability_class": perm_class,
    }


def logd_protein_binding_correction(
    logD_74: float,
    protein_conc_g_L: float = 45.0,
) -> dict:
    """Estimate plasma protein binding (fu) from logD at pH 7.4.

    Empirical model:
      pKd_ppb = 0.72 * logD_74 + 2.1
      fu = 1 / (1 + 10^pKd_ppb * protein_conc_g_L / 66500)

    fu is clamped to (0, 1].

    Returns: logD_74, fu_predicted, pKd_ppb, notes.
    """
    pkd_ppb = 0.72 * logD_74 + 2.1
    kd = 10.0**pkd_ppb
    # protein_conc in mol/L: protein_conc_g_L / MW_albumin(66500 g/mol)
    protein_conc_mol_L = protein_conc_g_L / 66500.0
    fu = 1.0 / (1.0 + kd * protein_conc_mol_L)

    # Clamp to (0, 1]
    fu = max(1e-9, min(1.0, fu))

    notes = (
        "Rough estimate based on empirical logD-PPB correlation. "
        "Validate with experimental measurements."
    )

    return {
        "logD_74": logD_74,
        "fu_predicted": fu,
        "pKd_ppb": pkd_ppb,
        "notes": notes,
    }
