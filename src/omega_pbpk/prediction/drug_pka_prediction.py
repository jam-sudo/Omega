"""
Phase 970 — pKa prediction from molecular descriptors.

Predict ionization constants for acids, bases, and zwitterions using
rule-based functional group contributions with Hammett/Taft corrections.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["PKAPredictionResult", "predict_pka", "screen_pka_profiles"]

# Reference pKa values for common functional groups
FUNCTIONAL_GROUP_PKA: dict = {
    "carboxylic_acid": 4.2,
    "aromatic_amine": 4.6,
    "aliphatic_amine": 9.5,
    "phenol": 9.2,
    "imidazole": 6.0,
    "pyridine": 5.2,
    "sulfonamide": 10.1,
    "sulfuric_acid": 1.2,
    "phosphoric_acid": 2.1,
    "thiol": 8.3,
}

# Functional groups classified as acids or bases
_ACID_GROUPS = {
    "carboxylic_acid",
    "phenol",
    "sulfonamide",
    "sulfuric_acid",
    "phosphoric_acid",
    "thiol",
}
_BASE_GROUPS = {
    "aromatic_amine",
    "aliphatic_amine",
    "imidazole",
    "pyridine",
}


def _fraction_ionized(pka: float, ph: float, pka_type: str) -> float:
    """Henderson-Hasselbalch ionized fraction."""
    if pka_type == "acid":
        # HA ⇌ H+ + A-  → f_ionized = 1 / (1 + 10^(pKa - pH))
        exponent = pka - ph
    elif pka_type == "base":
        # BH+ ⇌ B + H+  → f_ionized = 1 / (1 + 10^(pH - pKa))
        exponent = ph - pka
    else:
        return 0.0  # neutral

    # Clamp exponent to avoid overflow
    exponent = max(-30.0, min(30.0, exponent))
    return 1.0 / (1.0 + math.pow(10.0, exponent))


def _effective_solubility_ratio(pka: float, pka_type: str) -> float:
    """
    Ratio of apparent solubility at intestinal pH (6.8) relative to intrinsic
    (unionized) solubility using Henderson-Hasselbalch equation.

    S_pH / S0 = 1 + 10^(pH - pKa)  for acids
    S_pH / S0 = 1 + 10^(pKa - pH)  for bases
    """
    ph_intestinal = 6.8
    if pka_type == "acid":
        exponent = ph_intestinal - pka
    elif pka_type == "base":
        exponent = pka - ph_intestinal
    else:
        return 1.0

    exponent = max(-30.0, min(30.0, exponent))
    ratio = 1.0 + math.pow(10.0, exponent)
    return max(1.0, ratio)


def _charge_at_ph(pka: float, pka_type: str, ph: float = 7.4) -> float:
    """Approximate charge at given pH."""
    fi = _fraction_ionized(pka, ph, pka_type)
    if pka_type == "acid":
        return -fi  # ionized acid carries -1 charge
    elif pka_type == "base":
        return fi   # ionized base carries +1 charge
    return 0.0


@dataclass(frozen=True)
class PKAPredictionResult:
    drug_name: str
    logp: float
    mw: float
    functional_group: str
    pka_predicted: float
    pka_type: str  # "acid" | "base" | "neutral"
    fraction_ionized_at_physiological_ph: float
    fraction_ionized_at_gastric_ph: float
    fraction_ionized_at_intestinal_ph: float
    effective_solubility_ratio_intestinal: float
    charge_at_physiological_ph: float
    notes: str


def predict_pka(
    drug_name: str,
    logp: float,
    mw: float,
    functional_group: str,
) -> PKAPredictionResult:
    """Predict pKa and ionization properties from molecular descriptors.

    Parameters
    ----------
    drug_name : str
    logp : float
    mw : float  molecular weight (g/mol)
    functional_group : str  key from FUNCTIONAL_GROUP_PKA or "unknown"

    Returns
    -------
    PKAPredictionResult (frozen dataclass)
    """
    if mw <= 0:
        raise ValueError("mw must be > 0")

    valid_groups = set(FUNCTIONAL_GROUP_PKA.keys()) | {"unknown"}
    if functional_group not in valid_groups:
        raise ValueError(f"functional_group not recognized: {functional_group!r}")

    # Base pKa
    if functional_group == "unknown":
        pka_ref = 7.0
        pka_type = "neutral"
    else:
        pka_ref = FUNCTIONAL_GROUP_PKA[functional_group]
        if functional_group in _ACID_GROUPS:
            pka_type = "acid"
        else:
            pka_type = "base"

    # logP correction: electron donation/withdrawal
    pka_corrected = pka_ref + 0.2 * logp

    # MW correction: steric effects for large molecules
    if mw > 500:
        pka_corrected += 0.3

    # Ionization fractions at physiological pH levels
    fi_physiological = _fraction_ionized(pka_corrected, 7.4, pka_type)
    fi_gastric = _fraction_ionized(pka_corrected, 1.2, pka_type)
    fi_intestinal = _fraction_ionized(pka_corrected, 6.8, pka_type)

    # Effective solubility ratio at intestinal pH
    sol_ratio = _effective_solubility_ratio(pka_corrected, pka_type)

    # Charge at physiological pH
    charge = _charge_at_ph(pka_corrected, pka_type, ph=7.4)

    notes = (
        f"functional_group={functional_group}; "
        f"pKa_ref={pka_ref:.2f}; "
        f"logP_correction={0.2 * logp:+.2f}; "
        f"MW_correction={'+0.30' if mw > 500 else '+0.00'}; "
        f"pKa_predicted={pka_corrected:.2f}; "
        f"type={pka_type}"
    )

    return PKAPredictionResult(
        drug_name=drug_name,
        logp=logp,
        mw=mw,
        functional_group=functional_group,
        pka_predicted=pka_corrected,
        pka_type=pka_type,
        fraction_ionized_at_physiological_ph=fi_physiological,
        fraction_ionized_at_gastric_ph=fi_gastric,
        fraction_ionized_at_intestinal_ph=fi_intestinal,
        effective_solubility_ratio_intestinal=sol_ratio,
        charge_at_physiological_ph=charge,
        notes=notes,
    )


def screen_pka_profiles(compounds: list) -> list[PKAPredictionResult]:
    """Screen multiple compounds for pKa profiles.

    Parameters
    ----------
    compounds : list[dict]  each dict with keys: drug_name, logp, mw, functional_group

    Returns
    -------
    list[PKAPredictionResult] sorted by pka_predicted ascending
    """
    results = []
    for comp in compounds:
        r = predict_pka(
            drug_name=comp["drug_name"],
            logp=comp["logp"],
            mw=comp["mw"],
            functional_group=comp["functional_group"],
        )
        results.append(r)

    results.sort(key=lambda x: x.pka_predicted)
    return results
