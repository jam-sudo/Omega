"""PAMPA (Parallel Artificial Membrane Permeability Assay) permeability prediction.

Predicts membrane permeability from physicochemical properties (logP, MW)
and classifies compounds for BCS permeability class and fraction absorbed.

References
----------
- Sugano K et al. Caco-2/PAMPA correlation. Eur J Pharm Sci. 2006.
- Sun D et al. Peff from Caco-2 for PBPK. Drug Metab Dispos. 2002.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PAMPAPermeabilityResult:
    """Result from PAMPA permeability prediction.

    Attributes
    ----------
    drug_name : Name of the drug compound.
    logP : Octanol-water partition coefficient.
    mw_Da : Molecular weight (Da).
    pKa : Acid/base dissociation constant.
    drug_type : Ionization class: 'acid', 'base', or 'neutral'.
    peff_cm_per_s : Predicted PAMPA Peff (cm/s).
    peff_apparent_cm_per_s : pH-corrected apparent Peff at intestinal pH.
    absorption_number : Dimensionless absorption number (An).
    fa_predicted : Predicted fraction absorbed (0-1).
    bcs_permeability_class : 'high' (BCS I/II) or 'low' (BCS III/IV).
    fa_classification : 'complete', 'moderate', or 'low' absorption.
    notes : Summary of key predictions.
    """

    drug_name: str
    logP: float
    mw_Da: float
    pKa: float
    drug_type: str
    peff_cm_per_s: float
    peff_apparent_cm_per_s: float
    absorption_number: float
    fa_predicted: float
    bcs_permeability_class: str
    fa_classification: str
    notes: str


# Constants
_R_INTESTINE_CM = 1.75  # intestinal radius (cm)
_T_TRANSIT_MIN = 200.0  # small intestinal transit time (min)
_T_TRANSIT_S = _T_TRANSIT_MIN * 60.0  # convert to seconds
_PEFF_MIN = 1e-8  # cm/s clamp lower bound
_PEFF_MAX = 1e-4  # cm/s clamp upper bound
_PEFF_HIGH_THRESHOLD = 1e-6  # cm/s BCS high/low boundary
_INTESTINAL_PH = 6.5  # default intestinal pH for ionization correction


def _predict_peff(logP: float, mw_Da: float) -> float:
    """Predict PAMPA Peff from logP and MW, clamped to [1e-8, 1e-4] cm/s."""
    log10_peff = 0.5 * logP - 0.01 * mw_Da - 4.5
    peff = 10.0**log10_peff
    return max(_PEFF_MIN, min(_PEFF_MAX, peff))


def _ionized_fraction_acid(ph: float, pka: float) -> float:
    """Fraction ionized for an acid at given pH using Henderson-Hasselbalch."""
    delta = ph - pka
    # ionized_fraction = 10^(pH-pKa) / (1 + 10^(pH-pKa))
    # Clamp exponent to avoid overflow
    delta_clamped = max(-30.0, min(30.0, delta))
    ratio = 10.0**delta_clamped
    return ratio / (1.0 + ratio)


def _absorption_number(peff_cm_per_s: float) -> float:
    """Compute dimensionless absorption number An = Peff * 2 * t_transit / R."""
    return peff_cm_per_s * 2.0 * _T_TRANSIT_S / _R_INTESTINE_CM


def _fa_from_an(an: float) -> float:
    """Fraction absorbed from absorption number: fa = 1 - exp(-2*An)."""
    if an <= 0.0:
        return 0.0
    return 1.0 - math.exp(-2.0 * an)


def predict_pampa_permeability(
    drug_name: str,
    logP: float,
    mw_Da: float,
    pKa: float,
    drug_type: str,
    ph_intestine: float = _INTESTINAL_PH,
) -> PAMPAPermeabilityResult:
    """Predict PAMPA permeability and absorption from physicochemical properties.

    Parameters
    ----------
    drug_name : Name of the drug compound.
    logP : Octanol-water partition coefficient.
    mw_Da : Molecular weight (Da), must be > 0.
    pKa : Ionization constant.
    drug_type : One of 'acid', 'base', or 'neutral'.
    ph_intestine : Intestinal pH for apparent permeability calculation.

    Returns
    -------
    PAMPAPermeabilityResult with all permeability and absorption metrics.

    Raises
    ------
    ValueError : If mw_Da <= 0 or drug_type is not valid.
    """
    if mw_Da <= 0:
        raise ValueError("mw_Da must be > 0")
    if drug_type not in {"acid", "base", "neutral"}:
        raise ValueError("drug_type must be 'acid', 'base', or 'neutral'")

    # Base Peff from logP and MW
    peff = _predict_peff(logP, mw_Da)

    # pH-corrected apparent Peff
    if drug_type == "acid":
        ion_frac = _ionized_fraction_acid(ph_intestine, pKa)
        peff_apparent = peff * (1.0 - ion_frac)
        # Clamp to minimum
        peff_apparent = max(_PEFF_MIN, peff_apparent)
    else:
        # bases and neutrals: neutral form permeates regardless
        peff_apparent = peff

    # Absorption number and fa
    an = _absorption_number(peff_apparent)
    fa = _fa_from_an(an)

    # BCS permeability classification
    bcs_class = "high" if peff_apparent > _PEFF_HIGH_THRESHOLD else "low"

    # fa classification
    if fa > 0.9:
        fa_class = "complete"
    elif fa >= 0.5:
        fa_class = "moderate"
    else:
        fa_class = "low"

    notes = (
        f"{drug_name}: logP={logP:.2f}, MW={mw_Da:.1f} Da, "
        f"pKa={pKa:.2f} ({drug_type}). "
        f"Peff={peff:.2e} cm/s, Peff_apparent={peff_apparent:.2e} cm/s "
        f"(pH {ph_intestine:.1f}). "
        f"An={an:.3f}, fa={fa:.3f} ({fa_class}). "
        f"BCS permeability: {bcs_class}."
    )

    return PAMPAPermeabilityResult(
        drug_name=drug_name,
        logP=logP,
        mw_Da=mw_Da,
        pKa=pKa,
        drug_type=drug_type,
        peff_cm_per_s=peff,
        peff_apparent_cm_per_s=peff_apparent,
        absorption_number=an,
        fa_predicted=fa,
        bcs_permeability_class=bcs_class,
        fa_classification=fa_class,
        notes=notes,
    )


def screen_pampa(compounds: list) -> list:
    """Screen a list of compounds for PAMPA permeability, sorted by Peff descending.

    Parameters
    ----------
    compounds : List of dicts with keys:
        - drug_name (str)
        - logP (float)
        - mw_Da (float)
        - pKa (float)
        - drug_type (str: 'acid'/'base'/'neutral')
        - ph_intestine (float, optional, default 6.5)

    Returns
    -------
    List of PAMPAPermeabilityResult sorted by peff_cm_per_s descending.
    """
    results: list[PAMPAPermeabilityResult] = []
    for cmpd in compounds:
        result = predict_pampa_permeability(
            drug_name=cmpd["drug_name"],
            logP=cmpd["logP"],
            mw_Da=cmpd["mw_Da"],
            pKa=cmpd["pKa"],
            drug_type=cmpd["drug_type"],
            ph_intestine=cmpd.get("ph_intestine", _INTESTINAL_PH),
        )
        results.append(result)
    results.sort(key=lambda r: r.peff_cm_per_s, reverse=True)
    return results


__all__ = [
    "PAMPAPermeabilityResult",
    "predict_pampa_permeability",
    "screen_pampa",
]
