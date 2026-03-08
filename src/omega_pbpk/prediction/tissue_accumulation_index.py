"""Phase 1038 — Drug Tissue Accumulation Index (prediction/tissue_accumulation_index.py).

Predict tissue accumulation potential based on physicochemical properties.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_VALID_TISSUES = {"fat", "muscle", "liver", "kidney", "heart", "lung"}
_VALID_IONIZATION = {"acid", "base", "neutral"}

# Tissue properties: f_water, f_lipid, tissue_pH, lysosomal
_TISSUE_PROPS: dict[str, dict] = {
    "fat": {"f_water": 0.10, "f_lipid": 0.80, "pH": 7.0, "lysosomal": False},
    "muscle": {"f_water": 0.75, "f_lipid": 0.05, "pH": 7.2, "lysosomal": False},
    "liver": {"f_water": 0.70, "f_lipid": 0.10, "pH": 7.2, "lysosomal": True},
    "kidney": {"f_water": 0.78, "f_lipid": 0.06, "pH": 7.3, "lysosomal": False},
    "heart": {"f_water": 0.76, "f_lipid": 0.06, "pH": 7.2, "lysosomal": False},
    "lung": {"f_water": 0.81, "f_lipid": 0.05, "pH": 6.9, "lysosomal": False},
}


@dataclass(frozen=True)
class TissueAccumulationResult:
    """Result of tissue accumulation prediction."""

    drug_name: str
    logp: float
    pka: float
    ionization_type: str
    tissue: str
    kp_predicted: float
    accumulation_index: float
    accumulation_class: str
    half_life_washout_h: float
    toxicity_concern: bool
    notes: str


def predict_tissue_accumulation(
    drug_name: str,
    logp: float,
    pka: float,
    ionization_type: str,
    tissue: str,
    fu_plasma: float = 0.3,
    cl_sys_L_per_h: float = 5.0,
    vd_sys_L: float = 30.0,
) -> TissueAccumulationResult:
    """Predict tissue accumulation index for a drug in a specific tissue.

    Parameters
    ----------
    drug_name: Name of the drug.
    logp: Octanol-water partition coefficient.
    pka: Acid/base pKa value.
    ionization_type: "acid", "base", or "neutral".
    tissue: One of fat/muscle/liver/kidney/heart/lung.
    fu_plasma: Unbound fraction in plasma (0, 1].
    cl_sys_L_per_h: Systemic clearance in L/h.
    vd_sys_L: Volume of distribution in L.

    Returns
    -------
    TissueAccumulationResult (frozen dataclass).
    """
    # Validation
    if fu_plasma <= 0 or fu_plasma > 1:
        raise ValueError("fu_plasma must be in (0, 1]")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    if ionization_type not in _VALID_IONIZATION:
        raise ValueError(f"ionization_type must be one of {_VALID_IONIZATION}")
    if tissue not in _VALID_TISSUES:
        raise ValueError(f"tissue must be one of {_VALID_TISSUES}")
    if pka < 0 or pka > 14:
        raise ValueError("pka must be in [0, 14]")

    props = _TISSUE_PROPS[tissue]
    f_water = props["f_water"]
    f_lipid = props["f_lipid"]
    tissue_pH = props["pH"]
    lysosomal = props["lysosomal"]

    # Base Kp from Rodgers-Rowland simplified
    kp = (f_water + f_lipid * (10**logp)) / fu_plasma

    # Ion trapping corrections
    if ionization_type == "base" and lysosomal:
        # Lysosomotropic amplification for basic drugs in liver
        kp_base = (1 + 10 ** (pka - 5.0)) / (1 + 10 ** (pka - 7.4))
        kp *= min(10.0, kp_base)
    elif ionization_type == "acid":
        # Acid ion trapping in tissues by pH difference
        kp_acid = (1 + 10 ** (7.4 - pka)) / (1 + 10 ** (tissue_pH - pka))
        kp *= max(0.1, kp_acid)

    # Clamp Kp
    kp = max(0.01, min(100.0, kp))

    # Accumulation index (0-5 scale)
    accumulation_index = min(5.0, math.log10(kp + 1) * 2.5)

    # Classification
    if accumulation_index < 0.5:
        accumulation_class = "negligible"
    elif accumulation_index < 1.5:
        accumulation_class = "low"
    elif accumulation_index < 2.5:
        accumulation_class = "moderate"
    elif accumulation_index < 3.5:
        accumulation_class = "high"
    else:
        accumulation_class = "very_high"

    # Washout half-life estimate
    half_life_washout_h = kp * vd_sys_L / cl_sys_L_per_h

    toxicity_concern = accumulation_index > 3.0

    notes = (
        f"Kp={kp:.3f}; ion_trap={'yes' if ionization_type != 'neutral' else 'no'}; "
        f"tissue_pH={tissue_pH}; lysosomal={lysosomal}; "
        f"accumulation_index={accumulation_index:.2f} ({accumulation_class})"
    )

    return TissueAccumulationResult(
        drug_name=drug_name,
        logp=logp,
        pka=pka,
        ionization_type=ionization_type,
        tissue=tissue,
        kp_predicted=kp,
        accumulation_index=accumulation_index,
        accumulation_class=accumulation_class,
        half_life_washout_h=half_life_washout_h,
        toxicity_concern=toxicity_concern,
        notes=notes,
    )


def screen_tissues(
    drug_name: str,
    logp: float,
    pka: float,
    ionization_type: str,
    fu_plasma: float = 0.3,
    cl_sys_L_per_h: float = 5.0,
    vd_sys_L: float = 30.0,
) -> list:
    """Screen all 6 tissues and return results sorted by kp_predicted descending.

    Parameters
    ----------
    drug_name: Name of the drug.
    logp: Octanol-water partition coefficient.
    pka: Acid/base pKa value.
    ionization_type: "acid", "base", or "neutral".
    fu_plasma: Unbound fraction in plasma (0, 1].
    cl_sys_L_per_h: Systemic clearance in L/h.
    vd_sys_L: Volume of distribution in L.

    Returns
    -------
    List of TissueAccumulationResult for all 6 tissues, sorted by kp_predicted desc.
    """
    results = []
    for tissue in _VALID_TISSUES:
        result = predict_tissue_accumulation(
            drug_name=drug_name,
            logp=logp,
            pka=pka,
            ionization_type=ionization_type,
            tissue=tissue,
            fu_plasma=fu_plasma,
            cl_sys_L_per_h=cl_sys_L_per_h,
            vd_sys_L=vd_sys_L,
        )
        results.append(result)
    results.sort(key=lambda r: r.kp_predicted, reverse=True)
    return results
