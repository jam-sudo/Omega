"""Volume of distribution prediction using tissue partition equations."""

from __future__ import annotations

import math
from dataclasses import dataclass


# Tissue volumes (L/70 kg human) and blood flows (L/h)
# From ICRP 2002 reference human
_TISSUE_DATA = {
    "adipose": {"volume_L": 18.2, "frac_water": 0.15, "frac_lipid": 0.80, "frac_protein": 0.02},
    "muscle": {"volume_L": 29.0, "frac_water": 0.76, "frac_lipid": 0.01, "frac_protein": 0.18},
    "liver": {"volume_L": 1.69, "frac_water": 0.71, "frac_lipid": 0.04, "frac_protein": 0.14},
    "brain": {"volume_L": 1.45, "frac_water": 0.77, "frac_lipid": 0.11, "frac_protein": 0.08},
    "kidney": {"volume_L": 0.31, "frac_water": 0.78, "frac_lipid": 0.01, "frac_protein": 0.16},
    "heart": {"volume_L": 0.33, "frac_water": 0.76, "frac_lipid": 0.01, "frac_protein": 0.17},
    "lung": {"volume_L": 0.50, "frac_water": 0.79, "frac_lipid": 0.02, "frac_protein": 0.13},
    "skin": {"volume_L": 3.71, "frac_water": 0.71, "frac_lipid": 0.01, "frac_protein": 0.21},
    "bone": {"volume_L": 10.5, "frac_water": 0.40, "frac_lipid": 0.03, "frac_protein": 0.24},
    "spleen": {"volume_L": 0.18, "frac_water": 0.78, "frac_lipid": 0.01, "frac_protein": 0.16},
}

# Plasma fractions for partitioning
_PLASMA_WATER = 0.935
_BLOOD_PLASMA_RATIO = 1.0  # simplified: assume BP ratio = 1


@dataclass(frozen=True)
class VolumePredictionResult:
    drug_name: str
    vd_L: float               # Vd at steady-state (L/70 kg)
    vd_per_kg_L: float        # Vd (L/kg)
    kp_tissue: dict           # tissue:Kp values
    classification: str       # low (<0.6 L/kg) / moderate (0.6–20) / high (>20)
    fup: float
    logP: float
    method: str


def _kp_regression(
    logP: float,
    fup: float,
    tissue: str,
    pka_acid: float | None = None,
    pka_basic: float | None = None,
    drug_type: str = "neutral",
) -> float:
    """Simplified Rodgers-Rowland Kp (tissue:plasma) for neutral/acid/base drugs.

    For neutral drugs: Kp = fup * (fw_t/fw_p + P * flp_t / flp_p)
    For basic drugs: additional lysosomal trapping correction.
    """
    td = _TISSUE_DATA.get(tissue, _TISSUE_DATA["muscle"])
    fw_t = td["frac_water"]
    flp_t = td["frac_lipid"]
    fp_t = td["frac_protein"]

    fw_p = _PLASMA_WATER
    P = 10 ** logP  # octanol:water partition coefficient

    # Neutral drug (Kp):
    # Kp = fup * (fw_t/fw_p + P*flp_t/flp_p + fp_t*fp_factor/fw_p)
    # Simplified: ignore protein binding saturation
    flp_p = 0.001  # plasma phospholipid fraction
    fp_factor = 0.035  # nonspecific protein binding factor

    kp = fup * (fw_t / fw_p + P * flp_t / max(flp_p, 1e-6) * 0.001 + fp_t * fp_factor / fw_p)

    # Basic drug: lysosomal trapping → higher tissue binding
    if drug_type == "basic" or (pka_basic is not None and pka_basic > 7.0):
        # pH-dependent trapping factor
        pka_b = pka_basic if pka_basic is not None else 9.0
        trapped = max(1.0, 1.0 + 10 ** (pka_b - 7.4))
        kp = kp * min(trapped, 20.0)

    # Acid: lower Kp in most tissues except albumin-rich
    if drug_type == "acid" or (pka_acid is not None and pka_acid < 6.0):
        kp = kp * 0.6

    return max(kp, 0.001)


def predict_vd_tissue(
    drug_name: str,
    logP: float,
    fup: float,
    pka_acid: float | None = None,
    pka_basic: float | None = None,
    drug_type: str = "neutral",
    body_weight_kg: float = 70.0,
) -> VolumePredictionResult:
    """Predict Vd using tissue partition coefficients.

    Vd = Vp + Σ(Vt * Kp_t)  where Kp_t = tissue:plasma partition.

    Parameters
    ----------
    drug_type : str
        'neutral', 'acid', or 'basic'.
    """
    fup = float(max(min(fup, 1.0), 0.001))

    kp_dict = {}
    for tissue in _TISSUE_DATA:
        kp_dict[tissue] = _kp_regression(logP, fup, tissue, pka_acid, pka_basic, drug_type)

    # Vd = Vplasma + Σ(Vt * Kp_t * (fup/1))
    # Volume of plasma ~3 L in 70 kg
    v_plasma = 3.0 * (body_weight_kg / 70.0)

    vd = v_plasma
    for tissue, td in _TISSUE_DATA.items():
        v_tissue = td["volume_L"] * (body_weight_kg / 70.0)
        kp = kp_dict[tissue]
        vd += v_tissue * kp * fup

    vd_per_kg = vd / body_weight_kg

    if vd_per_kg < 0.6:
        classification = "low"
    elif vd_per_kg < 20.0:
        classification = "moderate"
    else:
        classification = "high"

    return VolumePredictionResult(
        drug_name=drug_name,
        vd_L=float(vd),
        vd_per_kg_L=float(vd_per_kg),
        kp_tissue=kp_dict,
        classification=classification,
        fup=fup,
        logP=logP,
        method="Rodgers-Rowland simplified",
    )


def screen_vd(compounds: list[dict]) -> list[VolumePredictionResult]:
    """Predict Vd for a list of compounds."""
    results = []
    for c in compounds:
        r = predict_vd_tissue(
            drug_name=c.get("drug_name", "unknown"),
            logP=c.get("logP", 2.0),
            fup=c.get("fup", 0.5),
            pka_acid=c.get("pka_acid"),
            pka_basic=c.get("pka_basic"),
            drug_type=c.get("drug_type", "neutral"),
        )
        results.append(r)
    return results


__all__ = [
    "VolumePredictionResult",
    "predict_vd_tissue",
    "screen_vd",
]
