"""Volume of distribution estimation using multiple methods.

Methods implemented:
- Oie-Tozer equation (with lysosomal trapping for basic drugs)
- Poulin-Theil simplified tissue partition method
- Kp-based estimation from tissue partition coefficients
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "VdEstimationResult",
    "estimate_vd_oie",
    "estimate_vd_poulin_theil",
    "estimate_vd_from_kp",
    "predict_vd",
    "screen_vd",
]

# Standard tissue volumes (L/70 kg human)
_TISSUE_VOLUMES_L: dict[str, float] = {
    "adipose": 16.0,
    "muscle": 28.0,
    "liver": 1.5,
    "kidney": 0.3,
    "brain": 1.4,
}

_VD_CLASS_LOW = 30.0  # L
_VD_CLASS_HIGH = 100.0  # L


@dataclass(frozen=True)
class VdEstimationResult:
    """Result of volume of distribution estimation."""

    compound_name: str
    logP: float
    mw: float
    fup: float
    vd_oie_L: float
    vd_poulin_theil_L: float
    vd_ensemble_L: float
    vd_class: str
    notes: str


def estimate_vd_oie(
    logP: float,
    mw: float,
    fup: float,
    pka_base: float | None = None,
    pka_acid: float | None = None,
) -> float:
    """Oie-Tozer equation for volume of distribution (L, 70 kg human).

    Formulation:
        Vd = (Vp / fup + Vt / fut) × 70 kg

    where:
        - Vp = 0.042 L/kg (plasma volume)
        - Vt = 0.2 L/kg (accessible tissue volume)
        - fut = 0.5 / (1 + 0.1 × max(0, logP))  [tissue unbound fraction]

    Low fup → high Vd (extensive plasma protein binding → large distribution).
    High logP → low fut → high Vd (lipophilic drugs sequester in tissues).
    Basic drugs (pKa > 7.4) → lysosomal trapping → lower fut → higher Vd.

    Parameters
    ----------
    logP:
        Octanol-water partition coefficient.
    mw:
        Molecular weight (g/mol).
    fup:
        Fraction unbound in plasma (0 < fup <= 1).
    pka_base:
        pKa for basic drugs (optional). If >7.4, lysosomal trapping reduces fut.
    pka_acid:
        pKa for acidic drugs (optional, currently unused in calculation).

    Returns
    -------
    float
        Vd in liters (70 kg human).
    """
    if fup <= 0 or fup > 1:
        raise ValueError(f"fup must be in (0, 1], got {fup}")
    if mw <= 0:
        raise ValueError(f"mw must be > 0, got {mw}")

    # Tissue volumes per kg
    vp_per_kg = 0.042  # L/kg plasma
    vt_per_kg = 0.2  # L/kg accessible tissue volume

    # Fraction unbound in tissue: lipophilic drugs bind more to tissue lipids
    fut = 0.5 / (1.0 + 0.1 * max(0.0, logP))

    # Lysosomal trapping for basic drugs (pKa > 7.4 → drug accumulates in lysosomes)
    if pka_base is not None and pka_base > 7.4:
        ph_plasma = 7.4
        trapping_factor = 1.0 + math.pow(10.0, pka_base - ph_plasma)
        fut = fut / trapping_factor
        fut = max(0.001, fut)

    # Oie-Tozer: Vd = Vp/fup + Vt/fut  (low fup → high plasma term → high Vd)
    vd_per_kg = vp_per_kg / fup + vt_per_kg / fut

    return vd_per_kg * 70.0


def estimate_vd_poulin_theil(
    logP: float,
    fup: float,
    b_p_ratio: float = 0.65,
) -> float:
    """Poulin-Theil simplified tissue partition method (L, 70 kg human).

    Simplified approximation based on lipophilicity and plasma protein binding:
        Vd_L = 5.0 + max(0, logP) × 20.0 + fup × (-3.0)

    High logP → high Vd (lipophilic drugs partition into fat/tissues).
    Higher fup slightly reduces Vd (less plasma retention effect).

    Parameters
    ----------
    logP:
        Octanol-water partition coefficient.
    fup:
        Fraction unbound in plasma (0 < fup <= 1).
    b_p_ratio:
        Blood-to-plasma concentration ratio (unused in simplified form).

    Returns
    -------
    float
        Vd in liters (70 kg human), clamped to [5, 1000] L.
    """
    if fup <= 0 or fup > 1:
        raise ValueError(f"fup must be in (0, 1], got {fup}")

    # Simplified Poulin-Theil approximation
    vd_L = 5.0 + max(0.0, logP) * 20.0 + fup * (-3.0)
    return max(5.0, min(1000.0, vd_L))


def estimate_vd_from_kp(
    kp_dict: dict,
    fup: float,
    vt_L_per_kg: float = 0.38,
) -> float:
    """Estimate Vd from tissue partition coefficients.

    Parameters
    ----------
    kp_dict:
        Tissue-specific Kp values, e.g. {"adipose": 5.0, "muscle": 2.0, ...}.
    fup:
        Fraction unbound in plasma (0 < fup <= 1).
    vt_L_per_kg:
        Default tissue volume per kg for tissues not in standard table (L/kg).

    Returns
    -------
    float
        Vd in liters (70 kg human).
    """
    if fup <= 0 or fup > 1:
        raise ValueError(f"fup must be in (0, 1], got {fup}")

    vp_L = 0.042 * 70.0  # plasma volume

    # Fallback volume for unknown tissues
    n_tissues = len(kp_dict) if kp_dict else 1
    fallback_vt = vt_L_per_kg * 70.0 / n_tissues

    weighted_sum = 0.0
    for tissue, kp in kp_dict.items():
        tissue_vol = _TISSUE_VOLUMES_L.get(tissue.lower(), fallback_vt)
        weighted_sum += kp * tissue_vol

    vd_L = vp_L + fup * weighted_sum
    return vd_L


def _classify_vd(vd_L: float) -> str:
    """Classify Vd as low, moderate, or high."""
    if vd_L < _VD_CLASS_LOW:
        return "low"
    elif vd_L <= _VD_CLASS_HIGH:
        return "moderate"
    else:
        return "high"


def predict_vd(
    compound_name: str,
    logP: float,
    mw: float = 400.0,
    fup: float = 0.1,
    pka_base: float | None = None,
    pka_acid: float | None = None,
) -> VdEstimationResult:
    """Run both Oie-Tozer and Poulin-Theil methods and return ensemble.

    Parameters
    ----------
    compound_name:
        Name/identifier of the compound.
    logP:
        Octanol-water partition coefficient.
    mw:
        Molecular weight (g/mol).
    fup:
        Fraction unbound in plasma.
    pka_base:
        pKa for basic drugs (optional).
    pka_acid:
        pKa for acidic drugs (optional).

    Returns
    -------
    VdEstimationResult
    """
    if fup <= 0 or fup > 1:
        raise ValueError(f"fup must be in (0, 1], got {fup}")
    if mw <= 0:
        raise ValueError(f"mw must be > 0, got {mw}")

    vd_oie = estimate_vd_oie(logP, mw, fup, pka_base=pka_base, pka_acid=pka_acid)
    vd_pt = estimate_vd_poulin_theil(logP, fup)

    vd_ensemble = (vd_oie + vd_pt) / 2.0
    vd_class = _classify_vd(vd_ensemble)

    method_parts = [f"Oie-Tozer={vd_oie:.1f}L", f"Poulin-Theil={vd_pt:.1f}L"]
    if pka_base is not None:
        method_parts.append(f"pKa_base={pka_base} (lysosomal trapping applied)")
    notes = f"Ensemble Vd={vd_ensemble:.1f}L ({vd_class}); " + ", ".join(method_parts)

    return VdEstimationResult(
        compound_name=compound_name,
        logP=logP,
        mw=mw,
        fup=fup,
        vd_oie_L=vd_oie,
        vd_poulin_theil_L=vd_pt,
        vd_ensemble_L=vd_ensemble,
        vd_class=vd_class,
        notes=notes,
    )


def screen_vd(compounds: list[dict]) -> list[VdEstimationResult]:
    """Screen a list of compounds for Vd, sorted descending by ensemble Vd.

    Parameters
    ----------
    compounds:
        List of dicts with keys: compound_name, logP, mw (optional), fup,
        pka_base (optional), pka_acid (optional).

    Returns
    -------
    list[VdEstimationResult]
        Sorted by vd_ensemble_L descending.
    """
    if not compounds:
        return []

    results: list[VdEstimationResult] = []
    for cpd in compounds:
        result = predict_vd(
            compound_name=cpd.get("compound_name", "unknown"),
            logP=float(cpd["logP"]),
            mw=float(cpd.get("mw", 400.0)),
            fup=float(cpd["fup"]),
            pka_base=cpd.get("pka_base"),
            pka_acid=cpd.get("pka_acid"),
        )
        results.append(result)

    results.sort(key=lambda r: r.vd_ensemble_L, reverse=True)
    return results
