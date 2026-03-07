"""Phase 565 — Absorption enhancer effects on intestinal permeability."""

import math

# Enhancer parameters: (max_fold, EC50_mM, safety_limit_mM, safety_note)
_ENHANCER_PARAMS = {
    "EDTA": {
        "max_fold": 3.0,
        "ec50_mM": 2.0,
        "safety_limit_mM": 10.0,
        "safety_note": "Chelates calcium; disrupts tight junctions; reversible at low doses",
    },
    "sodium_caprate": {
        "max_fold": 5.0,
        "ec50_mM": 5.0,
        "safety_limit_mM": 15.0,
        "safety_note": "Medium-chain fatty acid; membrane fluidizer; mild irritant at high doses",
    },
    "chitosan": {
        "max_fold": 2.5,
        "ec50_mM": 1.0,
        "safety_limit_mM": 20.0,
        "safety_note": "Polycationic biopolymer; pH-dependent; generally biocompatible",
    },
    "SDS": {
        "max_fold": 4.0,
        "ec50_mM": 0.2,
        "safety_limit_mM": 1.0,
        "safety_note": "Anionic surfactant; cytotoxic above 1 mM; use with caution",
    },
    "zonula_occludens": {
        "max_fold": 2.0,
        "ec50_mM": 3.0,
        "safety_limit_mM": 20.0,
        "safety_note": "ZOT peptide; reversible tight junction modulator",
    },
}


def _validate_enhancer(enhancer: str) -> dict:
    """Return enhancer params or raise ValueError."""
    if enhancer not in _ENHANCER_PARAMS:
        raise ValueError(
            f"Unknown enhancer '{enhancer}'. Supported: {sorted(_ENHANCER_PARAMS.keys())}"
        )
    return _ENHANCER_PARAMS[enhancer]


def enhance_permeability(
    peff_baseline_cm_s: float,
    enhancer: str,
    concentration_mM: float,
) -> dict:
    """Calculate enhanced permeability for a given absorption enhancer.

    fold_enhancement = 1 + (max_fold - 1) * C / (EC50 + C)
    peff_enhanced = peff_baseline * fold_enhancement

    Parameters
    ----------
    peff_baseline_cm_s:
        Baseline effective permeability (cm/s). Must be >= 0.
    enhancer:
        Name of the enhancer (EDTA, sodium_caprate, chitosan, SDS, zonula_occludens).
    concentration_mM:
        Enhancer concentration (mM). Must be >= 0.

    Returns
    -------
    dict with keys: enhancer, concentration_mM, peff_baseline_cm_s,
                    peff_enhanced_cm_s, fold_enhancement, safety_note
    """
    if peff_baseline_cm_s < 0:
        raise ValueError("peff_baseline_cm_s must be >= 0")
    if concentration_mM < 0:
        raise ValueError("concentration_mM must be >= 0")

    params = _validate_enhancer(enhancer)
    max_fold = params["max_fold"]
    ec50 = params["ec50_mM"]

    fold_enhancement = 1.0 + (max_fold - 1.0) * concentration_mM / (ec50 + concentration_mM)
    peff_enhanced = peff_baseline_cm_s * fold_enhancement

    return {
        "enhancer": enhancer,
        "concentration_mM": concentration_mM,
        "peff_baseline_cm_s": peff_baseline_cm_s,
        "peff_enhanced_cm_s": peff_enhanced,
        "fold_enhancement": fold_enhancement,
        "safety_note": params["safety_note"],
    }


def fa_from_peff(
    peff_cm_s: float,
    transit_time_h: float = 3.5,
    intestinal_radius_cm: float = 1.75,
) -> float:
    """Calculate fraction absorbed from effective permeability.

    ka_abs (h^-1) = 2 * (peff_cm_s * 3600) / intestinal_radius_cm
    fa = 1 - exp(-ka_abs * transit_time_h)

    Parameters
    ----------
    peff_cm_s:
        Effective permeability (cm/s). Must be >= 0.
    transit_time_h:
        Small intestinal transit time (h).
    intestinal_radius_cm:
        Effective intestinal radius (cm).

    Returns
    -------
    float: fraction absorbed in [0, 1]
    """
    if peff_cm_s < 0:
        raise ValueError("peff_cm_s must be >= 0")

    peff_cm_h = peff_cm_s * 3600.0
    ka_abs = 2.0 * peff_cm_h / intestinal_radius_cm
    fa = 1.0 - math.exp(-ka_abs * transit_time_h)
    return max(0.0, min(1.0, fa))


def compare_enhancers(
    peff_baseline_cm_s: float,
    concentration_mM: float,
    transit_time_h: float = 3.5,
) -> list:
    """Evaluate all 5 enhancers at the given concentration.

    Parameters
    ----------
    peff_baseline_cm_s:
        Baseline effective permeability (cm/s).
    concentration_mM:
        Enhancer concentration (mM).
    transit_time_h:
        Small intestinal transit time (h).

    Returns
    -------
    list of dicts sorted by fa descending. Each dict contains:
    enhancer, fold_enhancement, peff_enhanced_cm_s, fa
    """
    results = []
    for enhancer_name in _ENHANCER_PARAMS:
        result = enhance_permeability(peff_baseline_cm_s, enhancer_name, concentration_mM)
        fa = fa_from_peff(result["peff_enhanced_cm_s"], transit_time_h)
        results.append(
            {
                "enhancer": enhancer_name,
                "fold_enhancement": result["fold_enhancement"],
                "peff_enhanced_cm_s": result["peff_enhanced_cm_s"],
                "fa": fa,
            }
        )

    results.sort(key=lambda x: x["fa"], reverse=True)
    return results


def enhancer_safety_margin(enhancer: str, concentration_mM: float) -> dict:
    """Assess safety margin for an enhancer at a given concentration.

    Parameters
    ----------
    enhancer:
        Name of the enhancer.
    concentration_mM:
        Enhancer concentration (mM).

    Returns
    -------
    dict with keys: enhancer, concentration_mM, limit_mM, safety_level
    ('safe', 'borderline', or 'toxic')
    """
    params = _validate_enhancer(enhancer)
    limit_mM = params["safety_limit_mM"]

    if concentration_mM > limit_mM:
        safety_level = "toxic"
    elif concentration_mM > 0.5 * limit_mM:
        safety_level = "borderline"
    else:
        safety_level = "safe"

    return {
        "enhancer": enhancer,
        "concentration_mM": concentration_mM,
        "limit_mM": limit_mM,
        "safety_level": safety_level,
    }
