"""Drug Photostability Assessment — Phase 382.

Assess photostability of drug substances and products under ICH Q1B light
exposure conditions.

References
----------
ICH Q1B (1996) Stability Testing: Photostability Testing of New Drug
Substances and Products.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "PhotostabilityResult",
    "assess_photostability",
    "compare_light_sources",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# ICH Q1B standard exposure for Option 2 (xenon)
_ICH_STANDARD_LUX_H = 1.2e6

# UV fraction multiplier for each light source (xenon = 1.0 reference)
_LIGHT_SOURCE_UV_FACTOR: dict[str, float] = {
    "xenon": 1.0,
    "fluorescent": 0.3,
    "D65": 0.8,
    "UV_only": 2.0,
}

_VALID_LIGHT_SOURCES = set(_LIGHT_SOURCE_UV_FACTOR.keys())

# Base first-order photodegradation rate (per lux-h) for a non-chromophoric drug
_BASE_RATE_PER_LUX_H = 1e-7


# ---------------------------------------------------------------------------
# Result dataclass (frozen — no list/dict fields)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PhotostabilityResult:
    """Result of ICH Q1B photostability assessment."""

    drug_name: str
    light_source: str
    total_lux_h: float
    total_uvvis_wh_per_m2: float
    degradation_pct: float
    photodegradation_rate_per_lux_h: float
    ich_q1b_compliant: bool
    risk_category: str
    chromophore_flag: bool
    recommended_packaging: str
    half_life_lux_h: float
    notes: str


# ---------------------------------------------------------------------------
# Core assessment function
# ---------------------------------------------------------------------------


def assess_photostability(
    drug_name: str,
    logp: float,
    mw_da: float,
    n_aromatic_rings: int,
    n_double_bonds: int,
    light_source: str = "xenon",
    exposure_lux_h: float = 1.2e6,
) -> PhotostabilityResult:
    """Assess drug photostability under ICH Q1B conditions.

    Parameters
    ----------
    drug_name:
        Name of the drug substance.
    logp:
        Octanol-water partition coefficient (dimensionless).
    mw_da:
        Molecular weight in Daltons. Must be >= 0.
    n_aromatic_rings:
        Number of aromatic rings in the structure. Must be >= 0.
    n_double_bonds:
        Number of non-aromatic double bonds. Must be >= 0.
    light_source:
        ICH Q1B light source: "xenon", "fluorescent", "D65", or "UV_only".
    exposure_lux_h:
        Total light exposure in lux-hours. Must be > 0.

    Returns
    -------
    PhotostabilityResult
    """
    # --- Validation ---
    if mw_da < 0:
        raise ValueError(f"mw_da must be >= 0, got {mw_da}")
    if n_aromatic_rings < 0:
        raise ValueError(f"n_aromatic_rings must be >= 0, got {n_aromatic_rings}")
    if n_double_bonds < 0:
        raise ValueError(f"n_double_bonds must be >= 0, got {n_double_bonds}")
    if light_source not in _VALID_LIGHT_SOURCES:
        raise ValueError(
            f"light_source must be one of {sorted(_VALID_LIGHT_SOURCES)}, got {light_source!r}"
        )
    if exposure_lux_h <= 0:
        raise ValueError(f"exposure_lux_h must be > 0, got {exposure_lux_h}")

    # --- Chromophore detection ---
    chromophore_flag = n_aromatic_rings >= 2 or n_double_bonds >= 4 or logp > 3.5

    # --- Photodegradation rate ---
    uv_factor = _LIGHT_SOURCE_UV_FACTOR[light_source]

    if chromophore_flag:
        rate_multiplier = (
            1.0 + 0.5 * n_aromatic_rings + 0.2 * n_double_bonds + 0.1 * max(0.0, logp - 2.0)
        )
    else:
        rate_multiplier = 0.3

    photodegradation_rate = _BASE_RATE_PER_LUX_H * rate_multiplier * uv_factor

    # --- Degradation at given exposure (first-order) ---
    degradation_pct = min(100.0, (1.0 - math.exp(-photodegradation_rate * exposure_lux_h)) * 100.0)

    # --- Half-life in lux-h ---
    half_life_lux_h = math.log(2.0) / photodegradation_rate

    # --- ICH Q1B compliance (< 5% at standard 1.2M lux-h) ---
    # Evaluate compliance at ICH standard exposure regardless of test exposure
    degradation_at_ich = min(
        100.0,
        (1.0 - math.exp(-photodegradation_rate * _ICH_STANDARD_LUX_H)) * 100.0,
    )
    ich_q1b_compliant = degradation_at_ich < 5.0

    # --- Risk category based on actual test degradation ---
    if degradation_pct < 1.0:
        risk_category = "stable"
    elif degradation_pct < 5.0:
        risk_category = "borderline"
    else:
        risk_category = "unstable"

    # --- UV/VIS energy (rough conversion) ---
    total_uvvis_wh_per_m2 = exposure_lux_h * 0.002 * uv_factor

    # --- Packaging recommendation ---
    if degradation_pct < 0.5:
        recommended_packaging = "clear"
    elif degradation_pct < 2.0:
        recommended_packaging = "amber"
    elif degradation_pct < 10.0:
        recommended_packaging = "opaque"
    else:
        recommended_packaging = "foil"

    # --- Notes ---
    notes = (
        f"Drug: {drug_name}; source={light_source}; "
        f"chromophore={chromophore_flag}; "
        f"rate={photodegradation_rate:.3e}/lux-h; "
        f"deg@ICH={degradation_at_ich:.2f}%; "
        f"risk={risk_category}; packaging={recommended_packaging}"
    )

    return PhotostabilityResult(
        drug_name=drug_name,
        light_source=light_source,
        total_lux_h=exposure_lux_h,
        total_uvvis_wh_per_m2=total_uvvis_wh_per_m2,
        degradation_pct=degradation_pct,
        photodegradation_rate_per_lux_h=photodegradation_rate,
        ich_q1b_compliant=ich_q1b_compliant,
        risk_category=risk_category,
        chromophore_flag=chromophore_flag,
        recommended_packaging=recommended_packaging,
        half_life_lux_h=half_life_lux_h,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Comparison across light sources
# ---------------------------------------------------------------------------


def compare_light_sources(
    drug_name: str,
    logp: float,
    mw_da: float,
    n_aromatic_rings: int,
    n_double_bonds: int,
    exposure_lux_h: float = 1.2e6,
) -> list:
    """Compare photostability across all four ICH Q1B light sources.

    Parameters
    ----------
    drug_name:
        Name of the drug substance.
    logp:
        Octanol-water partition coefficient.
    mw_da:
        Molecular weight in Daltons. Must be >= 0.
    n_aromatic_rings:
        Number of aromatic rings. Must be >= 0.
    n_double_bonds:
        Number of non-aromatic double bonds. Must be >= 0.
    exposure_lux_h:
        Total light exposure in lux-hours. Must be > 0.

    Returns
    -------
    list of PhotostabilityResult
        Sorted by degradation_pct ascending (most stable first).
    """
    results = []
    for source in _VALID_LIGHT_SOURCES:
        result = assess_photostability(
            drug_name=drug_name,
            logp=logp,
            mw_da=mw_da,
            n_aromatic_rings=n_aromatic_rings,
            n_double_bonds=n_double_bonds,
            light_source=source,
            exposure_lux_h=exposure_lux_h,
        )
        results.append(result)

    results.sort(key=lambda r: r.degradation_pct)
    return results
