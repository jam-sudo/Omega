"""Phase 1032 — Drug Photodegradation Kinetics.

Predicts drug degradation under light exposure for formulation/packaging design.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log

_VALID_CHROMOPHORES = {"none", "aromatic", "carbonyl", "alkene", "nitro", "phenol"}
_VALID_LIGHT_SOURCES = {"fluorescent", "UV", "sunlight", "incandescent"}

_CHROMOPHORE_K_BASE: dict[str, float] = {
    "none": 0.0001,
    "aromatic": 0.005,
    "carbonyl": 0.02,
    "alkene": 0.01,
    "nitro": 0.05,
    "phenol": 0.03,
}

_LIGHT_FACTOR: dict[str, float] = {
    "incandescent": 0.5,
    "fluorescent": 1.0,
    "sunlight": 5.0,
    "UV": 20.0,
}

_LN2 = log(2.0)
_LN10 = log(10.0)


@dataclass(frozen=True)
class PhotodegradationResult:
    """Result of photodegradation kinetics prediction."""

    drug_name: str
    chromophore: str
    light_source: str
    initial_conc_mg_mL: float
    exposure_h: float
    k_photo_per_h: float
    fraction_remaining: float
    t50_h: float
    t90_percent_loss_h: float
    photostability_class: str
    quantum_yield_estimate: float
    protection_needed: bool
    notes: str


def predict_photodegradation(
    drug_name: str,
    mw_Da: float,
    chromophore: str = "none",
    light_source: str = "fluorescent",
    initial_conc_mg_mL: float = 1.0,
    exposure_h: float = 24.0,
    oxygen_present: bool = True,
    ph: float = 7.0,
    antioxidant_conc_mM: float = 0.0,
) -> PhotodegradationResult:
    """Predict drug photodegradation under specified light conditions.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    mw_Da:
        Molecular weight (Da).
    chromophore:
        Type of light-absorbing group.
    light_source:
        Type of light exposure.
    initial_conc_mg_mL:
        Initial drug concentration (mg/mL).
    exposure_h:
        Duration of light exposure (h).
    oxygen_present:
        Whether oxygen is present (promotes photo-oxidation).
    ph:
        Solution pH.
    antioxidant_conc_mM:
        Antioxidant concentration (mM); reduces photodegradation.

    Returns
    -------
    PhotodegradationResult
    """
    # Validation
    if initial_conc_mg_mL <= 0:
        raise ValueError(f"initial_conc_mg_mL must be > 0, got {initial_conc_mg_mL}")
    if exposure_h <= 0:
        raise ValueError(f"exposure_h must be > 0, got {exposure_h}")
    if chromophore not in _VALID_CHROMOPHORES:
        raise ValueError(f"chromophore must be one of {_VALID_CHROMOPHORES}, got {chromophore!r}")
    if light_source not in _VALID_LIGHT_SOURCES:
        raise ValueError(
            f"light_source must be one of {_VALID_LIGHT_SOURCES}, got {light_source!r}"
        )
    if antioxidant_conc_mM < 0:
        raise ValueError(f"antioxidant_conc_mM must be >= 0, got {antioxidant_conc_mM}")

    # Build rate constant
    k_base = _CHROMOPHORE_K_BASE[chromophore]
    light_factor = _LIGHT_FACTOR[light_source]

    o2_factor = 1.5 if oxygen_present else 1.0
    ph_factor = 1.0 + 0.02 * abs(ph - 7.0)
    antioxidant_factor = exp(-antioxidant_conc_mM * 0.1)

    k_photo = k_base * light_factor * o2_factor * ph_factor * antioxidant_factor
    k_photo = max(1e-6, min(100.0, k_photo))

    # First-order kinetics
    fraction_remaining = max(0.0, min(1.0, exp(-k_photo * exposure_h)))

    t50_h = _LN2 / k_photo
    t90_loss_h = _LN10 / k_photo  # time to 90% degradation (10% remaining)

    # Photostability class
    if fraction_remaining > 0.95:
        photostability_class = "stable"
    elif fraction_remaining > 0.80:
        photostability_class = "slightly_unstable"
    elif fraction_remaining > 0.50:
        photostability_class = "unstable"
    else:
        photostability_class = "very_unstable"

    # Quantum yield estimate (dimensionless, scaled by light intensity)
    quantum_yield_estimate = min(1.0, k_photo / (light_factor * 10.0))

    protection_needed = photostability_class != "stable"

    # Build notes
    note_parts = [
        f"{drug_name}: photodegradation under {light_source} light ({chromophore} chromophore).",
        f"k_photo = {k_photo:.5f}/h; t½ = {t50_h:.2f} h; {fraction_remaining * 100:.1f}% remains after {exposure_h} h.",
        f"Classification: {photostability_class}.",
    ]
    if protection_needed:
        note_parts.append("Light protection recommended (amber glass or opaque packaging).")
    if antioxidant_conc_mM > 0:
        note_parts.append(
            f"Antioxidant at {antioxidant_conc_mM} mM reduces degradation rate by "
            f"{(1 - antioxidant_factor) * 100:.1f}%."
        )
    notes = " ".join(note_parts)

    return PhotodegradationResult(
        drug_name=drug_name,
        chromophore=chromophore,
        light_source=light_source,
        initial_conc_mg_mL=initial_conc_mg_mL,
        exposure_h=exposure_h,
        k_photo_per_h=k_photo,
        fraction_remaining=fraction_remaining,
        t50_h=t50_h,
        t90_percent_loss_h=t90_loss_h,
        photostability_class=photostability_class,
        quantum_yield_estimate=quantum_yield_estimate,
        protection_needed=protection_needed,
        notes=notes,
    )


def screen_packaging_conditions(
    drug_name: str,
    mw_Da: float,
    chromophore: str = "aromatic",
    exposure_h: float = 24.0,
) -> list:
    """Screen all light sources for packaging condition selection.

    Returns a list of PhotodegradationResult for all 4 light sources,
    sorted by fraction_remaining descending (most stable first).

    Parameters
    ----------
    drug_name:
        Name of the drug.
    mw_Da:
        Molecular weight (Da).
    chromophore:
        Chromophore type.
    exposure_h:
        Exposure duration (h).

    Returns
    -------
    list[PhotodegradationResult]
    """
    results = [
        predict_photodegradation(
            drug_name=drug_name,
            mw_Da=mw_Da,
            chromophore=chromophore,
            light_source=src,
            exposure_h=exposure_h,
        )
        for src in _VALID_LIGHT_SOURCES
    ]
    results.sort(key=lambda r: r.fraction_remaining, reverse=True)
    return results
