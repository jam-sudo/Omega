"""Drug crystallinity and polymorphism impact on dissolution and bioavailability.

Different polymorphic forms have different crystal lattice energies, leading to
different thermodynamic solubilities (described by the modified Yalkowsky equation)
and consequently different oral bioavailabilities for BCS II/IV compounds.

Key equation (Abraham-Le rule):
    log(S_A/S_B) = -(ΔHfus/R) * (1/Tm_A - 1/Tm_B) / 2.303
    ΔHfus ≈ 56.5 * Tm_B  (J/mol, Tm in Kelvin)
    R = 8.314 J/mol/K
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "PolymorphBioavailResult",
    "PolymorphRiskResult",
    "polymorph_solubility_ratio",
    "dissolution_rate_ratio",
    "bioavailability_impact",
    "assess_polymorph_risk",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_R_J_MOL_K = 8.314  # J/mol/K — universal gas constant
_DELTA_HF_FACTOR = 56.5  # J/mol/K — Abraham-Le proportionality constant
_KELVIN_OFFSET = 273.15  # °C → K


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolymorphBioavailResult:
    """Result of bioavailability impact assessment for two polymorphic forms."""

    solubility_ratio: float
    """S_A / S_B — ratio of Form A to Form B aqueous solubility."""

    bcs_class: int
    """BCS class (1, 2, 3, or 4)."""

    fa_form_a: float
    """Fraction absorbed for Form A (0–1)."""

    fa_form_b: float
    """Fraction absorbed for Form B (0–1)."""

    bioavailability_ratio: float
    """fa_form_a / fa_form_b — relative oral bioavailability."""

    bioavailability_change_pct: float
    """(bioavailability_ratio - 1) * 100 — % change vs Form B."""

    notes: str
    """Human-readable interpretation."""


@dataclass(frozen=True)
class PolymorphRiskResult:
    """Comprehensive polymorphism risk assessment for a drug."""

    drug_name: str
    form_a_tm_C: float
    """Form A melting point (°C)."""

    form_b_tm_C: float
    """Form B melting point (°C)."""

    solubility_ratio: float
    """Thermodynamic S_A / S_B."""

    bioavailability_ratio: float
    """Oral F_A / F_B (1.0 for BCS I/III)."""

    risk_category: str
    """'low' (<10% F change), 'moderate' (10–30%), or 'high' (>30%)."""

    notes: str


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def polymorph_solubility_ratio(
    form_a_melting_point_C: float,
    form_b_melting_point_C: float,
    temperature_C: float = 37.0,
) -> float:
    """Ratio S_A/S_B using the modified Yalkowsky equation.

    Uses ΔHfus ≈ 56.5 * Tm_B (Abraham-Le rule, Tm in Kelvin).

    Parameters
    ----------
    form_a_melting_point_C:
        Melting point of Form A in degrees Celsius.
    form_b_melting_point_C:
        Melting point of Form B in degrees Celsius (reference form).
    temperature_C:
        Temperature at which solubility is assessed (default 37°C, body temp).

    Returns
    -------
    float
        Solubility ratio S_A / S_B (dimensionless, > 0).
    """
    if form_a_melting_point_C <= -_KELVIN_OFFSET:
        raise ValueError(
            f"form_a_melting_point_C must be > {-_KELVIN_OFFSET}, "
            f"got {form_a_melting_point_C}"
        )
    if form_b_melting_point_C <= -_KELVIN_OFFSET:
        raise ValueError(
            f"form_b_melting_point_C must be > {-_KELVIN_OFFSET}, "
            f"got {form_b_melting_point_C}"
        )

    tm_a_k = form_a_melting_point_C + _KELVIN_OFFSET
    tm_b_k = form_b_melting_point_C + _KELVIN_OFFSET

    # ΔHfus estimated from Form B melting point (reference form)
    delta_hfus = _DELTA_HF_FACTOR * tm_b_k  # J/mol

    # log10(S_A/S_B) = -(ΔHfus/R) * (1/Tm_A - 1/Tm_B) / 2.303
    log10_ratio = -(delta_hfus / _R_J_MOL_K) * (1.0 / tm_a_k - 1.0 / tm_b_k) / 2.303
    return 10.0 ** log10_ratio


def dissolution_rate_ratio(
    form_a_solubility_mg_mL: float,
    form_b_solubility_mg_mL: float,
    particle_size_um: float = 10.0,
) -> float:
    """Ratio of Noyes-Whitney dissolution rates, Form A / Form B.

    Under sink conditions, rate ∝ solubility → ratio = S_A / S_B.
    Particle size affects absolute rates but not the ratio between forms.

    Parameters
    ----------
    form_a_solubility_mg_mL:
        Aqueous solubility of Form A (mg/mL).
    form_b_solubility_mg_mL:
        Aqueous solubility of Form B (mg/mL).
    particle_size_um:
        Mean particle diameter (µm). Included for API completeness; does not
        affect the ratio under sink conditions.

    Returns
    -------
    float
        Dissolution rate ratio (Form A / Form B).
    """
    if form_a_solubility_mg_mL <= 0:
        raise ValueError(
            f"form_a_solubility_mg_mL must be > 0, got {form_a_solubility_mg_mL}"
        )
    if form_b_solubility_mg_mL <= 0:
        raise ValueError(
            f"form_b_solubility_mg_mL must be > 0, got {form_b_solubility_mg_mL}"
        )
    if particle_size_um <= 0:
        raise ValueError(f"particle_size_um must be > 0, got {particle_size_um}")

    return form_a_solubility_mg_mL / form_b_solubility_mg_mL


def bioavailability_impact(
    solubility_ratio: float,
    bcs_class: int,
    absorption_window_h: float = 4.0,
    ka_base_per_h: float = 1.0,
) -> PolymorphBioavailResult:
    """Estimate bioavailability change between Form A and Form B.

    For BCS I/III: dissolution is not rate-limiting → no change.
    For BCS II/IV: ka ∝ solubility → fa = 1 - exp(-ka * t_window).

    Parameters
    ----------
    solubility_ratio:
        S_A / S_B (dimensionless).
    bcs_class:
        BCS class (1, 2, 3, or 4).
    absorption_window_h:
        Effective absorption window in the GI tract (hours).
    ka_base_per_h:
        First-order absorption rate constant for Form B (/h).

    Returns
    -------
    PolymorphBioavailResult
    """
    if solubility_ratio <= 0:
        raise ValueError(f"solubility_ratio must be > 0, got {solubility_ratio}")
    if bcs_class not in (1, 2, 3, 4):
        raise ValueError(f"bcs_class must be 1, 2, 3, or 4, got {bcs_class}")
    if absorption_window_h <= 0:
        raise ValueError(
            f"absorption_window_h must be > 0, got {absorption_window_h}"
        )
    if ka_base_per_h <= 0:
        raise ValueError(f"ka_base_per_h must be > 0, got {ka_base_per_h}")

    if bcs_class in (1, 3):
        # Not dissolution-limited; polymorphism has no bioavailability impact
        fa_a = 1.0 - math.exp(-ka_base_per_h * absorption_window_h)
        fa_b = fa_a
        f_ratio = 1.0
        notes = (
            f"BCS class {bcs_class}: absorption is not dissolution-limited. "
            "Polymorphism has no bioavailability impact."
        )
    else:
        # BCS II/IV: ka scales with solubility ratio
        ka_a = ka_base_per_h * solubility_ratio
        ka_b = ka_base_per_h  # Form B is the reference
        fa_a = 1.0 - math.exp(-ka_a * absorption_window_h)
        fa_b = 1.0 - math.exp(-ka_b * absorption_window_h)
        f_ratio = fa_a / fa_b if fa_b > 0 else float("inf")
        notes = (
            f"BCS class {bcs_class}: dissolution-limited absorption. "
            f"ka_A={ka_a:.3f}/h, ka_B={ka_b:.3f}/h; "
            f"fa_A={fa_a:.3f}, fa_B={fa_b:.3f}."
        )

    f_change_pct = (f_ratio - 1.0) * 100.0

    return PolymorphBioavailResult(
        solubility_ratio=solubility_ratio,
        bcs_class=bcs_class,
        fa_form_a=fa_a,
        fa_form_b=fa_b,
        bioavailability_ratio=f_ratio,
        bioavailability_change_pct=f_change_pct,
        notes=notes,
    )


def assess_polymorph_risk(
    drug_name: str,
    form_a_tm_C: float,
    form_b_tm_C: float,
    bcs_class: int,
    daily_dose_mg: float,
) -> PolymorphRiskResult:
    """Comprehensive polymorphism risk assessment.

    Combines thermodynamic solubility ratio, dissolution ratio, and
    bioavailability modelling to classify risk.

    Parameters
    ----------
    drug_name:
        Name of the drug substance.
    form_a_tm_C:
        Melting point of Form A (°C).
    form_b_tm_C:
        Melting point of Form B reference (°C).
    bcs_class:
        BCS class (1, 2, 3, or 4).
    daily_dose_mg:
        Daily dose in mg (used for contextual notes).

    Returns
    -------
    PolymorphRiskResult
    """
    if form_a_tm_C <= -_KELVIN_OFFSET:
        raise ValueError(
            f"form_a_tm_C must be > {-_KELVIN_OFFSET} °C, got {form_a_tm_C}"
        )
    if form_b_tm_C <= -_KELVIN_OFFSET:
        raise ValueError(
            f"form_b_tm_C must be > {-_KELVIN_OFFSET} °C, got {form_b_tm_C}"
        )
    if bcs_class not in (1, 2, 3, 4):
        raise ValueError(f"bcs_class must be 1, 2, 3, or 4, got {bcs_class}")
    if daily_dose_mg <= 0:
        raise ValueError(f"daily_dose_mg must be > 0, got {daily_dose_mg}")

    sol_ratio = polymorph_solubility_ratio(form_a_tm_C, form_b_tm_C)
    ba_result = bioavailability_impact(sol_ratio, bcs_class)

    f_change_abs = abs(ba_result.bioavailability_change_pct)
    if f_change_abs < 10.0:
        risk = "low"
    elif f_change_abs < 30.0:
        risk = "moderate"
    else:
        risk = "high"

    notes = (
        f"Drug: {drug_name}, dose={daily_dose_mg:.1f} mg/day, BCS {bcs_class}. "
        f"S_A/S_B={sol_ratio:.3f}; F change={ba_result.bioavailability_change_pct:+.1f}%. "
        f"Risk: {risk}."
    )

    return PolymorphRiskResult(
        drug_name=drug_name,
        form_a_tm_C=form_a_tm_C,
        form_b_tm_C=form_b_tm_C,
        solubility_ratio=sol_ratio,
        bioavailability_ratio=ba_result.bioavailability_ratio,
        risk_category=risk,
        notes=notes,
    )
