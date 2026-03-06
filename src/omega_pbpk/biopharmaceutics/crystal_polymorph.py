"""Crystal polymorph impact on drug solubility and dissolution.

Different polymorphic forms have different crystal lattice energies,
leading to different thermodynamic solubilities and dissolution rates.
"""

from __future__ import annotations

__all__ = [
    "PolymorphResult",
    "CrystalForm",
    "assess_polymorph_impact",
    "compare_polymorphs",
]

from dataclasses import dataclass


@dataclass(frozen=True)
class CrystalForm:
    """Describes a crystal polymorphic form of a drug."""

    form_name: str  # e.g. "Form I", "Form II", "Amorphous"
    relative_solubility: float  # vs Form I (1.0 = reference)
    melting_point_C: float
    is_amorphous: bool


@dataclass(frozen=True)
class PolymorphResult:
    """Result of polymorph impact assessment."""

    drug_name: str
    form: CrystalForm
    solubility_mg_mL: float  # absolute solubility
    dissolution_rate_mg_per_min: float
    bioavailability_impact_pct: float  # vs Form I reference
    stability_risk: str  # "low"/"moderate"/"high" (amorphous = high)
    notes: str


def assess_polymorph_impact(
    drug_name: str,
    reference_solubility_mg_mL: float,  # Form I solubility
    crystal_form: CrystalForm,
    surface_area_cm2: float = 1.0,
    dose_mg: float = 100.0,
    bcs_class: str = "II",  # BCS class affects whether solubility matters
) -> PolymorphResult:
    """Assess the impact of a crystal polymorphic form on PK.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    reference_solubility_mg_mL:
        Solubility of Form I (reference form) in mg/mL.
    crystal_form:
        The crystal form to assess.
    surface_area_cm2:
        Particle surface area available for dissolution (cm^2).
    dose_mg:
        Dose in mg.
    bcs_class:
        BCS classification ("I", "II", "III", "IV").

    Returns
    -------
    PolymorphResult
    """
    if reference_solubility_mg_mL <= 0:
        raise ValueError(
            f"reference_solubility_mg_mL must be > 0, got {reference_solubility_mg_mL}"
        )
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    bcs_class = bcs_class.upper()
    if bcs_class not in ("I", "II", "III", "IV"):
        raise ValueError(
            f"bcs_class must be one of 'I','II','III','IV', got '{bcs_class}'"
        )

    # Absolute solubility
    solubility = reference_solubility_mg_mL * crystal_form.relative_solubility

    # Dissolution rate: Noyes-Whitney simplified (sink conditions)
    # dM/dt ≈ k_diss * Cs   where k_diss = D*A/h ≈ 0.01 * surface_area_cm2
    k_diss = 0.01 * surface_area_cm2  # cm/min approximation
    dissolution_rate = k_diss * solubility  # mg/min

    # Bioavailability impact relative to Form I
    if bcs_class in ("II", "IV"):
        # Solubility-limited: BA proportional to sqrt(relative_solubility)
        ba_impact = (crystal_form.relative_solubility**0.5 - 1.0) * 100.0
    else:
        # BCS I/III: absorption not solubility-limited
        ba_impact = 0.0

    # Stability risk
    if crystal_form.is_amorphous:
        stability_risk = "high"
    elif crystal_form.relative_solubility > 2.0:
        stability_risk = "moderate"
    else:
        stability_risk = "low"

    notes_parts = [
        f"Form: {crystal_form.form_name}",
        f"Relative solubility: {crystal_form.relative_solubility:.2f}x vs Form I",
        f"Solubility: {solubility:.3f} mg/mL",
        f"BCS class {bcs_class}: "
        + (
            "solubility-limited absorption"
            if bcs_class in ("II", "IV")
            else "not solubility-limited"
        ),
    ]
    notes = "; ".join(notes_parts)

    return PolymorphResult(
        drug_name=drug_name,
        form=crystal_form,
        solubility_mg_mL=solubility,
        dissolution_rate_mg_per_min=dissolution_rate,
        bioavailability_impact_pct=ba_impact,
        stability_risk=stability_risk,
        notes=notes,
    )


def compare_polymorphs(
    drug_name: str,
    reference_solubility_mg_mL: float,
    forms: list[CrystalForm],
    **kwargs,
) -> list[PolymorphResult]:
    """Compare multiple polymorphic forms.

    Returns list sorted by bioavailability_impact_pct descending
    (highest impact first).
    """
    results = [
        assess_polymorph_impact(
            drug_name=drug_name,
            reference_solubility_mg_mL=reference_solubility_mg_mL,
            crystal_form=form,
            **kwargs,
        )
        for form in forms
    ]
    return sorted(results, key=lambda r: r.bioavailability_impact_pct, reverse=True)
