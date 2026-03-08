"""Phase 241: Intrinsic dissolution rate prediction from drug properties."""

from dataclasses import dataclass

# Diffusion coefficient (cm2/min) = 7.4e-6 cm2/s * 60 s/min
_D_CM2_MIN = 7.4e-6 * 60.0
# Diffusion layer thickness (cm) = 25 um
_H_CM = 25e-4

_VALID_CRYSTAL_FORMS = {"amorphous", "crystalline", "hydrate", "polymorph_A", "polymorph_B"}
_CRYSTAL_FORM_MODIFIERS = {
    "amorphous": 1.0,
    "crystalline": 0.8,
    "hydrate": 0.9,
    "polymorph_A": 1.0,
    "polymorph_B": 0.85,
}


@dataclass(frozen=True)
class DissolutionRateResult:
    drug_name: str
    crystal_form: str
    solubility_mg_mL: float
    mw: float
    logp: float
    intrinsic_dissolution_rate_mg_cm2_min: float
    apparent_dissolution_rate_mg_min: float
    surface_area_cm2: float
    dissolution_class: str
    estimated_t90_min: float
    improvement_needed: bool
    notes: str


def _compute_dissolution_class(idr: float) -> str:
    if idr > 1.0:
        return "class1"
    elif idr >= 0.1:
        return "class2"
    else:
        return "class3"


def predict_dissolution_rate(
    drug_name: str,
    solubility_mg_mL: float,
    mw: float,
    logp: float,
    crystal_form: str = "amorphous",
    surface_area_cm2: float = 10.0,
) -> DissolutionRateResult:
    """Predict intrinsic dissolution rate from drug physicochemical properties.

    Args:
        drug_name: Name of the drug.
        solubility_mg_mL: Aqueous solubility in mg/mL.
        mw: Molecular weight in g/mol.
        logp: Octanol-water partition coefficient (log scale).
        crystal_form: One of amorphous, crystalline, hydrate, polymorph_A, polymorph_B.
        surface_area_cm2: Surface area of tablet/particle in cm2.

    Returns:
        DissolutionRateResult with IDR and related metrics.
    """
    if solubility_mg_mL <= 0:
        raise ValueError("solubility_mg_mL must be > 0")
    if mw <= 0:
        raise ValueError("mw must be > 0")
    if surface_area_cm2 <= 0:
        raise ValueError("surface_area_cm2 must be > 0")
    if crystal_form not in _VALID_CRYSTAL_FORMS:
        raise ValueError(
            f"crystal_form must be one of {_VALID_CRYSTAL_FORMS}, got '{crystal_form}'"
        )

    modifier = _CRYSTAL_FORM_MODIFIERS[crystal_form]
    effective_solubility = solubility_mg_mL * modifier

    # IDR = D * Cs / h  (mg/cm2/min)
    idr = _D_CM2_MIN * effective_solubility / _H_CM

    # Apparent dissolution rate (mg/min)
    apparent = idr * surface_area_cm2

    # Time to 90% dissolution for 100 mg dose (volume = 900 mL sink)
    dose_mg = 100.0
    if apparent > 0:
        t90 = 0.9 * dose_mg / apparent
        if t90 > 999.0:
            t90 = 999.0
    else:
        t90 = 999.0

    diss_class = _compute_dissolution_class(idr)
    improvement_needed = diss_class == "class3"

    notes_parts = [
        f"Crystal form modifier: {modifier:.2f}",
        f"Effective solubility: {effective_solubility:.4f} mg/mL",
        f"IDR: {idr:.4f} mg/cm2/min",
    ]
    if improvement_needed:
        notes_parts.append("Consider amorphous dispersion or particle size reduction.")

    notes = "; ".join(notes_parts)

    return DissolutionRateResult(
        drug_name=drug_name,
        crystal_form=crystal_form,
        solubility_mg_mL=solubility_mg_mL,
        mw=mw,
        logp=logp,
        intrinsic_dissolution_rate_mg_cm2_min=idr,
        apparent_dissolution_rate_mg_min=apparent,
        surface_area_cm2=surface_area_cm2,
        dissolution_class=diss_class,
        estimated_t90_min=t90,
        improvement_needed=improvement_needed,
        notes=notes,
    )


def compare_crystal_forms(
    drug_name: str,
    solubility_mg_mL: float,
    mw: float,
    logp: float,
) -> list:
    """Compare all 5 crystal forms, sorted by IDR descending.

    Returns:
        List of DissolutionRateResult for each crystal form.
    """
    results = [
        predict_dissolution_rate(drug_name, solubility_mg_mL, mw, logp, form)
        for form in _VALID_CRYSTAL_FORMS
    ]
    results.sort(key=lambda r: r.intrinsic_dissolution_rate_mg_cm2_min, reverse=True)
    return results
