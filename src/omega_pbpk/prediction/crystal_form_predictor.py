"""
Phase 804 — Crystal Form Prediction and Polymorph Selection

Predicts physicochemical properties for different crystal forms and polymorphs
of drug compounds, including solubility ratio, dissolution rate, thermodynamic
stability, and formulation recommendations.
"""

import math
from dataclasses import dataclass

# Gas constant in kJ/mol/K
_R_kJ = 8.314e-3


@dataclass(frozen=True)
class CrystalFormResult:
    compound_name: str
    polymorph_name: str
    melting_point_celsius: float
    enthalpy_of_fusion_kJ_mol: float
    solubility_ratio: float
    dissolution_rate_ratio: float
    thermodynamic_stability: str
    bioavailability_advantage: float
    shelf_life_stability: str
    formulation_recommendation: str
    notes: str


def predict_crystal_form_properties(
    compound_name: str,
    polymorph_name: str = "Form I",
    melting_point_celsius: float = 150.0,
    enthalpy_of_fusion_kJ_mol: float = 20.0,
    is_amorphous: bool = False,
    logp: float = 2.0,
    mw_da: float = 300.0,
    is_reference_form: bool = True,
) -> CrystalFormResult:
    """
    Predict crystal form properties for a drug polymorph.

    Parameters
    ----------
    compound_name : str
        Name of the compound.
    polymorph_name : str
        Name of the polymorph (e.g., "Form I", "Form II", "Amorphous").
    melting_point_celsius : float
        Melting point in degrees Celsius.
    enthalpy_of_fusion_kJ_mol : float
        Enthalpy of fusion (crystalline) in kJ/mol.
    is_amorphous : bool
        True if this form is amorphous (not crystalline).
    logp : float
        Octanol-water partition coefficient.
    mw_da : float
        Molecular weight in Daltons. Must be > 0.
    is_reference_form : bool
        True if this is the thermodynamically stable reference form (Form I).

    Returns
    -------
    CrystalFormResult
    """
    if mw_da <= 0:
        raise ValueError("mw_da must be positive")
    if enthalpy_of_fusion_kJ_mol < 0:
        raise ValueError("enthalpy_of_fusion_kJ_mol must be non-negative")

    # Reference melting point (Form I thermodynamically stable form)
    tm_ref_K = 150.0 + 273.15  # 423.15 K

    # Melting point in Kelvin for this form
    tm_K = melting_point_celsius + 273.15

    # --- Solubility ratio ---
    if is_amorphous:
        # Amorphous: 10-100x solubility advantage; default 30x
        solubility_ratio = 30.0
    elif is_reference_form:
        # Reference form: solubility ratio = 1.0 by definition
        solubility_ratio = 1.0
    else:
        # Crystalline polymorph: van't Hoff approach
        # Solubility_ratio = exp(ΔHf/R * (1/Tm_ref - 1/Tm))
        # Lower MP polymorphs have higher solubility
        if enthalpy_of_fusion_kJ_mol > 0 and tm_K > 0 and tm_ref_K > 0:
            # van't Hoff: ln(S/S_ref) = (ΔHf/R) * (1/Tm - 1/Tm_ref)
            # Lower MP → 1/Tm > 1/Tm_ref → positive exponent → higher solubility
            exponent = (enthalpy_of_fusion_kJ_mol / _R_kJ) * (1.0 / tm_K - 1.0 / tm_ref_K)
            solubility_ratio = math.exp(exponent)
        else:
            solubility_ratio = 1.0

    # Ensure solubility ratio is positive
    solubility_ratio = max(solubility_ratio, 1e-6)

    # --- Dissolution rate ratio (approx sqrt of solubility ratio) ---
    dissolution_rate_ratio = math.sqrt(solubility_ratio)

    # --- Thermodynamic stability ---
    if is_reference_form:
        thermodynamic_stability = "stable"
    elif is_amorphous:
        thermodynamic_stability = "metastable"
    else:
        # Compare MP: forms with lower MP than reference are less stable
        if melting_point_celsius >= 150.0:
            thermodynamic_stability = "stable"
        elif melting_point_celsius >= 100.0:
            thermodynamic_stability = "metastable"
        else:
            thermodynamic_stability = "unstable"

    # --- Bioavailability advantage ---
    # Absorption rate limited: BF = solubility_ratio^0.5
    # Permeability limited (logP > 3): BF = solubility_ratio^0.3
    if logp > 3.0:
        bioavailability_advantage = solubility_ratio**0.3
    else:
        bioavailability_advantage = solubility_ratio**0.5

    # Cap at reasonable maximum (amorphous typically gives 5-10x in vivo)
    bioavailability_advantage = min(bioavailability_advantage, 100.0)

    # --- Shelf life stability ---
    if is_amorphous and logp < 1.0:
        shelf_life_stability = "risk_of_conversion"
    elif melting_point_celsius < 100.0:
        shelf_life_stability = "unstable"
    else:
        shelf_life_stability = "stable"

    # --- Formulation recommendation ---
    if is_amorphous:
        if logp < 1.0:
            formulation_recommendation = (
                "Amorphous solid dispersion (ASD) with polymer stabilizer "
                "(HPMC, PVP) to prevent crystallization. Monitor stability."
            )
        else:
            formulation_recommendation = (
                "Amorphous solid dispersion (ASD) recommended for solubility "
                "enhancement. Use HPMC-AS or PVP-VA as carrier."
            )
    elif thermodynamic_stability == "stable":
        formulation_recommendation = (
            "Stable crystalline form. Standard formulation approaches suitable."
        )
    elif thermodynamic_stability == "metastable":
        formulation_recommendation = (
            "Metastable polymorph: use with caution. Monitor for form conversion "
            "during processing. Consider ASD or co-crystal alternatives."
        )
    else:
        formulation_recommendation = (
            "Unstable form: high risk of conversion during storage/processing. "
            "Not recommended for pharmaceutical development without stabilization."
        )

    # --- Notes ---
    notes_parts = []
    if solubility_ratio > 5:
        notes_parts.append(f"High solubility advantage ({solubility_ratio:.1f}x).")
    if bioavailability_advantage > 2.0:
        notes_parts.append(
            f"Significant bioavailability improvement predicted ({bioavailability_advantage:.2f}x)."
        )
    if is_amorphous:
        notes_parts.append("Amorphous form assumed; solubility ratio fixed at 30x.")
    if logp > 3.0:
        notes_parts.append("Permeability-limited absorption assumed (logP > 3).")
    notes = " ".join(notes_parts) if notes_parts else "Standard polymorph properties."

    return CrystalFormResult(
        compound_name=compound_name,
        polymorph_name=polymorph_name,
        melting_point_celsius=melting_point_celsius,
        enthalpy_of_fusion_kJ_mol=enthalpy_of_fusion_kJ_mol,
        solubility_ratio=solubility_ratio,
        dissolution_rate_ratio=dissolution_rate_ratio,
        thermodynamic_stability=thermodynamic_stability,
        bioavailability_advantage=bioavailability_advantage,
        shelf_life_stability=shelf_life_stability,
        formulation_recommendation=formulation_recommendation,
        notes=notes,
    )


def compare_polymorphs(
    compound_name: str,
    forms: list,
) -> list:
    """
    Compare multiple polymorphs of a compound.

    Parameters
    ----------
    compound_name : str
        Name of the compound.
    forms : list[dict]
        List of dicts, each with keys:
          - polymorph_name (str)
          - mp (float): melting point in Celsius
          - dhf (float): enthalpy of fusion kJ/mol
          - is_amorphous (bool, optional, default False)
          - is_reference (bool, optional, default False)
          - logp (float, optional, default 2.0)
          - mw_da (float, optional, default 300.0)

    Returns
    -------
    list[CrystalFormResult]
        Sorted by solubility_ratio descending (highest bioavailability first).
    """
    if not forms:
        return []

    results = []
    for form in forms:
        result = predict_crystal_form_properties(
            compound_name=compound_name,
            polymorph_name=form.get("polymorph_name", "Unknown"),
            melting_point_celsius=form.get("mp", 150.0),
            enthalpy_of_fusion_kJ_mol=form.get("dhf", 20.0),
            is_amorphous=form.get("is_amorphous", False),
            logp=form.get("logp", 2.0),
            mw_da=form.get("mw_da", 300.0),
            is_reference_form=form.get("is_reference", False),
        )
        results.append(result)

    # Sort by solubility_ratio descending
    results.sort(key=lambda r: r.solubility_ratio, reverse=True)
    return results
