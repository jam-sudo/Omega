"""Phase 670 — Drug Formulation pH Stability.

Predict chemical stability of drugs as a function of pH and temperature
in formulation. Models acid/base hydrolysis and oxidation degradation pathways.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FormulationpHStabilityResult:
    """Result of formulation pH stability prediction."""

    drug_name: str
    optimal_ph: float
    ph_tested: float
    temperature_C: float
    t_half_degradation_h: float
    k_degradation_per_h: float
    degradation_pathway: str
    shelf_life_years: float
    potency_at_2y_pct: float
    potency_at_5y_pct: float
    stability_class: str
    formulation_recommendation: str
    notes: str


def predict_ph_stability(
    drug_name: str,
    logP: float,
    mw_Da: float,
    pka: float | None = None,
    is_base: bool = True,
    ph_tested: float = 7.0,
    temperature_C: float = 25.0,
    susceptibility_hydrolysis: float = 0.5,
    susceptibility_oxidation: float = 0.3,
) -> FormulationpHStabilityResult:
    """Predict pH-dependent stability of a drug in formulation.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    logP : float
        Partition coefficient.
    mw_Da : float
        Molecular weight in Daltons.
    pka : float or None
        Acid dissociation constant.
    is_base : bool
        Whether the drug is a base.
    ph_tested : float
        pH of formulation to test.
    temperature_C : float
        Storage temperature in Celsius.
    susceptibility_hydrolysis : float
        Hydrolysis susceptibility (0-1).
    susceptibility_oxidation : float
        Oxidation susceptibility (0-1).

    Returns
    -------
    FormulationpHStabilityResult
    """
    # Validation
    if not 0 <= susceptibility_hydrolysis <= 1:
        raise ValueError("susceptibility_hydrolysis must be in [0, 1]")
    if not 0 <= susceptibility_oxidation <= 1:
        raise ValueError("susceptibility_oxidation must be in [0, 1]")
    if temperature_C <= 0:
        raise ValueError("temperature_C must be > 0")

    # Optimal pH
    if is_base:
        if pka is not None:
            optimal_ph = 5.0 + max(0, pka - 8.0) * 0.5
        else:
            optimal_ph = 5.5
    else:
        optimal_ph = 7.5
    optimal_ph = max(3.0, min(9.0, optimal_ph))

    # pH-dependent hydrolysis rate
    if is_base:
        k_hyd = susceptibility_hydrolysis * 10 ** (ph_tested - 7.0) * 0.001
    else:
        k_hyd = susceptibility_hydrolysis * 10 ** (7.0 - ph_tested) * 0.001
    k_hyd = max(1e-6, min(0.1, k_hyd))

    # Oxidation rate
    k_ox = susceptibility_oxidation * 0.0001 * temperature_C / 25.0
    k_ox = max(1e-7, min(0.05, k_ox))

    # Total degradation
    k_deg = k_hyd + k_ox

    # Arrhenius temperature correction
    q10 = 2.0
    temp_correction = q10 ** ((temperature_C - 25.0) / 10.0)
    k_deg *= temp_correction

    # Half-life
    t_half_degradation_h = 0.693 / k_deg

    # Shelf life: time to 5% loss
    shelf_life_h = -math.log(0.95) / k_deg
    shelf_life_years = shelf_life_h / (24 * 365)

    # Potency
    potency_at_2y_pct = math.exp(-k_deg * 2 * 365 * 24) * 100
    potency_at_5y_pct = math.exp(-k_deg * 5 * 365 * 24) * 100

    # Degradation pathway
    if k_hyd > 1e-5 and k_ox > 1e-5 and k_ox / k_hyd > 0.5 and k_hyd / k_ox > 0.5:
        degradation_pathway = "combined"
    elif k_hyd > k_ox:
        degradation_pathway = "hydrolysis"
    else:
        degradation_pathway = "oxidation"

    # Stability class
    if shelf_life_years >= 3:
        stability_class = "excellent"
    elif shelf_life_years >= 2:
        stability_class = "good"
    elif shelf_life_years >= 1:
        stability_class = "fair"
    else:
        stability_class = "unstable"

    # Formulation recommendation
    if stability_class == "unstable":
        formulation_recommendation = "lyophilize or reformulate"
    elif degradation_pathway in ("hydrolysis", "combined"):
        formulation_recommendation = f"optimize pH to {optimal_ph:.1f}"
    elif degradation_pathway == "oxidation":
        formulation_recommendation = "add antioxidant, nitrogen purge"
    else:
        formulation_recommendation = "standard liquid formulation"

    return FormulationpHStabilityResult(
        drug_name=drug_name,
        optimal_ph=optimal_ph,
        ph_tested=ph_tested,
        temperature_C=temperature_C,
        t_half_degradation_h=t_half_degradation_h,
        k_degradation_per_h=k_deg,
        degradation_pathway=degradation_pathway,
        shelf_life_years=shelf_life_years,
        potency_at_2y_pct=potency_at_2y_pct,
        potency_at_5y_pct=potency_at_5y_pct,
        stability_class=stability_class,
        formulation_recommendation=formulation_recommendation,
        notes=f"pH stability prediction for {drug_name} at pH {ph_tested}, {temperature_C}C",
    )


def screen_ph_stability(
    compounds: list,
    ph_range: list | None = None,
) -> dict:
    """Screen multiple compounds across pH values to find optimal pH.

    Parameters
    ----------
    compounds : list of dict
        Each dict must have 'drug_name', 'logP', 'mw_Da' and optionally
        'pka', 'is_base', 'susceptibility_hydrolysis', 'susceptibility_oxidation'.
    ph_range : list of float or None
        pH values to test. Default [3, 4, 5, 6, 7, 8, 9].

    Returns
    -------
    dict
        Mapping drug_name -> optimal pH (pH with longest shelf life).
    """
    if ph_range is None:
        ph_range = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]

    result = {}
    for compound in compounds:
        best_ph = ph_range[0]
        best_shelf_life = -1.0
        for ph in ph_range:
            r = predict_ph_stability(
                drug_name=compound["drug_name"],
                logP=compound["logP"],
                mw_Da=compound["mw_Da"],
                pka=compound.get("pka"),
                is_base=compound.get("is_base", True),
                ph_tested=ph,
                susceptibility_hydrolysis=compound.get("susceptibility_hydrolysis", 0.5),
                susceptibility_oxidation=compound.get("susceptibility_oxidation", 0.3),
            )
            if r.shelf_life_years > best_shelf_life:
                best_shelf_life = r.shelf_life_years
                best_ph = ph
        result[compound["drug_name"]] = best_ph
    return result
