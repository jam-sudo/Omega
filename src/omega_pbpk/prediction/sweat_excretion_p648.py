"""Drug excretion via sweat glands.

Predicts drug partitioning into sweat based on physicochemical properties
(logP, MW, pKa) and physiological parameters (sweat rate, sweat pH).
Relevant for forensic toxicology, wearable biosensors, and transdermal
drug monitoring.

Key mechanisms
--------------
- pH-partition (ion trapping): basic drugs concentrate in acidic sweat
- Lipophilicity: passive diffusion favours moderate-to-high logP
- Molecular weight: larger molecules penetrate less efficiently

References
----------
- Kidwell DA et al., Forensic Sci Int. 1998;84(1-3):75-84 (sweat drug testing)
- de la Torre R et al., Clin Pharmacol Ther. 2004;75(5):P51 (MDMA in sweat)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SweatExcretionResult:
    """Result container for sweat excretion prediction."""

    drug_name: str
    logP: float
    mw_Da: float
    plasma_concentration_mg_L: float
    sweat_concentration_mg_L: float
    sweat_to_plasma_ratio: float
    daily_excretion_mg: float
    excretion_rate_mg_h: float
    detection_window_h: float
    sweat_patch_accumulation_mg: float
    ionization_effect: str
    excretion_class: str
    notes: str


def predict_sweat_excretion(
    drug_name: str,
    logP: float,
    mw_Da: float,
    pka: float = 7.4,
    is_base: bool = True,
    plasma_concentration_mg_L: float = 1.0,
    sweat_rate_mL_h: float = 20.0,
    sweat_ph: float = 5.5,
    body_surface_area_m2: float = 1.8,
) -> SweatExcretionResult:
    """Predict drug excretion into sweat.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    logP : float
        Octanol-water partition coefficient (log scale).
    mw_Da : float
        Molecular weight in Daltons.
    pka : float
        Acid dissociation constant.
    is_base : bool
        True if the drug is a base, False if acid.
    plasma_concentration_mg_L : float
        Steady-state plasma concentration (mg/L).
    sweat_rate_mL_h : float
        Sweat production rate (mL/h).
    sweat_ph : float
        Sweat pH (typically ~5.5).
    body_surface_area_m2 : float
        Body surface area (m^2).

    Returns
    -------
    SweatExcretionResult
    """
    # --- validation ---
    if mw_Da <= 0:
        raise ValueError("mw_Da must be > 0")
    if plasma_concentration_mg_L <= 0:
        raise ValueError("plasma_concentration_mg_L must be > 0")
    if sweat_rate_mL_h <= 0:
        raise ValueError("sweat_rate_mL_h must be > 0")

    # --- pH partition (ion trapping for bases) ---
    if pka > 0 and is_base:
        ratio_ph = (1 + 10 ** (pka - sweat_ph)) / (1 + 10 ** (pka - 7.4))
        ratio_ph = min(100.0, ratio_ph)
    else:
        ratio_ph = 1.0

    # --- lipophilicity contribution ---
    ratio_logP = max(0.01, min(5.0, 0.1 * 10 ** (logP * 0.5)))

    # --- MW correction ---
    ratio_mw = max(0.01, 1.0 - mw_Da / 1000.0)

    # --- total sweat/plasma ratio ---
    sweat_to_plasma = ratio_ph * ratio_logP * ratio_mw
    sweat_to_plasma = max(0.001, min(50.0, sweat_to_plasma))

    # --- derived quantities ---
    sweat_conc = plasma_concentration_mg_L * sweat_to_plasma
    excretion_rate = sweat_conc * sweat_rate_mL_h / 1000.0  # mL/h -> L/h
    daily_excretion = excretion_rate * 24.0
    sweat_patch_accumulation = excretion_rate * 24.0  # 24h patch

    # --- detection window (surrogate) ---
    if logP > 3:
        detection_window = 48.0
    elif logP >= 1:
        detection_window = 24.0
    else:
        detection_window = 12.0

    # --- ionization effect classification ---
    if ratio_ph > 5:
        ionization_effect = "strong"
    elif ratio_ph > 2:
        ionization_effect = "moderate"
    else:
        ionization_effect = "weak"

    # --- excretion class ---
    if sweat_to_plasma > 1.0:
        excretion_class = "high"
    elif sweat_to_plasma > 0.1:
        excretion_class = "medium"
    else:
        excretion_class = "low"

    notes_parts = [
        f"ratio_ph={ratio_ph:.2f}",
        f"ratio_logP={ratio_logP:.3f}",
        f"ratio_mw={ratio_mw:.3f}",
    ]

    return SweatExcretionResult(
        drug_name=drug_name,
        logP=logP,
        mw_Da=mw_Da,
        plasma_concentration_mg_L=plasma_concentration_mg_L,
        sweat_concentration_mg_L=sweat_conc,
        sweat_to_plasma_ratio=sweat_to_plasma,
        daily_excretion_mg=daily_excretion,
        excretion_rate_mg_h=excretion_rate,
        detection_window_h=detection_window,
        sweat_patch_accumulation_mg=sweat_patch_accumulation,
        ionization_effect=ionization_effect,
        excretion_class=excretion_class,
        notes="; ".join(notes_parts),
    )


def screen_sweat_excretion(
    compounds: list[dict],
    plasma_concentration_mg_L: float = 1.0,
) -> list[SweatExcretionResult]:
    """Screen multiple compounds for sweat excretion potential.

    Parameters
    ----------
    compounds : list[dict]
        Each dict must have keys: drug_name, logP, mw_Da.
        Optional keys: pka, is_base.
    plasma_concentration_mg_L : float
        Plasma concentration to use for all compounds.

    Returns
    -------
    list[SweatExcretionResult]
        Sorted by sweat_to_plasma_ratio descending.
    """
    results: list[SweatExcretionResult] = []
    for comp in compounds:
        kwargs: dict = {
            "drug_name": comp["drug_name"],
            "logP": comp["logP"],
            "mw_Da": comp["mw_Da"],
            "plasma_concentration_mg_L": plasma_concentration_mg_L,
        }
        if "pka" in comp:
            kwargs["pka"] = comp["pka"]
        if "is_base" in comp:
            kwargs["is_base"] = comp["is_base"]
        results.append(predict_sweat_excretion(**kwargs))

    results.sort(key=lambda r: r.sweat_to_plasma_ratio, reverse=True)
    return results
