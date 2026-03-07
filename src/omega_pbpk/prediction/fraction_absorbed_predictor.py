"""Phase 706 — Fraction absorbed (fa) prediction from biopharmaceutical properties."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log10

__all__ = [
    "FaResult",
    "predict_fa_from_peff",
    "predict_fa_from_papp",
    "predict_fa_bcs",
    "screen_fa",
]


@dataclass(frozen=True)
class FaResult:
    """Result from BCS-based fraction absorbed prediction."""

    compound_name: str
    logP: float
    mw: float
    psa: float
    dose_mg: float
    fa_predicted: float
    fa_permeability: float
    fa_dissolution: float
    papp_estimate: float
    do_number: float
    bcs_class: str
    absorption_class: str
    notes: str


def predict_fa_from_peff(
    peff_cm_per_s: float,
    ka_per_h: float = None,
    transit_time_h: float = 3.0,
) -> float:
    """Predict fraction absorbed from intestinal effective permeability.

    Uses a simplified cylindrical intestinal model:
        fa = min(1.0, 2 * Peff * transit_time_s / radius_cm)
    where radius_cm = 1.75 cm, transit_time_s = transit_time_h * 3600.

    Args:
        peff_cm_per_s: Effective intestinal permeability (cm/s).
        ka_per_h: Absorption rate constant (1/h), unused (kept for API compatibility).
        transit_time_h: Small intestine transit time (h). Default 3.0 h.

    Returns:
        Predicted fraction absorbed in [0, 1].
    """
    radius_cm = 1.75
    transit_time_s = transit_time_h * 3600.0
    fa = 2.0 * peff_cm_per_s * transit_time_s / radius_cm
    return max(0.0, min(1.0, fa))


def predict_fa_from_papp(papp_1e6_cm_per_s: float) -> float:
    """Predict fraction absorbed from Caco-2 Papp value.

    Converts Caco-2 Papp to human intestinal Peff using Sun et al. 2004:
        log_Peff = 0.6 * log_Papp + 0.364
    where Papp is in x10^-6 cm/s and Peff result is in x10^-4 cm/s.

    Then predicts fa from Peff using predict_fa_from_peff().

    Args:
        papp_1e6_cm_per_s: Caco-2 Papp in units of x10^-6 cm/s.

    Returns:
        Predicted fraction absorbed in [0, 1].
    """
    if papp_1e6_cm_per_s <= 0:
        return 0.0

    log_papp = log10(papp_1e6_cm_per_s)
    log_peff = 0.6 * log_papp + 0.364  # Peff in x10^-4 cm/s
    peff_1e4 = 10.0**log_peff  # value in units of 1e-4 cm/s
    peff_cm_per_s = peff_1e4 * 1e-4  # convert to cm/s

    return predict_fa_from_peff(peff_cm_per_s)


def _estimate_papp(logP: float, mw: float, psa: float) -> float:
    """Estimate Caco-2 Papp (x10^-6 cm/s) from physicochemical descriptors.

    Uses QSAR equation:
        papp = 10^(0.5*logP - 1.0) * exp(-max(0, psa-60)*0.02) * exp(-max(0, mw-300)*0.003)

    Returns:
        Papp in x10^-6 cm/s, clamped to [0.01, 300].
    """
    base = 10.0 ** (0.5 * logP - 1.0)
    psa_penalty = exp(-max(0.0, psa - 60.0) * 0.02)
    mw_penalty = exp(-max(0.0, mw - 300.0) * 0.003)
    papp = base * psa_penalty * mw_penalty
    return max(0.01, min(300.0, papp))


def _classify_bcs(papp: float, do_number: float) -> str:
    """Classify BCS class based on permeability (Papp) and dissolution (Do) number.

    BCS classification:
        Class I: high permeability (Papp >= 1), high solubility (Do <= 1)
        Class II: high permeability (Papp >= 1), low solubility (Do > 1)
        Class III: low permeability (Papp < 1), high solubility (Do <= 1)
        Class IV: low permeability (Papp < 1), low solubility (Do > 1)

    Args:
        papp: Caco-2 Papp in x10^-6 cm/s.
        do_number: Dissolution number (dose / (solubility * volume)).

    Returns:
        BCS class string: "I", "II", "III", or "IV".
    """
    high_perm = papp >= 1.0  # threshold 1 x10^-6 cm/s
    high_sol = do_number <= 1.0

    if high_perm and high_sol:
        return "I"
    elif high_perm and not high_sol:
        return "II"
    elif not high_perm and high_sol:
        return "III"
    else:
        return "IV"


def _absorption_class(fa: float) -> str:
    """Classify absorption extent from fa value."""
    if fa > 0.9:
        return "complete"
    elif fa >= 0.7:
        return "high"
    elif fa >= 0.4:
        return "moderate"
    elif fa >= 0.2:
        return "low"
    else:
        return "poor"


def predict_fa_bcs(
    logP: float,
    mw: float,
    psa: float,
    solubility_mg_per_mL: float,
    dose_mg: float = 100.0,
    dose_volume_mL: float = 250.0,
    compound_name: str = "compound",
) -> FaResult:
    """Predict fraction absorbed using BCS-based framework.

    Combines permeability-limited fa (from Papp QSAR) and dissolution-limited fa
    (from Dissolution number), taking the product as combined fa.

    Args:
        logP: Octanol-water partition coefficient.
        mw: Molecular weight (Da).
        psa: Polar surface area (Angstrom^2).
        solubility_mg_per_mL: Aqueous solubility (mg/mL).
        dose_mg: Oral dose (mg). Default 100.0 mg.
        dose_volume_mL: Gastric volume (mL). Default 250.0 mL.
        compound_name: Name of the compound. Default "compound".

    Returns:
        FaResult dataclass with predicted fa and BCS classification.

    Raises:
        ValueError: If dose_mg <= 0 or solubility_mg_per_mL <= 0 or mw <= 0 or psa < 0.
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be positive, got {dose_mg}")
    if solubility_mg_per_mL <= 0:
        raise ValueError(f"solubility_mg_per_mL must be positive, got {solubility_mg_per_mL}")
    if mw <= 0:
        raise ValueError(f"mw must be positive, got {mw}")
    if psa < 0:
        raise ValueError(f"psa must be >= 0, got {psa}")

    # Step 1: Dissolution number (Do)
    # Do = dose / (solubility * dose_volume)
    do_number = dose_mg / (solubility_mg_per_mL * dose_volume_mL)

    # Step 2: Estimate Papp from physicochemical properties
    papp = _estimate_papp(logP, mw, psa)

    # Step 3: fa from permeability
    fa_permeability = predict_fa_from_papp(papp)

    # Step 4: fa from dissolution
    if do_number <= 1.0:
        fa_dissolution = 1.0
    else:
        fa_dissolution = min(1.0, 1.0 / do_number)

    # Step 5: Combined fa (both permeability and dissolution limiting)
    fa_predicted = fa_permeability * fa_dissolution
    fa_predicted = max(0.0, min(1.0, fa_predicted))

    # Step 6: BCS classification
    bcs_class = _classify_bcs(papp, do_number)

    # Step 7: Absorption classification
    absorption_class = _absorption_class(fa_predicted)

    notes = (
        f"Do={do_number:.3f}, Papp={papp:.2f}e-6 cm/s, "
        f"fa_perm={fa_permeability:.3f}, fa_diss={fa_dissolution:.3f}"
    )

    return FaResult(
        compound_name=compound_name,
        logP=logP,
        mw=mw,
        psa=psa,
        dose_mg=dose_mg,
        fa_predicted=fa_predicted,
        fa_permeability=fa_permeability,
        fa_dissolution=fa_dissolution,
        papp_estimate=papp,
        do_number=do_number,
        bcs_class=bcs_class,
        absorption_class=absorption_class,
        notes=notes,
    )


def screen_fa(compounds: list[dict]) -> list[FaResult]:
    """Screen a list of compounds for fraction absorbed.

    Each compound dict must contain:
        - logP (float)
        - mw (float)
        - psa (float)
        - solubility_mg_per_mL (float)
    And optionally:
        - dose_mg (float, default 100.0)
        - dose_volume_mL (float, default 250.0)
        - compound_name (str, default "compound")

    Args:
        compounds: List of compound property dicts.

    Returns:
        List of FaResult sorted by fa_predicted descending.
    """
    results = []
    for comp in compounds:
        result = predict_fa_bcs(
            logP=comp["logP"],
            mw=comp["mw"],
            psa=comp["psa"],
            solubility_mg_per_mL=comp["solubility_mg_per_mL"],
            dose_mg=comp.get("dose_mg", 100.0),
            dose_volume_mL=comp.get("dose_volume_mL", 250.0),
            compound_name=comp.get("compound_name", "compound"),
        )
        results.append(result)

    # Sort by fa_predicted descending
    results.sort(key=lambda r: r.fa_predicted, reverse=True)
    return results
