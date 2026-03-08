"""Phase 863 — Solid Dispersion AUC Predictor.

Predict AUC enhancement from amorphous solid dispersion (ASD) formulations
vs crystalline drug.
"""

from dataclasses import dataclass

# Polymer base supersaturation ratios
_POLYMER_SR: dict = {
    "HPMC": 3.0,
    "PVP": 2.5,
    "HPMCAS": 4.0,
    "Eudragit": 2.0,
}

_VALID_POLYMERS = set(_POLYMER_SR.keys())


@dataclass(frozen=True)
class SolidDispersionAUCResult:
    """Result of solid dispersion AUC prediction."""

    drug_name: str
    polymer_type: str
    drug_loading_pct: float
    dose_mg: float
    cs_crystalline_mg_mL: float
    supersaturation_ratio: float
    auc_ratio: float
    auc_enhancement_pct: float
    spring_duration_h: float
    recrystallization_risk: str
    dose_limited: bool
    notes: str


def predict_solid_dispersion_auc(
    drug_name: str,
    dose_mg: float,
    cs_crystalline_mg_mL: float,
    logp: float,
    polymer_type: str,
    drug_loading_pct: float = 20.0,
    v_gi_mL: float = 250.0,
) -> SolidDispersionAUCResult:
    """Predict AUC enhancement from amorphous solid dispersion formulation.

    Parameters
    ----------
    drug_name:
        Name of the drug compound.
    dose_mg:
        Dose in mg.
    cs_crystalline_mg_mL:
        Crystalline solubility in mg/mL.
    logp:
        Lipophilicity (log P).
    polymer_type:
        Polymer carrier. One of: HPMC, PVP, HPMCAS, Eudragit.
    drug_loading_pct:
        Drug loading percentage (0-100).
    v_gi_mL:
        GI fluid volume in mL (default 250 mL).

    Returns
    -------
    SolidDispersionAUCResult
    """
    # Validation
    if cs_crystalline_mg_mL <= 0:
        raise ValueError("cs_crystalline_mg_mL must be > 0")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if not (0 < drug_loading_pct < 100):
        raise ValueError("drug_loading_pct must be between 0 and 100 (exclusive)")
    if polymer_type not in _VALID_POLYMERS:
        raise ValueError(
            f"polymer_type must be one of {sorted(_VALID_POLYMERS)}, got '{polymer_type}'"
        )

    # Base supersaturation ratio from polymer
    base_sr = _POLYMER_SR[polymer_type]

    # Loading penalty: higher loading → less polymer → less inhibition
    sr = base_sr * (1.0 - drug_loading_pct / 200.0)
    sr = max(sr, 1.0)

    # Check whether dose is already fully dissolved without ASD
    dissolved_dose_without_asd = cs_crystalline_mg_mL * v_gi_mL
    dose_limited = dose_mg < dissolved_dose_without_asd

    if dose_limited:
        # Drug already fully dissolved → SR has no benefit
        auc_ratio = 1.0
        auc_enhancement_pct = 0.0
    else:
        # AUC_ratio = SR (linear PK: AUC ∝ absorbed dose)
        auc_ratio = sr
        auc_enhancement_pct = (auc_ratio - 1.0) * 100.0

    # Spring duration: estimated duration of supersaturation
    spring_duration_h = 2.0 * sr

    # Recrystallization risk based on SR
    if sr < 2.0:
        recrystallization_risk = "low"
    elif sr < 4.0:
        recrystallization_risk = "moderate"
    else:
        recrystallization_risk = "high"

    notes_parts = []
    if dose_limited:
        notes_parts.append(
            "Dose is below crystalline solubility limit; ASD provides no AUC benefit."
        )
    else:
        notes_parts.append(
            f"ASD with {polymer_type} predicts {auc_enhancement_pct:.1f}% AUC enhancement."
        )
    notes_parts.append(f"logP={logp:.2f} affects membrane permeability.")
    notes = " ".join(notes_parts)

    return SolidDispersionAUCResult(
        drug_name=drug_name,
        polymer_type=polymer_type,
        drug_loading_pct=drug_loading_pct,
        dose_mg=dose_mg,
        cs_crystalline_mg_mL=cs_crystalline_mg_mL,
        supersaturation_ratio=sr,
        auc_ratio=auc_ratio,
        auc_enhancement_pct=auc_enhancement_pct,
        spring_duration_h=spring_duration_h,
        recrystallization_risk=recrystallization_risk,
        dose_limited=dose_limited,
        notes=notes,
    )


def screen_polymers(
    drug_name: str,
    dose_mg: float,
    cs_crystalline_mg_mL: float,
    logp: float,
    drug_loading_pct: float = 20.0,
) -> list[SolidDispersionAUCResult]:
    """Run all 4 polymer types and return results sorted by auc_ratio descending.

    Parameters
    ----------
    drug_name:
        Name of the drug compound.
    dose_mg:
        Dose in mg.
    cs_crystalline_mg_mL:
        Crystalline solubility in mg/mL.
    logp:
        Lipophilicity (log P).
    drug_loading_pct:
        Drug loading percentage (0-100).

    Returns
    -------
    List of SolidDispersionAUCResult sorted by auc_ratio descending.
    """
    results = []
    for polymer in _VALID_POLYMERS:
        result = predict_solid_dispersion_auc(
            drug_name=drug_name,
            dose_mg=dose_mg,
            cs_crystalline_mg_mL=cs_crystalline_mg_mL,
            logp=logp,
            polymer_type=polymer,
            drug_loading_pct=drug_loading_pct,
        )
        results.append(result)

    results.sort(key=lambda r: r.auc_ratio, reverse=True)
    return results
