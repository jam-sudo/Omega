"""Adverse effect predictor — predict likelihood of common adverse effects from PK exposure."""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class AdverseEffectResult:
    """Predicted adverse effects based on PK exposure metrics."""

    drug_name: str
    drug_class: str
    cmax_ratio: float  # cmax / therapeutic_cmax
    auc_ratio: float  # auc_24h / therapeutic_auc
    predicted_effects: list[dict]  # [{"effect": ..., "likelihood": ..., "rationale": ...}]
    overall_safety_concern: str  # "acceptable" / "monitor" / "high_risk"
    notes: str


# ---------------------------------------------------------------------------
# Supported drug classes
# ---------------------------------------------------------------------------

_VALID_DRUG_CLASSES = {
    "nsaid",
    "opioid",
    "antibiotic",
    "antihypertensive",
    "anticoagulant",
    "immunosuppressant",
}


# ---------------------------------------------------------------------------
# Likelihood helpers
# ---------------------------------------------------------------------------


def _likelihood(ratio: float, moderate_thresh: float, high_thresh: float) -> str:
    """Map an exposure ratio to a risk likelihood string."""
    if ratio >= high_thresh:
        return "high"
    if ratio >= moderate_thresh:
        return "moderate"
    return "low"


# ---------------------------------------------------------------------------
# Per-class adverse effect logic
# ---------------------------------------------------------------------------


def _predict_nsaid(cmax_ratio: float, auc_ratio: float) -> list[dict]:
    effects = []

    # GI irritation — driven by Cmax (mucosal exposure)
    gi_lkl = _likelihood(cmax_ratio, moderate_thresh=1.5, high_thresh=2.0)
    effects.append(
        {
            "effect": "GI_irritation",
            "likelihood": gi_lkl,
            "rationale": (
                f"Cmax ratio {cmax_ratio:.2f}: threshold >1.5x moderate, >2.0x high; "
                "mucosal exposure drives GI risk"
            ),
        }
    )

    # Renal toxicity — driven by AUC (sustained prostaglandin inhibition)
    renal_lkl = _likelihood(auc_ratio, moderate_thresh=1.5, high_thresh=2.5)
    effects.append(
        {
            "effect": "renal_toxicity",
            "likelihood": renal_lkl,
            "rationale": (
                f"AUC ratio {auc_ratio:.2f}: sustained exposure reduces renal prostaglandin synthesis"
            ),
        }
    )

    # Cardiovascular risk at very high exposures
    cv_lkl = _likelihood(auc_ratio, moderate_thresh=2.0, high_thresh=3.0)
    effects.append(
        {
            "effect": "cardiovascular_risk",
            "likelihood": cv_lkl,
            "rationale": (
                f"AUC ratio {auc_ratio:.2f}: elevated COX-2 inhibition increases thrombotic risk"
            ),
        }
    )

    return effects


def _predict_opioid(cmax_ratio: float, auc_ratio: float) -> list[dict]:
    effects = []

    # Respiratory depression — driven by Cmax
    resp_lkl = _likelihood(cmax_ratio, moderate_thresh=2.0, high_thresh=3.0)
    effects.append(
        {
            "effect": "respiratory_depression",
            "likelihood": resp_lkl,
            "rationale": (
                f"Cmax ratio {cmax_ratio:.2f}: threshold >2x moderate, >3x high; "
                "CNS mu-receptor saturation at peak"
            ),
        }
    )

    # Sedation — driven by AUC
    sed_lkl = _likelihood(auc_ratio, moderate_thresh=1.5, high_thresh=2.5)
    effects.append(
        {
            "effect": "sedation",
            "likelihood": sed_lkl,
            "rationale": (
                f"AUC ratio {auc_ratio:.2f}: total CNS exposure correlates with sedation duration"
            ),
        }
    )

    # Nausea/vomiting — moderate even at therapeutic
    nausea_lkl = _likelihood(cmax_ratio, moderate_thresh=1.0, high_thresh=2.0)
    effects.append(
        {
            "effect": "nausea_vomiting",
            "likelihood": nausea_lkl,
            "rationale": (
                f"Cmax ratio {cmax_ratio:.2f}: opioid-induced emesis related to peak exposure"
            ),
        }
    )

    return effects


def _predict_antibiotic(cmax_ratio: float, auc_ratio: float, drug_name: str) -> list[dict]:
    effects = []
    name_lower = drug_name.lower()

    # Nephrotoxicity — aminoglycoside pattern (Cmax-driven)
    if any(k in name_lower for k in ("gentamicin", "tobramycin", "amikacin", "aminoglycoside")):
        nephro_lkl = _likelihood(cmax_ratio, moderate_thresh=1.5, high_thresh=2.0)
    else:
        nephro_lkl = _likelihood(auc_ratio, moderate_thresh=2.0, high_thresh=3.5)
    effects.append(
        {
            "effect": "nephrotoxicity",
            "likelihood": nephro_lkl,
            "rationale": (
                f"Ratio (Cmax {cmax_ratio:.2f}, AUC {auc_ratio:.2f}): "
                "renal tubular accumulation risk"
            ),
        }
    )

    # Hepatotoxicity — fluoroquinolone pattern (AUC-driven)
    if any(
        k in name_lower
        for k in ("ciprofloxacin", "levofloxacin", "moxifloxacin", "fluoroquinolone")
    ):
        hep_lkl = _likelihood(auc_ratio, moderate_thresh=1.5, high_thresh=2.5)
    else:
        hep_lkl = _likelihood(auc_ratio, moderate_thresh=2.5, high_thresh=4.0)
    effects.append(
        {
            "effect": "hepatotoxicity",
            "likelihood": hep_lkl,
            "rationale": (
                f"AUC ratio {auc_ratio:.2f}: hepatic enzyme elevation at supra-therapeutic exposure"
            ),
        }
    )

    # C. difficile — AUC-driven (broad-spectrum disruption)
    cdiff_lkl = _likelihood(auc_ratio, moderate_thresh=1.0, high_thresh=2.0)
    effects.append(
        {
            "effect": "c_difficile_risk",
            "likelihood": cdiff_lkl,
            "rationale": (
                f"AUC ratio {auc_ratio:.2f}: gut microbiome disruption proportional to exposure"
            ),
        }
    )

    return effects


def _predict_antihypertensive(
    cmax_ratio: float, auc_ratio: float, trough_mg_L: float
) -> list[dict]:
    effects = []

    # Hypotension — trough-based (sustained effect at nadir)
    # Risk increases when trough itself is significant relative to target Cmax context
    # Use a trough-derived index: if trough is high relative to expected, risk is elevated
    # We use auc_ratio as proxy for overall sustained exposure burden
    hypo_lkl = _likelihood(auc_ratio, moderate_thresh=1.5, high_thresh=2.5)
    effects.append(
        {
            "effect": "hypotension",
            "likelihood": hypo_lkl,
            "rationale": (
                f"AUC ratio {auc_ratio:.2f}, trough {trough_mg_L:.3f} mg/L: "
                "sustained BP reduction risk from excessive trough exposure"
            ),
        }
    )

    # Reflex tachycardia — Cmax-driven (rapid onset vasodilation)
    tachy_lkl = _likelihood(cmax_ratio, moderate_thresh=1.5, high_thresh=2.5)
    effects.append(
        {
            "effect": "reflex_tachycardia",
            "likelihood": tachy_lkl,
            "rationale": (
                f"Cmax ratio {cmax_ratio:.2f}: rapid peak vasodilation triggers baroreceptor reflex"
            ),
        }
    )

    # Electrolyte disturbance (diuretic class)
    electro_lkl = _likelihood(auc_ratio, moderate_thresh=2.0, high_thresh=3.0)
    effects.append(
        {
            "effect": "electrolyte_disturbance",
            "likelihood": electro_lkl,
            "rationale": (
                f"AUC ratio {auc_ratio:.2f}: excessive diuresis may cause hypokalemia/hyponatremia"
            ),
        }
    )

    return effects


def _predict_anticoagulant(cmax_ratio: float, auc_ratio: float, trough_mg_L: float) -> list[dict]:
    effects = []

    # Bleeding risk — AUC is primary driver (> 1.5x → high)
    bleed_lkl = _likelihood(auc_ratio, moderate_thresh=1.2, high_thresh=1.5)
    effects.append(
        {
            "effect": "bleeding_risk",
            "likelihood": bleed_lkl,
            "rationale": (
                f"AUC ratio {auc_ratio:.2f}: threshold >1.2x moderate, >1.5x high; "
                "sustained anticoagulation increases hemorrhagic risk"
            ),
        }
    )

    # Major bleeding at very high exposure
    major_bleed_lkl = _likelihood(cmax_ratio, moderate_thresh=1.5, high_thresh=2.0)
    effects.append(
        {
            "effect": "major_bleeding",
            "likelihood": major_bleed_lkl,
            "rationale": (
                f"Cmax ratio {cmax_ratio:.2f}: peak anticoagulant activity "
                "at Cmax drives major bleed risk"
            ),
        }
    )

    # Hematoma formation — trough-dependent
    hematoma_lkl = _likelihood(auc_ratio, moderate_thresh=1.8, high_thresh=2.5)
    effects.append(
        {
            "effect": "hematoma_formation",
            "likelihood": hematoma_lkl,
            "rationale": (
                f"AUC ratio {auc_ratio:.2f}, trough {trough_mg_L:.3f} mg/L: "
                "residual anticoagulation at trough increases wound hematoma risk"
            ),
        }
    )

    return effects


def _predict_immunosuppressant(
    cmax_ratio: float, auc_ratio: float, trough_mg_L: float
) -> list[dict]:
    effects = []

    # Infection risk — AUC-driven (> 2x → high)
    infect_lkl = _likelihood(auc_ratio, moderate_thresh=1.5, high_thresh=2.0)
    effects.append(
        {
            "effect": "infection_risk",
            "likelihood": infect_lkl,
            "rationale": (
                f"AUC ratio {auc_ratio:.2f}: threshold >1.5x moderate, >2.0x high; "
                "over-immunosuppression increases opportunistic infection risk"
            ),
        }
    )

    # Nephrotoxicity — trough-driven (calcineurin inhibitor pattern)
    nephro_lkl = _likelihood(auc_ratio, moderate_thresh=1.3, high_thresh=1.8)
    effects.append(
        {
            "effect": "nephrotoxicity",
            "likelihood": nephro_lkl,
            "rationale": (
                f"AUC ratio {auc_ratio:.2f}, trough {trough_mg_L:.3f} mg/L: "
                "tubular toxicity correlated with sustained trough levels"
            ),
        }
    )

    # Malignancy risk at chronic high exposure
    malig_lkl = _likelihood(auc_ratio, moderate_thresh=2.0, high_thresh=3.0)
    effects.append(
        {
            "effect": "malignancy_risk",
            "likelihood": malig_lkl,
            "rationale": (
                f"AUC ratio {auc_ratio:.2f}: prolonged immunosuppression impairs cancer surveillance"
            ),
        }
    )

    return effects


# ---------------------------------------------------------------------------
# Overall safety classification
# ---------------------------------------------------------------------------


def _overall_safety(effects: list[dict]) -> str:
    """Derive overall safety concern from predicted effects."""
    counts = {"high": 0, "moderate": 0, "low": 0}
    for eff in effects:
        counts[eff["likelihood"]] += 1

    if counts["high"] >= 1:
        return "high_risk"
    if counts["moderate"] >= 2:
        return "high_risk"
    if counts["moderate"] >= 1:
        return "monitor"
    return "acceptable"


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------


def predict_adverse_effects(
    drug_name: str,
    drug_class: str,
    cmax_mg_L: float,
    auc_24h_mg_h_L: float,
    trough_mg_L: float,
    therapeutic_cmax: float,
    therapeutic_auc: float,
) -> AdverseEffectResult:
    """Predict likelihood of common adverse effects from PK exposure.

    Parameters
    ----------
    drug_name : str
        Name of the drug (used for class-specific heuristics, e.g. aminoglycosides).
    drug_class : str
        One of: "nsaid", "opioid", "antibiotic", "antihypertensive",
        "anticoagulant", "immunosuppressant".
    cmax_mg_L : float
        Observed/predicted peak plasma concentration (mg/L).
    auc_24h_mg_h_L : float
        Observed/predicted 24-h AUC (mg·h/L).
    trough_mg_L : float
        Observed/predicted trough concentration (mg/L).
    therapeutic_cmax : float
        Reference therapeutic Cmax (mg/L).
    therapeutic_auc : float
        Reference therapeutic 24-h AUC (mg·h/L).

    Returns
    -------
    AdverseEffectResult
    """
    if not drug_name or not drug_name.strip():
        raise ValueError("drug_name must be a non-empty string")
    if drug_class not in _VALID_DRUG_CLASSES:
        raise ValueError(
            f"drug_class '{drug_class}' not supported; choose from {sorted(_VALID_DRUG_CLASSES)}"
        )
    if cmax_mg_L < 0:
        raise ValueError("cmax_mg_L must be >= 0")
    if auc_24h_mg_h_L < 0:
        raise ValueError("auc_24h_mg_h_L must be >= 0")
    if trough_mg_L < 0:
        raise ValueError("trough_mg_L must be >= 0")
    if therapeutic_cmax <= 0:
        raise ValueError("therapeutic_cmax must be > 0")
    if therapeutic_auc <= 0:
        raise ValueError("therapeutic_auc must be > 0")

    cmax_ratio = cmax_mg_L / therapeutic_cmax
    auc_ratio = auc_24h_mg_h_L / therapeutic_auc

    if drug_class == "nsaid":
        effects = _predict_nsaid(cmax_ratio, auc_ratio)
        notes = "NSAIDs: GI and renal risk scale with supra-therapeutic exposure."
    elif drug_class == "opioid":
        effects = _predict_opioid(cmax_ratio, auc_ratio)
        notes = "Opioids: respiratory depression is the critical safety threshold."
    elif drug_class == "antibiotic":
        effects = _predict_antibiotic(cmax_ratio, auc_ratio, drug_name)
        notes = (
            "Antibiotics: nephrotoxicity and hepatotoxicity patterns differ by class. "
            "Aminoglycoside names trigger stricter renal threshold."
        )
    elif drug_class == "antihypertensive":
        effects = _predict_antihypertensive(cmax_ratio, auc_ratio, trough_mg_L)
        notes = "Antihypertensives: hypotension risk is trough/AUC-dependent."
    elif drug_class == "anticoagulant":
        effects = _predict_anticoagulant(cmax_ratio, auc_ratio, trough_mg_L)
        notes = "Anticoagulants: AUC > 1.5x therapeutic strongly predicts bleeding risk."
    else:  # immunosuppressant
        effects = _predict_immunosuppressant(cmax_ratio, auc_ratio, trough_mg_L)
        notes = (
            "Immunosuppressants: infection and nephrotoxicity are primary concerns; "
            "trough monitoring essential."
        )

    overall = _overall_safety(effects)

    return AdverseEffectResult(
        drug_name=drug_name,
        drug_class=drug_class,
        cmax_ratio=cmax_ratio,
        auc_ratio=auc_ratio,
        predicted_effects=effects,
        overall_safety_concern=overall,
        notes=notes,
    )


__all__ = [
    "AdverseEffectResult",
    "predict_adverse_effects",
]
