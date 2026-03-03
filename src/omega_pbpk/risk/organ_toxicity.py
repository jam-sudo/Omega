"""Organ-specific toxicity risk scoring.

Estimates organ-level drug exposure using tissue partition coefficients (Kp)
applied to plasma concentration profiles, then scores toxicity risk against
established thresholds for hepatotoxicity, nephrotoxicity, cardiac risk,
and pulmonary toxicity.

Approach:
  C_organ(t) = Kp_organ × C_plasma(t)
  Cmax_unbound_uM = Cmax_mg_L × fup × 1000 / MW
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from omega_pbpk._compat import np_trapz
from omega_pbpk.core.body import SimulationResult
from omega_pbpk.drugs.drug import Drug

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
DILI_DOSE_THRESHOLD_MG = 100.0
DILI_FUP_THRESHOLD = 0.1
DILI_LIVER_CMAX_UNBOUND_THRESHOLD_UM = 50.0
NEPHRO_KIDNEY_CMAX_THRESHOLD_UM = 100.0
HERG_MARGIN_THRESHOLD = 30.0
HERG_MARGIN_CAUTION = 10.0
LUNG_ACCUMULATION_RATIO_THRESHOLD = 10.0
PHOSPHOLIPIDOSIS_LOGP_THRESHOLD = 2.0

# Organs we score toxicity for, mapped to Kp lookup keys.
_SCORED_ORGANS = ("liver", "kidney", "heart", "lung")

# Default Kp values when drug.default_kp is unavailable
_DEFAULT_KP: dict[str, float] = {
    "liver": 4.0,
    "kidney": 3.5,
    "heart": 1.5,
    "lung": 1.2,
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OrganExposure:
    """Pharmacokinetic exposure metrics for a single organ."""

    organ: str
    cmax_mg_L: float
    auc_mg_h_L: float
    tmax_h: float
    cmax_unbound_uM: float
    cave_mg_L: float


@dataclass(frozen=True)
class OrganToxicityScore:
    """Toxicity risk score for one organ–endpoint pair."""

    organ: str
    endpoint: str
    exposure_metric: float
    threshold: float
    margin: float
    risk_level: str  # "low", "moderate", "high"
    reference: str
    warning: str


@dataclass(frozen=True)
class OrganToxicityReport:
    """Aggregated organ toxicity assessment."""

    drug_name: str
    organ_exposures: list[OrganExposure]
    toxicity_scores: list[OrganToxicityScore]
    overall_risk: str  # "low", "moderate", "high"
    risk_count: int
    max_concern_organ: str
    summary: dict[str, object] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Exposure extraction
# ---------------------------------------------------------------------------


def _mg_L_to_unbound_uM(conc_mg_L: float, fup: float, mw: float) -> float:
    """Convert mg/L total concentration to unbound µM."""
    return conc_mg_L * fup * 1000.0 / max(mw, 1e-12)


def extract_organ_exposures(
    result: SimulationResult,
    fup: float,
    mw: float,
) -> list[OrganExposure]:
    """Extract organ exposures from a SimulationResult using Kp-based estimation.

    C_organ(t) = Kp_organ × C_plasma(t)

    For organs with explicit state variables in the ODE system, we compute
    concentration directly from amounts / volume. For simplicity and
    robustness, we use the Kp × plasma approach for all scored organs.
    """
    time = result.time_h
    cp = result.plasma_concentration()  # mg/L

    exposures: list[OrganExposure] = []
    for organ_name in _SCORED_ORGANS:
        kp = _get_organ_kp(result, organ_name)
        c_organ = kp * cp  # mg/L in organ

        cmax = float(np.max(c_organ))
        tmax = float(time[int(np.argmax(c_organ))])
        auc = float(np_trapz(c_organ, time))
        duration = float(time[-1] - time[0]) if len(time) > 1 else 1.0
        cave = auc / max(duration, 1e-12)
        cmax_unbound_uM = _mg_L_to_unbound_uM(cmax, fup, mw)

        exposures.append(
            OrganExposure(
                organ=organ_name,
                cmax_mg_L=round(cmax, 6),
                auc_mg_h_L=round(auc, 4),
                tmax_h=round(tmax, 4),
                cmax_unbound_uM=round(cmax_unbound_uM, 6),
                cave_mg_L=round(cave, 6),
            )
        )
    return exposures


def _get_organ_kp(result: SimulationResult, organ_name: str) -> float:
    """Retrieve Kp for an organ from the SimulationResult's organ dict or drug."""
    # Try organs dict (populated by WholeBodyPBPK)
    if result.organs and organ_name in result.organs:
        kp = result.organs[organ_name].kp
        if kp > 0:
            return kp
    # Fall back to drug's default_kp
    try:
        kp = result.drug.default_kp(organ_name)
        if kp > 0:
            return kp
    except Exception:
        pass
    return _DEFAULT_KP.get(organ_name, 1.0)


# ---------------------------------------------------------------------------
# Individual organ scoring functions
# ---------------------------------------------------------------------------


def score_hepatotoxicity(
    liver_exposure: OrganExposure,
    dose_mg: float,
    fup: float,
    logP: float,
) -> OrganToxicityScore:
    """Score hepatotoxicity (DILI) risk.

    Rules:
      1. Daily dose > 100 mg AND fup < 0.1 → "high" (Lammert 2008)
      2. Liver Cmax_unbound > 50 µM → "high"
      3. Dose > 100 mg OR logP > 3 → "moderate"
      4. Otherwise → "low"
    """
    cmax_ub = liver_exposure.cmax_unbound_uM
    margin = DILI_LIVER_CMAX_UNBOUND_THRESHOLD_UM / max(cmax_ub, 1e-12)
    warning = ""

    if dose_mg > DILI_DOSE_THRESHOLD_MG and fup < DILI_FUP_THRESHOLD:
        risk = "high"
        warning = (
            f"High DILI risk: dose {dose_mg:.0f} mg > {DILI_DOSE_THRESHOLD_MG} mg "
            f"and fup {fup:.3f} < {DILI_FUP_THRESHOLD}"
        )
    elif cmax_ub > DILI_LIVER_CMAX_UNBOUND_THRESHOLD_UM:
        risk = "high"
        warning = (
            f"Liver Cmax_unbound {cmax_ub:.1f} µM exceeds "
            f"{DILI_LIVER_CMAX_UNBOUND_THRESHOLD_UM} µM threshold"
        )
    elif dose_mg > DILI_DOSE_THRESHOLD_MG or logP > 3.0:
        risk = "moderate"
        warning = "Moderate DILI concern: high dose or lipophilicity"
    else:
        risk = "low"

    return OrganToxicityScore(
        organ="liver",
        endpoint="DILI",
        exposure_metric=round(cmax_ub, 4),
        threshold=DILI_LIVER_CMAX_UNBOUND_THRESHOLD_UM,
        margin=round(margin, 2),
        risk_level=risk,
        reference="Lammert et al. 2008; FDA DILI guidance",
        warning=warning,
    )


def score_nephrotoxicity(kidney_exposure: OrganExposure) -> OrganToxicityScore:
    """Score nephrotoxicity risk based on kidney Cmax_unbound.

    Kidney Cmax_unbound > 100 µM → "high"
    > 50 µM → "moderate"
    Otherwise → "low"
    """
    cmax_ub = kidney_exposure.cmax_unbound_uM
    margin = NEPHRO_KIDNEY_CMAX_THRESHOLD_UM / max(cmax_ub, 1e-12)
    warning = ""

    if cmax_ub > NEPHRO_KIDNEY_CMAX_THRESHOLD_UM:
        risk = "high"
        warning = (
            f"Kidney Cmax_unbound {cmax_ub:.1f} µM exceeds "
            f"{NEPHRO_KIDNEY_CMAX_THRESHOLD_UM} µM threshold"
        )
    elif cmax_ub > NEPHRO_KIDNEY_CMAX_THRESHOLD_UM / 2:
        risk = "moderate"
        warning = f"Kidney Cmax_unbound {cmax_ub:.1f} µM approaching threshold"
    else:
        risk = "low"

    return OrganToxicityScore(
        organ="kidney",
        endpoint="nephrotoxicity",
        exposure_metric=round(cmax_ub, 4),
        threshold=NEPHRO_KIDNEY_CMAX_THRESHOLD_UM,
        margin=round(margin, 2),
        risk_level=risk,
        reference="FDA renal safety guidance",
        warning=warning,
    )


def score_cardiac_risk(
    heart_exposure: OrganExposure,
    herg_ic50_uM: float | None,
    cmax_plasma_unbound_uM: float,
) -> OrganToxicityScore:
    """Score cardiac (hERG) risk.

    hERG margin = IC50 / Cmax_unbound_plasma
    Margin < 10 → "high", < 30 → "moderate", else → "low"
    """
    warning = ""
    if herg_ic50_uM is not None and herg_ic50_uM > 0:
        margin = herg_ic50_uM / max(cmax_plasma_unbound_uM, 1e-12)
        if margin < HERG_MARGIN_CAUTION:
            risk = "high"
            warning = (
                f"hERG margin {margin:.1f}× < {HERG_MARGIN_CAUTION}× "
                f"(IC50={herg_ic50_uM:.1f} µM, Cmax_ub={cmax_plasma_unbound_uM:.2f} µM)"
            )
        elif margin < HERG_MARGIN_THRESHOLD:
            risk = "moderate"
            warning = f"hERG margin {margin:.1f}× below {HERG_MARGIN_THRESHOLD}× safety threshold"
        else:
            risk = "low"
    else:
        # No hERG data — cannot assess
        margin = float("inf")
        risk = "low"
        warning = "No hERG IC50 data provided; cardiac risk not fully assessed"

    return OrganToxicityScore(
        organ="heart",
        endpoint="hERG_cardiac",
        exposure_metric=round(cmax_plasma_unbound_uM, 4),
        threshold=HERG_MARGIN_THRESHOLD,
        margin=round(margin, 2) if margin < 1e6 else float("inf"),
        risk_level=risk,
        reference="ICH S7B; hERG safety margin ≥30×",
        warning=warning,
    )


def score_pulmonary_risk(
    lung_exposure: OrganExposure,
    plasma_cmax: float,
    logP: float,
) -> OrganToxicityScore:
    """Score pulmonary toxicity risk.

    Lung accumulation ratio = lung_Cmax / plasma_Cmax
    High ratio + high logP → phospholipidosis concern
    """
    ratio = lung_exposure.cmax_mg_L / max(plasma_cmax, 1e-12)
    warning = ""

    if ratio > LUNG_ACCUMULATION_RATIO_THRESHOLD and logP > PHOSPHOLIPIDOSIS_LOGP_THRESHOLD:
        risk = "high"
        warning = (
            f"Lung accumulation ratio {ratio:.1f}× with logP {logP:.1f} "
            f"suggests phospholipidosis risk"
        )
    elif ratio > LUNG_ACCUMULATION_RATIO_THRESHOLD or logP > 3.0:
        risk = "moderate"
        warning = f"Lung accumulation ratio {ratio:.1f}×; monitor for pulmonary effects"
    else:
        risk = "low"

    return OrganToxicityScore(
        organ="lung",
        endpoint="pulmonary_accumulation",
        exposure_metric=round(ratio, 4),
        threshold=LUNG_ACCUMULATION_RATIO_THRESHOLD,
        margin=round(LUNG_ACCUMULATION_RATIO_THRESHOLD / max(ratio, 1e-12), 2),
        risk_level=risk,
        reference="Phospholipidosis guidance; logP-based accumulation",
        warning=warning,
    )


# ---------------------------------------------------------------------------
# Master assessment function
# ---------------------------------------------------------------------------


def run_organ_toxicity_assessment(
    result: SimulationResult,
    drug: Drug,
    dose_mg: float,
    herg_ic50_uM: float | None = None,
) -> OrganToxicityReport:
    """Run complete organ-specific toxicity assessment.

    Args:
        result: SimulationResult from PBPK simulation.
        drug: Drug object with physicochemical properties.
        dose_mg: Administered dose (mg).
        herg_ic50_uM: Optional hERG IC50 value (µM).

    Returns:
        OrganToxicityReport with per-organ exposures, scores, and summary.
    """
    fup = drug.fup
    mw = drug.mw
    logP = drug.logP

    # 1. Extract organ exposures
    exposures = extract_organ_exposures(result, fup, mw)
    exposure_map = {e.organ: e for e in exposures}

    # Plasma Cmax unbound (µM) for cardiac risk
    cp = result.plasma_concentration()
    plasma_cmax = float(np.max(cp))
    cmax_plasma_unbound_uM = _mg_L_to_unbound_uM(plasma_cmax, fup, mw)

    # 2. Score each organ
    scores: list[OrganToxicityScore] = []

    if "liver" in exposure_map:
        scores.append(score_hepatotoxicity(exposure_map["liver"], dose_mg, fup, logP))

    if "kidney" in exposure_map:
        scores.append(score_nephrotoxicity(exposure_map["kidney"]))

    if "heart" in exposure_map:
        scores.append(
            score_cardiac_risk(exposure_map["heart"], herg_ic50_uM, cmax_plasma_unbound_uM)
        )

    if "lung" in exposure_map:
        scores.append(score_pulmonary_risk(exposure_map["lung"], plasma_cmax, logP))

    # 3. Aggregate
    warnings = [s.warning for s in scores if s.warning]

    risk_levels = [s.risk_level for s in scores]
    n_high = risk_levels.count("high")
    n_moderate = risk_levels.count("moderate")

    if n_high >= 2:
        overall = "high"
    elif n_high >= 1:
        overall = "high"
    elif n_moderate >= 2:
        overall = "moderate"
    elif n_moderate >= 1:
        overall = "moderate"
    else:
        overall = "low"

    risk_count = n_high + n_moderate

    # Identify max-concern organ
    risk_priority = {"high": 3, "moderate": 2, "low": 1}
    max_concern = max(scores, key=lambda s: risk_priority.get(s.risk_level, 0))
    max_concern_organ = max_concern.organ if scores else "none"

    summary = {
        "n_organs_assessed": len(exposures),
        "n_high_risk": n_high,
        "n_moderate_risk": n_moderate,
        "overall_risk": overall,
        "max_concern_organ": max_concern_organ,
        "plasma_cmax_mg_L": round(plasma_cmax, 6),
        "plasma_cmax_unbound_uM": round(cmax_plasma_unbound_uM, 6),
    }

    return OrganToxicityReport(
        drug_name=drug.name,
        organ_exposures=exposures,
        toxicity_scores=scores,
        overall_risk=overall,
        risk_count=risk_count,
        max_concern_organ=max_concern_organ,
        summary=summary,
        warnings=warnings,
    )


__all__ = [
    "DILI_DOSE_THRESHOLD_MG",
    "DILI_FUP_THRESHOLD",
    "DILI_LIVER_CMAX_UNBOUND_THRESHOLD_UM",
    "NEPHRO_KIDNEY_CMAX_THRESHOLD_UM",
    "HERG_MARGIN_THRESHOLD",
    "HERG_MARGIN_CAUTION",
    "LUNG_ACCUMULATION_RATIO_THRESHOLD",
    "PHOSPHOLIPIDOSIS_LOGP_THRESHOLD",
    "OrganExposure",
    "OrganToxicityScore",
    "OrganToxicityReport",
    "extract_organ_exposures",
    "score_hepatotoxicity",
    "score_nephrotoxicity",
    "score_cardiac_risk",
    "score_pulmonary_risk",
    "run_organ_toxicity_assessment",
]
