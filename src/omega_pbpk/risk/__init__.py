"""PK-based risk flag assessment.

Provides simple threshold-based risk flags from pharmacokinetic summary
parameters. These complement the molecular safety panel (off-target IC50)
with PK-derived clinical risk indicators.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskFlags:
    """Boolean risk flags derived from PK parameters.

    Attributes:
        high_cmax_risk: Cmax ≥ threshold (potential toxicity).
        long_half_life_risk: t½ ≥ threshold (accumulation concern).
        high_variability_risk: Exposure CV ≥ threshold (dose-finding difficulty).
        high_ddi_susceptibility_risk: DDI score ≥ threshold (metabolic vulnerability).
    """

    high_cmax_risk: bool
    long_half_life_risk: bool
    high_variability_risk: bool
    high_ddi_susceptibility_risk: bool

    @property
    def any_risk(self) -> bool:
        """True if any risk flag is raised."""
        return (
            self.high_cmax_risk
            or self.long_half_life_risk
            or self.high_variability_risk
            or self.high_ddi_susceptibility_risk
        )

    @property
    def risk_count(self) -> int:
        """Number of raised risk flags."""
        return sum(
            [
                self.high_cmax_risk,
                self.long_half_life_risk,
                self.high_variability_risk,
                self.high_ddi_susceptibility_risk,
            ]
        )

    def summary(self) -> dict[str, bool]:
        """Return risk flags as a dict."""
        return {
            "high_cmax_risk": self.high_cmax_risk,
            "long_half_life_risk": self.long_half_life_risk,
            "high_variability_risk": self.high_variability_risk,
            "high_ddi_susceptibility_risk": self.high_ddi_susceptibility_risk,
        }


def compute_risk_flags(
    cmax_mg_per_l: float,
    half_life_h: float,
    exposure_cv: float = 0.0,
    ddi_risk_score: float = 0.0,
    *,
    cmax_threshold: float = 5.0,
    half_life_threshold_h: float = 24.0,
    cv_threshold: float = 0.4,
    ddi_threshold: float = 0.5,
) -> RiskFlags:
    """Compute PK-based risk flags with configurable thresholds.

    Args:
        cmax_mg_per_l: Maximum plasma concentration (mg/L).
        half_life_h: Terminal half-life (h).
        exposure_cv: Coefficient of variation of AUC (0–1).
        ddi_risk_score: DDI susceptibility score (0–1).
        cmax_threshold: Cmax threshold for high_cmax_risk.
        half_life_threshold_h: Half-life threshold for long_half_life_risk.
        cv_threshold: CV threshold for high_variability_risk.
        ddi_threshold: DDI score threshold for high_ddi_susceptibility_risk.

    Returns:
        RiskFlags dataclass with boolean risk indicators.
    """
    return RiskFlags(
        high_cmax_risk=cmax_mg_per_l >= cmax_threshold,
        long_half_life_risk=half_life_h >= half_life_threshold_h,
        high_variability_risk=exposure_cv >= cv_threshold,
        high_ddi_susceptibility_risk=ddi_risk_score >= ddi_threshold,
    )


from omega_pbpk.risk.mist_assessment import (  # noqa: E402
    MIST_THRESHOLD,
    MetaboliteMISTResult,
    MetaboliteSpec,
    MISTReport,
    classify_metabolite,
    estimate_metabolite_auc,
    format_mist_table,
    run_mist_assessment,
)
from omega_pbpk.risk.organ_toxicity import (  # noqa: E402
    OrganExposure,
    OrganToxicityReport,
    OrganToxicityScore,
    run_organ_toxicity_assessment,
)

__all__ = [
    "RiskFlags",
    "compute_risk_flags",
    "OrganExposure",
    "OrganToxicityScore",
    "OrganToxicityReport",
    "run_organ_toxicity_assessment",
    "MIST_THRESHOLD",
    "MetaboliteSpec",
    "MetaboliteMISTResult",
    "MISTReport",
    "estimate_metabolite_auc",
    "classify_metabolite",
    "run_mist_assessment",
    "format_mist_table",
]
