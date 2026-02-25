"""Off-target safety panel: 11 FDA safety targets + 5 CYP IC50 prediction.

Safety targets (FDA guidance):
  1. hERG (cardiac) — IC50 threshold: 30× Cmax
  2. Nav1.5 (cardiac sodium channel)
  3. Cav1.2 (cardiac calcium channel)
  4. 5-HT2B (valvulopathy)
  5. D2 (extrapyramidal)
  6. M2 (muscarinic, cardiac)
  7. α1A (orthostatic hypotension)
  8. H1 (sedation)
  9. MOR (opioid, respiratory depression)
  10. GABA-A (sedation)
  11. PXR (enzyme induction liability)

CYP inhibition panel:
  CYP1A2, CYP2C9, CYP2C19, CYP2D6, CYP3A4
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TargetResult:
    """Safety assessment result for a single target."""

    target: str
    predicted_ic50_uM: float
    margin: float  # IC50 / Cmax_unbound
    flag: str  # "safe", "caution", "risk"
    threshold_margin: float = 30.0


@dataclass(frozen=True)
class CYPInhibitionResult:
    """CYP inhibition prediction for a single enzyme."""

    enzyme: str
    predicted_ic50_uM: float
    ddi_risk: str  # "unlikely", "possible", "likely"


@dataclass(frozen=True)
class SafetyReport:
    """Complete safety assessment report."""

    compound_name: str
    cmax_total_uM: float
    cmax_unbound_uM: float
    target_results: list[TargetResult]
    cyp_results: list[CYPInhibitionResult]
    overall_risk: str  # "low", "moderate", "high"
    flags: list[str] = field(default_factory=list)


# QSPR coefficients for safety target IC50 prediction
# Format: target_name → (intercept, coeff_logP, coeff_MW, coeff_TPSA, coeff_pKa)
SAFETY_TARGET_MODELS: dict[str, dict[str, float]] = {
    "hERG": {"intercept": 1.8, "logP": -0.45, "MW": 0.001, "TPSA": 0.008, "pKa": -0.05},
    "Nav1.5": {"intercept": 2.5, "logP": -0.30, "MW": 0.001, "TPSA": 0.005, "pKa": -0.03},
    "Cav1.2": {"intercept": 2.3, "logP": -0.35, "MW": 0.001, "TPSA": 0.006, "pKa": -0.04},
    "5-HT2B": {"intercept": 2.0, "logP": -0.40, "MW": 0.002, "TPSA": 0.005, "pKa": -0.08},
    "D2": {"intercept": 2.2, "logP": -0.38, "MW": 0.001, "TPSA": 0.004, "pKa": -0.06},
    "M2": {"intercept": 2.4, "logP": -0.32, "MW": 0.001, "TPSA": 0.007, "pKa": -0.04},
    "alpha1A": {"intercept": 2.1, "logP": -0.35, "MW": 0.001, "TPSA": 0.005, "pKa": -0.05},
    "H1": {"intercept": 2.3, "logP": -0.30, "MW": 0.001, "TPSA": 0.006, "pKa": -0.03},
    "MOR": {"intercept": 2.6, "logP": -0.25, "MW": 0.001, "TPSA": 0.004, "pKa": -0.07},
    "GABA-A": {"intercept": 2.8, "logP": -0.20, "MW": 0.001, "TPSA": 0.005, "pKa": -0.02},
    "PXR": {"intercept": 1.5, "logP": -0.50, "MW": 0.002, "TPSA": 0.003, "pKa": -0.01},
}

CYP_INHIBITION_MODELS: dict[str, dict[str, float]] = {
    "CYP1A2": {"intercept": 2.5, "logP": -0.30, "MW": 0.001, "TPSA": 0.005},
    "CYP2C9": {"intercept": 2.2, "logP": -0.35, "MW": 0.001, "TPSA": 0.004},
    "CYP2C19": {"intercept": 2.3, "logP": -0.33, "MW": 0.001, "TPSA": 0.004},
    "CYP2D6": {"intercept": 2.0, "logP": -0.40, "MW": 0.002, "TPSA": 0.005},
    "CYP3A4": {"intercept": 1.8, "logP": -0.45, "MW": 0.002, "TPSA": 0.003},
}


class SafetyPanel:
    """Off-target safety and CYP inhibition assessment.

    Usage:
        panel = SafetyPanel()
        report = panel.assess(
            compound_name="Midazolam",
            logP=3.89, mw=325.77, tpsa=30.2, pka=6.15,
            cmax_uM=0.15, fup=0.032,
        )
    """

    def assess(
        self,
        compound_name: str,
        logP: float,
        mw: float,
        tpsa: float = 50.0,
        pka: float = 7.0,
        cmax_uM: float = 1.0,
        fup: float = 0.5,
    ) -> SafetyReport:
        """Run full safety assessment."""
        cmax_unbound = cmax_uM * fup
        flags: list[str] = []

        # Safety targets
        target_results: list[TargetResult] = []
        for target, coeffs in SAFETY_TARGET_MODELS.items():
            log_ic50 = (
                coeffs["intercept"]
                + coeffs["logP"] * logP
                + coeffs["MW"] * mw
                + coeffs["TPSA"] * tpsa
                + coeffs["pKa"] * pka
            )
            ic50 = 10**log_ic50
            margin = ic50 / max(cmax_unbound, 1e-12)

            threshold = 30.0 if target == "hERG" else 10.0

            if margin < threshold:
                flag = "risk"
                flags.append(f"{target}: margin {margin:.1f}× (< {threshold}×)")
            elif margin < threshold * 3:
                flag = "caution"
            else:
                flag = "safe"

            target_results.append(
                TargetResult(
                    target=target,
                    predicted_ic50_uM=round(float(ic50), 2),
                    margin=round(float(margin), 1),
                    flag=flag,
                    threshold_margin=threshold,
                )
            )

        # CYP inhibition
        cyp_results: list[CYPInhibitionResult] = []
        for enzyme, coeffs in CYP_INHIBITION_MODELS.items():
            log_ic50 = (
                coeffs["intercept"]
                + coeffs["logP"] * logP
                + coeffs["MW"] * mw
                + coeffs["TPSA"] * tpsa
            )
            ic50 = 10**log_ic50

            ratio = cmax_unbound / max(ic50, 1e-12)
            if ratio > 0.1:
                risk = "likely"
                flags.append(f"{enzyme} DDI: [I]u/IC50 = {ratio:.3f}")
            elif ratio > 0.02:
                risk = "possible"
            else:
                risk = "unlikely"

            cyp_results.append(
                CYPInhibitionResult(
                    enzyme=enzyme,
                    predicted_ic50_uM=round(float(ic50), 2),
                    ddi_risk=risk,
                )
            )

        # Overall risk
        n_risk = sum(1 for r in target_results if r.flag == "risk")
        n_cyp_risk = sum(1 for r in cyp_results if r.ddi_risk == "likely")
        if n_risk >= 2 or (n_risk >= 1 and n_cyp_risk >= 1):
            overall = "high"
        elif n_risk >= 1 or n_cyp_risk >= 1:
            overall = "moderate"
        else:
            overall = "low"

        return SafetyReport(
            compound_name=compound_name,
            cmax_total_uM=round(cmax_uM, 4),
            cmax_unbound_uM=round(cmax_unbound, 6),
            target_results=target_results,
            cyp_results=cyp_results,
            overall_risk=overall,
            flags=flags,
        )
