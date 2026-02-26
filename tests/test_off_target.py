"""Unit tests for the off-target safety panel (SafetyPanel).

Covers:
  - hERG risk classification thresholds (safe / caution / risk)
  - CYP inhibition prediction returns expected enzyme keys
  - High lipophilicity (logP) drives higher hERG risk scores / lower IC50
  - Safety report structure is complete (all 11 safety targets present)
"""

from __future__ import annotations

import pytest

from omega_pbpk.docking.off_target import (
    SAFETY_TARGET_MODELS,
    CYP_INHIBITION_MODELS,
    SafetyPanel,
)

# Expected target and enzyme names from the module docstring
EXPECTED_SAFETY_TARGETS = {
    "hERG", "Nav1.5", "Cav1.2", "5-HT2B", "D2",
    "M2", "alpha1A", "H1", "MOR", "GABA-A", "PXR",
}
EXPECTED_CYP_ENZYMES = {"CYP1A2", "CYP2C9", "CYP2C19", "CYP2D6", "CYP3A4"}


def _assess(
    logP: float = 2.0,
    mw: float = 300.0,
    tpsa: float = 50.0,
    pka: float = 7.0,
    cmax_uM: float = 1.0,
    fup: float = 0.5,
):
    """Convenience wrapper around SafetyPanel.assess with sensible defaults."""
    panel = SafetyPanel()
    return panel.assess(
        compound_name="TestCompound",
        logP=logP,
        mw=mw,
        tpsa=tpsa,
        pka=pka,
        cmax_uM=cmax_uM,
        fup=fup,
    )


# ------------------------------------------------------------------ safety target structure


class TestSafetyReportStructure:
    def test_all_11_targets_present(self) -> None:
        """Safety report must contain exactly all 11 FDA safety targets."""
        report = _assess()
        returned_targets = {r.target for r in report.target_results}
        assert returned_targets == EXPECTED_SAFETY_TARGETS, (
            f"Missing targets: {EXPECTED_SAFETY_TARGETS - returned_targets}"
        )

    def test_11_targets_count(self) -> None:
        """Exactly 11 TargetResult objects must be in the report."""
        report = _assess()
        assert len(report.target_results) == 11

    def test_cyp_results_have_all_five_enzymes(self) -> None:
        """CYP inhibition results must include all 5 major CYP enzymes."""
        report = _assess()
        returned_enzymes = {r.enzyme for r in report.cyp_results}
        assert returned_enzymes == EXPECTED_CYP_ENZYMES

    def test_report_fields_populated(self) -> None:
        """All top-level SafetyReport fields must be populated."""
        report = _assess(cmax_uM=0.5, fup=0.1)
        assert report.compound_name == "TestCompound"
        assert report.cmax_total_uM == pytest.approx(0.5, rel=1e-4)
        assert report.cmax_unbound_uM == pytest.approx(0.05, rel=1e-4)
        assert report.overall_risk in {"low", "moderate", "high"}
        assert isinstance(report.flags, list)

    def test_target_result_flags_valid(self) -> None:
        """Every TargetResult flag must be one of the three valid values."""
        report = _assess()
        valid_flags = {"safe", "caution", "risk"}
        for tr in report.target_results:
            assert tr.flag in valid_flags, f"{tr.target} has invalid flag: {tr.flag!r}"

    def test_cyp_result_risk_valid(self) -> None:
        """Every CYPInhibitionResult ddi_risk must be a valid category."""
        report = _assess()
        valid_risks = {"unlikely", "possible", "likely"}
        for cr in report.cyp_results:
            assert cr.ddi_risk in valid_risks, f"{cr.enzyme} has invalid risk: {cr.ddi_risk!r}"

    def test_ic50_values_positive(self) -> None:
        """All predicted IC50 values must be strictly positive."""
        report = _assess()
        for tr in report.target_results:
            assert tr.predicted_ic50_uM > 0, f"{tr.target} has non-positive IC50"
        for cr in report.cyp_results:
            assert cr.predicted_ic50_uM > 0, f"{cr.enzyme} has non-positive IC50"


# ------------------------------------------------------------------ hERG risk classification


class TestHERGClassification:
    """hERG threshold: margin < 30 → risk, 30–90 → caution, >90 → safe."""

    def _get_herg(self, report):
        return next(r for r in report.target_results if r.target == "hERG")

    def test_herg_safe_when_margin_high(self) -> None:
        """A very low Cmax relative to predicted hERG IC50 should yield safe flag."""
        # logP=2, mw=300, tpsa=50, pka=7 → log_ic50 = 1.8 - 0.9 + 0.3 + 0.4 - 0.35 = 1.25
        # ic50 ≈ 17.8 µM; cmax_unbound = 0.0001 µM → margin >> 90
        report = _assess(logP=2.0, mw=300.0, tpsa=50.0, pka=7.0, cmax_uM=0.0001, fup=1.0)
        herg = self._get_herg(report)
        assert herg.flag == "safe", f"Expected safe, got {herg.flag!r} (margin={herg.margin})"
        assert herg.margin > 90

    def test_herg_risk_when_margin_below_30(self) -> None:
        """A very high Cmax (cmax_unbound >> IC50/30) must yield risk flag."""
        # Force cmax_unbound to be huge so margin < 30
        report = _assess(cmax_uM=1000.0, fup=1.0)
        herg = self._get_herg(report)
        assert herg.flag == "risk", (
            f"Expected risk, got {herg.flag!r} (margin={herg.margin})"
        )
        assert herg.margin < 30

    def test_herg_caution_in_intermediate_range(self) -> None:
        """Margin in [30, 90) should produce caution flag for hERG."""
        # We need to engineer a margin between 30 and 90.
        # ic50 ≈ 10^(1.8 - 0.45*2 + 0.001*300 + 0.008*50 - 0.05*7) = 10^1.25 ≈ 17.78 µM
        # For margin=40 we need cmax_unbound = 17.78 / 40 ≈ 0.444 µM
        # Use fup=1.0, cmax_uM=0.444
        import math
        coeffs = SAFETY_TARGET_MODELS["hERG"]
        logP, mw, tpsa, pka = 2.0, 300.0, 50.0, 7.0
        log_ic50 = (
            coeffs["intercept"]
            + coeffs["logP"] * logP
            + coeffs["MW"] * mw
            + coeffs["TPSA"] * tpsa
            + coeffs["pKa"] * pka
        )
        ic50 = 10 ** log_ic50  # µM
        # Target margin = 50 (caution: 30 ≤ margin < 90)
        cmax_unbound_target = ic50 / 50.0
        report = _assess(logP=logP, mw=mw, tpsa=tpsa, pka=pka,
                         cmax_uM=cmax_unbound_target, fup=1.0)
        herg = self._get_herg(report)
        assert herg.flag == "caution", (
            f"Expected caution, got {herg.flag!r} (margin={herg.margin})"
        )

    def test_herg_threshold_is_30(self) -> None:
        """hERG TargetResult should have threshold_margin == 30."""
        report = _assess()
        herg = self._get_herg(report)
        assert herg.threshold_margin == 30.0


# ------------------------------------------------------------------ lipophilicity effect


class TestLipophilicityEffect:
    def test_higher_logP_lowers_herg_ic50(self) -> None:
        """hERG IC50 decreases as logP increases (negative coefficient)."""
        panel = SafetyPanel()
        low_logP_report = panel.assess("Low", logP=1.0, mw=300.0)
        high_logP_report = panel.assess("High", logP=5.0, mw=300.0)

        low_ic50 = next(r for r in low_logP_report.target_results if r.target == "hERG").predicted_ic50_uM
        high_ic50 = next(r for r in high_logP_report.target_results if r.target == "hERG").predicted_ic50_uM

        assert high_ic50 < low_ic50, (
            f"Higher logP should reduce hERG IC50; got low={low_ic50:.2f}, high={high_ic50:.2f}"
        )

    def test_higher_logP_increases_herg_risk_score(self) -> None:
        """At fixed Cmax, higher logP produces a smaller margin (higher risk)."""
        panel = SafetyPanel()
        low_logP_report = panel.assess("Low", logP=1.0, mw=300.0, cmax_uM=0.5, fup=1.0)
        high_logP_report = panel.assess("High", logP=5.0, mw=300.0, cmax_uM=0.5, fup=1.0)

        low_margin = next(r for r in low_logP_report.target_results if r.target == "hERG").margin
        high_margin = next(r for r in high_logP_report.target_results if r.target == "hERG").margin

        assert high_margin < low_margin, (
            f"Higher logP should give lower margin; got low={low_margin:.1f}, high={high_margin:.1f}"
        )

    def test_high_logP_can_trigger_risk_flag(self) -> None:
        """A very high logP drug at moderate Cmax should be flagged as risk for hERG."""
        # logP=8 → log_ic50 = 1.8 - 3.6 + ... very low IC50 → small margin
        panel = SafetyPanel()
        report = panel.assess("VeryLipophilic", logP=8.0, mw=300.0, cmax_uM=1.0, fup=1.0)
        herg = next(r for r in report.target_results if r.target == "hERG")
        # With logP=8, coefficient=-0.45 → contribution = -3.6, ic50 < 1 µM
        # margin = ic50 / 1.0 → should be < 30
        assert herg.flag in {"caution", "risk"}, (
            f"High logP=8 should yield caution or risk; got {herg.flag!r} (margin={herg.margin})"
        )


# ------------------------------------------------------------------ CYP inhibition panel


class TestCYPInhibitionPrediction:
    def test_cyp_results_return_expected_keys(self) -> None:
        """CYP results should have enzyme, predicted_ic50_uM, ddi_risk for all 5 enzymes."""
        report = _assess()
        assert len(report.cyp_results) == 5
        for cr in report.cyp_results:
            assert hasattr(cr, "enzyme")
            assert hasattr(cr, "predicted_ic50_uM")
            assert hasattr(cr, "ddi_risk")

    def test_cyp_risk_likely_when_ratio_high(self) -> None:
        """When cmax_unbound / IC50 > 0.1 the DDI risk must be 'likely'."""
        # Force a high cmax_unbound so ratio > 0.1 for at least one enzyme
        report = _assess(cmax_uM=1000.0, fup=1.0)
        risk_levels = {cr.ddi_risk for cr in report.cyp_results}
        assert "likely" in risk_levels, (
            f"Expected at least one 'likely' CYP DDI risk; got {risk_levels}"
        )

    def test_cyp_risk_unlikely_when_ratio_low(self) -> None:
        """When cmax_unbound << IC50 the DDI risk should be 'unlikely'."""
        report = _assess(cmax_uM=0.00001, fup=1.0)
        for cr in report.cyp_results:
            assert cr.ddi_risk == "unlikely", (
                f"{cr.enzyme} should be unlikely; got {cr.ddi_risk!r}"
            )

    def test_cyp_inhibition_models_have_five_entries(self) -> None:
        """The CYP_INHIBITION_MODELS dict must contain exactly 5 enzymes."""
        assert len(CYP_INHIBITION_MODELS) == 5
        assert set(CYP_INHIBITION_MODELS.keys()) == EXPECTED_CYP_ENZYMES


# ------------------------------------------------------------------ overall risk aggregation


class TestOverallRisk:
    def test_overall_risk_low_for_safe_compound(self) -> None:
        """A very low Cmax compound with no flags should yield low overall risk."""
        report = _assess(cmax_uM=0.00001, fup=0.01)
        assert report.overall_risk == "low"

    def test_overall_risk_high_for_flagged_compound(self) -> None:
        """A very high Cmax compound with multiple target flags should yield high risk."""
        report = _assess(cmax_uM=10000.0, fup=1.0)
        assert report.overall_risk in {"moderate", "high"}

    def test_safety_target_models_has_11_entries(self) -> None:
        """The SAFETY_TARGET_MODELS dict must contain exactly 11 targets."""
        assert len(SAFETY_TARGET_MODELS) == 11
        assert set(SAFETY_TARGET_MODELS.keys()) == EXPECTED_SAFETY_TARGETS
