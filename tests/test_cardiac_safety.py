"""Tests for comprehensive cardiac safety assessment (Phase 178)."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.cardiac_safety import CardiacSafetyResult, assess_cardiac_safety

# ── Helpers ──────────────────────────────────────────────────────────────────


def _safe_drug(**kwargs):
    params = dict(
        drug_name="safe_compound",
        smiles="CC(=O)Oc1ccccc1C(=O)O",  # aspirin-like, no alerts
        herg_ic50_uM=100.0,  # high IC50 → low inhibition
        concentration_uM=0.1,
        logP=1.5,
        mw=300.0,
        daily_dose_mg=100.0,
    )
    params.update(kwargs)
    return assess_cardiac_safety(**params)


def _risky_drug(**kwargs):
    params = dict(
        drug_name="risky_compound",
        smiles="CN(C)c1ccc(cc1)C(=O)O",  # tertiary amine, maybe basic
        herg_ic50_uM=0.5,  # low IC50 → high inhibition
        concentration_uM=5.0,
        logP=4.5,
        mw=450.0,
        daily_dose_mg=800.0,
    )
    params.update(kwargs)
    return assess_cardiac_safety(**params)


# ── Result structure ──────────────────────────────────────────────────────────


class TestResultStructure:
    def test_returns_cardiac_safety_result(self):
        r = _safe_drug()
        assert isinstance(r, CardiacSafetyResult)

    def test_fields_populated(self):
        r = _safe_drug()
        assert r.drug_name == "safe_compound"
        assert r.herg_ic50_uM == pytest.approx(100.0)
        assert isinstance(r.structural_alerts, list)
        assert isinstance(r.mitigation, list)
        assert isinstance(r.notes, str)

    def test_qt_risk_score_bounded(self):
        r = _risky_drug()
        assert 0.0 <= r.qt_risk_score <= 100.0


# ── QT risk ───────────────────────────────────────────────────────────────────


class TestQTRisk:
    def test_low_inhibition_low_qt_risk(self):
        """High IC50 relative to concentration → low QT risk."""
        r = assess_cardiac_safety(
            "safe",
            "C",
            herg_ic50_uM=1000.0,
            concentration_uM=1.0,
            logP=1.0,
            mw=200.0,
            daily_dose_mg=50.0,
        )
        assert r.qt_risk_category == "low"

    def test_high_inhibition_high_qt_risk(self):
        """concentration >> IC50 → high/critical QT risk."""
        r = assess_cardiac_safety(
            "risky",
            "C",
            herg_ic50_uM=0.01,
            concentration_uM=100.0,
            logP=4.0,
            mw=500.0,
            daily_dose_mg=1000.0,
        )
        assert r.qt_risk_category in ("high", "critical")

    def test_delta_qtc_proportional_to_inhibition(self):
        """delta_qtc_ms = inhibition_pct * 0.8."""
        r = assess_cardiac_safety(
            "drug",
            "C",
            herg_ic50_uM=10.0,
            concentration_uM=10.0,
            logP=1.0,
            mw=200.0,
            daily_dose_mg=10.0,
        )
        # inhibition = 50%  → delta_qtc = 40 ms
        assert r.delta_qtc_ms == pytest.approx(40.0, rel=1e-4)

    def test_zero_concentration_zero_inhibition(self):
        r = assess_cardiac_safety(
            "drug",
            "C",
            herg_ic50_uM=10.0,
            concentration_uM=0.0,
        )
        assert r.delta_qtc_ms == pytest.approx(0.0)
        assert r.qt_risk_category == "low"

    def test_qt_categories_correct_boundaries(self):
        # score just under 20 → low
        r_low = assess_cardiac_safety(
            "d",
            "C",
            herg_ic50_uM=1000.0,
            concentration_uM=0.01,
            logP=0.0,
            mw=100.0,
            daily_dose_mg=0.0,
        )
        assert r_low.qt_risk_category == "low"

    def test_critical_qt_risk(self):
        r = assess_cardiac_safety(
            "d",
            "C",
            herg_ic50_uM=0.001,
            concentration_uM=1000.0,
            logP=10.0,
            mw=200.0,
            daily_dose_mg=5000.0,
        )
        assert r.qt_risk_category == "critical"


# ── delta QTc ─────────────────────────────────────────────────────────────────


class TestDeltaQTc:
    def test_delta_qtc_formula(self):
        """delta_qtc_ms = hERG_inhibition_pct * 0.8."""
        ic50, conc = 5.0, 5.0  # 50% inhibition
        r = assess_cardiac_safety("d", "C", herg_ic50_uM=ic50, concentration_uM=conc)
        expected_delta = 50.0 * 0.8
        assert r.delta_qtc_ms == pytest.approx(expected_delta, rel=1e-4)

    def test_delta_qtc_nonnegative(self):
        r = _safe_drug()
        assert r.delta_qtc_ms >= 0.0


# ── Structural alerts ─────────────────────────────────────────────────────────


class TestStructuralAlerts:
    def test_no_alerts_clean_smiles(self):
        r = assess_cardiac_safety(
            "aspirin",
            "CC(=O)Oc1ccccc1C(=O)O",
            herg_ic50_uM=100.0,
            concentration_uM=0.1,
        )
        assert len(r.structural_alerts) == 0

    def test_anthraquinone_alert_detected(self):
        # "C(=O)c1" pattern — carbonyl carbon directly bonded to aromatic c1
        r = assess_cardiac_safety(
            "quinone_drug",
            "C(=O)c1ccccc1",
            herg_ic50_uM=100.0,
            concentration_uM=0.1,
        )
        assert "anthraquinone" in r.structural_alerts

    def test_tertiary_amine_alert_detected(self):
        # "N(C)(C)C" pattern
        r = assess_cardiac_safety(
            "tert_amine_drug",
            "CN(C)(C)CCc1ccccc1",
            herg_ic50_uM=100.0,
            concentration_uM=0.1,
        )
        assert "tertiary_amine_basic" in r.structural_alerts

    def test_biguanide_alert_detected(self):
        # "NC(=N)N" pattern
        r = assess_cardiac_safety(
            "biguanide_drug",
            "NC(=N)NC(=N)N",
            herg_ic50_uM=100.0,
            concentration_uM=0.1,
        )
        assert "biguanide" in r.structural_alerts


# ── Inotropy ──────────────────────────────────────────────────────────────────


class TestInotropy:
    def test_low_inotropy_risk_low_logP(self):
        r = assess_cardiac_safety(
            "d",
            "C",
            herg_ic50_uM=100.0,
            concentration_uM=1.0,
            logP=1.0,
            mw=200.0,
        )
        assert r.inotropy_risk == "low"

    def test_moderate_inotropy_high_logP_and_mw(self):
        r = assess_cardiac_safety(
            "d",
            "C",
            herg_ic50_uM=100.0,
            concentration_uM=1.0,
            logP=4.0,
            mw=400.0,
        )
        assert r.inotropy_risk == "moderate"


# ── Mitigation ────────────────────────────────────────────────────────────────


class TestMitigation:
    def test_mitigation_empty_for_safe_drug(self):
        r = _safe_drug()
        # Safe drug: low QT risk and low hERG inhibition → no mitigation needed
        assert r.mitigation == []

    def test_cardiac_monitoring_required_for_high_qt(self):
        r = assess_cardiac_safety(
            "d",
            "C",
            herg_ic50_uM=0.01,
            concentration_uM=100.0,
            logP=4.0,
            mw=500.0,
            daily_dose_mg=1000.0,
        )
        assert "Comprehensive cardiac monitoring required" in r.mitigation

    def test_safety_pharmacology_required_above_50pct_inhibition(self):
        # 50% inhibition: concentration == IC50
        r = assess_cardiac_safety(
            "d",
            "C",
            herg_ic50_uM=5.0,
            concentration_uM=5.5,
            logP=1.0,
            mw=200.0,
            daily_dose_mg=10.0,
        )
        assert "Evaluate safety pharmacology studies" in r.mitigation


# ── Validation errors ─────────────────────────────────────────────────────────


class TestValidation:
    def test_zero_herg_ic50_raises(self):
        with pytest.raises(ValueError, match="herg_ic50_uM"):
            assess_cardiac_safety("d", "C", herg_ic50_uM=0.0, concentration_uM=1.0)

    def test_negative_herg_ic50_raises(self):
        with pytest.raises(ValueError, match="herg_ic50_uM"):
            assess_cardiac_safety("d", "C", herg_ic50_uM=-1.0, concentration_uM=1.0)

    def test_negative_concentration_raises(self):
        with pytest.raises(ValueError, match="concentration_uM"):
            assess_cardiac_safety("d", "C", herg_ic50_uM=1.0, concentration_uM=-0.1)
