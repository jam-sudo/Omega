"""Tests for Phase 392 — Drug-Induced Hepatotoxicity (DILI) Risk Score."""

import pytest

from omega_pbpk.risk.dili_risk_score import (
    DILIRiskResult,
    predict_dili_risk,
    screen_dili_risk,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def safe_drug(**kwargs):
    defaults = dict(
        drug_name="SafeDrug",
        max_daily_dose_mg=10.0,
        logp=1.0,
        mw_da=200.0,
        n_reactive_groups=0,
        n_aromatic_rings=1,
    )
    defaults.update(kwargs)
    return predict_dili_risk(**defaults)


def risky_drug(**kwargs):
    defaults = dict(
        drug_name="RiskyDrug",
        max_daily_dose_mg=1000.0,
        logp=4.0,
        mw_da=500.0,
        n_reactive_groups=2,
        n_aromatic_rings=4,
    )
    defaults.update(kwargs)
    return predict_dili_risk(**defaults)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_dataclass(self):
        r = safe_drug()
        assert isinstance(r, DILIRiskResult)

    def test_frozen(self):
        r = safe_drug()
        with pytest.raises((AttributeError, TypeError)):
            r.dili_score = 99.0  # type: ignore[misc]

    def test_all_fields_present(self):
        r = safe_drug()
        assert hasattr(r, "drug_name")
        assert hasattr(r, "dili_score")
        assert hasattr(r, "dili_risk_category")
        assert hasattr(r, "reactive_metabolite_risk")
        assert hasattr(r, "mitochondrial_liability")
        assert hasattr(r, "bile_salt_export_pump_inhibition")
        assert hasattr(r, "cholestatic_risk")
        assert hasattr(r, "hepatocellular_risk")
        assert hasattr(r, "daily_dose_contribution")
        assert hasattr(r, "physicochemical_contribution")
        assert hasattr(r, "structural_contribution")
        assert hasattr(r, "max_daily_dose_mg")
        assert hasattr(r, "idiosyncratic_risk")
        assert hasattr(r, "notes")


# ---------------------------------------------------------------------------
# Score components
# ---------------------------------------------------------------------------


class TestScoreComponents:
    def test_score_in_0_to_100(self):
        r = safe_drug()
        assert 0.0 <= r.dili_score <= 100.0

    def test_risky_score_higher_than_safe(self):
        r_safe = safe_drug()
        r_risky = risky_drug()
        assert r_risky.dili_score > r_safe.dili_score

    def test_high_dose_increases_score(self):
        r_low = safe_drug(max_daily_dose_mg=10.0)
        r_high = safe_drug(max_daily_dose_mg=400.0)
        assert r_high.daily_dose_contribution > r_low.daily_dose_contribution

    def test_dose_contribution_capped_at_40(self):
        r = safe_drug(max_daily_dose_mg=10000.0)
        assert r.daily_dose_contribution == pytest.approx(40.0)

    def test_physicochemical_zero_for_low_logp_low_mw(self):
        r = predict_dili_risk(
            drug_name="Tiny",
            max_daily_dose_mg=10.0,
            logp=1.0,  # below 2 → 0 logp points
            mw_da=200.0,  # below 300 → 0 mw points
            n_reactive_groups=0,
            n_aromatic_rings=0,
        )
        assert r.physicochemical_contribution == pytest.approx(0.0)

    def test_physicochemical_capped_at_30(self):
        r = predict_dili_risk(
            drug_name="LipophilicGiant",
            max_daily_dose_mg=10.0,
            logp=20.0,
            mw_da=2000.0,
            n_reactive_groups=0,
            n_aromatic_rings=0,
        )
        assert r.physicochemical_contribution == pytest.approx(30.0)

    def test_structural_contribution_reactive_groups(self):
        r = predict_dili_risk(
            drug_name="Reactive",
            max_daily_dose_mg=10.0,
            logp=0.0,
            mw_da=200.0,
            n_reactive_groups=2,
            n_aromatic_rings=0,
        )
        # 2 * 8 = 16 points
        assert r.structural_contribution == pytest.approx(16.0)

    def test_structural_capped_at_30(self):
        r = predict_dili_risk(
            drug_name="HighlyReactive",
            max_daily_dose_mg=10.0,
            logp=0.0,
            mw_da=200.0,
            n_reactive_groups=10,
            n_aromatic_rings=10,
        )
        assert r.structural_contribution == pytest.approx(30.0)

    def test_score_sum_of_components(self):
        r = safe_drug(max_daily_dose_mg=100.0, logp=3.0, mw_da=350.0)
        expected = min(
            100.0,
            r.daily_dose_contribution + r.physicochemical_contribution + r.structural_contribution,
        )
        assert r.dili_score == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Risk flags
# ---------------------------------------------------------------------------


class TestRiskFlags:
    def test_reactive_metabolite_risk_when_reactive_groups(self):
        r = safe_drug(n_reactive_groups=1)
        assert r.reactive_metabolite_risk is True

    def test_no_reactive_metabolite_risk_when_zero_groups(self):
        r = safe_drug(n_reactive_groups=0)
        assert r.reactive_metabolite_risk is False

    def test_mitochondrial_liability_high_logp_high_mw(self):
        r = predict_dili_risk(
            drug_name="MitoLiable",
            max_daily_dose_mg=100.0,
            logp=4.0,
            mw_da=500.0,
            n_reactive_groups=0,
            n_aromatic_rings=0,
        )
        assert r.mitochondrial_liability is True

    def test_no_mitochondrial_liability_low_logp(self):
        r = predict_dili_risk(
            drug_name="Safe",
            max_daily_dose_mg=100.0,
            logp=2.0,
            mw_da=500.0,
            n_reactive_groups=0,
            n_aromatic_rings=0,
        )
        assert r.mitochondrial_liability is False

    def test_bsep_inhibition_high_logp_high_dose(self):
        r = predict_dili_risk(
            drug_name="BSEPDrug",
            max_daily_dose_mg=200.0,
            logp=3.0,
            mw_da=300.0,
            n_reactive_groups=0,
            n_aromatic_rings=0,
        )
        assert r.bile_salt_export_pump_inhibition is True
        assert r.cholestatic_risk is True

    def test_no_bsep_inhibition_low_dose(self):
        r = predict_dili_risk(
            drug_name="LowDose",
            max_daily_dose_mg=50.0,
            logp=3.0,
            mw_da=300.0,
            n_reactive_groups=0,
            n_aromatic_rings=0,
        )
        assert r.bile_salt_export_pump_inhibition is False

    def test_idiosyncratic_risk_with_reactive_groups(self):
        r = safe_drug(n_reactive_groups=1)
        assert r.idiosyncratic_risk is True

    def test_no_idiosyncratic_risk_without_reactive_groups(self):
        r = safe_drug(n_reactive_groups=0)
        assert r.idiosyncratic_risk is False

    def test_hepatocellular_risk_via_reactive(self):
        r = safe_drug(n_reactive_groups=1)
        assert r.hepatocellular_risk is True


# ---------------------------------------------------------------------------
# Risk category
# ---------------------------------------------------------------------------


class TestRiskCategory:
    def test_low_category(self):
        r = safe_drug()
        # Small dose, low logp, no reactive groups → should be low
        assert r.dili_risk_category == "low"

    def test_very_high_category(self):
        # dose=1000 (40pts) + logp=12 (30pts physico) + 1 reactive group (8pts) = 78 → very_high
        r = predict_dili_risk(
            drug_name="VeryHighDrug",
            max_daily_dose_mg=1000.0,
            logp=12.0,
            mw_da=300.0,
            n_reactive_groups=1,
            n_aromatic_rings=0,
        )
        assert r.dili_risk_category == "very_high"

    def test_valid_categories(self):
        valid = {"low", "moderate", "high", "very_high"}
        for n_rg in (0, 1, 2):
            for dose in (10.0, 100.0, 500.0, 1000.0):
                r = safe_drug(max_daily_dose_mg=dose, n_reactive_groups=n_rg, logp=2.0)
                assert r.dili_risk_category in valid


# ---------------------------------------------------------------------------
# cLogP parameter
# ---------------------------------------------------------------------------


class TestCLogP:
    def test_clogp_used_when_provided(self):
        r1 = predict_dili_risk(
            drug_name="D",
            max_daily_dose_mg=100.0,
            logp=1.0,
            mw_da=300.0,
            n_reactive_groups=0,
            n_aromatic_rings=0,
            cLogP=5.0,
        )
        r2 = predict_dili_risk(
            drug_name="D",
            max_daily_dose_mg=100.0,
            logp=1.0,
            mw_da=300.0,
            n_reactive_groups=0,
            n_aromatic_rings=0,
            cLogP=None,
        )
        # r1 uses cLogP=5.0, r2 uses logp=1.0; r1 should have higher physico contribution
        assert r1.physicochemical_contribution > r2.physicochemical_contribution


# ---------------------------------------------------------------------------
# screen_dili_risk
# ---------------------------------------------------------------------------


class TestScreenDILIRisk:
    def _make_library(self):
        return [
            {
                "name": "SafeDrug",
                "max_daily_dose_mg": 10.0,
                "logp": 1.0,
                "mw_da": 200.0,
                "n_reactive_groups": 0,
                "n_aromatic_rings": 1,
            },
            {
                "name": "ModerateDrug",
                "max_daily_dose_mg": 200.0,
                "logp": 3.0,
                "mw_da": 350.0,
                "n_reactive_groups": 0,
                "n_aromatic_rings": 2,
            },
            {
                "name": "HighDrug",
                "max_daily_dose_mg": 500.0,
                "logp": 4.0,
                "mw_da": 450.0,
                "n_reactive_groups": 1,
                "n_aromatic_rings": 3,
            },
            {
                "name": "VeryHighDrug",
                "max_daily_dose_mg": 1000.0,
                "logp": 5.0,
                "mw_da": 600.0,
                "n_reactive_groups": 3,
                "n_aromatic_rings": 5,
            },
        ]

    def test_screen_returns_list(self):
        results = screen_dili_risk(self._make_library())
        assert isinstance(results, list)

    def test_screen_length(self):
        results = screen_dili_risk(self._make_library())
        assert len(results) == 4

    def test_screen_sorted_descending(self):
        results = screen_dili_risk(self._make_library())
        scores = [r.dili_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_screen_all_correct_type(self):
        results = screen_dili_risk(self._make_library())
        for r in results:
            assert isinstance(r, DILIRiskResult)

    def test_screen_with_clogp(self):
        drugs = [
            {
                "name": "D1",
                "max_daily_dose_mg": 100.0,
                "logp": 2.0,
                "mw_da": 300.0,
                "n_reactive_groups": 0,
                "n_aromatic_rings": 1,
                "cLogP": 4.0,
            },
        ]
        results = screen_dili_risk(drugs)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="max_daily_dose_mg"):
            predict_dili_risk(
                drug_name="D",
                max_daily_dose_mg=0.0,
                logp=1.0,
                mw_da=300.0,
                n_reactive_groups=0,
                n_aromatic_rings=0,
            )

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="max_daily_dose_mg"):
            predict_dili_risk(
                drug_name="D",
                max_daily_dose_mg=-100.0,
                logp=1.0,
                mw_da=300.0,
                n_reactive_groups=0,
                n_aromatic_rings=0,
            )

    def test_zero_mw_raises(self):
        with pytest.raises(ValueError, match="mw_da"):
            predict_dili_risk(
                drug_name="D",
                max_daily_dose_mg=100.0,
                logp=1.0,
                mw_da=0.0,
                n_reactive_groups=0,
                n_aromatic_rings=0,
            )

    def test_negative_reactive_groups_raises(self):
        with pytest.raises(ValueError, match="n_reactive_groups"):
            predict_dili_risk(
                drug_name="D",
                max_daily_dose_mg=100.0,
                logp=1.0,
                mw_da=300.0,
                n_reactive_groups=-1,
                n_aromatic_rings=0,
            )

    def test_negative_aromatic_rings_raises(self):
        with pytest.raises(ValueError, match="n_aromatic_rings"):
            predict_dili_risk(
                drug_name="D",
                max_daily_dose_mg=100.0,
                logp=1.0,
                mw_da=300.0,
                n_reactive_groups=0,
                n_aromatic_rings=-1,
            )
