"""Tests for DILI scoring module."""

import pytest

from omega_pbpk.risk.dili_score import DILIResult, score_dili, screen_dili


# ── Validation tests ──────────────────────────────────────────────────


class TestValidation:
    def test_mw_zero_raises(self):
        with pytest.raises(ValueError, match="mw"):
            score_dili("X", "CCO", 0, 1.0, 50)

    def test_mw_negative_raises(self):
        with pytest.raises(ValueError, match="mw"):
            score_dili("X", "CCO", -100, 1.0, 50)

    def test_dose_zero_raises(self):
        with pytest.raises(ValueError, match="daily_dose_mg"):
            score_dili("X", "CCO", 300, 1.0, 0)

    def test_dose_negative_raises(self):
        with pytest.raises(ValueError, match="daily_dose_mg"):
            score_dili("X", "CCO", 300, 1.0, -10)

    def test_empty_smiles_raises(self):
        with pytest.raises(ValueError, match="smiles"):
            score_dili("X", "", 300, 1.0, 50)

    def test_whitespace_smiles_raises(self):
        with pytest.raises(ValueError, match="smiles"):
            score_dili("X", "   ", 300, 1.0, 50)


# ── Result structure tests ────────────────────────────────────────────


class TestResultStructure:
    def test_returns_dili_result(self):
        r = score_dili("Aspirin", "CC(=O)Oc1ccccc1C(=O)O", 180, 1.2, 500)
        assert isinstance(r, DILIResult)

    def test_frozen_dataclass(self):
        r = score_dili("X", "CCO", 300, 1.0, 50)
        with pytest.raises(AttributeError):
            r.dili_score = 99  # type: ignore[misc]

    def test_compound_name_preserved(self):
        r = score_dili("TestDrug", "CCO", 300, 1.0, 50)
        assert r.compound_name == "TestDrug"

    def test_score_clamped_0_100(self):
        # Trigger everything: high dose, logP>3, mw<300, reactive met, mito, bile
        r = score_dili(
            "Bad", "CCNS", 250, 4.0, 200,
            is_mitochondrial_toxin=True,
            bile_salt_export_inhibitor=True,
        )
        assert 0 <= r.dili_score <= 100


# ── Scoring logic tests ──────────────────────────────────────────────


class TestScoringLogic:
    def test_clean_compound_low_score(self):
        # Low dose, low logP, high MW, no flags
        r = score_dili("Clean", "CCCCCCCC", 500, 1.0, 5)
        assert r.dili_score < 25
        assert r.dili_category == "no_concern"

    def test_high_dose_adds_20(self):
        r_low = score_dili("X", "CCCCCCCC", 500, 1.0, 5)
        r_high = score_dili("X", "CCCCCCCC", 500, 1.0, 200)
        assert r_high.dili_score - r_low.dili_score == 20

    def test_moderate_dose_adds_10(self):
        r_low = score_dili("X", "CCCCCCCC", 500, 1.0, 5)
        r_mod = score_dili("X", "CCCCCCCC", 500, 1.0, 50)
        assert r_mod.dili_score - r_low.dili_score == 10

    def test_high_logp_adds_15(self):
        r_low = score_dili("X", "CCCCCCCC", 500, 1.0, 5)
        r_high = score_dili("X", "CCCCCCCC", 500, 4.0, 5)
        assert r_high.dili_score - r_low.dili_score == 15

    def test_low_mw_adds_10(self):
        r_high_mw = score_dili("X", "CCCCCCCC", 500, 1.0, 5)
        r_low_mw = score_dili("X", "CCCCCCCC", 250, 1.0, 5)
        assert r_low_mw.dili_score - r_high_mw.dili_score == 10

    def test_mito_toxin_adds_25(self):
        r_no = score_dili("X", "CCCCCCCC", 500, 1.0, 5)
        r_yes = score_dili("X", "CCCCCCCC", 500, 1.0, 5, is_mitochondrial_toxin=True)
        assert r_yes.dili_score - r_no.dili_score == 25

    def test_bile_salt_adds_20(self):
        r_no = score_dili("X", "CCCCCCCC", 500, 1.0, 5)
        r_yes = score_dili("X", "CCCCCCCC", 500, 1.0, 5, bile_salt_export_inhibitor=True)
        assert r_yes.dili_score - r_no.dili_score == 20

    def test_reactive_metabolite_risk_triggered(self):
        # logP>2, mw<400, N in smiles
        r = score_dili("X", "CCN", 350, 3.0, 5)
        assert r.reactive_metabolite_risk is True

    def test_reactive_metabolite_risk_not_triggered(self):
        # logP<2
        r = score_dili("X", "CCN", 350, 1.0, 5)
        assert r.reactive_metabolite_risk is False

    def test_reactive_metabolite_risk_sulfur(self):
        # S in smiles
        r = score_dili("X", "CCS", 350, 3.0, 5)
        assert r.reactive_metabolite_risk is True

    def test_reactive_metabolite_no_n_or_s(self):
        r = score_dili("X", "CCCCCCCC", 350, 3.0, 5)
        assert r.reactive_metabolite_risk is False


# ── Category tests ────────────────────────────────────────────────────


class TestCategories:
    def test_no_concern_category(self):
        r = score_dili("X", "CCCCCCCC", 500, 1.0, 5)
        assert r.dili_category == "no_concern"

    def test_high_category(self):
        r = score_dili(
            "X", "CCNS", 250, 4.0, 200,
            is_mitochondrial_toxin=True,
            bile_salt_export_inhibitor=True,
        )
        assert r.dili_category == "high"

    def test_dose_dependency_high_dose(self):
        r = score_dili("X", "CCO", 300, 1.0, 200)
        assert r.dose_dependency == "high_dose_only"

    def test_dose_dependency_all_doses(self):
        r = score_dili("X", "CCO", 300, 1.0, 50)
        assert r.dose_dependency == "all_doses"

    def test_mitochondrial_liability_flag(self):
        r = score_dili("X", "CCO", 300, 1.0, 50, is_mitochondrial_toxin=True)
        assert r.mitochondrial_liability is True

    def test_bile_salt_inhibition_flag(self):
        r = score_dili("X", "CCO", 300, 1.0, 50, bile_salt_export_inhibitor=True)
        assert r.bile_salt_inhibition is True


# ── Screen tests ──────────────────────────────────────────────────────


class TestScreen:
    def test_screen_sorts_by_score_desc(self):
        compounds = [
            {"compound_name": "Safe", "smiles": "CCCCCCCC", "mw": 500, "logP": 1.0, "daily_dose_mg": 5},
            {"compound_name": "Risky", "smiles": "CCNS", "mw": 250, "logP": 4.0, "daily_dose_mg": 200,
             "is_mitochondrial_toxin": True},
        ]
        results = screen_dili(compounds)
        assert results[0].compound_name == "Risky"
        assert results[0].dili_score >= results[1].dili_score

    def test_screen_returns_all(self):
        compounds = [
            {"compound_name": "A", "smiles": "CCO", "mw": 300, "logP": 1.0, "daily_dose_mg": 50},
            {"compound_name": "B", "smiles": "CCO", "mw": 300, "logP": 1.0, "daily_dose_mg": 50},
            {"compound_name": "C", "smiles": "CCO", "mw": 300, "logP": 1.0, "daily_dose_mg": 50},
        ]
        assert len(screen_dili(compounds)) == 3

    def test_screen_empty_raises(self):
        with pytest.raises(ValueError, match="compounds"):
            screen_dili([])
