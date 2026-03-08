"""Tests for Phase 623 — P-glycoprotein Drug Interaction Prediction."""

from dataclasses import fields

import pytest

from omega_pbpk.prediction.pgp_interaction_p623 import (
    PgpInteractionResult,
    predict_pgp_interaction,
    screen_pgp_substrates,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _digoxin():
    """Digoxin: classic P-gp substrate. MW=781, logP=1.26, PSA=202, HBD=5, HBA=12."""
    return predict_pgp_interaction(
        drug_name="Digoxin",
        mw_Da=781.0,
        logP=1.26,
        psa_A2=202.0,
        n_hbd=5,
        n_hba=12,
        concentration_uM=1.0,
    )


def _verapamil():
    """Verapamil: P-gp inhibitor. MW=455, logP=3.79, PSA=63, HBD=0, HBA=7."""
    return predict_pgp_interaction(
        drug_name="Verapamil",
        mw_Da=455.0,
        logP=3.79,
        psa_A2=63.0,
        n_hbd=0,
        n_hba=7,
    )


def _aspirin():
    """Aspirin: small non-P-gp substrate. MW=180, logP=1.19, PSA=63, HBD=1, HBA=4."""
    return predict_pgp_interaction(
        drug_name="Aspirin",
        mw_Da=180.0,
        logP=1.19,
        psa_A2=63.0,
        n_hbd=1,
        n_hba=4,
    )


# ---------------------------------------------------------------------------
# Return type and frozen dataclass
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_pgp_interaction_result(self):
        result = _digoxin()
        assert isinstance(result, PgpInteractionResult)

    def test_frozen_dataclass_immutable(self):
        result = _digoxin()
        with pytest.raises((AttributeError, TypeError)):
            result.drug_name = "Modified"  # type: ignore[misc]

    def test_all_fields_present(self):
        result = _digoxin()
        expected_fields = {
            "drug_name",
            "is_pgp_substrate",
            "is_pgp_inhibitor",
            "pgp_substrate_score",
            "pgp_inhibitor_score",
            "efflux_ratio",
            "fa_without_pgp",
            "fa_with_pgp",
            "fa_ratio",
            "ddi_auc_change_pct",
            "bbb_pgp_effect",
            "intestinal_pgp_effect",
            "notes",
        }
        actual_fields = {f.name for f in fields(result)}
        assert expected_fields == actual_fields

    def test_drug_name_preserved(self):
        result = _digoxin()
        assert result.drug_name == "Digoxin"


# ---------------------------------------------------------------------------
# Substrate classification
# ---------------------------------------------------------------------------


class TestSubstrateClassification:
    def test_digoxin_is_substrate(self):
        result = _digoxin()
        assert result.is_pgp_substrate is True

    def test_aspirin_not_substrate(self):
        result = _aspirin()
        assert result.is_pgp_substrate is False

    def test_substrate_score_range(self):
        result = _digoxin()
        assert 0.0 <= result.pgp_substrate_score <= 1.0

    def test_digoxin_high_substrate_score(self):
        result = _digoxin()
        assert result.pgp_substrate_score > 0.5

    def test_aspirin_low_substrate_score(self):
        result = _aspirin()
        assert result.pgp_substrate_score <= 0.5

    def test_large_mw_increases_substrate_score(self):
        low_mw = predict_pgp_interaction("A", mw_Da=200.0, logP=2.0, psa_A2=50.0, n_hbd=1, n_hba=3)
        high_mw = predict_pgp_interaction("B", mw_Da=700.0, logP=2.0, psa_A2=50.0, n_hbd=1, n_hba=3)
        assert high_mw.pgp_substrate_score > low_mw.pgp_substrate_score


# ---------------------------------------------------------------------------
# Inhibitor classification
# ---------------------------------------------------------------------------


class TestInhibitorClassification:
    def test_verapamil_is_inhibitor(self):
        result = _verapamil()
        assert result.is_pgp_inhibitor is True

    def test_inhibitor_score_range(self):
        result = _verapamil()
        assert 0.0 <= result.pgp_inhibitor_score <= 1.0

    def test_high_logp_increases_inhibitor_score(self):
        low_lp = predict_pgp_interaction("X", mw_Da=450.0, logP=1.5, psa_A2=60.0, n_hbd=1, n_hba=4)
        high_lp = predict_pgp_interaction("Y", mw_Da=450.0, logP=4.5, psa_A2=60.0, n_hbd=1, n_hba=4)
        assert high_lp.pgp_inhibitor_score > low_lp.pgp_inhibitor_score

    def test_inhibitor_score_max_1(self):
        result = predict_pgp_interaction(
            "Z", mw_Da=2000.0, logP=10.0, psa_A2=20.0, n_hbd=0, n_hba=1
        )
        assert result.pgp_inhibitor_score <= 1.0


# ---------------------------------------------------------------------------
# Efflux ratio
# ---------------------------------------------------------------------------


class TestEffluxRatio:
    def test_efflux_ratio_minimum_is_1(self):
        result = _aspirin()
        assert result.efflux_ratio >= 1.0

    def test_digoxin_efflux_ratio_above_2(self):
        result = _digoxin()
        assert result.efflux_ratio > 2.0

    def test_efflux_ratio_formula(self):
        result = _digoxin()
        expected = 1.0 + result.pgp_substrate_score * 4.0
        assert abs(result.efflux_ratio - expected) < 1e-9

    def test_efflux_ratio_max_is_5(self):
        result = predict_pgp_interaction(
            "Max", mw_Da=2000.0, logP=5.0, psa_A2=300.0, n_hbd=5, n_hba=10
        )
        assert result.efflux_ratio <= 5.0


# ---------------------------------------------------------------------------
# Fraction absorbed
# ---------------------------------------------------------------------------


class TestFractionAbsorbed:
    def test_fa_without_pgp_in_range(self):
        result = _digoxin()
        assert 0.1 <= result.fa_without_pgp <= 0.95

    def test_fa_with_pgp_le_fa_without_pgp(self):
        result = _digoxin()
        assert result.fa_with_pgp <= result.fa_without_pgp

    def test_fa_with_pgp_minimum(self):
        result = _digoxin()
        assert result.fa_with_pgp >= 0.01

    def test_fa_ratio_ge_1(self):
        result = _digoxin()
        assert result.fa_ratio >= 1.0

    def test_aspirin_fa_ratio_close_to_1(self):
        result = _aspirin()
        # Low substrate score → small reduction → fa_ratio close to 1
        assert result.fa_ratio < 2.0


# ---------------------------------------------------------------------------
# DDI AUC change
# ---------------------------------------------------------------------------


class TestDDIAUCChange:
    def test_ddi_auc_change_nonnegative(self):
        result = _digoxin()
        assert result.ddi_auc_change_pct >= 0.0

    def test_ddi_auc_change_larger_for_strong_substrate(self):
        strong = _digoxin()
        weak = _aspirin()
        assert strong.ddi_auc_change_pct > weak.ddi_auc_change_pct


# ---------------------------------------------------------------------------
# BBB and intestinal effects
# ---------------------------------------------------------------------------


class TestPgpEffects:
    def test_bbb_effect_valid_values(self):
        result = _digoxin()
        assert result.bbb_pgp_effect in ("significant", "moderate", "minimal")

    def test_intestinal_effect_valid_values(self):
        result = _digoxin()
        assert result.intestinal_pgp_effect in ("significant", "moderate", "minimal")

    def test_strong_substrate_significant_bbb(self):
        # Digoxin has substrate_score > 0.7 if score is computed correctly
        result = _digoxin()
        if result.pgp_substrate_score > 0.7:
            assert result.bbb_pgp_effect == "significant"

    def test_minimal_effect_for_weak_substrate(self):
        result = _aspirin()
        if result.pgp_substrate_score <= 0.4:
            assert result.bbb_pgp_effect == "minimal"

    def test_bbb_equals_intestinal_effect(self):
        result = _digoxin()
        assert result.bbb_pgp_effect == result.intestinal_pgp_effect


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


class TestNotes:
    def test_notes_is_string(self):
        result = _digoxin()
        assert isinstance(result.notes, str)

    def test_notes_nonempty(self):
        result = _digoxin()
        assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Screening
# ---------------------------------------------------------------------------


class TestScreening:
    def _make_library(self):
        return [
            {
                "drug_name": "DrugA",
                "mw_Da": 750.0,
                "logP": 2.5,
                "psa_A2": 180.0,
                "n_hbd": 4,
                "n_hba": 10,
            },
            {
                "drug_name": "DrugB",
                "mw_Da": 200.0,
                "logP": 1.5,
                "psa_A2": 40.0,
                "n_hbd": 1,
                "n_hba": 2,
            },
            {
                "drug_name": "DrugC",
                "mw_Da": 550.0,
                "logP": 3.0,
                "psa_A2": 100.0,
                "n_hbd": 3,
                "n_hba": 7,
            },
        ]

    def test_screen_returns_list(self):
        result = screen_pgp_substrates(self._make_library())
        assert isinstance(result, list)

    def test_screen_length(self):
        lib = self._make_library()
        result = screen_pgp_substrates(lib)
        assert len(result) == len(lib)

    def test_screen_sorted_descending(self):
        result = screen_pgp_substrates(self._make_library())
        scores = [r.pgp_substrate_score for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_screen_all_pgp_interaction_results(self):
        result = screen_pgp_substrates(self._make_library())
        assert all(isinstance(r, PgpInteractionResult) for r in result)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_zero_mw_raises(self):
        with pytest.raises(ValueError, match="mw_Da"):
            predict_pgp_interaction("X", mw_Da=0.0, logP=2.0, psa_A2=50.0, n_hbd=1, n_hba=3)

    def test_negative_mw_raises(self):
        with pytest.raises(ValueError, match="mw_Da"):
            predict_pgp_interaction("X", mw_Da=-100.0, logP=2.0, psa_A2=50.0, n_hbd=1, n_hba=3)

    def test_negative_psa_raises(self):
        with pytest.raises(ValueError, match="psa_A2"):
            predict_pgp_interaction("X", mw_Da=300.0, logP=2.0, psa_A2=-10.0, n_hbd=1, n_hba=3)

    def test_negative_hbd_raises(self):
        with pytest.raises(ValueError, match="n_hbd"):
            predict_pgp_interaction("X", mw_Da=300.0, logP=2.0, psa_A2=50.0, n_hbd=-1, n_hba=3)

    def test_negative_hba_raises(self):
        with pytest.raises(ValueError, match="n_hba"):
            predict_pgp_interaction("X", mw_Da=300.0, logP=2.0, psa_A2=50.0, n_hbd=1, n_hba=-1)

    def test_zero_concentration_raises(self):
        with pytest.raises(ValueError, match="concentration_uM"):
            predict_pgp_interaction(
                "X", mw_Da=300.0, logP=2.0, psa_A2=50.0, n_hbd=1, n_hba=3, concentration_uM=0.0
            )
