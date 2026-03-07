"""Tests for Phase 291: Drug-induced phospholipidosis risk prediction."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.risk.phospholipidosis import (
    PhospholipidosisResult,
    assess_phospholipidosis_risk,
    compute_cad_score,
    screen_phospholipidosis,
)

# ---------------------------------------------------------------------------
# compute_cad_score tests
# ---------------------------------------------------------------------------


class TestComputeCadScore:
    def test_non_cad_no_nitrogen_low_logp_low_pka(self):
        # No nitrogen, logP <= 2, pKa <= 6 → score = 0
        score = compute_cad_score("CCC", logP=1.0, pka_basic=4.0, mw=300.0)
        assert score == pytest.approx(0.0)

    def test_nitrogen_flag_lowercase_n(self):
        # 'n' in SMILES → nitrogen_flag=1.0 → contributes 0.5
        score = compute_cad_score("CCn1cccc1", logP=1.0, pka_basic=4.0, mw=300.0)
        assert score == pytest.approx(0.5)

    def test_nitrogen_flag_uppercase_n(self):
        # 'N' in SMILES → 'n' in smiles.lower() → contributes 0.5
        score = compute_cad_score("CCN", logP=1.0, pka_basic=4.0, mw=300.0)
        assert score == pytest.approx(0.5)

    def test_logp_contribution(self):
        # logP=4, no nitrogen, pKa=4 → (4-2)*0.3 = 0.6
        score = compute_cad_score("CCC", logP=4.0, pka_basic=4.0, mw=300.0)
        assert score == pytest.approx(0.6)

    def test_pka_contribution(self):
        # pKa=8, no nitrogen, logP=1 → (8-6)*0.2 = 0.4
        score = compute_cad_score("CCC", logP=1.0, pka_basic=8.0, mw=300.0)
        assert score == pytest.approx(0.4)

    def test_all_contributions(self):
        # logP=4, pKa=8, nitrogen → 0.6 + 0.4 + 0.5 = 1.5
        score = compute_cad_score("CCN", logP=4.0, pka_basic=8.0, mw=300.0)
        assert score == pytest.approx(1.5)

    def test_clamp_upper_bound(self):
        # Extreme values → clamped to 3.0
        score = compute_cad_score("CCN", logP=20.0, pka_basic=20.0, mw=300.0)
        assert score == pytest.approx(3.0)

    def test_clamp_lower_bound(self):
        # No contributions → 0.0
        score = compute_cad_score("CCC", logP=-5.0, pka_basic=0.0, mw=300.0)
        assert score == pytest.approx(0.0)

    def test_negative_logp_allowed(self):
        # logP can be any float; negative logP → no logP contribution
        score = compute_cad_score("CCC", logP=-2.0, pka_basic=4.0, mw=300.0)
        assert score == pytest.approx(0.0)

    def test_invalid_pka_raises(self):
        with pytest.raises(ValueError, match="pka_basic"):
            compute_cad_score("CCC", logP=2.0, pka_basic=-1.0, mw=300.0)

    def test_invalid_mw_raises(self):
        with pytest.raises(ValueError, match="mw"):
            compute_cad_score("CCC", logP=2.0, pka_basic=6.0, mw=0.0)

    def test_mw_not_used_in_score(self):
        # mw does not affect the score (only used for validation)
        s1 = compute_cad_score("CCC", logP=3.0, pka_basic=7.0, mw=200.0)
        s2 = compute_cad_score("CCC", logP=3.0, pka_basic=7.0, mw=800.0)
        assert s1 == pytest.approx(s2)

    def test_exact_boundary_logp_equals_2(self):
        # logP exactly 2 → no logP contribution
        score = compute_cad_score("CCC", logP=2.0, pka_basic=4.0, mw=300.0)
        assert score == pytest.approx(0.0)

    def test_exact_boundary_pka_equals_6(self):
        # pKa exactly 6 → no pKa contribution
        score = compute_cad_score("CCC", logP=1.0, pka_basic=6.0, mw=300.0)
        assert score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# assess_phospholipidosis_risk tests
# ---------------------------------------------------------------------------


class TestAssessPhospholipidosisRisk:
    def _make_result(self, **kwargs):
        defaults = dict(
            compound_name="TestDrug",
            smiles="CCN",
            logP=3.0,
            pka_basic=7.0,
            mw=300.0,
            daily_dose_mg=100.0,
        )
        defaults.update(kwargs)
        return assess_phospholipidosis_risk(**defaults)

    def test_returns_correct_type(self):
        result = self._make_result()
        assert isinstance(result, PhospholipidosisResult)

    def test_frozen_dataclass(self):
        result = self._make_result()
        with pytest.raises((AttributeError, TypeError)):
            result.risk_category = "new"  # type: ignore[misc]

    def test_low_risk(self):
        # No nitrogen, low logP, low pKa, low dose → low risk
        result = assess_phospholipidosis_risk(
            "LowRisk", "CCC", logP=1.0, pka_basic=4.0, mw=200.0, daily_dose_mg=1.0
        )
        assert result.risk_category == "low"
        assert result.dipl_score < 0.5

    def test_moderate_risk(self):
        # Moderate CAD score at reference dose
        result = assess_phospholipidosis_risk(
            "Moderate", "CCN", logP=3.0, pka_basic=7.0, mw=300.0, daily_dose_mg=100.0
        )
        # cad_score = (3-2)*0.3 + (7-6)*0.2 + 0.5 = 0.3+0.2+0.5 = 1.0
        # dose_factor = log10(100)/log10(100) = 1.0
        # dipl_score = 1.0 -> moderate (0.5–1.5)
        assert result.risk_category == "moderate"

    def test_high_risk(self):
        # Strong CAD properties, high dose
        result = assess_phospholipidosis_risk(
            "HighRisk", "CCN", logP=5.0, pka_basic=9.0, mw=400.0, daily_dose_mg=1000.0
        )
        assert result.risk_category == "high"

    def test_cad_score_correct(self):
        result = assess_phospholipidosis_risk(
            "Test", "CCN", logP=4.0, pka_basic=8.0, mw=300.0, daily_dose_mg=100.0
        )
        expected_cad = compute_cad_score("CCN", logP=4.0, pka_basic=8.0, mw=300.0)
        assert result.cad_score == pytest.approx(expected_cad)

    def test_dipl_score_formula(self):
        result = self._make_result(logP=3.0, pka_basic=7.0, daily_dose_mg=100.0)
        # dose_factor = log10(100)/log10(100) = 1.0
        assert result.dipl_score == pytest.approx(result.cad_score * 1.0)

    def test_ec50_formula(self):
        result = self._make_result()
        expected_ec50 = 10.0 ** (3.0 - result.cad_score)
        assert result.ec50_lysosome_nM == pytest.approx(expected_ec50)

    def test_ec50_non_cad_is_1000nM(self):
        # cad_score=0 → EC50 = 10^3 = 1000 nM
        result = assess_phospholipidosis_risk(
            "NonCAD", "CCC", logP=1.0, pka_basic=4.0, mw=200.0, daily_dose_mg=10.0
        )
        assert result.ec50_lysosome_nM == pytest.approx(1000.0)

    def test_dose_factor_below_100mg(self):
        # dose_factor < 1 for daily_dose < 100 mg
        result = assess_phospholipidosis_risk(
            "LowDose", "CCN", logP=3.0, pka_basic=7.0, mw=300.0, daily_dose_mg=10.0
        )
        expected_factor = math.log10(10.0) / math.log10(100.0)  # = 0.5
        assert result.dipl_score == pytest.approx(result.cad_score * expected_factor)

    def test_invalid_pka_raises(self):
        with pytest.raises(ValueError, match="pka_basic"):
            assess_phospholipidosis_risk("X", "CCC", 2.0, -1.0, 300.0, 50.0)

    def test_invalid_mw_raises(self):
        with pytest.raises(ValueError, match="mw"):
            assess_phospholipidosis_risk("X", "CCC", 2.0, 6.0, 0.0, 50.0)

    def test_invalid_dose_raises(self):
        with pytest.raises(ValueError, match="daily_dose_mg"):
            assess_phospholipidosis_risk("X", "CCC", 2.0, 6.0, 300.0, 0.0)

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="daily_dose_mg"):
            assess_phospholipidosis_risk("X", "CCC", 2.0, 6.0, 300.0, -10.0)

    def test_notes_not_empty(self):
        result = self._make_result()
        assert isinstance(result.notes, str)
        assert len(result.notes) > 0

    def test_no_risk_factors_note(self):
        result = assess_phospholipidosis_risk(
            "Clean", "CCC", logP=1.0, pka_basic=4.0, mw=200.0, daily_dose_mg=10.0
        )
        assert "No significant" in result.notes


# ---------------------------------------------------------------------------
# screen_phospholipidosis tests
# ---------------------------------------------------------------------------


class TestScreenPhospholipidosis:
    def _make_compound(self, name, smiles, logP, pka_basic, mw, daily_dose_mg):
        return {
            "compound_name": name,
            "smiles": smiles,
            "logP": logP,
            "pka_basic": pka_basic,
            "mw": mw,
            "daily_dose_mg": daily_dose_mg,
        }

    def test_returns_list(self):
        compounds = [self._make_compound("A", "CCC", 1.0, 4.0, 200.0, 10.0)]
        results = screen_phospholipidosis(compounds)
        assert isinstance(results, list)

    def test_sorted_by_dipl_score_descending(self):
        compounds = [
            self._make_compound("Low", "CCC", 1.0, 4.0, 200.0, 1.0),
            self._make_compound("High", "CCN", 5.0, 9.0, 400.0, 500.0),
            self._make_compound("Mid", "CCN", 3.0, 7.0, 300.0, 100.0),
        ]
        results = screen_phospholipidosis(compounds)
        scores = [r.dipl_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_list(self):
        results = screen_phospholipidosis([])
        assert results == []

    def test_single_compound(self):
        compounds = [self._make_compound("Only", "CCN", 3.0, 7.0, 300.0, 100.0)]
        results = screen_phospholipidosis(compounds)
        assert len(results) == 1

    def test_all_results_are_phospholipidosis_result(self):
        compounds = [
            self._make_compound("A", "CCC", 2.0, 5.0, 250.0, 50.0),
            self._make_compound("B", "CCN", 4.0, 8.0, 350.0, 200.0),
        ]
        results = screen_phospholipidosis(compounds)
        for r in results:
            assert isinstance(r, PhospholipidosisResult)
