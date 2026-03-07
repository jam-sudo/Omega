"""Tests for src/omega_pbpk/prediction/synthetic_accessibility.py (Phase 446)."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.synthetic_accessibility import (
    DrugLikenessResult,
    _calc_qed,
    _norm_score,
    fragment_analysis,
    predict_drug_likeness,
)

# ---------------------------------------------------------------------------
# Helper defaults — aspirin-like compound (MW=180, logP=1.2, PSA=63, HBD=1, HBA=4)
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    drug_name="Aspirin",
    mw=180.0,
    logP=1.2,
    psa=63.0,
    n_hbd=1,
    n_hba=4,
    n_rot_bonds=3,
    n_rings=1,
    n_stereocenters=0,
    n_violations_allowed=1,
)


def run(**overrides):
    kwargs = {**DEFAULTS, **overrides}
    return predict_drug_likeness(**kwargs)


# ---------------------------------------------------------------------------
# _norm_score tests
# ---------------------------------------------------------------------------

class TestNormScore:
    def test_at_optimal_is_1(self):
        score = _norm_score(330.0, 330.0, 120.0)
        assert score == pytest.approx(1.0)

    def test_far_from_optimal_is_low(self):
        score = _norm_score(1000.0, 330.0, 120.0)
        assert score < 0.01

    def test_symmetric(self):
        a = _norm_score(330.0 + 100, 330.0, 120.0)
        b = _norm_score(330.0 - 100, 330.0, 120.0)
        assert a == pytest.approx(b)


# ---------------------------------------------------------------------------
# _calc_qed tests
# ---------------------------------------------------------------------------

class TestCalcQED:
    def test_returns_value_in_0_1(self):
        q = _calc_qed(300, 2.5, 80, 1, 4, 4)
        assert 0.0 <= q <= 1.0

    def test_optimal_params_give_high_qed(self):
        q = _calc_qed(330, 2.5, 80, 1, 4, 4)
        assert q > 0.5

    def test_poor_params_give_low_qed(self):
        q = _calc_qed(700, 8.0, 200, 8, 12, 15)
        assert q < 0.3


# ---------------------------------------------------------------------------
# predict_drug_likeness — basic correctness
# ---------------------------------------------------------------------------

class TestPredictDrugLikeness:
    def test_returns_drug_likeness_result(self):
        result = run()
        assert isinstance(result, DrugLikenessResult)

    def test_aspirin_lipinski_pass(self):
        result = run()
        assert result.lipinski_pass is True

    def test_aspirin_veber_pass(self):
        result = run()
        assert result.veber_pass is True

    def test_aspirin_egan_pass(self):
        result = run()
        assert result.egan_pass is True

    def test_aspirin_lead_like(self):
        result = run()
        assert result.lead_like is True

    def test_aspirin_zero_ro5_violations(self):
        result = run()
        assert result.n_ro5_violations == 0

    def test_lipinski_fail_high_mw(self):
        result = run(mw=700, logP=6, n_hbd=6, n_hba=11, n_violations_allowed=0)
        assert result.lipinski_pass is False

    def test_veber_fail_high_psa(self):
        result = run(psa=150)
        assert result.veber_pass is False

    def test_veber_fail_high_rot_bonds(self):
        result = run(n_rot_bonds=12)
        assert result.veber_pass is False

    def test_egan_fail_high_psa(self):
        result = run(psa=140)
        assert result.egan_pass is False

    def test_egan_fail_high_logP(self):
        result = run(logP=6.0)
        assert result.egan_pass is False

    def test_lead_like_fail_high_mw(self):
        result = run(mw=400)
        assert result.lead_like is False

    def test_lead_like_fail_high_logP(self):
        result = run(logP=4.0)
        assert result.lead_like is False

    def test_sa_score_increases_with_stereocenters(self):
        r1 = run(n_stereocenters=0)
        r2 = run(n_stereocenters=5)
        assert r2.sa_score > r1.sa_score

    def test_sa_score_increases_with_rings(self):
        r1 = run(n_rings=1)
        r2 = run(n_rings=6)
        assert r2.sa_score > r1.sa_score

    def test_sa_score_clamped_1_to_10(self):
        r = run(n_stereocenters=20, n_rings=20)
        assert 1.0 <= r.sa_score <= 10.0

    def test_sa_score_minimum_at_zero(self):
        r = run(n_stereocenters=0, n_rings=0)
        assert r.sa_score == pytest.approx(1.0)

    def test_bioavailability_score_6_for_no_violations(self):
        result = run(mw=180, logP=1.2, psa=63, n_rot_bonds=3)
        assert result.bioavailability_score == 6

    def test_bioavailability_score_range_2_to_6(self):
        result = run()
        assert 2 <= result.bioavailability_score <= 6

    def test_qed_score_in_0_1(self):
        result = run()
        assert 0.0 <= result.qed_score <= 1.0

    def test_good_drug_likeness_for_aspirin(self):
        result = run()
        assert result.overall_drug_likeness in {"good", "excellent"}

    def test_poor_drug_likeness_for_ugly_molecule(self):
        result = run(
            mw=700, logP=7.0, psa=200, n_hbd=8, n_hba=12,
            n_rot_bonds=15, n_rings=8, n_stereocenters=10,
            n_violations_allowed=0
        )
        assert result.overall_drug_likeness in {"poor", "acceptable"}

    def test_notes_is_str(self):
        result = run()
        assert isinstance(result.notes, str)
        assert len(result.notes) > 0

    def test_ro5_violations_count_four_properties(self):
        # All 4 RO5 rules violated
        result = run(mw=600, logP=6, n_hbd=7, n_hba=12, n_violations_allowed=5)
        assert result.n_ro5_violations == 4

    def test_violations_allowed_zero_fails_with_one_violation(self):
        result = run(mw=600, n_violations_allowed=0)
        assert result.lipinski_pass is False

    def test_violations_allowed_1_passes_with_one_violation(self):
        result = run(mw=600, n_violations_allowed=1)
        # mw=600 causes 1 violation → still passes with tolerance=1
        assert result.lipinski_pass is True

    def test_result_is_frozen(self):
        result = run()
        with pytest.raises((AttributeError, TypeError)):
            result.mw = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# predict_drug_likeness — validation errors
# ---------------------------------------------------------------------------

class TestPredictDrugLikenessValidation:
    def test_empty_drug_name(self):
        with pytest.raises(ValueError, match="drug_name"):
            run(drug_name="")

    def test_zero_mw(self):
        with pytest.raises(ValueError, match="mw"):
            run(mw=0)

    def test_negative_mw(self):
        with pytest.raises(ValueError, match="mw"):
            run(mw=-1)

    def test_negative_psa(self):
        with pytest.raises(ValueError, match="psa"):
            run(psa=-1)

    def test_negative_hbd(self):
        with pytest.raises(ValueError, match="n_hbd"):
            run(n_hbd=-1)

    def test_negative_hba(self):
        with pytest.raises(ValueError, match="n_hba"):
            run(n_hba=-1)

    def test_negative_rot_bonds(self):
        with pytest.raises(ValueError, match="n_rot_bonds"):
            run(n_rot_bonds=-1)

    def test_negative_rings(self):
        with pytest.raises(ValueError, match="n_rings"):
            run(n_rings=-1)

    def test_negative_stereocenters(self):
        with pytest.raises(ValueError, match="n_stereocenters"):
            run(n_stereocenters=-1)

    def test_negative_violations_allowed(self):
        with pytest.raises(ValueError, match="n_violations_allowed"):
            run(n_violations_allowed=-1)


# ---------------------------------------------------------------------------
# fragment_analysis tests
# ---------------------------------------------------------------------------

class TestFragmentAnalysis:
    def test_returns_dict(self):
        result = fragment_analysis("Frag", 1, 0, 2, 150, 1.5)
        assert isinstance(result, dict)

    def test_fragment_like_small_molecule(self):
        result = fragment_analysis("Small", 1, 0, 1, 180, 1.5)
        assert result["fragment_likeness"] == "fragment-like"

    def test_lead_like_medium_molecule(self):
        result = fragment_analysis("Lead", 2, 1, 3, 320, 3.0)
        assert result["fragment_likeness"] == "lead-like"

    def test_drug_like_normal_drug(self):
        result = fragment_analysis("Drug", 2, 1, 5, 450, 3.5)
        assert result["fragment_likeness"] == "drug-like"

    def test_complex_large_molecule(self):
        result = fragment_analysis("Big", 4, 2, 10, 700, 5.0)
        assert result["fragment_likeness"] == "complex"

    def test_ring_count_class_no_rings(self):
        result = fragment_analysis("Linear", 0, 0, 2, 150, 1.0)
        assert result["ring_count_class"] == "no_rings"

    def test_ring_count_class_monocyclic(self):
        result = fragment_analysis("Mono", 1, 0, 2, 150, 1.5)
        assert result["ring_count_class"] == "monocyclic"

    def test_ring_count_class_bicyclic(self):
        result = fragment_analysis("Bi", 1, 1, 3, 200, 2.0)
        assert result["ring_count_class"] == "bicyclic"

    def test_ring_count_class_polycyclic(self):
        result = fragment_analysis("Poly", 2, 1, 4, 300, 2.5)
        assert result["ring_count_class"] == "polycyclic"

    def test_heteroatom_fraction_between_0_and_1(self):
        result = fragment_analysis("X", 1, 0, 3, 200, 2.0)
        assert 0.0 <= result["heteroatom_fraction"] <= 1.0

    def test_total_rings_field(self):
        result = fragment_analysis("X", 2, 3, 4, 300, 2.5)
        assert result["total_rings"] == 5

    def test_drug_name_preserved(self):
        result = fragment_analysis("MyDrug", 1, 0, 2, 200, 2.0)
        assert result["drug_name"] == "MyDrug"

    def test_invalid_drug_name(self):
        with pytest.raises(ValueError, match="drug_name"):
            fragment_analysis("", 1, 0, 2, 200, 2.0)

    def test_negative_aromatic_rings(self):
        with pytest.raises(ValueError, match="n_aromatic_rings"):
            fragment_analysis("X", -1, 0, 2, 200, 2.0)

    def test_negative_aliphatic_rings(self):
        with pytest.raises(ValueError, match="n_aliphatic_rings"):
            fragment_analysis("X", 1, -1, 2, 200, 2.0)

    def test_negative_heteroatoms(self):
        with pytest.raises(ValueError, match="n_heteroatoms"):
            fragment_analysis("X", 1, 0, -1, 200, 2.0)

    def test_zero_mw(self):
        with pytest.raises(ValueError, match="mw"):
            fragment_analysis("X", 1, 0, 2, 0, 2.0)
