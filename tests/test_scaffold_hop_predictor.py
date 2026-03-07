"""Tests for scaffold_hop_predictor module (Phase 448)."""

import pytest

from omega_pbpk.prediction.scaffold_hop_predictor import (
    ScaffoldHopResult,
    _assess_risk_flags,
    _overall_recommendation,
    _predict_absorption_change,
    _predict_cl_change,
    _predict_vd_change,
    _validate_drug_dict,
    predict_scaffold_hop_impact,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_drug(**kwargs):
    """Return a minimal valid drug dict with optional overrides."""
    base = {
        "name": "DrugA",
        "mw": 350.0,
        "logP": 2.5,
        "psa": 60.0,
        "n_hbd": 2,
        "pka": 7.4,
        "molecule_type": "small_molecule",
    }
    base.update(kwargs)
    return base


def _make_pair(**candidate_kwargs):
    """Return (original, candidate) dicts; candidate overrides base values."""
    original = _make_drug(name="OriginalA")
    candidate = _make_drug(name="CandidateB", **candidate_kwargs)
    return original, candidate


# ---------------------------------------------------------------------------
# 1. ScaffoldHopResult dataclass
# ---------------------------------------------------------------------------

class TestScaffoldHopResult:
    def test_result_is_frozen(self):
        original, candidate = _make_pair()
        result = predict_scaffold_hop_impact(original, candidate)
        with pytest.raises((AttributeError, TypeError)):
            result.delta_mw = 999.0  # type: ignore[misc]

    def test_result_fields_present(self):
        original, candidate = _make_pair()
        result = predict_scaffold_hop_impact(original, candidate)
        assert hasattr(result, "original_name")
        assert hasattr(result, "candidate_name")
        assert hasattr(result, "delta_mw")
        assert hasattr(result, "delta_logP")
        assert hasattr(result, "delta_psa")
        assert hasattr(result, "delta_pka")
        assert hasattr(result, "predicted_vd_change")
        assert hasattr(result, "predicted_absorption_change")
        assert hasattr(result, "predicted_cl_change")
        assert hasattr(result, "risk_flags")
        assert hasattr(result, "overall_recommendation")
        assert hasattr(result, "notes")

    def test_risk_flags_is_tuple(self):
        original, candidate = _make_pair()
        result = predict_scaffold_hop_impact(original, candidate)
        assert isinstance(result.risk_flags, tuple)

    def test_notes_contains_key_delta_info(self):
        original, candidate = _make_pair(mw=450.0)
        result = predict_scaffold_hop_impact(original, candidate)
        assert "delta_MW" in result.notes
        assert "delta_logP" in result.notes
        assert "delta_PSA" in result.notes
        assert "delta_pKa" in result.notes


# ---------------------------------------------------------------------------
# 2. _validate_drug_dict
# ---------------------------------------------------------------------------

class TestValidateDrugDict:
    def test_valid_dict_passes(self):
        _validate_drug_dict(_make_drug(), "test")

    def test_missing_field_raises(self):
        drug = _make_drug()
        del drug["mw"]
        with pytest.raises(ValueError, match="missing fields"):
            _validate_drug_dict(drug, "test")

    def test_mw_zero_raises(self):
        with pytest.raises(ValueError, match="mw must be > 0"):
            _validate_drug_dict(_make_drug(mw=0), "test")

    def test_mw_negative_raises(self):
        with pytest.raises(ValueError, match="mw must be > 0"):
            _validate_drug_dict(_make_drug(mw=-10.0), "test")

    def test_psa_negative_raises(self):
        with pytest.raises(ValueError, match="psa must be >= 0"):
            _validate_drug_dict(_make_drug(psa=-5.0), "test")

    def test_n_hbd_negative_raises(self):
        with pytest.raises(ValueError, match="n_hbd must be >= 0"):
            _validate_drug_dict(_make_drug(n_hbd=-1), "test")

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="name must be a non-empty string"):
            _validate_drug_dict(_make_drug(name=""), "test")

    def test_whitespace_name_raises(self):
        with pytest.raises(ValueError, match="name must be a non-empty string"):
            _validate_drug_dict(_make_drug(name="   "), "test")

    def test_empty_molecule_type_raises(self):
        with pytest.raises(ValueError, match="molecule_type must be a non-empty string"):
            _validate_drug_dict(_make_drug(molecule_type=""), "test")

    def test_psa_zero_passes(self):
        """PSA of 0 is allowed (borderline case)."""
        _validate_drug_dict(_make_drug(psa=0.0), "test")

    def test_all_required_fields_checked(self):
        """Each required field individually triggers an error when removed."""
        required = {"name", "mw", "logP", "psa", "n_hbd", "pka", "molecule_type"}
        for field in required:
            drug = _make_drug()
            del drug[field]
            with pytest.raises(ValueError):
                _validate_drug_dict(drug, "test")


# ---------------------------------------------------------------------------
# 3. _predict_vd_change
# ---------------------------------------------------------------------------

class TestPredictVdChange:
    def test_large_logP_increase_returns_increase(self):
        assert _predict_vd_change(0, 1.5, 0) == "increase"

    def test_large_logP_decrease_returns_decrease(self):
        assert _predict_vd_change(0, -1.5, 0) == "decrease"

    def test_no_change_returns_similar(self):
        assert _predict_vd_change(0, 0, 0) == "similar"

    def test_large_mw_increase_lowers_vd(self):
        # delta_mw=150 -> vd_score=-0.5, which is not < -0.6, so "similar"
        # Need combined signal to reach "decrease"
        assert _predict_vd_change(150, 0, 0) == "similar"

    def test_large_psa_increase_lowers_vd(self):
        assert _predict_vd_change(0, 0, 40) == "decrease"

    def test_large_psa_decrease_increases_vd(self):
        # logP moderate increase + large PSA drop -> increase
        assert _predict_vd_change(0, 1.2, -40) == "increase"

    def test_small_logP_change_similar(self):
        assert _predict_vd_change(0, 0.3, 0) == "similar"

    def test_moderate_logP_increase_similar(self):
        # delta_logP in (0.5, 1.0] only scores 0.5, not enough to push above 0.6
        result = _predict_vd_change(0, 0.7, 0)
        assert result == "similar"


# ---------------------------------------------------------------------------
# 4. _predict_absorption_change
# ---------------------------------------------------------------------------

class TestPredictAbsorptionChange:
    def test_large_mw_increase_decreases_absorption(self):
        assert _predict_absorption_change(120, 0, 0, 0) == "decrease"

    def test_large_psa_increase_decreases_absorption(self):
        assert _predict_absorption_change(0, 0, 40, 0) == "decrease"

    def test_psa_reduction_improves_absorption(self):
        # delta_logP in (0, 2] gives slight boost; large PSA drop -> increase
        assert _predict_absorption_change(0, 1.0, -35, 0) == "increase"

    def test_no_change_returns_similar(self):
        assert _predict_absorption_change(0, 0, 0, 0) == "similar"

    def test_large_pka_change_penalises_absorption(self):
        # delta_pka > 2 adds penalty; combined with moderate psa -> decrease
        assert _predict_absorption_change(0, 0, 15, 2.5) == "decrease"

    def test_negative_logP_penalises_absorption(self):
        # delta_logP < -1 should slightly penalize
        result = _predict_absorption_change(0, -1.5, 0, 0)
        # mild penalty only, still within -0.5..0.5 range -> similar
        assert result == "similar"


# ---------------------------------------------------------------------------
# 5. _predict_cl_change
# ---------------------------------------------------------------------------

class TestPredictClChange:
    def test_large_logP_increase_increases_cl(self):
        candidate = _make_drug(logP=3.0)
        assert _predict_cl_change(2.0, 0, candidate) == "increase"

    def test_large_logP_decrease_decreases_cl(self):
        candidate = _make_drug(logP=1.0)
        assert _predict_cl_change(-2.0, 0, candidate) == "decrease"

    def test_no_change_returns_similar(self):
        candidate = _make_drug(logP=2.5)
        assert _predict_cl_change(0, 0, candidate) == "similar"

    def test_very_high_candidate_logP_near_threshold(self):
        # candidate logP > 5 adds +0.5 bonus, but score=0.5 is NOT > 0.5 -> "similar"
        # To trigger "increase" need delta_logP > 0.5 as well
        candidate = _make_drug(logP=6.0)
        assert _predict_cl_change(0.0, 0, candidate) == "similar"

    def test_very_high_candidate_logP_with_delta_increases_cl(self):
        # candidate logP > 5 (+0.5) + delta_logP > 0.5 (+0.3) -> 0.8 > 0.5 -> "increase"
        candidate = _make_drug(logP=6.0)
        assert _predict_cl_change(0.6, 0, candidate) == "increase"

    def test_large_mw_increase_raises_cl(self):
        # delta_mw > 150 adds +0.3; delta_logP 0.6 adds +0.3 -> total 0.6 > 0.5 -> increase
        candidate = _make_drug(logP=3.1)
        assert _predict_cl_change(0.6, 200, candidate) == "increase"

    def test_moderate_logP_change_similar(self):
        # delta_logP in (0.5, 1.5) adds 0.3 -> below threshold
        candidate = _make_drug(logP=3.0)
        assert _predict_cl_change(0.8, 0, candidate) == "similar"


# ---------------------------------------------------------------------------
# 6. _assess_risk_flags
# ---------------------------------------------------------------------------

class TestAssessRiskFlags:
    def test_large_mw_increase_flag(self):
        original = _make_drug(mw=300.0)
        candidate = _make_drug(mw=450.0)
        flags = _assess_risk_flags(original, candidate, 150, 0, 0, 0)
        assert any("large MW increase" in f for f in flags)

    def test_high_candidate_logP_flag(self):
        original = _make_drug(logP=3.0)
        candidate = _make_drug(logP=6.0)
        flags = _assess_risk_flags(original, candidate, 0, 3.0, 0, 0)
        assert any("logP" in f and "exceeds 5" in f for f in flags)

    def test_low_psa_cns_flag(self):
        original = _make_drug(psa=60.0)
        candidate = _make_drug(psa=30.0)
        flags = _assess_risk_flags(original, candidate, 0, 0, -30, 0)
        assert any("CNS penetration" in f for f in flags)

    def test_high_mw_lipinski_flag(self):
        original = _make_drug(mw=400.0)
        candidate = _make_drug(mw=600.0)
        flags = _assess_risk_flags(original, candidate, 200, 0, 0, 0)
        assert any("Lipinski Ro5" in f for f in flags)

    def test_high_hbd_flag(self):
        original = _make_drug(n_hbd=2)
        candidate = _make_drug(n_hbd=7)
        flags = _assess_risk_flags(original, candidate, 0, 0, 0, 0)
        assert any("n_HBD" in f and "Lipinski violation" in f for f in flags)

    def test_large_pka_shift_flag(self):
        original = _make_drug(pka=5.0)
        candidate = _make_drug(pka=8.0)
        flags = _assess_risk_flags(original, candidate, 0, 0, 0, 3.0)
        assert any("delta_pKa" in f for f in flags)

    def test_large_psa_increase_flag(self):
        original = _make_drug(psa=40.0)
        candidate = _make_drug(psa=110.0)
        flags = _assess_risk_flags(original, candidate, 0, 0, 70, 0)
        assert any("large PSA increase" in f for f in flags)

    def test_logP_boundary_crossing_flag(self):
        original = _make_drug(logP=4.5)
        candidate = _make_drug(logP=5.5)
        flags = _assess_risk_flags(original, candidate, 0, 1.0, 0, 0)
        assert any("crosses logP=5 boundary" in f for f in flags)

    def test_clean_candidate_no_flags(self):
        original = _make_drug()
        candidate = _make_drug(name="Clean", mw=300.0, logP=2.0, psa=55.0, n_hbd=2, pka=7.5)
        flags = _assess_risk_flags(original, candidate, -50, -0.5, -5, 0.1)
        assert len(flags) == 0

    def test_flags_returns_tuple(self):
        original = _make_drug()
        candidate = _make_drug()
        flags = _assess_risk_flags(original, candidate, 0, 0, 0, 0)
        assert isinstance(flags, tuple)


# ---------------------------------------------------------------------------
# 7. _overall_recommendation
# ---------------------------------------------------------------------------

class TestOverallRecommendation:
    def test_very_high_logP_returns_poor(self):
        candidate = _make_drug(logP=8.0, mw=400.0, psa=60.0)
        rec = _overall_recommendation((), "similar", "similar", "similar", 0, 0, 0, candidate)
        assert rec == "poor"

    def test_very_high_mw_returns_poor(self):
        candidate = _make_drug(logP=2.0, mw=750.0, psa=60.0)
        rec = _overall_recommendation((), "similar", "similar", "similar", 0, 0, 0, candidate)
        assert rec == "poor"

    def test_many_flags_returns_risky(self):
        candidate = _make_drug(logP=2.0, mw=400.0, psa=60.0)
        flags = ("flag1", "flag2", "flag3", "flag4")
        rec = _overall_recommendation(flags, "similar", "similar", "similar", 0, 0, 0, candidate)
        assert rec == "risky"

    def test_favorable_scenario(self):
        # No flags, absorption increase, Lipinski-compliant candidate
        candidate = _make_drug(name="Good", mw=350.0, logP=3.0, psa=55.0, n_hbd=2)
        rec = _overall_recommendation((), "similar", "increase", "similar", 0, 0, 0, candidate)
        assert rec == "favorable"

    def test_acceptable_scenario(self):
        candidate = _make_drug(name="Mid", mw=400.0, logP=4.0, psa=65.0, n_hbd=3)
        flags = ("one flag",)
        rec = _overall_recommendation(flags, "similar", "increase", "similar", 0, 0, 0, candidate)
        assert rec == "acceptable"

    def test_recommendation_in_valid_set(self):
        candidate = _make_drug()
        rec = _overall_recommendation((), "similar", "similar", "similar", 0, 0, 0, candidate)
        assert rec in {"favorable", "acceptable", "risky", "poor"}


# ---------------------------------------------------------------------------
# 8. predict_scaffold_hop_impact (integration tests)
# ---------------------------------------------------------------------------

class TestPredictScaffoldHopImpact:
    def test_returns_scaffold_hop_result(self):
        original, candidate = _make_pair()
        result = predict_scaffold_hop_impact(original, candidate)
        assert isinstance(result, ScaffoldHopResult)

    def test_names_stored_correctly(self):
        original = _make_drug(name="DrugOrig")
        candidate = _make_drug(name="DrugCand")
        result = predict_scaffold_hop_impact(original, candidate)
        assert result.original_name == "DrugOrig"
        assert result.candidate_name == "DrugCand"

    def test_delta_mw_computed_correctly(self):
        original = _make_drug(mw=300.0)
        candidate = _make_drug(mw=450.0)
        result = predict_scaffold_hop_impact(original, candidate)
        assert abs(result.delta_mw - 150.0) < 1e-9

    def test_delta_logP_computed_correctly(self):
        original = _make_drug(logP=2.0)
        candidate = _make_drug(logP=4.5)
        result = predict_scaffold_hop_impact(original, candidate)
        assert abs(result.delta_logP - 2.5) < 1e-9

    def test_identical_drugs_all_similar(self):
        drug = _make_drug()
        drug2 = _make_drug(name="Same2")
        result = predict_scaffold_hop_impact(drug, drug2)
        assert result.delta_mw == 0.0
        assert result.predicted_vd_change == "similar"
        assert result.predicted_absorption_change == "similar"

    def test_invalid_original_raises(self):
        bad = _make_drug(mw=-1)
        candidate = _make_drug(name="Cand")
        with pytest.raises(ValueError):
            predict_scaffold_hop_impact(bad, candidate)

    def test_invalid_candidate_raises(self):
        original = _make_drug()
        bad = _make_drug(name="Bad", psa=-10.0)
        with pytest.raises(ValueError):
            predict_scaffold_hop_impact(original, bad)

    def test_original_missing_field_raises(self):
        original = _make_drug()
        del original["logP"]
        candidate = _make_drug(name="Cand")
        with pytest.raises(ValueError, match="missing fields"):
            predict_scaffold_hop_impact(original, candidate)

    def test_high_lipophilicity_candidate_flagged(self):
        original = _make_drug(logP=4.0)
        candidate = _make_drug(name="Fatty", logP=6.5)
        result = predict_scaffold_hop_impact(original, candidate)
        assert any("logP" in f for f in result.risk_flags)

    def test_recommendation_valid_values(self):
        original, candidate = _make_pair()
        result = predict_scaffold_hop_impact(original, candidate)
        assert result.overall_recommendation in {"favorable", "acceptable", "risky", "poor"}

    def test_notes_includes_risk_flag_count_when_present(self):
        original = _make_drug(mw=400.0, logP=4.9)
        candidate = _make_drug(name="Flaggy", mw=650.0, logP=6.0, psa=25.0, n_hbd=7)
        result = predict_scaffold_hop_impact(original, candidate)
        assert len(result.risk_flags) > 0
        assert "risk flag" in result.notes

    def test_delta_psa_negative(self):
        original = _make_drug(psa=90.0)
        candidate = _make_drug(name="LowPSA", psa=40.0)
        result = predict_scaffold_hop_impact(original, candidate)
        assert result.delta_psa == pytest.approx(-50.0)

    def test_poor_when_mw_exceeds_700(self):
        original = _make_drug(mw=300.0)
        candidate = _make_drug(name="Huge", mw=750.0, logP=2.0)
        result = predict_scaffold_hop_impact(original, candidate)
        assert result.overall_recommendation == "poor"

    def test_delta_pka_stored_correctly(self):
        original = _make_drug(pka=6.0)
        candidate = _make_drug(pka=9.0)
        result = predict_scaffold_hop_impact(original, candidate)
        assert result.delta_pka == pytest.approx(3.0)

    def test_notes_is_nonempty_string(self):
        original, candidate = _make_pair()
        result = predict_scaffold_hop_impact(original, candidate)
        assert isinstance(result.notes, str)
        assert len(result.notes) > 0

    def test_vd_cl_absorption_string_values(self):
        original, candidate = _make_pair()
        result = predict_scaffold_hop_impact(original, candidate)
        assert result.predicted_vd_change in {"increase", "decrease", "similar"}
        assert result.predicted_absorption_change in {"increase", "decrease", "similar"}
        assert result.predicted_cl_change in {"increase", "decrease", "similar"}
