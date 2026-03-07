"""Tests for bioequivalence_analysis module — Phase 704."""

import pytest

from omega_pbpk.analysis.bioequivalence_analysis import (
    BEResult,
    compute_gmr_and_ci,
    simulate_be_study,
    tost_test,
)

# ── compute_gmr_and_ci tests ─────────────────────────────────────────────────


def test_compute_gmr_returns_be_result():
    test_aucs = [100.0, 105.0, 98.0, 102.0]
    ref_aucs = [100.0, 103.0, 99.0, 101.0]
    result = compute_gmr_and_ci(test_aucs, ref_aucs)
    assert isinstance(result, BEResult)


def test_gmr_approx_1_when_identical():
    """GMR ≈ 1.0 when test == ref."""
    data = [100.0, 105.0, 98.0, 102.0, 99.0]
    result = compute_gmr_and_ci(data, data)
    assert abs(result.gmr_auc - 1.0) < 1e-10


def test_is_bioequivalent_true_for_near_identical():
    """is_bioequivalent = True when GMR very close to 1.0 with tight CI."""
    test_aucs = [100.0, 101.0, 99.0, 100.5, 99.5] * 4
    ref_aucs = [100.0, 100.0, 100.0, 100.0, 100.0] * 4
    result = compute_gmr_and_ci(test_aucs, ref_aucs)
    assert result.is_bioequivalent is True


def test_is_bioequivalent_false_for_large_ratio():
    """is_bioequivalent = False when GMR = 1.5 (outside 80-125%)."""
    test_aucs = [150.0, 155.0, 148.0, 152.0, 149.0]
    ref_aucs = [100.0, 103.0, 98.0, 101.0, 99.0]
    result = compute_gmr_and_ci(test_aucs, ref_aucs)
    assert result.is_bioequivalent is False
    assert result.gmr_auc > 1.25


def test_ci_bounds_contain_gmr():
    """CI_lower < GMR < CI_upper."""
    test_aucs = [110.0, 112.0, 108.0, 111.0, 109.0]
    ref_aucs = [100.0, 103.0, 98.0, 101.0, 99.0]
    result = compute_gmr_and_ci(test_aucs, ref_aucs)
    assert result.ci_lower_auc < result.gmr_auc < result.ci_upper_auc


def test_n_test_n_ref_match_input_lengths():
    test_aucs = [100.0, 105.0, 98.0]
    ref_aucs = [100.0, 103.0, 99.0, 101.0]
    result = compute_gmr_and_ci(test_aucs, ref_aucs)
    assert result.n_test == 3
    assert result.n_ref == 4


def test_ci_lower_less_than_ci_upper():
    test_aucs = [100.0, 110.0, 95.0, 105.0]
    ref_aucs = [100.0, 100.0, 100.0, 100.0]
    result = compute_gmr_and_ci(test_aucs, ref_aucs)
    assert result.ci_lower_auc < result.ci_upper_auc


def test_gmr_below_1_for_lower_test():
    """GMR < 1 when test AUCs are lower than reference."""
    test_aucs = [70.0, 72.0, 68.0, 71.0]
    ref_aucs = [100.0, 102.0, 99.0, 101.0]
    result = compute_gmr_and_ci(test_aucs, ref_aucs)
    assert result.gmr_auc < 1.0
    assert result.is_bioequivalent is False


def test_validation_fewer_than_2_test_raises():
    with pytest.raises(ValueError):
        compute_gmr_and_ci([100.0], [100.0, 102.0])


def test_validation_fewer_than_2_ref_raises():
    with pytest.raises(ValueError):
        compute_gmr_and_ci([100.0, 102.0], [100.0])


def test_validation_zero_auc_raises():
    with pytest.raises(ValueError):
        compute_gmr_and_ci([0.0, 100.0], [100.0, 102.0])


def test_validation_negative_auc_raises():
    with pytest.raises(ValueError):
        compute_gmr_and_ci([100.0, -1.0], [100.0, 102.0])


# ── tost_test tests ──────────────────────────────────────────────────────────


def test_tost_returns_dict():
    test_vals = [100.0, 105.0, 98.0, 102.0]
    ref_vals = [100.0, 103.0, 99.0, 101.0]
    result = tost_test(test_vals, ref_vals)
    assert isinstance(result, dict)


def test_tost_has_required_keys():
    test_vals = [100.0, 105.0, 98.0, 102.0]
    ref_vals = [100.0, 103.0, 99.0, 101.0]
    result = tost_test(test_vals, ref_vals)
    for key in ["t_lower", "t_upper", "t_crit", "df", "be_demonstrated"]:
        assert key in result


def test_tost_be_demonstrated_for_near_identical():
    """Nearly identical groups → be_demonstrated True."""
    test_vals = [100.0, 101.0, 99.0, 100.5, 99.5] * 5
    ref_vals = [100.0, 100.0, 100.0, 100.0, 100.0] * 5
    result = tost_test(test_vals, ref_vals)
    assert result["be_demonstrated"] is True


def test_tost_be_not_demonstrated_for_divergent():
    """GMR = 1.5 → be_demonstrated False."""
    test_vals = [150.0, 152.0, 148.0, 151.0]
    ref_vals = [100.0, 102.0, 99.0, 101.0]
    result = tost_test(test_vals, ref_vals)
    assert result["be_demonstrated"] is False


def test_tost_t_crit_positive():
    test_vals = [100.0, 105.0, 98.0, 102.0]
    ref_vals = [100.0, 103.0, 99.0, 101.0]
    result = tost_test(test_vals, ref_vals)
    assert result["t_crit"] > 0


# ── simulate_be_study tests ──────────────────────────────────────────────────


def test_simulate_be_study_returns_dict():
    ref_params = {"cl": 5.0, "vd": 50.0, "ka": 1.0, "dose": 100.0, "f_oral": 0.8}
    tst_params = {"cl": 5.0, "vd": 50.0, "ka": 1.0, "dose": 100.0, "f_oral": 0.8}
    result = simulate_be_study(ref_params, tst_params)
    assert isinstance(result, dict)


def test_simulate_be_study_has_be_result_key():
    ref_params = {"cl": 5.0, "vd": 50.0, "ka": 1.0, "dose": 100.0, "f_oral": 0.8}
    tst_params = {"cl": 5.0, "vd": 50.0, "ka": 1.0, "dose": 100.0, "f_oral": 0.8}
    result = simulate_be_study(ref_params, tst_params)
    assert "be_result" in result
    assert isinstance(result["be_result"], BEResult)


def test_simulate_be_study_identical_formulations_tend_to_be():
    """Identical formulations: GMR should be near 1.0."""
    ref_params = {"cl": 5.0, "vd": 50.0, "ka": 1.0, "dose": 100.0, "f_oral": 0.8}
    tst_params = {"cl": 5.0, "vd": 50.0, "ka": 1.0, "dose": 100.0, "f_oral": 0.8}
    result = simulate_be_study(ref_params, tst_params, n_subjects=24, seed=42)
    gmr = result["be_result"].gmr_auc
    # GMR should be reasonably close to 1.0 (within ±30% even with BSV)
    assert 0.7 < gmr < 1.3


def test_simulate_be_study_aucs_lists():
    ref_params = {"cl": 5.0, "vd": 50.0, "ka": 1.0, "dose": 100.0, "f_oral": 0.8}
    tst_params = {"cl": 5.0, "vd": 50.0, "ka": 1.0, "dose": 100.0, "f_oral": 0.8}
    result = simulate_be_study(ref_params, tst_params, n_subjects=12)
    assert len(result["test_aucs"]) == 12
    assert len(result["ref_aucs"]) == 12


def test_simulate_be_study_all_aucs_positive():
    ref_params = {"cl": 5.0, "vd": 50.0, "ka": 1.0, "dose": 100.0, "f_oral": 0.8}
    tst_params = {"cl": 5.0, "vd": 50.0, "ka": 1.0, "dose": 100.0, "f_oral": 0.8}
    result = simulate_be_study(ref_params, tst_params, n_subjects=10)
    assert all(a > 0 for a in result["test_aucs"])
    assert all(a > 0 for a in result["ref_aucs"])


def test_simulate_be_study_summary_has_gmr():
    ref_params = {"cl": 5.0, "vd": 50.0, "ka": 1.0, "dose": 100.0, "f_oral": 0.8}
    tst_params = {"cl": 5.0, "vd": 50.0, "ka": 1.0, "dose": 100.0, "f_oral": 0.8}
    result = simulate_be_study(ref_params, tst_params, n_subjects=10)
    assert "summary" in result
    assert "gmr" in result["summary"]
