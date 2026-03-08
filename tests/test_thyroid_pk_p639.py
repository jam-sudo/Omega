"""Tests for Phase 639 — Drug Thyroid Gland Accumulation."""

import pytest

from omega_pbpk.core.thyroid_pk_p639 import ThyroidPKResult, simulate_thyroid_pk

# --- Fixture helpers ---


def default_result(**kwargs):
    params = dict(
        drug_name="TestDrug",
        dose_mg=200.0,
        logP=2.0,
        mw_Da=300.0,
        cl_sys_L_per_h=5.0,
        vd_sys_L=50.0,
    )
    params.update(kwargs)
    return simulate_thyroid_pk(**params)


# --- Return type and structure ---


def test_returns_thyroid_pk_result():
    result = default_result()
    assert isinstance(result, ThyroidPKResult)


def test_result_has_correct_drug_name():
    result = default_result(drug_name="Amiodarone")
    assert result.drug_name == "Amiodarone"


def test_result_has_correct_dose():
    result = default_result(dose_mg=100.0)
    assert result.dose_mg == 100.0


def test_times_list_is_nonempty():
    result = default_result()
    assert len(result.times_h) > 0


def test_times_start_at_zero():
    result = default_result()
    assert result.times_h[0] == pytest.approx(0.0)


def test_times_end_near_t_end():
    result = default_result(t_end_h=24.0)
    assert result.times_h[-1] == pytest.approx(24.0, abs=0.2)


def test_plasma_and_thyroid_lists_same_length():
    result = default_result()
    assert len(result.c_plasma_mg_L) == len(result.c_thyroid_mg_g)
    assert len(result.c_plasma_mg_L) == len(result.times_h)


def test_plasma_concentrations_nonnegative():
    result = default_result()
    assert all(c >= 0.0 for c in result.c_plasma_mg_L)


def test_thyroid_concentrations_nonnegative():
    result = default_result()
    assert all(c >= 0.0 for c in result.c_thyroid_mg_g)


# --- Kp and accumulation ---


def test_kp_thyroid_in_valid_range():
    result = default_result()
    assert 0.2 <= result.kp_thyroid <= 50.0


def test_iodine_content_increases_kp():
    low_iodine = default_result(iodine_content_pct=0.0)
    high_iodine = default_result(iodine_content_pct=37.0)
    assert high_iodine.kp_thyroid > low_iodine.kp_thyroid


def test_iodine_content_increases_thyroid_accumulation():
    low = default_result(iodine_content_pct=0.0)
    high = default_result(iodine_content_pct=37.0)
    assert high.cmax_thyroid_mg_g > low.cmax_thyroid_mg_g


def test_high_logP_increases_kp():
    low_logP = default_result(logP=0.0)
    high_logP = default_result(logP=5.0)
    assert high_logP.kp_thyroid >= low_logP.kp_thyroid


def test_large_mw_reduces_kp():
    small_mw = default_result(mw_Da=100.0)
    large_mw = default_result(mw_Da=1000.0)
    assert small_mw.kp_thyroid >= large_mw.kp_thyroid


# --- AUC and ratios ---


def test_auc_plasma_positive():
    result = default_result()
    assert result.auc_plasma_mg_h_per_L > 0.0


def test_auc_thyroid_positive():
    result = default_result()
    assert result.auc_thyroid_mg_h_per_g > 0.0


def test_thyroid_to_plasma_ratio_positive():
    result = default_result()
    assert result.thyroid_to_plasma_ratio > 0.0


def test_higher_iodine_increases_auc_ratio():
    low = default_result(iodine_content_pct=0.0)
    high = default_result(iodine_content_pct=37.0)
    assert high.thyroid_to_plasma_ratio > low.thyroid_to_plasma_ratio


# --- Toxicity index ---


def test_toxicity_index_equals_cmax_thyroid():
    result = default_result()
    assert result.thyroid_toxicity_index == pytest.approx(result.cmax_thyroid_mg_g, rel=1e-6)


def test_high_dose_raises_toxicity_index():
    low = default_result(dose_mg=50.0)
    high = default_result(dose_mg=500.0)
    assert high.thyroid_toxicity_index > low.thyroid_toxicity_index


# --- Iodine competition flag ---


def test_iodine_competition_false_when_no_iodine():
    result = default_result(iodine_content_pct=0.0)
    assert result.iodine_uptake_competition is False


def test_iodine_competition_false_at_threshold():
    result = default_result(iodine_content_pct=0.5)
    assert result.iodine_uptake_competition is False


def test_iodine_competition_true_above_threshold():
    result = default_result(iodine_content_pct=1.0)
    assert result.iodine_uptake_competition is True


def test_iodine_competition_true_for_amiodarone():
    result = default_result(
        drug_name="Amiodarone",
        iodine_content_pct=37.0,
        logP=7.0,
        mw_Da=645.0,
    )
    assert result.iodine_uptake_competition is True


# --- Half-life ---


def test_t_half_thyroid_positive():
    result = default_result()
    assert result.t_half_thyroid_h > 0.0


def test_t_half_thyroid_reasonable_range():
    # Thyroid t_half should be finite and reasonable for a 48h simulation
    result = default_result()
    assert 0.001 < result.t_half_thyroid_h < 1000.0


# --- Validation errors ---


def test_raises_on_zero_dose():
    with pytest.raises(ValueError):
        simulate_thyroid_pk("X", 0.0, 2.0, 300.0, 5.0, 50.0)


def test_raises_on_negative_dose():
    with pytest.raises(ValueError):
        simulate_thyroid_pk("X", -10.0, 2.0, 300.0, 5.0, 50.0)


def test_raises_on_zero_cl():
    with pytest.raises(ValueError):
        simulate_thyroid_pk("X", 100.0, 2.0, 300.0, 0.0, 50.0)


def test_raises_on_zero_vd():
    with pytest.raises(ValueError):
        simulate_thyroid_pk("X", 100.0, 2.0, 300.0, 5.0, 0.0)


def test_raises_on_negative_iodine():
    with pytest.raises(ValueError):
        simulate_thyroid_pk("X", 100.0, 2.0, 300.0, 5.0, 50.0, iodine_content_pct=-1.0)


def test_raises_on_zero_thyroid_weight():
    with pytest.raises(ValueError):
        simulate_thyroid_pk("X", 100.0, 2.0, 300.0, 5.0, 50.0, thyroid_weight_g=0.0)


def test_notes_is_string():
    result = default_result()
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0
