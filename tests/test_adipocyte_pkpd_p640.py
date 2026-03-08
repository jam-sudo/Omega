"""Tests for Phase 640 — Drug Adipocyte PK/PD."""

import pytest

from omega_pbpk.core.adipocyte_pkpd_p640 import AdipocytePKPDResult, simulate_adipocyte_pkpd

# --- Fixture helpers ---


def default_result(**kwargs):
    params = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        logP=2.0,
        mw_Da=300.0,
        cl_sys_L_per_h=5.0,
        vd_sys_L=50.0,
    )
    params.update(kwargs)
    return simulate_adipocyte_pkpd(**params)


# --- Return type and structure ---


def test_returns_adipocyte_pkpd_result():
    result = default_result()
    assert isinstance(result, AdipocytePKPDResult)


def test_result_has_correct_drug_name():
    result = default_result(drug_name="Niacin")
    assert result.drug_name == "Niacin"


def test_result_has_correct_dose():
    result = default_result(dose_mg=200.0)
    assert result.dose_mg == 200.0


def test_times_list_is_nonempty():
    result = default_result()
    assert len(result.times_h) > 0


def test_times_start_at_zero():
    result = default_result()
    assert result.times_h[0] == pytest.approx(0.0)


def test_times_end_near_t_end():
    result = default_result(t_end_h=12.0)
    assert result.times_h[-1] == pytest.approx(12.0, abs=0.2)


def test_all_lists_same_length():
    result = default_result()
    n = len(result.times_h)
    assert len(result.c_plasma_mg_L) == n
    assert len(result.c_adipocyte_mg_g) == n
    assert len(result.lipase_activity) == n
    assert len(result.lipolysis_rate) == n
    assert len(result.fatty_acid_release_mg_h) == n


def test_plasma_concentrations_nonnegative():
    result = default_result()
    assert all(c >= 0.0 for c in result.c_plasma_mg_L)


def test_adipocyte_concentrations_nonnegative():
    result = default_result()
    assert all(c >= 0.0 for c in result.c_adipocyte_mg_g)


# --- Lipase activity bounds ---


def test_lipase_activity_between_0_and_2():
    result = default_result()
    for la in result.lipase_activity:
        assert 0.0 < la <= 2.0


def test_lipase_activity_at_t0_near_baseline():
    result = default_result()
    # At t=0, no drug, so lipase activity should be near 1.0
    assert result.lipase_activity[0] == pytest.approx(1.0, abs=0.05)


def test_lipase_activity_decreases_with_higher_dose():
    low = default_result(dose_mg=10.0, ic50_lipase_mg_L=0.5)
    high = default_result(dose_mg=1000.0, ic50_lipase_mg_L=0.5)
    min_low = min(low.lipase_activity)
    min_high = min(high.lipase_activity)
    assert min_high < min_low


# --- Lipolysis ---


def test_lipolysis_rate_nonnegative():
    result = default_result()
    assert all(r >= 0.0 for r in result.lipolysis_rate)


def test_fatty_acid_release_is_90pct_of_lipolysis():
    result = default_result()
    for fa, lys in zip(result.fatty_acid_release_mg_h, result.lipolysis_rate):
        assert fa == pytest.approx(lys * 0.9, rel=1e-6)


def test_cumulative_lipolysis_positive():
    result = default_result()
    assert result.cumulative_lipolysis_mg > 0.0


def test_cumulative_lipolysis_less_than_baseline_times_duration():
    # With inhibition, total lipolysis < baseline * t_end
    result = default_result(t_end_h=24.0, baseline_lipolysis_mg_h=500.0)
    max_possible = 500.0 * 24.0
    assert result.cumulative_lipolysis_mg <= max_possible + 1.0  # allow tiny rounding


# --- Lipase inhibition % ---


def test_lipase_inhibition_pct_range():
    result = default_result()
    assert 0.0 <= result.lipase_inhibition_pct <= 100.0


def test_higher_dose_increases_inhibition():
    low = default_result(dose_mg=10.0, ic50_lipase_mg_L=0.5)
    high = default_result(dose_mg=500.0, ic50_lipase_mg_L=0.5)
    assert high.lipase_inhibition_pct >= low.lipase_inhibition_pct


def test_zero_emax_gives_near_zero_inhibition():
    result = default_result(emax_lipase=0.0, dose_mg=100.0)
    assert result.lipase_inhibition_pct < 1.0


# --- Drug effect classification ---


def test_drug_effect_class_is_string():
    result = default_result()
    assert isinstance(result.drug_effect_class, str)


def test_antilipolytic_classification():
    # Very potent drug: low IC50, high dose, high emax → strong inhibition
    result = default_result(
        dose_mg=500.0,
        logP=3.0,
        cl_sys_L_per_h=1.0,
        vd_sys_L=10.0,
        ic50_lipase_mg_L=0.01,
        emax_lipase=0.95,
        baseline_lipolysis_mg_h=500.0,
    )
    assert result.drug_effect_class == "antilipolytic"


def test_neutral_classification_low_drug():
    # Very weak drug: very high IC50 → minimal inhibition
    result = default_result(
        dose_mg=10.0,
        logP=1.0,
        cl_sys_L_per_h=20.0,
        vd_sys_L=50.0,
        ic50_lipase_mg_L=1000.0,
        emax_lipase=0.1,
        baseline_lipolysis_mg_h=500.0,
    )
    assert result.drug_effect_class == "neutral"


def test_effect_class_one_of_valid_options():
    result = default_result()
    assert result.drug_effect_class in {"antilipolytic", "moderate_inhibition", "neutral"}


# --- IC50 estimation from logP ---


def test_ic50_estimated_from_logP_when_none():
    # Should not raise; IC50 should be embedded in notes
    result = simulate_adipocyte_pkpd(
        drug_name="X",
        dose_mg=100.0,
        logP=2.0,
        mw_Da=300.0,
        cl_sys_L_per_h=5.0,
        vd_sys_L=50.0,
        ic50_lipase_mg_L=None,
    )
    assert result is not None
    assert "ic50_lipase" in result.notes


def test_high_logP_lowers_estimated_ic50():
    # High logP → lower estimated IC50 (more lipophilic = more lipase-penetrating)
    r_low = simulate_adipocyte_pkpd("X", 100.0, 0.0, 300.0, 5.0, 50.0, ic50_lipase_mg_L=None)
    r_high = simulate_adipocyte_pkpd("X", 100.0, 5.0, 300.0, 5.0, 50.0, ic50_lipase_mg_L=None)

    # Lower logP → higher IC50; higher logP → lower IC50
    # Extract from notes
    def get_ic50(notes):
        for part in notes.split(";"):
            if "ic50_lipase" in part:
                return float(part.split("=")[1].strip().split(" ")[0])
        return None

    assert get_ic50(r_high.notes) < get_ic50(r_low.notes)


# --- AUC ---


def test_auc_plasma_positive():
    result = default_result()
    assert result.auc_plasma_mg_h_per_L > 0.0


def test_higher_dose_gives_higher_auc():
    low = default_result(dose_mg=50.0)
    high = default_result(dose_mg=200.0)
    assert high.auc_plasma_mg_h_per_L > low.auc_plasma_mg_h_per_L


# --- Validation errors ---


def test_raises_on_zero_dose():
    with pytest.raises(ValueError):
        simulate_adipocyte_pkpd("X", 0.0, 2.0, 300.0, 5.0, 50.0)


def test_raises_on_negative_dose():
    with pytest.raises(ValueError):
        simulate_adipocyte_pkpd("X", -50.0, 2.0, 300.0, 5.0, 50.0)


def test_raises_on_zero_cl():
    with pytest.raises(ValueError):
        simulate_adipocyte_pkpd("X", 100.0, 2.0, 300.0, 0.0, 50.0)


def test_raises_on_zero_vd():
    with pytest.raises(ValueError):
        simulate_adipocyte_pkpd("X", 100.0, 2.0, 300.0, 5.0, 0.0)


def test_raises_on_negative_ic50():
    with pytest.raises(ValueError):
        simulate_adipocyte_pkpd("X", 100.0, 2.0, 300.0, 5.0, 50.0, ic50_lipase_mg_L=-1.0)


def test_raises_on_invalid_emax_above_1():
    with pytest.raises(ValueError):
        simulate_adipocyte_pkpd("X", 100.0, 2.0, 300.0, 5.0, 50.0, emax_lipase=1.5)


def test_raises_on_invalid_emax_below_0():
    with pytest.raises(ValueError):
        simulate_adipocyte_pkpd("X", 100.0, 2.0, 300.0, 5.0, 50.0, emax_lipase=-0.1)


def test_raises_on_zero_baseline_lipolysis():
    with pytest.raises(ValueError):
        simulate_adipocyte_pkpd("X", 100.0, 2.0, 300.0, 5.0, 50.0, baseline_lipolysis_mg_h=0.0)


def test_notes_is_string():
    result = default_result()
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0
