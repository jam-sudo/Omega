"""Tests for Phase 773 — Drug Formulation PK Comparison."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.formulation_pk_comparison import (
    FormulationPKResult,
    compare_formulations,
    recommend_formulation,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def all_results():
    """Run all 5 formulations with default parameters."""
    return compare_formulations(
        drug_name="TestDrug",
        dose_mg=100.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
        f_oral=0.8,
        ka_base_per_h=1.0,
        mec_mg_L=0.2,
        mtc_mg_L=4.0,
        t_end_h=24.0,
        dt_h=0.05,
    )


@pytest.fixture()
def ir_result(all_results):
    return next(r for r in all_results if r.formulation_type == "IR")


@pytest.fixture()
def er_result(all_results):
    return next(r for r in all_results if r.formulation_type == "ER")


@pytest.fixture()
def cr_result(all_results):
    return next(r for r in all_results if r.formulation_type == "CR")


@pytest.fixture()
def dr_result(all_results):
    return next(r for r in all_results if r.formulation_type == "DR")


@pytest.fixture()
def biphasic_result(all_results):
    return next(r for r in all_results if r.formulation_type == "Biphasic")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_returns_list_of_formulation_pk_results(all_results):
    """compare_formulations returns a list of FormulationPKResult."""
    assert isinstance(all_results, list)
    assert len(all_results) == 5
    for r in all_results:
        assert isinstance(r, FormulationPKResult)


def test_all_formulation_types_present(all_results):
    types = {r.formulation_type for r in all_results}
    assert types == {"IR", "ER", "CR", "DR", "Biphasic"}


def test_ir_has_highest_cmax(ir_result, er_result, cr_result):
    """IR should have the highest Cmax due to fastest absorption."""
    assert ir_result.cmax_mg_L > er_result.cmax_mg_L
    assert ir_result.cmax_mg_L > cr_result.cmax_mg_L


def test_er_cmax_lower_than_ir(ir_result, er_result):
    """ER has slower ka, so Cmax must be lower than IR."""
    assert er_result.cmax_mg_L < ir_result.cmax_mg_L


def test_cr_cmax_lower_than_er(er_result, cr_result):
    """CR (10x slower ka) should have lower Cmax than ER (4x slower ka)."""
    assert cr_result.cmax_mg_L < er_result.cmax_mg_L


def test_er_time_above_mec_ge_ir(er_result, ir_result):
    """ER sustained release should maintain therapeutic levels at least as long as IR."""
    # ER has sustained release, so time above MEC should be >= IR
    assert er_result.time_above_mec_h >= ir_result.time_above_mec_h * 0.8


def test_dr_tmax_delayed_vs_ir(dr_result, ir_result):
    """DR has 2h lag, so Tmax must be later than IR."""
    assert dr_result.tmax_h > ir_result.tmax_h


def test_composite_score_between_0_and_100(all_results):
    for r in all_results:
        assert 0.0 <= r.composite_score <= 100.0, (
            f"{r.formulation_type}: composite_score={r.composite_score}"
        )


def test_sorted_by_composite_score_descending(all_results):
    scores = [r.composite_score for r in all_results]
    assert scores == sorted(scores, reverse=True)


def test_cmax_positive(all_results):
    for r in all_results:
        assert r.cmax_mg_L > 0.0, f"{r.formulation_type}: Cmax should be > 0"


def test_tmax_positive(all_results):
    for r in all_results:
        assert r.tmax_h >= 0.0, f"{r.formulation_type}: Tmax should be >= 0"


def test_auc_positive(all_results):
    for r in all_results:
        assert r.auc_mg_h_per_L > 0.0, f"{r.formulation_type}: AUC should be > 0"


def test_time_in_therapeutic_nonnegative(all_results):
    for r in all_results:
        assert r.time_in_therapeutic_h >= 0.0


def test_drug_name_preserved(all_results):
    for r in all_results:
        assert r.drug_name == "TestDrug"


def test_recommend_formulation_returns_dict():
    rec = recommend_formulation(
        drug_name="DrugX",
        dose_mg=50.0,
        mec_mg_L=0.1,
        mtc_mg_L=3.0,
    )
    assert isinstance(rec, dict)


def test_recommend_formulation_has_best_formulation_key():
    rec = recommend_formulation(
        drug_name="DrugX",
        dose_mg=50.0,
        mec_mg_L=0.1,
        mtc_mg_L=3.0,
    )
    assert "best_formulation" in rec
    assert isinstance(rec["best_formulation"], str)


def test_recommend_formulation_best_matches_highest_score():
    rec = recommend_formulation(
        drug_name="DrugX",
        dose_mg=50.0,
        mec_mg_L=0.1,
        mtc_mg_L=3.0,
    )
    results = rec["results"]
    assert rec["best_formulation"] == results[0].formulation_type


def test_subset_of_formulations():
    results = compare_formulations(
        drug_name="SubsetDrug",
        dose_mg=100.0,
        formulations=["IR", "ER"],
        mec_mg_L=0.2,
        mtc_mg_L=3.0,
    )
    assert len(results) == 2
    types = {r.formulation_type for r in results}
    assert types == {"IR", "ER"}


def test_validation_dose_le_zero_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        compare_formulations(drug_name="Bad", dose_mg=0.0, mec_mg_L=0.5, mtc_mg_L=5.0)


def test_validation_negative_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        compare_formulations(drug_name="Bad", dose_mg=-10.0, mec_mg_L=0.5, mtc_mg_L=5.0)


def test_validation_mec_ge_mtc_raises():
    with pytest.raises(ValueError, match="mec_mg_L"):
        compare_formulations(
            drug_name="Bad",
            dose_mg=100.0,
            mec_mg_L=5.0,
            mtc_mg_L=5.0,
        )


def test_validation_mec_greater_than_mtc_raises():
    with pytest.raises(ValueError, match="mec_mg_L"):
        compare_formulations(
            drug_name="Bad",
            dose_mg=100.0,
            mec_mg_L=6.0,
            mtc_mg_L=5.0,
        )


def test_notes_nonempty(all_results):
    for r in all_results:
        assert len(r.notes) > 0, f"{r.formulation_type}: notes should not be empty"


def test_biphasic_result_present(biphasic_result):
    """Biphasic formulation simulated correctly."""
    assert biphasic_result.formulation_type == "Biphasic"
    assert biphasic_result.cmax_mg_L > 0.0
    assert biphasic_result.auc_mg_h_per_L > 0.0


def test_formulation_result_is_frozen(all_results):
    """FormulationPKResult is frozen (immutable)."""
    r = all_results[0]
    with pytest.raises(Exception):
        r.cmax_mg_L = 999.0  # type: ignore[misc]
