"""Tests for Phase 443 — bile acid sequestrant drug interaction PK model."""

import pytest

from omega_pbpk.core.bile_acid_interaction_pk import (
    BileAcidInteractionResult,
    compare_sequestrants,
    simulate_bile_acid_interaction,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _default_result(**kwargs):
    """Run simulation with warfarin-like acid drug defaults."""
    defaults = dict(
        drug_name="warfarin",
        bile_sequestrant="cholestyramine",
        dose_mg=5.0,
        sequestrant_dose_g=4.0,
        ionization_type="acid",
        ka_per_h=1.0,
        cl_L_per_h=0.2,
        vd_L=10.0,
        f_oral_baseline=0.95,
    )
    defaults.update(kwargs)
    return simulate_bile_acid_interaction(**defaults)


# ---------------------------------------------------------------------------
# Return type and structure
# ---------------------------------------------------------------------------


def test_return_type():
    result = _default_result()
    assert isinstance(result, BileAcidInteractionResult)


def test_arrays_consistent_length():
    result = _default_result()
    n = len(result.times_h)
    assert len(result.c_plasma_without_seq_mg_L) == n
    assert len(result.c_plasma_with_seq_mg_L) == n
    assert n > 0


def test_times_start_at_zero():
    result = _default_result()
    assert result.times_h[0] == pytest.approx(0.0)


def test_plasma_without_seq_starts_at_zero():
    """At t=0, oral drug not yet absorbed → concentration = 0."""
    result = _default_result()
    assert result.c_plasma_without_seq_mg_L[0] == pytest.approx(0.0)


def test_plasma_with_seq_starts_at_zero():
    result = _default_result()
    assert result.c_plasma_with_seq_mg_L[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# AUC and sequestrant effect
# ---------------------------------------------------------------------------


def test_auc_without_greater_than_auc_with():
    """Sequestrant should reduce AUC."""
    result = _default_result()
    assert result.auc_without_seq_mg_h_per_L > result.auc_with_seq_mg_h_per_L


def test_auc_reduction_pct_between_0_and_100():
    result = _default_result()
    assert 0.0 <= result.auc_reduction_pct <= 100.0


def test_no_sequestrant_gives_zero_reduction():
    result = _default_result(bile_sequestrant="none", sequestrant_dose_g=0.0)
    assert result.auc_reduction_pct == pytest.approx(0.0, abs=1e-6)


def test_acid_drug_higher_reduction_than_neutral():
    """Acidic drugs bind sequestrants more strongly than neutral drugs."""
    result_acid = _default_result(ionization_type="acid")
    result_neutral = _default_result(ionization_type="neutral")
    assert result_acid.auc_reduction_pct > result_neutral.auc_reduction_pct


def test_acid_drug_higher_reduction_than_base():
    result_acid = _default_result(ionization_type="acid")
    result_base = _default_result(ionization_type="base")
    assert result_acid.auc_reduction_pct > result_base.auc_reduction_pct


def test_cholestyramine_more_reduction_than_colesevelam():
    """Cholestyramine has higher binding_fraction_per_g than colesevelam."""
    result_chol = _default_result(bile_sequestrant="cholestyramine")
    result_col = _default_result(bile_sequestrant="colesevelam")
    assert result_chol.auc_reduction_pct > result_col.auc_reduction_pct


def test_cholestyramine_more_reduction_than_colestipol():
    """Cholestyramine (0.08) > colestipol (0.06)."""
    result_chol = _default_result(bile_sequestrant="cholestyramine")
    result_col = _default_result(bile_sequestrant="colestipol")
    assert result_chol.auc_reduction_pct > result_col.auc_reduction_pct


# ---------------------------------------------------------------------------
# Bioavailability fields
# ---------------------------------------------------------------------------


def test_f_oral_without_matches_baseline():
    result = _default_result(f_oral_baseline=0.95)
    assert result.f_oral_without == pytest.approx(0.95)


def test_f_oral_with_less_than_without():
    result = _default_result()
    assert result.f_oral_with < result.f_oral_without


# ---------------------------------------------------------------------------
# Severity classification
# ---------------------------------------------------------------------------


def test_interaction_severity_severe_for_high_binding():
    """Large dose of cholestyramine with acidic drug should be severe."""
    result = _default_result(
        bile_sequestrant="cholestyramine",
        sequestrant_dose_g=12.0,  # large dose
        ionization_type="acid",
    )
    assert result.interaction_severity == "severe"


def test_interaction_severity_none_for_no_sequestrant():
    result = _default_result(bile_sequestrant="none", sequestrant_dose_g=0.0)
    assert result.interaction_severity == "none"


def test_interaction_severity_none_for_neutral_low_dose():
    """Neutral drug + small sequestrant dose = minimal interaction."""
    result = _default_result(
        bile_sequestrant="colesevelam",
        sequestrant_dose_g=1.0,
        ionization_type="neutral",
    )
    assert result.interaction_severity == "none"


# ---------------------------------------------------------------------------
# Timing recommendation
# ---------------------------------------------------------------------------


def test_timing_avoid_for_severe():
    result = _default_result(
        bile_sequestrant="cholestyramine",
        sequestrant_dose_g=12.0,
        ionization_type="acid",
    )
    assert result.timing_recommendation == "avoid"


def test_timing_no_restriction_for_none_sequestrant():
    result = _default_result(bile_sequestrant="none", sequestrant_dose_g=0.0)
    assert result.timing_recommendation == "no_restriction"


# ---------------------------------------------------------------------------
# compare_sequestrants
# ---------------------------------------------------------------------------


def test_compare_sequestrants_returns_three():
    results = compare_sequestrants(
        drug_name="warfarin",
        dose_mg=5.0,
        sequestrant_dose_g=4.0,
        ionization_type="acid",
        ka_per_h=1.0,
        cl_L_per_h=0.2,
        vd_L=10.0,
        f_oral_baseline=0.95,
    )
    assert len(results) == 3


def test_compare_sequestrants_sorted_descending():
    results = compare_sequestrants(
        drug_name="warfarin",
        dose_mg=5.0,
        sequestrant_dose_g=4.0,
        ionization_type="acid",
        ka_per_h=1.0,
        cl_L_per_h=0.2,
        vd_L=10.0,
        f_oral_baseline=0.95,
    )
    reductions = [r.auc_reduction_pct for r in results]
    assert reductions == sorted(reductions, reverse=True)


def test_compare_sequestrants_all_active():
    """All 3 results should be for active sequestrants, not 'none'."""
    results = compare_sequestrants(
        drug_name="warfarin",
        dose_mg=5.0,
        sequestrant_dose_g=4.0,
        ionization_type="acid",
        ka_per_h=1.0,
        cl_L_per_h=0.2,
        vd_L=10.0,
        f_oral_baseline=0.95,
    )
    for r in results:
        assert r.bile_sequestrant != "none"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_sequestrant_raises():
    with pytest.raises(ValueError, match="Unknown sequestrant"):
        simulate_bile_acid_interaction(
            drug_name="warfarin",
            bile_sequestrant="bad_seq",
            dose_mg=5.0,
            sequestrant_dose_g=4.0,
            ionization_type="acid",
            ka_per_h=1.0,
            cl_L_per_h=0.2,
            vd_L=10.0,
            f_oral_baseline=0.95,
        )


def test_invalid_ionization_type_raises():
    with pytest.raises(ValueError, match="Unknown ionization_type"):
        simulate_bile_acid_interaction(
            drug_name="warfarin",
            bile_sequestrant="cholestyramine",
            dose_mg=5.0,
            sequestrant_dose_g=4.0,
            ionization_type="zwitterion",
            ka_per_h=1.0,
            cl_L_per_h=0.2,
            vd_L=10.0,
            f_oral_baseline=0.95,
        )


def test_negative_dose_raises():
    with pytest.raises(ValueError):
        simulate_bile_acid_interaction(
            drug_name="warfarin",
            bile_sequestrant="cholestyramine",
            dose_mg=-5.0,
            sequestrant_dose_g=4.0,
            ionization_type="acid",
            ka_per_h=1.0,
            cl_L_per_h=0.2,
            vd_L=10.0,
            f_oral_baseline=0.95,
        )


def test_zero_cl_raises():
    with pytest.raises(ValueError):
        simulate_bile_acid_interaction(
            drug_name="warfarin",
            bile_sequestrant="cholestyramine",
            dose_mg=5.0,
            sequestrant_dose_g=4.0,
            ionization_type="acid",
            ka_per_h=1.0,
            cl_L_per_h=0.0,
            vd_L=10.0,
            f_oral_baseline=0.95,
        )


def test_f_oral_out_of_range_raises():
    with pytest.raises(ValueError):
        simulate_bile_acid_interaction(
            drug_name="warfarin",
            bile_sequestrant="cholestyramine",
            dose_mg=5.0,
            sequestrant_dose_g=4.0,
            ionization_type="acid",
            ka_per_h=1.0,
            cl_L_per_h=0.2,
            vd_L=10.0,
            f_oral_baseline=1.5,
        )
