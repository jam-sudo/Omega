"""Tests for Phase 185 — enterohepatic circulation PK model."""

import pytest

from omega_pbpk.core.enterohepatic_circulation_pk_p185 import (
    EHCPKResult,
    compare_ehc_levels,
    simulate_ehc_pk,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

COMMON_KWARGS = dict(
    drug_name="TestDrug",
    dose_mg=100.0,
    k_bile_per_h=0.05,
    f_reabs=0.5,
    meal_times_h=[4.0, 12.0, 24.0],
    cl_L_per_h=5.0,
    vd_L=50.0,
    t_end_h=48.0,
    dt_h=0.05,
)


def _default_result() -> EHCPKResult:
    return simulate_ehc_pk(**COMMON_KWARGS)


# ---------------------------------------------------------------------------
# 1. Return type
# ---------------------------------------------------------------------------


def test_return_type():
    res = _default_result()
    assert isinstance(res, EHCPKResult)


# ---------------------------------------------------------------------------
# 2. Field preservation
# ---------------------------------------------------------------------------


def test_field_drug_name():
    res = _default_result()
    assert res.drug_name == "TestDrug"


def test_field_dose_mg():
    res = _default_result()
    assert res.dose_mg == 100.0


def test_field_f_reabs():
    res = _default_result()
    assert res.f_reabs == 0.5


# ---------------------------------------------------------------------------
# 3. Cmax and AUC positivity
# ---------------------------------------------------------------------------


def test_cmax_positive():
    res = _default_result()
    assert res.cmax_mg_L > 0.0


def test_auc_positive():
    res = _default_result()
    assert res.auc_mg_h_per_L > 0.0


# ---------------------------------------------------------------------------
# 4. Times and concentration lists
# ---------------------------------------------------------------------------


def test_times_list_non_empty():
    res = _default_result()
    assert len(res.times_h) > 0


def test_concs_list_non_empty():
    res = _default_result()
    assert len(res.c_plasma_mg_L) > 0


def test_times_concs_same_length():
    res = _default_result()
    assert len(res.times_h) == len(res.c_plasma_mg_L)


def test_times_start_at_zero():
    res = _default_result()
    assert res.times_h[0] == pytest.approx(0.0, abs=1e-6)


def test_concentrations_non_negative():
    res = _default_result()
    assert all(c >= 0.0 for c in res.c_plasma_mg_L)


# ---------------------------------------------------------------------------
# 5. EHC behaviour — with EHC, secondary peaks exist
# ---------------------------------------------------------------------------


def test_with_ehc_has_secondary_peaks():
    """With f_reabs > 0 and meal times, secondary peaks should appear."""
    res = simulate_ehc_pk(
        drug_name="EHCDrug",
        dose_mg=200.0,
        k_bile_per_h=0.15,
        f_reabs=0.8,
        meal_times_h=[2.0, 6.0, 12.0, 24.0],
        cl_L_per_h=3.0,
        vd_L=30.0,
        t_end_h=48.0,
        dt_h=0.05,
    )
    assert res.secondary_peaks_count > 0


# ---------------------------------------------------------------------------
# 6. No EHC — zero f_reabs produces no secondary peaks
# ---------------------------------------------------------------------------


def test_no_ehc_no_secondary_peaks():
    res = simulate_ehc_pk(
        drug_name="NoDrug",
        dose_mg=100.0,
        k_bile_per_h=0.1,
        f_reabs=0.0,
        meal_times_h=[4.0, 12.0],
        cl_L_per_h=5.0,
        vd_L=50.0,
        t_end_h=48.0,
        dt_h=0.05,
    )
    assert res.secondary_peaks_count == 0


# ---------------------------------------------------------------------------
# 7. Higher f_reabs → higher AUC
# ---------------------------------------------------------------------------


def test_higher_freabs_higher_auc():
    low = simulate_ehc_pk(
        drug_name="D",
        dose_mg=100.0,
        k_bile_per_h=0.05,
        f_reabs=0.1,
        meal_times_h=[6.0, 18.0],
        cl_L_per_h=5.0,
        vd_L=50.0,
        t_end_h=48.0,
        dt_h=0.05,
    )
    high = simulate_ehc_pk(
        drug_name="D",
        dose_mg=100.0,
        k_bile_per_h=0.05,
        f_reabs=0.9,
        meal_times_h=[6.0, 18.0],
        cl_L_per_h=5.0,
        vd_L=50.0,
        t_end_h=48.0,
        dt_h=0.05,
    )
    assert high.auc_mg_h_per_L > low.auc_mg_h_per_L


# ---------------------------------------------------------------------------
# 8. Tmax is within simulation window
# ---------------------------------------------------------------------------


def test_tmax_within_simulation():
    res = _default_result()
    assert 0.0 <= res.tmax_h <= 48.0


# ---------------------------------------------------------------------------
# 9. Effective half-life is positive
# ---------------------------------------------------------------------------


def test_effective_half_life_positive():
    res = _default_result()
    assert res.t_half_effective_h >= 0.0


# ---------------------------------------------------------------------------
# 10. No meal → notes indicate no EHC when f_reabs=0
# ---------------------------------------------------------------------------


def test_notes_no_ehc():
    res = simulate_ehc_pk(
        drug_name="X",
        dose_mg=50.0,
        k_bile_per_h=0.05,
        f_reabs=0.0,
        meal_times_h=[],
        cl_L_per_h=4.0,
        vd_L=40.0,
        t_end_h=24.0,
        dt_h=0.1,
    )
    assert "No EHC" in res.notes


# ---------------------------------------------------------------------------
# 11. compare_ehc_levels — return list
# ---------------------------------------------------------------------------


def test_compare_returns_list():
    result = compare_ehc_levels(
        drug_name="D",
        dose_mg=100.0,
        f_reabs_values=[0.0, 0.5, 0.9],
    )
    assert isinstance(result, list)
    assert len(result) == 3


# ---------------------------------------------------------------------------
# 12. compare_ehc_levels — sorted by AUC descending
# ---------------------------------------------------------------------------


def test_compare_sorted_descending():
    result = compare_ehc_levels(
        drug_name="D",
        dose_mg=100.0,
        f_reabs_values=[0.0, 0.3, 0.7, 0.95],
    )
    aucs = [r.auc_mg_h_per_L for r in result]
    assert aucs == sorted(aucs, reverse=True)


# ---------------------------------------------------------------------------
# 13. compare_ehc_levels — all results are EHCPKResult
# ---------------------------------------------------------------------------


def test_compare_result_types():
    result = compare_ehc_levels(
        drug_name="D",
        dose_mg=100.0,
        f_reabs_values=[0.2, 0.8],
    )
    for r in result:
        assert isinstance(r, EHCPKResult)


# ---------------------------------------------------------------------------
# 14. Validation — dose <= 0
# ---------------------------------------------------------------------------


def test_validation_dose_zero():
    with pytest.raises((ValueError, Exception)):
        simulate_ehc_pk(
            drug_name="X",
            dose_mg=0.0,
            k_bile_per_h=0.05,
            f_reabs=0.5,
            meal_times_h=[],
            cl_L_per_h=5.0,
            vd_L=50.0,
        )


def test_validation_dose_negative():
    with pytest.raises((ValueError, Exception)):
        simulate_ehc_pk(
            drug_name="X",
            dose_mg=-10.0,
            k_bile_per_h=0.05,
            f_reabs=0.5,
            meal_times_h=[],
            cl_L_per_h=5.0,
            vd_L=50.0,
        )


# ---------------------------------------------------------------------------
# 15. Validation — cl <= 0
# ---------------------------------------------------------------------------


def test_validation_cl_zero():
    with pytest.raises((ValueError, Exception)):
        simulate_ehc_pk(
            drug_name="X",
            dose_mg=100.0,
            k_bile_per_h=0.05,
            f_reabs=0.5,
            meal_times_h=[],
            cl_L_per_h=0.0,
            vd_L=50.0,
        )


def test_validation_cl_negative():
    with pytest.raises((ValueError, Exception)):
        simulate_ehc_pk(
            drug_name="X",
            dose_mg=100.0,
            k_bile_per_h=0.05,
            f_reabs=0.5,
            meal_times_h=[],
            cl_L_per_h=-1.0,
            vd_L=50.0,
        )


# ---------------------------------------------------------------------------
# 16. Validation — f_reabs out of range
# ---------------------------------------------------------------------------


def test_validation_freabs_above_one():
    with pytest.raises((ValueError, Exception)):
        simulate_ehc_pk(
            drug_name="X",
            dose_mg=100.0,
            k_bile_per_h=0.05,
            f_reabs=1.5,
            meal_times_h=[],
            cl_L_per_h=5.0,
            vd_L=50.0,
        )


def test_validation_freabs_negative():
    with pytest.raises((ValueError, Exception)):
        simulate_ehc_pk(
            drug_name="X",
            dose_mg=100.0,
            k_bile_per_h=0.05,
            f_reabs=-0.1,
            meal_times_h=[],
            cl_L_per_h=5.0,
            vd_L=50.0,
        )


# ---------------------------------------------------------------------------
# 17. Cmax equals max of c_plasma_mg_L
# ---------------------------------------------------------------------------


def test_cmax_is_max_of_list():
    res = _default_result()
    assert res.cmax_mg_L == pytest.approx(max(res.c_plasma_mg_L), rel=1e-6)


# ---------------------------------------------------------------------------
# 18. IV vs oral — IV produces Cmax immediately
# ---------------------------------------------------------------------------


def test_iv_mode_cmax_at_t0():
    res = simulate_ehc_pk(
        drug_name="IV",
        dose_mg=100.0,
        k_bile_per_h=0.05,
        f_reabs=0.0,
        meal_times_h=[],
        cl_L_per_h=5.0,
        vd_L=50.0,
        t_end_h=24.0,
        dt_h=0.05,
        iv=True,
    )
    assert res.tmax_h == pytest.approx(0.0, abs=0.1)


# ---------------------------------------------------------------------------
# 19. secondary_peaks_count is non-negative integer
# ---------------------------------------------------------------------------


def test_secondary_peaks_count_non_negative_int():
    res = _default_result()
    assert isinstance(res.secondary_peaks_count, int)
    assert res.secondary_peaks_count >= 0


# ---------------------------------------------------------------------------
# 20. Meal times noted in result
# ---------------------------------------------------------------------------


def test_meal_times_in_notes():
    res = simulate_ehc_pk(
        drug_name="D",
        dose_mg=100.0,
        k_bile_per_h=0.05,
        f_reabs=0.5,
        meal_times_h=[4.0, 8.0],
        cl_L_per_h=5.0,
        vd_L=50.0,
        t_end_h=24.0,
        dt_h=0.1,
    )
    assert "Meals" in res.notes or "meal" in res.notes.lower() or "4.0" in res.notes


# ---------------------------------------------------------------------------
# 21. AUC increases with longer simulation time
# ---------------------------------------------------------------------------


def test_auc_increases_with_time():
    short = simulate_ehc_pk(
        drug_name="D",
        dose_mg=100.0,
        k_bile_per_h=0.05,
        f_reabs=0.3,
        meal_times_h=[6.0],
        cl_L_per_h=5.0,
        vd_L=50.0,
        t_end_h=12.0,
        dt_h=0.1,
    )
    long_ = simulate_ehc_pk(
        drug_name="D",
        dose_mg=100.0,
        k_bile_per_h=0.05,
        f_reabs=0.3,
        meal_times_h=[6.0],
        cl_L_per_h=5.0,
        vd_L=50.0,
        t_end_h=48.0,
        dt_h=0.1,
    )
    assert long_.auc_mg_h_per_L > short.auc_mg_h_per_L


# ---------------------------------------------------------------------------
# 22. Different drug names are preserved in compare results
# ---------------------------------------------------------------------------


def test_compare_drug_name_preserved():
    result = compare_ehc_levels(
        drug_name="Morphine",
        dose_mg=50.0,
        f_reabs_values=[0.0, 0.6],
    )
    for r in result:
        assert r.drug_name == "Morphine"


# ---------------------------------------------------------------------------
# 23. Zero bile rate → minimal EHC effect
# ---------------------------------------------------------------------------


def test_zero_k_bile_no_bile_accumulation():
    """With k_bile=0 no drug enters bile, no secondary peaks."""
    res = simulate_ehc_pk(
        drug_name="D",
        dose_mg=100.0,
        k_bile_per_h=0.0,
        f_reabs=0.9,
        meal_times_h=[4.0, 12.0],
        cl_L_per_h=5.0,
        vd_L=50.0,
        t_end_h=48.0,
        dt_h=0.05,
    )
    assert res.secondary_peaks_count == 0


# ---------------------------------------------------------------------------
# 24. compare_ehc_levels single value
# ---------------------------------------------------------------------------


def test_compare_single_value():
    result = compare_ehc_levels(
        drug_name="D",
        dose_mg=100.0,
        f_reabs_values=[0.5],
    )
    assert len(result) == 1


# ---------------------------------------------------------------------------
# 25. High bile secretion → higher AUC with EHC than without
# ---------------------------------------------------------------------------


def test_high_bile_ehc_increases_auc():
    no_ehc = simulate_ehc_pk(
        drug_name="D",
        dose_mg=100.0,
        k_bile_per_h=0.2,
        f_reabs=0.0,
        meal_times_h=[4.0, 12.0, 24.0],
        cl_L_per_h=3.0,
        vd_L=30.0,
        t_end_h=48.0,
        dt_h=0.05,
    )
    with_ehc = simulate_ehc_pk(
        drug_name="D",
        dose_mg=100.0,
        k_bile_per_h=0.2,
        f_reabs=0.9,
        meal_times_h=[4.0, 12.0, 24.0],
        cl_L_per_h=3.0,
        vd_L=30.0,
        t_end_h=48.0,
        dt_h=0.05,
    )
    assert with_ehc.auc_mg_h_per_L > no_ehc.auc_mg_h_per_L
