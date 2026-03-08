"""Tests for Phase 226 — wound_pk."""

import pytest

from omega_pbpk.core.wound_pk import (
    WoundPKResult,
    compare_wound_depths,
    simulate_wound_pk,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_result(**kwargs):
    defaults = dict(
        drug_name="Mupirocin",
        dose_mg=10.0,
        wound_depth="partial_thickness",
        formulation_viscosity_cp=1.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
        t_end_h=24.0,
    )
    defaults.update(kwargs)
    return simulate_wound_pk(**defaults)


# ---------------------------------------------------------------------------
# 1. Return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = _default_result()
    assert isinstance(result, WoundPKResult)


# ---------------------------------------------------------------------------
# 2. Field preservation
# ---------------------------------------------------------------------------


def test_field_preservation():
    result = _default_result(drug_name="TestDrug", dose_mg=20.0, wound_depth="superficial")
    assert result.drug_name == "TestDrug"
    assert result.dose_mg == pytest.approx(20.0)
    assert result.wound_depth == "superficial"


# ---------------------------------------------------------------------------
# 3. c_surface starts > 0 (= dose / V_surface = 10/2 = 5)
# ---------------------------------------------------------------------------


def test_c_surface_initial_positive():
    result = _default_result(dose_mg=10.0)
    assert result.c_surface_mg_mL[0] > 0.0


def test_c_surface_initial_value():
    # V_surface = 2.0 mL, dose = 10 mg → C_surface(0) = 5 mg/mL
    result = _default_result(dose_mg=10.0)
    assert result.c_surface_mg_mL[0] == pytest.approx(5.0, rel=1e-3)


# ---------------------------------------------------------------------------
# 4. c_surface decreases monotonically (it only has outflow)
# ---------------------------------------------------------------------------


def test_c_surface_decreases():
    result = _default_result()
    surface = result.c_surface_mg_mL
    # Check surface is non-increasing (small floating point tolerance)
    for i in range(1, min(len(surface), 100)):
        assert surface[i] <= surface[i - 1] + 1e-9


# ---------------------------------------------------------------------------
# 5. c_wound starts at 0 and rises above 0
# ---------------------------------------------------------------------------


def test_c_wound_starts_zero():
    result = _default_result()
    assert result.c_wound_mg_g[0] == pytest.approx(0.0, abs=1e-12)


def test_c_wound_rises():
    result = _default_result()
    # Should be > 0 at some point before 24 h
    assert max(result.c_wound_mg_g) > 0.0


# ---------------------------------------------------------------------------
# 6. c_systemic starts at 0
# ---------------------------------------------------------------------------


def test_c_systemic_starts_zero():
    result = _default_result()
    assert result.c_systemic_mg_L[0] == pytest.approx(0.0, abs=1e-12)


def test_c_systemic_eventually_positive():
    result = _default_result()
    assert result.cmax_systemic_mg_L >= 0.0  # may be very small but should be >= 0


# ---------------------------------------------------------------------------
# 7. full_thickness → higher wound AUC than superficial
# ---------------------------------------------------------------------------


def test_full_thickness_higher_auc_than_superficial():
    full = simulate_wound_pk(
        drug_name="Drug",
        dose_mg=10.0,
        wound_depth="full_thickness",
        cl_L_per_h=5.0,
        vd_L=50.0,
    )
    superficial = simulate_wound_pk(
        drug_name="Drug",
        dose_mg=10.0,
        wound_depth="superficial",
        cl_L_per_h=5.0,
        vd_L=50.0,
    )
    assert full.auc_wound_mg_h_per_g > superficial.auc_wound_mg_h_per_g


# ---------------------------------------------------------------------------
# 8. cmax_wound and tmax_wound consistency
# ---------------------------------------------------------------------------


def test_cmax_wound_matches_max():
    result = _default_result()
    assert result.cmax_wound_mg_g == pytest.approx(max(result.c_wound_mg_g), rel=1e-6)


def test_tmax_wound_in_time_range():
    result = _default_result()
    assert 0.0 <= result.tmax_wound_h <= 24.0


# ---------------------------------------------------------------------------
# 9. auc_wound > 0
# ---------------------------------------------------------------------------


def test_auc_wound_positive():
    result = _default_result()
    assert result.auc_wound_mg_h_per_g > 0.0


# ---------------------------------------------------------------------------
# 10. compare_wound_depths returns 3 results sorted by wound AUC descending
# ---------------------------------------------------------------------------


def test_compare_returns_3_results():
    results = compare_wound_depths(
        drug_name="Drug",
        dose_mg=10.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
    )
    assert len(results) == 3


def test_compare_sorted_by_auc_descending():
    results = compare_wound_depths(
        drug_name="Drug",
        dose_mg=10.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
    )
    aucs = [r.auc_wound_mg_h_per_g for r in results]
    assert aucs == sorted(aucs, reverse=True)


def test_compare_all_wound_depths_represented():
    results = compare_wound_depths(
        drug_name="Drug",
        dose_mg=10.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
    )
    depths = {r.wound_depth for r in results}
    assert depths == {"superficial", "partial_thickness", "full_thickness"}


# ---------------------------------------------------------------------------
# 11. mic_coverage_h >= 0
# ---------------------------------------------------------------------------


def test_mic_coverage_non_negative():
    result = _default_result()
    assert result.mic_coverage_h >= 0.0


def test_mic_coverage_low_mic_gives_higher_coverage():
    result_low_mic = simulate_wound_pk(
        drug_name="Drug",
        dose_mg=50.0,
        wound_depth="full_thickness",
        cl_L_per_h=5.0,
        vd_L=50.0,
        mic_mg_g=0.001,  # very low MIC
    )
    result_high_mic = simulate_wound_pk(
        drug_name="Drug",
        dose_mg=50.0,
        wound_depth="full_thickness",
        cl_L_per_h=5.0,
        vd_L=50.0,
        mic_mg_g=1000.0,  # very high MIC
    )
    assert result_low_mic.mic_coverage_h >= result_high_mic.mic_coverage_h


# ---------------------------------------------------------------------------
# 12. Time arrays have correct length
# ---------------------------------------------------------------------------


def test_time_array_length_consistency():
    result = _default_result()
    n = len(result.times_h)
    assert len(result.c_surface_mg_mL) == n
    assert len(result.c_wound_mg_g) == n
    assert len(result.c_systemic_mg_L) == n


def test_time_starts_at_zero():
    result = _default_result()
    assert result.times_h[0] == pytest.approx(0.0)


def test_time_ends_at_t_end():
    result = _default_result(t_end_h=12.0)
    assert result.times_h[-1] == pytest.approx(12.0, rel=1e-3)


# ---------------------------------------------------------------------------
# 13. Viscosity effect — higher viscosity → lower wound AUC
# ---------------------------------------------------------------------------


def test_higher_viscosity_lower_wound_auc():
    low_visc = simulate_wound_pk(
        drug_name="Drug",
        dose_mg=10.0,
        wound_depth="partial_thickness",
        formulation_viscosity_cp=1.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
    )
    high_visc = simulate_wound_pk(
        drug_name="Drug",
        dose_mg=10.0,
        wound_depth="partial_thickness",
        formulation_viscosity_cp=100.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
    )
    assert low_visc.auc_wound_mg_h_per_g > high_visc.auc_wound_mg_h_per_g


# ---------------------------------------------------------------------------
# 14. Dose scaling — higher dose → higher wound Cmax
# ---------------------------------------------------------------------------


def test_higher_dose_higher_cmax():
    low = simulate_wound_pk(
        "Drug", dose_mg=5.0, wound_depth="partial_thickness", cl_L_per_h=5.0, vd_L=50.0
    )
    high = simulate_wound_pk(
        "Drug", dose_mg=50.0, wound_depth="partial_thickness", cl_L_per_h=5.0, vd_L=50.0
    )
    assert high.cmax_wound_mg_g > low.cmax_wound_mg_g


# ---------------------------------------------------------------------------
# 15. Validation errors
# ---------------------------------------------------------------------------


def test_dose_zero_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_wound_pk("Drug", dose_mg=0.0, wound_depth="superficial", cl_L_per_h=5.0, vd_L=50.0)


def test_dose_negative_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_wound_pk(
            "Drug", dose_mg=-1.0, wound_depth="superficial", cl_L_per_h=5.0, vd_L=50.0
        )


def test_invalid_wound_depth_raises():
    with pytest.raises(ValueError, match="wound_depth"):
        simulate_wound_pk(
            "Drug", dose_mg=10.0, wound_depth="deep_tissue", cl_L_per_h=5.0, vd_L=50.0
        )


def test_cl_zero_raises():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_wound_pk(
            "Drug", dose_mg=10.0, wound_depth="superficial", cl_L_per_h=0.0, vd_L=50.0
        )


def test_cl_negative_raises():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_wound_pk(
            "Drug", dose_mg=10.0, wound_depth="superficial", cl_L_per_h=-5.0, vd_L=50.0
        )


def test_compare_dose_zero_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        compare_wound_depths("Drug", dose_mg=0.0, cl_L_per_h=5.0, vd_L=50.0)


def test_compare_cl_zero_raises():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        compare_wound_depths("Drug", dose_mg=10.0, cl_L_per_h=0.0, vd_L=50.0)


# ---------------------------------------------------------------------------
# 16. notes field is non-empty string
# ---------------------------------------------------------------------------


def test_notes_non_empty():
    result = _default_result()
    assert isinstance(result.notes, str)
    assert len(result.notes) > 5
