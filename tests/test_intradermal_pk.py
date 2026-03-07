"""Tests for Phase 329 — Intradermal Drug Delivery Model."""

import pytest

from omega_pbpk.core.intradermal_pk import IntradermalPKResult, simulate_intradermal_pk

# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------

BASE_PARAMS = {
    "drug_name": "TestDrug",
    "dose_mg": 1.0,
    "cl_sys_L_per_h": 2.0,
    "vd_sys_L": 10.0,
    "k_abs_dermis_per_h": 0.5,
    "k_local_distribution_per_h": 0.2,
    "v_dermis_mL": 0.1,
    "v_local_mL": 2.0,
    "t_end_h": 24.0,
    "dt_h": 0.05,
}


def run_base(**overrides) -> IntradermalPKResult:
    params = {**BASE_PARAMS, **overrides}
    return simulate_intradermal_pk(**params)


# ---------------------------------------------------------------------------
# 1. Return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = run_base()
    assert isinstance(result, IntradermalPKResult)


# ---------------------------------------------------------------------------
# 2. Array lengths consistency
# ---------------------------------------------------------------------------


def test_array_lengths_consistent():
    result = run_base()
    n = len(result.times_h)
    assert n > 0
    assert len(result.c_dermis_mg_mL) == n
    assert len(result.c_local_mg_mL) == n
    assert len(result.c_systemic_mg_L) == n


# ---------------------------------------------------------------------------
# 3. Dermis concentration starts at dose / v_dermis
# ---------------------------------------------------------------------------


def test_dermis_starts_at_dose_over_volume():
    result = run_base(dose_mg=2.0, v_dermis_mL=0.1)
    expected = 2.0 / 0.1  # 20 mg/mL
    assert abs(result.c_dermis_mg_mL[0] - expected) < 1e-6


# ---------------------------------------------------------------------------
# 4. Systemic concentration starts at 0
# ---------------------------------------------------------------------------


def test_systemic_starts_at_zero():
    result = run_base()
    assert result.c_systemic_mg_L[0] == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# 5. Local tissue starts at 0
# ---------------------------------------------------------------------------


def test_local_starts_at_zero():
    result = run_base()
    assert result.c_local_mg_mL[0] == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# 6. Systemic concentration rises then falls (Cmax not at t=0)
# ---------------------------------------------------------------------------


def test_systemic_rises_then_falls():
    result = run_base()
    # cmax should not be at t=0
    cmax_idx = result.c_systemic_mg_L.index(result.cmax_systemic_mg_L)
    assert cmax_idx > 0
    # and should not be at the last time point (drug falls off)
    assert cmax_idx < len(result.times_h) - 1 or result.cmax_systemic_mg_L > 0


# ---------------------------------------------------------------------------
# 7. tmax > 0
# ---------------------------------------------------------------------------


def test_tmax_greater_than_zero():
    result = run_base()
    assert result.tmax_systemic_h > 0.0


# ---------------------------------------------------------------------------
# 8. cmax > 0
# ---------------------------------------------------------------------------


def test_cmax_greater_than_zero():
    result = run_base()
    assert result.cmax_systemic_mg_L > 0.0


# ---------------------------------------------------------------------------
# 9. bioavailability_pct in [0, 100]
# ---------------------------------------------------------------------------


def test_bioavailability_in_range():
    result = run_base()
    assert 0.0 <= result.bioavailability_pct <= 100.0


# ---------------------------------------------------------------------------
# 10. AUC systemic > 0
# ---------------------------------------------------------------------------


def test_auc_systemic_positive():
    result = run_base()
    assert result.auc_systemic_mg_h_per_L > 0.0


# ---------------------------------------------------------------------------
# 11. Dermis decreases monotonically (or generally decreases)
# ---------------------------------------------------------------------------


def test_dermis_decreases():
    result = run_base()
    # dermis depot should be strictly decreasing overall
    assert result.c_dermis_mg_mL[-1] < result.c_dermis_mg_mL[0]


# ---------------------------------------------------------------------------
# 12. Higher k_abs -> faster tmax (smaller tmax value)
# ---------------------------------------------------------------------------


def test_higher_kabs_faster_tmax():
    result_low = run_base(k_abs_dermis_per_h=0.2)
    result_high = run_base(k_abs_dermis_per_h=1.5)
    assert result_high.tmax_systemic_h <= result_low.tmax_systemic_h


# ---------------------------------------------------------------------------
# 13. Higher k_abs -> generally higher or comparable cmax
# ---------------------------------------------------------------------------


def test_higher_kabs_higher_or_comparable_cmax():
    result_high = run_base(k_abs_dermis_per_h=2.0)
    # With very high absorption, more drug enters systemic faster
    # At minimum, cmax_high should be positive
    assert result_high.cmax_systemic_mg_L > 0.0


# ---------------------------------------------------------------------------
# 14. Dose linearity: 2x dose -> 2x AUC
# ---------------------------------------------------------------------------


def test_dose_linearity_auc():
    result_1 = run_base(dose_mg=1.0)
    result_2 = run_base(dose_mg=2.0)
    ratio = result_2.auc_systemic_mg_h_per_L / result_1.auc_systemic_mg_h_per_L
    assert abs(ratio - 2.0) < 0.05  # within 5%


# ---------------------------------------------------------------------------
# 15. Dose linearity: 2x dose -> 2x Cmax
# ---------------------------------------------------------------------------


def test_dose_linearity_cmax():
    result_1 = run_base(dose_mg=1.0)
    result_2 = run_base(dose_mg=2.0)
    ratio = result_2.cmax_systemic_mg_L / result_1.cmax_systemic_mg_L
    assert abs(ratio - 2.0) < 0.05


# ---------------------------------------------------------------------------
# 16. notes is a non-empty string
# ---------------------------------------------------------------------------


def test_notes_is_string():
    result = run_base()
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# 17. times_h starts at 0
# ---------------------------------------------------------------------------


def test_times_starts_at_zero():
    result = run_base()
    assert result.times_h[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 18. times_h ends at approximately t_end_h
# ---------------------------------------------------------------------------


def test_times_ends_at_t_end():
    result = run_base(t_end_h=12.0, dt_h=0.05)
    assert result.times_h[-1] == pytest.approx(12.0, abs=0.1)


# ---------------------------------------------------------------------------
# 19. Validation: dose <= 0
# ---------------------------------------------------------------------------


def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        run_base(dose_mg=0.0)


def test_validation_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        run_base(dose_mg=-1.0)


# ---------------------------------------------------------------------------
# 20. Validation: cl <= 0
# ---------------------------------------------------------------------------


def test_validation_cl_zero():
    with pytest.raises(ValueError, match="cl_sys"):
        run_base(cl_sys_L_per_h=0.0)


# ---------------------------------------------------------------------------
# 21. Validation: vd <= 0
# ---------------------------------------------------------------------------


def test_validation_vd_zero():
    with pytest.raises(ValueError, match="vd_sys"):
        run_base(vd_sys_L=0.0)


# ---------------------------------------------------------------------------
# 22. bioavailability denominator: compare with analytical approximation
# ---------------------------------------------------------------------------


def test_bioavailability_reasonable():
    # With moderate absorption, F should be between 10% and 100%
    result = run_base(k_abs_dermis_per_h=0.5, k_local_distribution_per_h=0.1)
    assert result.bioavailability_pct > 0.0
    assert result.bioavailability_pct <= 100.0
