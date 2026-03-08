"""Tests for Phase 414 — Drug Binding Kinetics to Target Receptor."""

import math

import pytest

from omega_pbpk.core.target_receptor_binding_kinetics import (
    ReceptorBindingResult,
    compare_koff_rates,
    simulate_receptor_binding,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DRUG = "DrugA"
RECEPTOR = "Beta1AR"
DRUG_CONC = 100.0  # nM
REC_CONC = 10.0  # nM
KON = 1e-3  # nM^-1 s^-1
KOFF = 1e-2  # s^-1   → Kd = 10 nM, MRT = 100 s


def _sim(
    drug_conc=DRUG_CONC,
    rec_conc=REC_CONC,
    kon=KON,
    koff=KOFF,
    t_end=3600.0,
    dt=1.0,
):
    return simulate_receptor_binding(
        drug_name=DRUG,
        receptor_name=RECEPTOR,
        drug_conc_nM=drug_conc,
        receptor_conc_nM=rec_conc,
        kon_per_nM_per_s=kon,
        koff_per_s=koff,
        t_end_s=t_end,
        dt_s=dt,
    )


# ---------------------------------------------------------------------------
# 1. Return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = _sim()
    assert isinstance(result, ReceptorBindingResult)


# ---------------------------------------------------------------------------
# 2. Time series have equal length
# ---------------------------------------------------------------------------


def test_time_series_equal_length():
    result = _sim()
    n = len(result.times_s)
    assert len(result.c_free_drug_nM) == n
    assert len(result.c_bound_drug_nM) == n
    assert len(result.receptor_occupancy_pct) == n


# ---------------------------------------------------------------------------
# 3. Initial conditions
# ---------------------------------------------------------------------------


def test_initial_free_drug_equals_input():
    result = _sim()
    assert abs(result.c_free_drug_nM[0] - DRUG_CONC) < 1e-9


def test_initial_bound_drug_is_zero():
    result = _sim()
    assert abs(result.c_bound_drug_nM[0]) < 1e-9


def test_initial_occupancy_is_zero():
    result = _sim()
    assert abs(result.receptor_occupancy_pct[0]) < 1e-9


# ---------------------------------------------------------------------------
# 4. DR increases from zero over time
# ---------------------------------------------------------------------------


def test_DR_increases_from_zero():
    result = _sim()
    assert result.c_bound_drug_nM[-1] > 0.0
    # monotonically non-decreasing early on (at least first half)
    mid = len(result.c_bound_drug_nM) // 4
    for i in range(1, mid):
        assert result.c_bound_drug_nM[i] >= result.c_bound_drug_nM[i - 1] - 1e-9


# ---------------------------------------------------------------------------
# 5. Occupancy rises toward equilibrium
# ---------------------------------------------------------------------------


def test_occupancy_rises_to_near_equilibrium():
    # Run long enough to approach equilibrium
    result = _sim(t_end=50000.0, dt=10.0)
    final_occ = result.receptor_occupancy_pct[-1]
    eq_occ = result.equilibrium_occupancy_pct
    # Should be within 5 percentage points
    assert abs(final_occ - eq_occ) < 5.0


# ---------------------------------------------------------------------------
# 6. Equilibrium occupancy formula: D / (D + Kd) * 100
# ---------------------------------------------------------------------------


def test_equilibrium_occupancy_formula():
    result = _sim()
    kd = KOFF / KON
    expected = DRUG_CONC / (DRUG_CONC + kd) * 100.0
    assert abs(result.equilibrium_occupancy_pct - expected) < 1e-9


# ---------------------------------------------------------------------------
# 7. Kd = koff / kon
# ---------------------------------------------------------------------------


def test_kd_equals_koff_over_kon():
    result = _sim()
    expected_kd = KOFF / KON
    assert abs(result.kd_nM - expected_kd) < 1e-12


# ---------------------------------------------------------------------------
# 8. Mass conservation: D + DR (approximately constant) ≤ initial drug
# ---------------------------------------------------------------------------


def test_mass_conservation_drug():
    result = _sim()
    total_0 = result.c_free_drug_nM[0] + result.c_bound_drug_nM[0]
    # At steady state, free + bound should still sum to initial (receptor << drug)
    for i in range(len(result.times_s)):
        total_i = result.c_free_drug_nM[i] + result.c_bound_drug_nM[i]
        assert total_i <= total_0 + 1e-6


# ---------------------------------------------------------------------------
# 9. t_half_dissociation = ln(2) / koff
# ---------------------------------------------------------------------------


def test_t_half_dissociation_formula():
    result = _sim()
    expected = math.log(2.0) / KOFF
    assert abs(result.t_half_dissociation_s - expected) < 1e-9


# ---------------------------------------------------------------------------
# 10. residence_time = 1 / koff
# ---------------------------------------------------------------------------


def test_residence_time_formula():
    result = _sim()
    expected = 1.0 / KOFF
    assert abs(result.residence_time_s - expected) < 1e-9


# ---------------------------------------------------------------------------
# 11. target_engagement_class based on residence_time
# ---------------------------------------------------------------------------


def test_target_engagement_class_transient():
    # koff = 1.0 s^-1 → residence = 1 s → transient
    result = _sim(koff=1.0, t_end=100.0)
    assert result.target_engagement_class == "transient"


def test_target_engagement_class_moderate():
    # koff = 0.01 → residence = 100 s → moderate
    result = _sim(koff=0.01, t_end=3600.0)
    assert result.target_engagement_class == "moderate"


def test_target_engagement_class_prolonged():
    # koff = 1e-4 → residence = 10000 s → prolonged
    result = _sim(koff=1e-4, t_end=3600.0)
    assert result.target_engagement_class == "prolonged"


# ---------------------------------------------------------------------------
# 12. clinical_duration_class
# ---------------------------------------------------------------------------


def test_clinical_duration_class_short():
    # koff = 1.0 → residence = 1 s → short
    result = _sim(koff=1.0, t_end=100.0)
    assert result.clinical_duration_class == "short"


def test_clinical_duration_class_medium():
    # koff = 1e-3 → residence = 1000 s ≈ 0.28 h → still short
    # koff = 1e-4 → residence = 10000 s ≈ 2.78 h → medium
    result = _sim(koff=1e-4, t_end=3600.0)
    assert result.clinical_duration_class == "medium"


def test_clinical_duration_class_long():
    # koff = 1e-5 → residence = 100000 s ≈ 27.8 h → long
    result = _sim(koff=1e-5, t_end=3600.0)
    assert result.clinical_duration_class == "long"


# ---------------------------------------------------------------------------
# 13. compare_koff_rates returns 5 results sorted by residence_time descending
# ---------------------------------------------------------------------------


def test_compare_koff_returns_five():
    results = compare_koff_rates(DRUG, RECEPTOR, DRUG_CONC, REC_CONC, KON)
    assert len(results) == 5


def test_compare_koff_sorted_descending_residence_time():
    results = compare_koff_rates(DRUG, RECEPTOR, DRUG_CONC, REC_CONC, KON)
    rts = [r.residence_time_s for r in results]
    assert rts == sorted(rts, reverse=True)


def test_compare_koff_custom_values():
    results = compare_koff_rates(
        DRUG,
        RECEPTOR,
        DRUG_CONC,
        REC_CONC,
        KON,
        koff_values=[0.01, 0.1],
    )
    assert len(results) == 2
    # First should have longer residence time (smaller koff)
    assert results[0].residence_time_s > results[1].residence_time_s


# ---------------------------------------------------------------------------
# 14. Validation errors
# ---------------------------------------------------------------------------


def test_validation_negative_drug_conc():
    with pytest.raises(ValueError, match="drug_conc_nM"):
        _sim(drug_conc=-1.0)


def test_validation_zero_receptor_conc():
    with pytest.raises(ValueError, match="receptor_conc_nM"):
        _sim(rec_conc=0.0)


def test_validation_negative_kon():
    with pytest.raises(ValueError, match="kon_per_nM_per_s"):
        _sim(kon=-1e-3)


def test_validation_negative_koff():
    with pytest.raises(ValueError, match="koff_per_s"):
        _sim(koff=-1e-2)


def test_validation_zero_t_end():
    with pytest.raises(ValueError, match="t_end_s"):
        _sim(t_end=0.0)


def test_validation_zero_dt():
    with pytest.raises(ValueError, match="dt_s"):
        _sim(dt=0.0)
