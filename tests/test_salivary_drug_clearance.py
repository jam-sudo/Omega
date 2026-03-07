"""Tests for Phase 325 — Salivary Drug Clearance Model."""

import pytest

from omega_pbpk.clinical.salivary_drug_clearance import (
    SalivaryDrugResult,
    _henderson_hasselbalch_fraction_unionized,
    _oral_exposure_risk,
    simulate_salivary_clearance,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

BASE_KWARGS = dict(
    dose_mg=100.0,
    cl_plasma_L_per_h=5.0,
    vd_plasma_L=20.0,
    f_oral=1.0,
    pKa=7.4,
    drug_type="neutral",
    saliva_flow_mL_h=30.0,
    saliva_pH=6.8,
    plasma_pH=7.4,
    t_end_h=24.0,
    dt_h=0.1,
)


# ---------------------------------------------------------------------------
# 1. Return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = simulate_salivary_clearance(**BASE_KWARGS)
    assert isinstance(result, SalivaryDrugResult)


# ---------------------------------------------------------------------------
# 2. Array lengths consistent
# ---------------------------------------------------------------------------


def test_array_lengths_consistent():
    result = simulate_salivary_clearance(**BASE_KWARGS)
    n = len(result.times_h)
    assert n == len(result.c_plasma_mg_L)
    assert n == len(result.c_saliva_mg_L)
    assert n >= 2


# ---------------------------------------------------------------------------
# 3. Times start at 0 and are monotonically increasing
# ---------------------------------------------------------------------------


def test_times_start_at_zero():
    result = simulate_salivary_clearance(**BASE_KWARGS)
    assert result.times_h[0] == pytest.approx(0.0)


def test_times_monotonically_increasing():
    result = simulate_salivary_clearance(**BASE_KWARGS)
    for i in range(1, len(result.times_h)):
        assert result.times_h[i] > result.times_h[i - 1]


# ---------------------------------------------------------------------------
# 4. Plasma starts at dose/Vd
# ---------------------------------------------------------------------------


def test_plasma_initial_concentration():
    result = simulate_salivary_clearance(**BASE_KWARGS)
    expected_c0 = 100.0 / 20.0  # dose/Vd = 5.0 mg/L
    assert result.c_plasma_mg_L[0] == pytest.approx(expected_c0, rel=1e-6)


# ---------------------------------------------------------------------------
# 5. Plasma decreases monotonically (1-cpt elimination)
# ---------------------------------------------------------------------------


def test_plasma_decreases_over_time():
    result = simulate_salivary_clearance(**BASE_KWARGS)
    # Check overall trend: last value < first value
    assert result.c_plasma_mg_L[-1] < result.c_plasma_mg_L[0]


# ---------------------------------------------------------------------------
# 6. Salivary concentration tracks plasma (neutral drug: ratio = 1)
# ---------------------------------------------------------------------------


def test_saliva_tracks_plasma_neutral():
    result = simulate_salivary_clearance(**BASE_KWARGS)
    for cp, cs in zip(result.c_plasma_mg_L, result.c_saliva_mg_L):
        assert cs == pytest.approx(cp, rel=1e-9)


# ---------------------------------------------------------------------------
# 7. Neutral drug: saliva/plasma ratio = 1.0
# ---------------------------------------------------------------------------


def test_neutral_drug_ratio_one():
    result = simulate_salivary_clearance(**BASE_KWARGS)
    assert result.saliva_to_plasma_ratio == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 8. Acid drug at high saliva pH → more ionized in saliva → lower ratio
# ---------------------------------------------------------------------------


def test_acid_drug_high_saliva_ph_lower_ratio():
    result_acid = simulate_salivary_clearance(
        **{**BASE_KWARGS, "drug_type": "acid", "pKa": 6.0, "saliva_pH": 8.0}
    )
    # At saliva pH 8, pKa 6 → very ionized → fu_saliva small → ratio < 1
    assert result_acid.saliva_to_plasma_ratio < 1.0


# ---------------------------------------------------------------------------
# 9. Acid drug at lower pH ratio closer to 1 than high-pH case
# ---------------------------------------------------------------------------


def test_acid_drug_lower_saliva_ph_higher_ratio():
    result_low = simulate_salivary_clearance(
        **{**BASE_KWARGS, "drug_type": "acid", "pKa": 7.4, "saliva_pH": 6.5}
    )
    result_high = simulate_salivary_clearance(
        **{**BASE_KWARGS, "drug_type": "acid", "pKa": 7.4, "saliva_pH": 8.5}
    )
    assert result_low.saliva_to_plasma_ratio > result_high.saliva_to_plasma_ratio


# ---------------------------------------------------------------------------
# 10. Base drug at higher saliva pH → more un-ionized in saliva → higher ratio
# ---------------------------------------------------------------------------


def test_base_drug_high_ph_higher_ratio():
    result_base_low_ph = simulate_salivary_clearance(
        **{**BASE_KWARGS, "drug_type": "base", "pKa": 8.0, "saliva_pH": 6.0}
    )
    result_base_high_ph = simulate_salivary_clearance(
        **{**BASE_KWARGS, "drug_type": "base", "pKa": 8.0, "saliva_pH": 8.0}
    )
    # Higher saliva pH → base more un-ionized in saliva → fu_saliva larger → ratio larger
    assert result_base_high_ph.saliva_to_plasma_ratio > result_base_low_ph.saliva_to_plasma_ratio


# ---------------------------------------------------------------------------
# 11. Higher saliva flow → higher cumulative excretion
# ---------------------------------------------------------------------------


def test_higher_flow_higher_excretion():
    result_low = simulate_salivary_clearance(**{**BASE_KWARGS, "saliva_flow_mL_h": 10.0})
    result_high = simulate_salivary_clearance(**{**BASE_KWARGS, "saliva_flow_mL_h": 100.0})
    assert result_high.cumulative_saliva_excretion_mg > result_low.cumulative_saliva_excretion_mg


# ---------------------------------------------------------------------------
# 12. AUC plasma > 0
# ---------------------------------------------------------------------------


def test_auc_plasma_positive():
    result = simulate_salivary_clearance(**BASE_KWARGS)
    assert result.auc_plasma_mg_h_per_L > 0.0


# ---------------------------------------------------------------------------
# 13. AUC saliva > 0
# ---------------------------------------------------------------------------


def test_auc_saliva_positive():
    result = simulate_salivary_clearance(**BASE_KWARGS)
    assert result.auc_saliva_mg_h_per_L > 0.0


# ---------------------------------------------------------------------------
# 14. For neutral drug: AUC saliva == AUC plasma
# ---------------------------------------------------------------------------


def test_neutral_drug_auc_saliva_equals_plasma():
    result = simulate_salivary_clearance(**BASE_KWARGS)
    assert result.auc_saliva_mg_h_per_L == pytest.approx(result.auc_plasma_mg_h_per_L, rel=1e-6)


# ---------------------------------------------------------------------------
# 15. Cumulative excretion > 0
# ---------------------------------------------------------------------------


def test_cumulative_excretion_positive():
    result = simulate_salivary_clearance(**BASE_KWARGS)
    assert result.cumulative_saliva_excretion_mg > 0.0


# ---------------------------------------------------------------------------
# 16. oral_exposure_risk is valid string
# ---------------------------------------------------------------------------


def test_oral_exposure_risk_valid():
    result = simulate_salivary_clearance(**BASE_KWARGS)
    assert result.oral_exposure_risk in ("low", "moderate", "high")


# ---------------------------------------------------------------------------
# 17. Low salivary AUC → "low" risk
# ---------------------------------------------------------------------------


def test_low_auc_low_risk():
    # Very small dose + large Vd
    result = simulate_salivary_clearance(
        dose_mg=1.0,
        cl_plasma_L_per_h=20.0,
        vd_plasma_L=200.0,
        drug_type="neutral",
        t_end_h=24.0,
        dt_h=0.1,
    )
    assert result.oral_exposure_risk == "low"


# ---------------------------------------------------------------------------
# 18. High salivary AUC → "high" risk
# ---------------------------------------------------------------------------


def test_high_auc_high_risk():
    # Large dose, slow clearance
    result = simulate_salivary_clearance(
        dose_mg=5000.0,
        cl_plasma_L_per_h=0.5,
        vd_plasma_L=10.0,
        drug_type="neutral",
        t_end_h=48.0,
        dt_h=0.1,
    )
    assert result.oral_exposure_risk == "high"


# ---------------------------------------------------------------------------
# 19. Validation: dose <= 0
# ---------------------------------------------------------------------------


def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_salivary_clearance(**{**BASE_KWARGS, "dose_mg": 0.0})


def test_validation_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_salivary_clearance(**{**BASE_KWARGS, "dose_mg": -10.0})


# ---------------------------------------------------------------------------
# 20. Validation: cl <= 0
# ---------------------------------------------------------------------------


def test_validation_cl_zero():
    with pytest.raises(ValueError, match="cl_plasma_L_per_h"):
        simulate_salivary_clearance(**{**BASE_KWARGS, "cl_plasma_L_per_h": 0.0})


# ---------------------------------------------------------------------------
# 21. Validation: vd <= 0
# ---------------------------------------------------------------------------


def test_validation_vd_zero():
    with pytest.raises(ValueError, match="vd_plasma_L"):
        simulate_salivary_clearance(**{**BASE_KWARGS, "vd_plasma_L": 0.0})


# ---------------------------------------------------------------------------
# 22. Validation: invalid drug_type
# ---------------------------------------------------------------------------


def test_validation_invalid_drug_type():
    with pytest.raises(ValueError, match="drug_type"):
        simulate_salivary_clearance(**{**BASE_KWARGS, "drug_type": "amphoteric"})


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------


def test_hh_neutral():
    assert _henderson_hasselbalch_fraction_unionized(7.4, 7.4, "neutral") == 1.0


def test_hh_acid_at_pka():
    # At pH = pKa, fraction unionized = 0.5
    fu = _henderson_hasselbalch_fraction_unionized(7.4, 7.4, "acid")
    assert fu == pytest.approx(0.5, rel=1e-6)


def test_hh_base_at_pka():
    fu = _henderson_hasselbalch_fraction_unionized(7.4, 7.4, "base")
    assert fu == pytest.approx(0.5, rel=1e-6)


def test_oral_exposure_risk_moderate():
    assert _oral_exposure_risk(5.0) == "moderate"
