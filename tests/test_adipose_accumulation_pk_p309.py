"""Tests for Phase 309 — Adipose Accumulation PK."""

import pytest

from omega_pbpk.core.adipose_accumulation_pk_p309 import (
    AdiposeAccumulationResult,
    predict_washout,
    simulate_adipose_accumulation,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

DRUG_PARAMS = dict(
    drug_name="TestDrug",
    dose_mg=100.0,
    dosing_interval_h=24.0,
    n_doses=7,
    cl_sys_L_per_h=2.0,
    vd_plasma_L=50.0,
    kp_adipose=10.0,
    adipose_volume_L=15.0,
    adipose_blood_flow_L_per_h=0.5,
    logp=4.0,
)

WASHOUT_PARAMS = dict(
    drug_name="TestDrug",
    dose_mg=100.0,
    cl_sys_L_per_h=2.0,
    vd_plasma_L=50.0,
    kp_adipose=2.0,  # lower kp → faster washout
    adipose_volume_L=15.0,
    adipose_blood_flow_L_per_h=5.0,  # higher flow → faster equilibration
    initial_adipose_conc_mg_L=5.0,
    t_end_h=200.0,
)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_simulate_returns_correct_type(self):
        result = simulate_adipose_accumulation(**DRUG_PARAMS)
        assert isinstance(result, AdiposeAccumulationResult)

    def test_washout_returns_correct_type(self):
        result = predict_washout(**WASHOUT_PARAMS)
        assert isinstance(result, AdiposeAccumulationResult)


# ---------------------------------------------------------------------------
# Field preservation
# ---------------------------------------------------------------------------


class TestFieldPreservation:
    def test_drug_name_preserved(self):
        result = simulate_adipose_accumulation(**DRUG_PARAMS)
        assert result.drug_name == "TestDrug"

    def test_dose_mg_preserved(self):
        result = simulate_adipose_accumulation(**DRUG_PARAMS)
        assert result.dose_mg == 100.0

    def test_kp_adipose_preserved(self):
        result = simulate_adipose_accumulation(**DRUG_PARAMS)
        assert result.kp_adipose == 10.0

    def test_logp_preserved(self):
        result = simulate_adipose_accumulation(**DRUG_PARAMS)
        assert result.logp == 4.0


# ---------------------------------------------------------------------------
# Concentrations > 0
# ---------------------------------------------------------------------------


class TestConcentrationsPositive:
    def test_cmax_plasma_positive(self):
        result = simulate_adipose_accumulation(**DRUG_PARAMS)
        assert result.cmax_plasma_mg_L > 0.0

    def test_cmax_adipose_positive(self):
        result = simulate_adipose_accumulation(**DRUG_PARAMS)
        assert result.cmax_adipose_mg_L > 0.0

    def test_auc_plasma_positive(self):
        result = simulate_adipose_accumulation(**DRUG_PARAMS)
        assert result.auc_plasma_mg_h_per_L > 0.0

    def test_steady_state_adipose_positive(self):
        result = simulate_adipose_accumulation(**DRUG_PARAMS)
        assert result.steady_state_adipose_mg_L > 0.0

    def test_washout_cmax_plasma_positive(self):
        result = predict_washout(**WASHOUT_PARAMS)
        assert result.cmax_plasma_mg_L > 0.0

    def test_washout_cmax_adipose_positive(self):
        result = predict_washout(**WASHOUT_PARAMS)
        assert result.cmax_adipose_mg_L > 0.0


# ---------------------------------------------------------------------------
# Higher kp_adipose → higher adipose accumulation
# ---------------------------------------------------------------------------


class TestKpEffect:
    def test_higher_kp_gives_higher_cmax_adipose(self):
        low_kp = simulate_adipose_accumulation(**{**DRUG_PARAMS, "kp_adipose": 2.0})
        high_kp = simulate_adipose_accumulation(**{**DRUG_PARAMS, "kp_adipose": 20.0})
        assert high_kp.cmax_adipose_mg_L > low_kp.cmax_adipose_mg_L

    def test_higher_kp_gives_higher_steady_state(self):
        low_kp = simulate_adipose_accumulation(**{**DRUG_PARAMS, "kp_adipose": 2.0})
        high_kp = simulate_adipose_accumulation(**{**DRUG_PARAMS, "kp_adipose": 20.0})
        assert high_kp.steady_state_adipose_mg_L > low_kp.steady_state_adipose_mg_L


# ---------------------------------------------------------------------------
# Washout: adipose decreases over time
# ---------------------------------------------------------------------------


class TestWashout:
    def test_adipose_decreases_over_time(self):
        result = predict_washout(**WASHOUT_PARAMS)
        # Sample a few early vs late concentrations
        mid = len(result.c_adipose_mg_L) // 2
        assert result.c_adipose_mg_L[0] >= result.c_adipose_mg_L[mid]
        assert result.c_adipose_mg_L[mid] >= result.c_adipose_mg_L[-1]

    def test_washout_t50_is_positive(self):
        result = predict_washout(**WASHOUT_PARAMS)
        assert result.washout_t50_h > 0.0

    def test_washout_t50_less_than_tend(self):
        result = predict_washout(**WASHOUT_PARAMS)
        assert result.washout_t50_h < WASHOUT_PARAMS["t_end_h"]

    def test_accumulation_sentinel_in_washout(self):
        result = predict_washout(**WASHOUT_PARAMS)
        assert result.accumulation_ratio == -1.0

    def test_washout_sentinel_in_accumulation(self):
        result = simulate_adipose_accumulation(**DRUG_PARAMS)
        assert result.washout_t50_h == -1.0


# ---------------------------------------------------------------------------
# Accumulation ratio >= 1.0
# ---------------------------------------------------------------------------


class TestAccumulationRatio:
    def test_accumulation_ratio_gte_one_multidose(self):
        result = simulate_adipose_accumulation(**DRUG_PARAMS)
        assert result.accumulation_ratio >= 1.0

    def test_accumulation_ratio_increases_with_more_doses(self):
        few = simulate_adipose_accumulation(**{**DRUG_PARAMS, "n_doses": 2})
        many = simulate_adipose_accumulation(**{**DRUG_PARAMS, "n_doses": 14})
        assert many.accumulation_ratio >= few.accumulation_ratio


# ---------------------------------------------------------------------------
# Times list
# ---------------------------------------------------------------------------


class TestTimesAndLists:
    def test_times_non_empty(self):
        result = simulate_adipose_accumulation(**DRUG_PARAMS)
        assert len(result.times_h) > 1

    def test_lists_same_length(self):
        result = simulate_adipose_accumulation(**DRUG_PARAMS)
        assert len(result.times_h) == len(result.c_plasma_mg_L) == len(result.c_adipose_mg_L)

    def test_times_monotone(self):
        result = simulate_adipose_accumulation(**DRUG_PARAMS)
        for i in range(1, len(result.times_h)):
            assert result.times_h[i] >= result.times_h[i - 1]


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidationErrors:
    def test_dose_zero_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_adipose_accumulation(**{**DRUG_PARAMS, "dose_mg": 0.0})

    def test_dose_negative_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_adipose_accumulation(**{**DRUG_PARAMS, "dose_mg": -1.0})

    def test_cl_zero_raises(self):
        with pytest.raises(ValueError):
            simulate_adipose_accumulation(**{**DRUG_PARAMS, "cl_sys_L_per_h": 0.0})

    def test_vd_zero_raises(self):
        with pytest.raises(ValueError):
            simulate_adipose_accumulation(**{**DRUG_PARAMS, "vd_plasma_L": 0.0})

    def test_kp_zero_raises(self):
        with pytest.raises(ValueError):
            simulate_adipose_accumulation(**{**DRUG_PARAMS, "kp_adipose": 0.0})

    def test_adipose_volume_zero_raises(self):
        with pytest.raises(ValueError):
            simulate_adipose_accumulation(**{**DRUG_PARAMS, "adipose_volume_L": 0.0})

    def test_blood_flow_zero_raises(self):
        with pytest.raises(ValueError):
            simulate_adipose_accumulation(**{**DRUG_PARAMS, "adipose_blood_flow_L_per_h": 0.0})

    def test_n_doses_zero_raises(self):
        with pytest.raises(ValueError, match="n_doses"):
            simulate_adipose_accumulation(**{**DRUG_PARAMS, "n_doses": 0})

    def test_interval_zero_raises(self):
        with pytest.raises(ValueError, match="dosing_interval_h"):
            simulate_adipose_accumulation(**{**DRUG_PARAMS, "dosing_interval_h": 0.0})

    def test_washout_tend_zero_raises(self):
        with pytest.raises(ValueError, match="t_end_h"):
            predict_washout(**{**WASHOUT_PARAMS, "t_end_h": 0.0})
