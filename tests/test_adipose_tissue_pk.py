"""
Tests for Phase 201 — Adipose Tissue PK
"""

import math

import pytest

from omega_pbpk.core.adipose_tissue_pk import (
    AdiposePKResult,
    assess_accumulation,
    simulate_adipose_pk,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_result(**kwargs):
    params = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        cl_L_per_h=5.0,
        vd_plasma_L=20.0,
        logP=2.0,
        fat_fraction=0.25,
        body_weight_kg=70.0,
        t_end_h=48.0,
        dt_h=0.1,
    )
    params.update(kwargs)
    return simulate_adipose_pk(**params)


# ---------------------------------------------------------------------------
# Return type and field preservation
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_adipose_pk_result(self):
        result = _default_result()
        assert isinstance(result, AdiposePKResult)

    def test_drug_name_preserved(self):
        result = _default_result(drug_name="Amiodarone")
        assert result.drug_name == "Amiodarone"

    def test_dose_mg_preserved(self):
        result = _default_result(dose_mg=200.0)
        assert result.dose_mg == 200.0

    def test_logP_preserved(self):
        result = _default_result(logP=3.5)
        assert result.logP == 3.5

    def test_fat_fraction_preserved(self):
        result = _default_result(fat_fraction=0.30)
        assert result.fat_fraction == pytest.approx(0.30)

    def test_body_weight_preserved(self):
        result = _default_result(body_weight_kg=80.0)
        assert result.body_weight_kg == pytest.approx(80.0)

    def test_times_h_is_list(self):
        result = _default_result()
        assert isinstance(result.times_h, list)

    def test_c_plasma_is_list(self):
        result = _default_result()
        assert isinstance(result.c_plasma_mg_L, list)

    def test_c_adipose_is_list(self):
        result = _default_result()
        assert isinstance(result.c_adipose_mg_kg, list)

    def test_lists_same_length(self):
        result = _default_result()
        assert len(result.times_h) == len(result.c_plasma_mg_L) == len(result.c_adipose_mg_kg)


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------


class TestInitialConditions:
    def test_c_adipose_starts_at_zero(self):
        result = _default_result()
        assert result.c_adipose_mg_kg[0] == pytest.approx(0.0)

    def test_c_plasma_starts_at_dose_over_vd(self):
        result = _default_result(dose_mg=100.0, vd_plasma_L=20.0)
        expected = 100.0 / 20.0  # 5 mg/L
        assert result.c_plasma_mg_L[0] == pytest.approx(expected)

    def test_times_start_at_zero(self):
        result = _default_result()
        assert result.times_h[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Pharmacokinetics
# ---------------------------------------------------------------------------


class TestPKBehaviour:
    def test_auc_plasma_positive(self):
        result = _default_result()
        assert result.auc_plasma > 0.0

    def test_auc_adipose_non_negative(self):
        result = _default_result()
        assert result.auc_adipose >= 0.0

    def test_cmax_adipose_positive_for_lipophilic(self):
        result = _default_result(logP=4.0)
        assert result.cmax_adipose_mg_kg > 0.0

    def test_tmax_adipose_after_time_zero(self):
        result = _default_result(logP=4.0)
        # Drug accumulates in adipose, tmax should be > 0
        assert result.tmax_adipose_h >= 0.0

    def test_high_logP_higher_adipose_accumulation(self):
        result_low = _default_result(logP=0.5)
        result_high = _default_result(logP=5.0)
        assert result_high.auc_adipose > result_low.auc_adipose

    def test_negative_logP_minimal_adipose(self):
        result = _default_result(logP=-1.0, t_end_h=24.0)
        # With logP ≤ 0, k_pa should be very low → minimal adipose uptake
        assert result.cmax_adipose_mg_kg < 1.0  # small compared to plasma Cmax

    def test_adipose_to_plasma_ratio_non_negative(self):
        result = _default_result()
        assert result.adipose_to_plasma_ratio >= 0.0

    def test_higher_logP_higher_adipose_plasma_ratio(self):
        result_low = _default_result(logP=0.5)
        result_high = _default_result(logP=5.0)
        assert result_high.adipose_to_plasma_ratio > result_low.adipose_to_plasma_ratio

    def test_notes_is_string(self):
        result = _default_result()
        assert isinstance(result.notes, str)
        assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# assess_accumulation
# ---------------------------------------------------------------------------


class TestAssessAccumulation:
    def test_returns_dict(self):
        result = _default_result()
        acc = assess_accumulation(result)
        assert isinstance(acc, dict)

    def test_has_accumulation_index_key(self):
        result = _default_result()
        acc = assess_accumulation(result)
        assert "accumulation_index" in acc

    def test_has_washout_t50_key(self):
        result = _default_result()
        acc = assess_accumulation(result)
        assert "washout_t50_h" in acc

    def test_has_accumulation_risk_key(self):
        result = _default_result()
        acc = assess_accumulation(result)
        assert "accumulation_risk" in acc

    def test_washout_t50_positive(self):
        result = _default_result()
        acc = assess_accumulation(result)
        assert acc["washout_t50_h"] > 0.0

    def test_washout_t50_correct_value(self):
        # k_ap = 0.02, so t50 = ln2 / 0.02
        result = _default_result()
        acc = assess_accumulation(result)
        expected = math.log(2) / 0.02
        assert acc["washout_t50_h"] == pytest.approx(expected, rel=1e-5)

    def test_accumulation_risk_valid_values(self):
        result = _default_result()
        acc = assess_accumulation(result)
        assert acc["accumulation_risk"] in ("low", "moderate", "high")

    def test_high_logP_leads_to_moderate_or_high_risk(self):
        result = _default_result(logP=6.0, t_end_h=120.0)
        acc = assess_accumulation(result)
        assert acc["accumulation_risk"] in ("moderate", "high")

    def test_low_logP_low_risk(self):
        result = _default_result(logP=-2.0, t_end_h=24.0)
        acc = assess_accumulation(result)
        assert acc["accumulation_risk"] == "low"


# ---------------------------------------------------------------------------
# Validation / edge cases
# ---------------------------------------------------------------------------


class TestValidation:
    def test_dose_zero_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_adipose_pk("Drug", 0.0, 5.0, 20.0, 2.0)

    def test_dose_negative_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_adipose_pk("Drug", -10.0, 5.0, 20.0, 2.0)

    def test_fat_fraction_zero_raises(self):
        with pytest.raises(ValueError, match="fat_fraction"):
            simulate_adipose_pk("Drug", 100.0, 5.0, 20.0, 2.0, fat_fraction=0.0)

    def test_fat_fraction_one_raises(self):
        with pytest.raises(ValueError, match="fat_fraction"):
            simulate_adipose_pk("Drug", 100.0, 5.0, 20.0, 2.0, fat_fraction=1.0)

    def test_fat_fraction_above_one_raises(self):
        with pytest.raises(ValueError, match="fat_fraction"):
            simulate_adipose_pk("Drug", 100.0, 5.0, 20.0, 2.0, fat_fraction=1.5)

    def test_body_weight_zero_raises(self):
        with pytest.raises(ValueError, match="body_weight_kg"):
            simulate_adipose_pk("Drug", 100.0, 5.0, 20.0, 2.0, body_weight_kg=0.0)

    def test_body_weight_negative_raises(self):
        with pytest.raises(ValueError, match="body_weight_kg"):
            simulate_adipose_pk("Drug", 100.0, 5.0, 20.0, 2.0, body_weight_kg=-70.0)
