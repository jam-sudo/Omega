"""Tests for cerebral blood flow PBPK model."""

import pytest

from omega_pbpk.core.cbf_pbpk import CBFPKResult, compare_cbf_conditions, simulate_cbf_pk

# ── Validation tests ──────────────────────────────────────────────────


class TestValidation:
    def test_dose_zero_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_cbf_pk("X", 0, 5, 50)

    def test_dose_negative_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_cbf_pk("X", -10, 5, 50)

    def test_cl_zero_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            simulate_cbf_pk("X", 100, 0, 50)

    def test_cl_negative_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            simulate_cbf_pk("X", 100, -1, 50)

    def test_vd_zero_raises(self):
        with pytest.raises(ValueError, match="vd_L"):
            simulate_cbf_pk("X", 100, 5, 0)

    def test_vd_negative_raises(self):
        with pytest.raises(ValueError, match="vd_L"):
            simulate_cbf_pk("X", 100, 5, -10)

    def test_cbf_factor_zero_raises(self):
        with pytest.raises(ValueError, match="cbf_factor"):
            simulate_cbf_pk("X", 100, 5, 50, cbf_factor=0)

    def test_cbf_factor_negative_raises(self):
        with pytest.raises(ValueError, match="cbf_factor"):
            simulate_cbf_pk("X", 100, 5, 50, cbf_factor=-0.5)


# ── Result structure tests ────────────────────────────────────────────


class TestResultStructure:
    def test_returns_cbfpk_result(self):
        r = simulate_cbf_pk("DrugA", 100, 5, 50)
        assert isinstance(r, CBFPKResult)

    def test_result_fields(self):
        r = simulate_cbf_pk("DrugA", 100, 5, 50)
        assert r.drug_name == "DrugA"
        assert r.dose_mg == 100
        assert r.cbf_factor == 1.0
        assert len(r.times_h) == len(r.c_plasma_mg_L)
        assert len(r.times_h) == len(r.c_brain_vasc_mg_L)
        assert len(r.times_h) == len(r.c_brain_tissue_mg_L)

    def test_frozen_dataclass(self):
        r = simulate_cbf_pk("DrugA", 100, 5, 50)
        with pytest.raises(AttributeError):
            r.dose_mg = 200  # type: ignore[misc]

    def test_times_start_at_zero(self):
        r = simulate_cbf_pk("DrugA", 100, 5, 50)
        assert r.times_h[0] == 0.0

    def test_times_end_at_t_end(self):
        r = simulate_cbf_pk("DrugA", 100, 5, 50, t_end_h=12.0)
        assert abs(r.times_h[-1] - 12.0) < 0.01


# ── PK behavior tests ────────────────────────────────────────────────


class TestPKBehavior:
    def test_higher_cbf_higher_brain_cmax(self):
        r_low = simulate_cbf_pk("X", 100, 5, 50, cbf_factor=0.5)
        r_high = simulate_cbf_pk("X", 100, 5, 50, cbf_factor=1.5)
        assert r_high.cmax_brain_tissue > r_low.cmax_brain_tissue

    def test_higher_cbf_higher_brain_vasc_cmax(self):
        r_low = simulate_cbf_pk("X", 100, 5, 50, cbf_factor=0.5)
        r_high = simulate_cbf_pk("X", 100, 5, 50, cbf_factor=1.5)
        assert r_high.cmax_brain_vasc > r_low.cmax_brain_vasc

    def test_linear_pk_double_dose(self):
        r1 = simulate_cbf_pk("X", 100, 5, 50)
        r2 = simulate_cbf_pk("X", 200, 5, 50)
        assert abs(r2.cmax_plasma / r1.cmax_plasma - 2.0) < 0.01

    def test_linear_pk_double_dose_brain(self):
        r1 = simulate_cbf_pk("X", 100, 5, 50)
        r2 = simulate_cbf_pk("X", 200, 5, 50)
        ratio = r2.cmax_brain_tissue / r1.cmax_brain_tissue
        assert abs(ratio - 2.0) < 0.1

    def test_auc_plasma_positive(self):
        r = simulate_cbf_pk("X", 100, 5, 50)
        assert r.auc_plasma > 0

    def test_auc_brain_positive(self):
        r = simulate_cbf_pk("X", 100, 5, 50)
        assert r.auc_brain_tissue > 0

    def test_brain_to_plasma_ratio_positive(self):
        r = simulate_cbf_pk("X", 100, 5, 50)
        assert r.brain_to_plasma_ratio > 0

    def test_brain_to_plasma_ratio_less_than_one(self):
        r = simulate_cbf_pk("X", 100, 5, 50)
        assert r.brain_to_plasma_ratio < 1.0

    def test_concentrations_nonnegative(self):
        r = simulate_cbf_pk("X", 100, 5, 50)
        assert all(c >= 0 for c in r.c_plasma_mg_L)
        assert all(c >= 0 for c in r.c_brain_vasc_mg_L)
        assert all(c >= 0 for c in r.c_brain_tissue_mg_L)

    def test_cmax_plasma_equals_dose_over_vd(self):
        r = simulate_cbf_pk("X", 100, 5, 50)
        assert abs(r.cmax_plasma - 100 / 50) < 0.1


# ── Notes tests ───────────────────────────────────────────────────────


class TestNotes:
    def test_normal_cbf_note(self):
        r = simulate_cbf_pk("X", 100, 5, 50, cbf_factor=1.0)
        assert "Normal CBF" in r.notes

    def test_reduced_cbf_note(self):
        r = simulate_cbf_pk("X", 100, 5, 50, cbf_factor=0.5)
        assert "Reduced CBF" in r.notes

    def test_increased_cbf_note(self):
        r = simulate_cbf_pk("X", 100, 5, 50, cbf_factor=1.5)
        assert "Increased CBF" in r.notes


# ── Compare conditions tests ─────────────────────────────────────────


class TestCompareConditions:
    def test_returns_correct_count(self):
        results = compare_cbf_conditions("X", 100, 5, 50, [0.5, 1.0, 1.5])
        assert len(results) == 3

    def test_cbf_factors_preserved(self):
        results = compare_cbf_conditions("X", 100, 5, 50, [0.5, 1.0, 1.5])
        assert [r.cbf_factor for r in results] == [0.5, 1.0, 1.5]

    def test_empty_cbf_factors_raises(self):
        with pytest.raises(ValueError, match="cbf_factors"):
            compare_cbf_conditions("X", 100, 5, 50, [])

    def test_compare_ordering(self):
        results = compare_cbf_conditions("X", 100, 5, 50, [0.5, 1.0, 2.0])
        cmaxes = [r.cmax_brain_tissue for r in results]
        assert cmaxes[0] < cmaxes[1] < cmaxes[2]
