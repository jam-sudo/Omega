"""Tests for gene therapy PK/PD module (AAV vector simulation)."""

from __future__ import annotations

import pytest

from omega_pbpk.core.gene_therapy_pk import (
    AAVPKResult,
    compare_doses,
    simulate_aav_pk,
)

# ── Input validation ────────────────────────────────────────────────


class TestValidation:
    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_vg_per_kg"):
            simulate_aav_pk("TestGT", dose_vg_per_kg=-1e13)

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_vg_per_kg"):
            simulate_aav_pk("TestGT", dose_vg_per_kg=0.0)

    def test_zero_bw_raises(self):
        with pytest.raises(ValueError, match="bw_kg"):
            simulate_aav_pk("TestGT", dose_vg_per_kg=1e13, bw_kg=0.0)

    def test_negative_bw_raises(self):
        with pytest.raises(ValueError, match="bw_kg"):
            simulate_aav_pk("TestGT", dose_vg_per_kg=1e13, bw_kg=-70.0)

    def test_zero_blood_clear_raises(self):
        with pytest.raises(ValueError, match="k_blood_clear"):
            simulate_aav_pk("TestGT", dose_vg_per_kg=1e13, k_blood_clear_per_h=0.0)

    def test_zero_k_transduce_raises(self):
        with pytest.raises(ValueError, match="k_transduce"):
            simulate_aav_pk("TestGT", dose_vg_per_kg=1e13, k_transduce_per_h=0.0)

    def test_zero_k_express_raises(self):
        with pytest.raises(ValueError, match="k_express"):
            simulate_aav_pk("TestGT", dose_vg_per_kg=1e13, k_express_per_day=0.0)

    def test_zero_k_protein_deg_raises(self):
        with pytest.raises(ValueError, match="k_protein_deg"):
            simulate_aav_pk("TestGT", dose_vg_per_kg=1e13, k_protein_deg_per_day=0.0)

    def test_zero_t_end_raises(self):
        with pytest.raises(ValueError, match="t_end_days"):
            simulate_aav_pk("TestGT", dose_vg_per_kg=1e13, t_end_days=0.0)

    def test_invalid_f_liver_tropic_raises(self):
        with pytest.raises(ValueError, match="f_liver_tropic"):
            simulate_aav_pk("TestGT", dose_vg_per_kg=1e13, f_liver_tropic=0.0)

    def test_f_liver_tropic_above_1_raises(self):
        with pytest.raises(ValueError, match="f_liver_tropic"):
            simulate_aav_pk("TestGT", dose_vg_per_kg=1e13, f_liver_tropic=1.5)


# ── Result type and structure ───────────────────────────────────────


class TestResultStructure:
    def setup_method(self):
        self.result = simulate_aav_pk(
            "AAV9-hFIX",
            dose_vg_per_kg=5e13,
            bw_kg=70.0,
            t_end_days=365.0,
            dt_days=1.0,
        )

    def test_returns_aav_pk_result(self):
        assert isinstance(self.result, AAVPKResult)

    def test_drug_name_stored(self):
        assert self.result.drug_name == "AAV9-hFIX"

    def test_dose_stored(self):
        assert self.result.dose_vg_per_kg == pytest.approx(5e13)

    def test_bw_stored(self):
        assert self.result.bw_kg == pytest.approx(70.0)

    def test_times_days_is_list(self):
        assert isinstance(self.result.times_days, list)

    def test_v_blood_is_list(self):
        assert isinstance(self.result.v_blood, list)

    def test_v_liver_copies_is_list(self):
        assert isinstance(self.result.v_liver_copies, list)

    def test_protein_expression_is_list(self):
        assert isinstance(self.result.protein_expression, list)

    def test_array_lengths_equal(self):
        n = len(self.result.times_days)
        assert len(self.result.v_blood) == n
        assert len(self.result.v_liver_copies) == n
        assert len(self.result.protein_expression) == n

    def test_notes_is_string(self):
        assert isinstance(self.result.notes, str)
        assert len(self.result.notes) > 0


# ── Vector blood dynamics ────────────────────────────────────────────


class TestBloodVectorDynamics:
    def setup_method(self):
        self.result = simulate_aav_pk(
            "AAV9-hFIX",
            dose_vg_per_kg=1e13,
            bw_kg=70.0,
            k_blood_clear_per_h=0.5,
            k_transduce_per_h=0.1,
            t_end_days=365.0,
            dt_days=1.0,
        )

    def test_v_blood_starts_high(self):
        # Blood concentration should be at maximum at t=0
        assert self.result.v_blood[0] > 0
        assert self.result.v_blood[0] >= max(self.result.v_blood[1:])

    def test_v_blood_decreases_monotonically(self):
        # Blood vector should decrease (clearance dominates)
        v = self.result.v_blood
        # After first few days, should be strictly decreasing
        for i in range(1, len(v) - 1):
            assert v[i + 1] <= v[i] + 1e-10  # allow numerical tolerance

    def test_v_blood_approaches_zero(self):
        # After 365 days at high clearance rate, blood vector ~0
        assert self.result.v_blood[-1] < self.result.v_blood[0] * 1e-6

    def test_t_half_blood_days_positive(self):
        assert self.result.t_half_blood_days > 0

    def test_t_half_blood_days_reasonable(self):
        # Half-life should be hours to days (not weeks or microseconds)
        # k_blood_clear=0.5/h → t½ ~ 1.38h = 0.058 days (plus transduction)
        assert 0.01 < self.result.t_half_blood_days < 5.0


# ── Liver transduction dynamics ──────────────────────────────────────


class TestLiverDynamics:
    def setup_method(self):
        self.result = simulate_aav_pk(
            "AAV9",
            dose_vg_per_kg=1e13,
            bw_kg=70.0,
            t_end_days=365.0,
            dt_days=0.5,
        )

    def test_v_liver_starts_at_zero(self):
        assert self.result.v_liver_copies[0] == pytest.approx(0.0)

    def test_v_liver_increases_then_plateaus(self):
        v = self.result.v_liver_copies
        # Should increase in early phase
        assert max(v) > v[0]
        # Eventually the liver copies plateau or peak
        max_idx = v.index(max(v))
        assert max_idx > 0  # not at time 0

    def test_liver_transduction_efficiency_in_0_1(self):
        eff = self.result.liver_transduction_efficiency
        assert 0.0 <= eff <= 1.0

    def test_liver_transduction_efficiency_positive(self):
        assert self.result.liver_transduction_efficiency > 0


# ── Protein expression ───────────────────────────────────────────────


class TestProteinExpression:
    def setup_method(self):
        self.result = simulate_aav_pk(
            "AAV9-hFIX",
            dose_vg_per_kg=1e13,
            bw_kg=70.0,
            k_express_per_day=100.0,
            k_protein_deg_per_day=0.2,
            t_end_days=365.0,
            dt_days=1.0,
        )

    def test_protein_starts_at_zero(self):
        assert self.result.protein_expression[0] == pytest.approx(0.0)

    def test_protein_increases(self):
        assert max(self.result.protein_expression) > 0

    def test_protein_peaks_after_days_not_immediately(self):
        # Protein buildup takes weeks due to transduction lag
        assert self.result.t_peak_protein_days > 0

    def test_protein_at_1year_positive(self):
        assert self.result.protein_at_1year > 0

    def test_peak_protein_positive(self):
        assert self.result.peak_protein > 0

    def test_all_protein_values_nonnegative(self):
        assert all(p >= 0 for p in self.result.protein_expression)


# ── Dose proportionality ─────────────────────────────────────────────


class TestDoseProportionality:
    def test_higher_dose_higher_protein_at_1year(self):
        r_low = simulate_aav_pk("AAV9", dose_vg_per_kg=1e13, bw_kg=70.0, dt_days=2.0)
        r_high = simulate_aav_pk("AAV9", dose_vg_per_kg=1e14, bw_kg=70.0, dt_days=2.0)
        assert r_high.protein_at_1year > r_low.protein_at_1year

    def test_10x_dose_roughly_10x_protein(self):
        r_low = simulate_aav_pk("AAV9", dose_vg_per_kg=1e13, bw_kg=70.0, dt_days=2.0)
        r_high = simulate_aav_pk("AAV9", dose_vg_per_kg=1e14, bw_kg=70.0, dt_days=2.0)
        ratio = r_high.protein_at_1year / r_low.protein_at_1year
        assert 5.0 < ratio < 20.0  # approximately proportional (model is linear)


# ── compare_doses ────────────────────────────────────────────────────


class TestCompareDoses:
    def test_sorted_by_peak_protein_descending(self):
        doses = [1e12, 1e13, 1e14]
        results = compare_doses("AAV9", doses, bw_kg=70.0, dt_days=5.0, t_end_days=365.0)
        peaks = [r.peak_protein for r in results]
        assert peaks == sorted(peaks, reverse=True)

    def test_returns_correct_count(self):
        doses = [1e12, 5e12, 1e13]
        results = compare_doses("AAV9", doses, bw_kg=70.0, dt_days=5.0, t_end_days=365.0)
        assert len(results) == 3

    def test_all_results_are_aav_pk_result(self):
        doses = [1e13, 5e13]
        results = compare_doses("AAV9", doses, bw_kg=70.0, dt_days=5.0, t_end_days=365.0)
        assert all(isinstance(r, AAVPKResult) for r in results)
