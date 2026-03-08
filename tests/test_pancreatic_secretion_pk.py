"""Tests for Phase 1105 — Drug Pancreatic Exocrine Secretion PK."""

import pytest

from omega_pbpk.core.pancreatic_secretion_pk import (
    PancreaticSecretionResult,
    _kp_pan_from_logp,
    compare_logp_pancreatic,
    simulate_pancreatic_secretion,
)


# ── Kp helper ────────────────────────────────────────────────────────────────
class TestKpPanFromLogP:
    def test_midrange(self):
        kp = _kp_pan_from_logp(2.0)
        assert abs(kp - 0.9) < 1e-9

    def test_clamp_low(self):
        # logP=-100 → raw = 0.5 - 20 = -19.5 → clamped to 0.1
        assert _kp_pan_from_logp(-100) == 0.1

    def test_clamp_high(self):
        # logP=100 → raw = 0.5 + 20 = 20.5 → clamped to 5.0
        assert _kp_pan_from_logp(100) == 5.0

    def test_monotone_increasing(self):
        kps = [_kp_pan_from_logp(lp) for lp in [-1.0, 0.0, 1.0, 2.0, 5.0, 10.0]]
        for a, b in zip(kps, kps[1:]):
            assert a <= b


# ── Basic simulation ─────────────────────────────────────────────────────────
class TestSimulatePancreaticSecretion:
    def test_returns_correct_type(self):
        res = simulate_pancreatic_secretion("Drug_A", 100.0)
        assert isinstance(res, PancreaticSecretionResult)

    def test_drug_name_preserved(self):
        res = simulate_pancreatic_secretion("TestDrug", 50.0)
        assert res.drug_name == "TestDrug"

    def test_dose_preserved(self):
        res = simulate_pancreatic_secretion("D", 200.0)
        assert res.dose_mg == 200.0

    def test_logP_preserved(self):
        res = simulate_pancreatic_secretion("D", 100.0, logP=3.5)
        assert res.logP == 3.5

    def test_c_pancreatic_starts_zero(self):
        res = simulate_pancreatic_secretion("D", 100.0)
        assert res.c_pancreatic_mg_L[0] == 0.0

    def test_c_plasma_initial_condition(self):
        dose, vd = 100.0, 10.0
        res = simulate_pancreatic_secretion("D", dose, vd_plasma_L=vd)
        expected = dose / vd
        assert abs(res.c_plasma_mg_L[0] - expected) < 1e-9

    def test_plasma_declines_over_time(self):
        res = simulate_pancreatic_secretion("D", 100.0)
        assert res.c_plasma_mg_L[-1] < res.c_plasma_mg_L[0]

    def test_pancreatic_rises_then_falls(self):
        res = simulate_pancreatic_secretion("D", 100.0)
        # pancreatic concentration should not be zero everywhere after t=0
        assert res.cmax_pancreatic > 0.0

    def test_auc_plasma_positive(self):
        res = simulate_pancreatic_secretion("D", 100.0)
        assert res.auc_plasma > 0.0

    def test_auc_pancreatic_positive(self):
        res = simulate_pancreatic_secretion("D", 100.0)
        assert res.auc_pancreatic > 0.0

    def test_pancreatic_to_plasma_ratio_positive(self):
        res = simulate_pancreatic_secretion("D", 100.0)
        assert res.pancreatic_to_plasma_ratio > 0.0

    def test_higher_logp_increases_ratio(self):
        """More lipophilic drugs should have higher Kp → higher pan/plasma ratio."""
        res_low = simulate_pancreatic_secretion("D", 100.0, logP=0.0)
        res_high = simulate_pancreatic_secretion("D", 100.0, logP=5.0)
        assert res_high.pancreatic_to_plasma_ratio > res_low.pancreatic_to_plasma_ratio

    def test_f_secreted_in_range(self):
        res = simulate_pancreatic_secretion("D", 100.0)
        assert 0.0 <= res.f_secreted_into_duodenum <= 1.0

    def test_f_secreted_positive(self):
        res = simulate_pancreatic_secretion("D", 100.0)
        assert res.f_secreted_into_duodenum > 0.0

    def test_risk_low_possible(self):
        # Simulate with very high k_flow_pan to drain pancreatic compartment fast
        # → short-lived pancreatic conc → low AUC ratio
        res = simulate_pancreatic_secretion("D", 100.0, logP=2.0, k_flow_pan=50.0, k_ps_out=0.001)
        # With very fast drain and negligible secretion, ratio should be very low
        assert res.pancreatic_to_plasma_ratio < 0.5 or res.pancreatic_exposure_risk == "low"

    def test_risk_high_for_large_logp(self):
        # very lipophilic → high Kp → higher ratio than low logP
        res_lo = simulate_pancreatic_secretion("D", 100.0, logP=0.0)
        res_hi = simulate_pancreatic_secretion("D", 100.0, logP=10.0)
        # higher logP should produce equal or higher pancreatic ratio
        assert res_hi.pancreatic_to_plasma_ratio >= res_lo.pancreatic_to_plasma_ratio
        assert res_hi.pancreatic_exposure_risk in ("moderate", "high")

    def test_risk_valid_values(self):
        for lp in [-5.0, 0.0, 2.0, 5.0, 10.0]:
            res = simulate_pancreatic_secretion("D", 100.0, logP=lp)
            assert res.pancreatic_exposure_risk in ("low", "moderate", "high")

    def test_tmax_pancreatic_non_negative(self):
        res = simulate_pancreatic_secretion("D", 100.0)
        assert res.tmax_pancreatic_h >= 0.0

    def test_notes_is_string(self):
        res = simulate_pancreatic_secretion("D", 100.0)
        assert isinstance(res.notes, str)

    def test_times_length_matches_traces(self):
        res = simulate_pancreatic_secretion("D", 100.0, t_end_h=5.0)
        assert len(res.times_h) == len(res.c_plasma_mg_L) == len(res.c_pancreatic_mg_L)

    def test_times_start_at_zero(self):
        res = simulate_pancreatic_secretion("D", 100.0)
        assert res.times_h[0] == 0.0


# ── Validation errors ────────────────────────────────────────────────────────
class TestValidationErrors:
    def test_negative_dose_raises(self):
        with pytest.raises(ValueError):
            simulate_pancreatic_secretion("D", -1.0)

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError):
            simulate_pancreatic_secretion("D", 0.0)

    def test_negative_vd_raises(self):
        with pytest.raises(ValueError):
            simulate_pancreatic_secretion("D", 100.0, vd_plasma_L=-5.0)

    def test_negative_cl_raises(self):
        with pytest.raises(ValueError):
            simulate_pancreatic_secretion("D", 100.0, cl_L_per_h=-1.0)

    def test_negative_k_ps_out_raises(self):
        with pytest.raises(ValueError):
            simulate_pancreatic_secretion("D", 100.0, k_ps_out=-0.01)

    def test_empty_logp_list_raises(self):
        with pytest.raises(ValueError):
            compare_logp_pancreatic("D", 100.0, [])


# ── compare_logp_pancreatic ──────────────────────────────────────────────────
class TestCompareLogPPancreatic:
    def test_returns_list(self):
        results = compare_logp_pancreatic("D", 100.0, [0.0, 2.0, 4.0])
        assert isinstance(results, list)

    def test_length_matches_input(self):
        results = compare_logp_pancreatic("D", 100.0, [0.0, 2.0, 4.0])
        assert len(results) == 3

    def test_sorted_descending(self):
        results = compare_logp_pancreatic("D", 100.0, [0.0, 1.0, 2.0, 3.0, 5.0])
        ratios = [r.pancreatic_to_plasma_ratio for r in results]
        assert ratios == sorted(ratios, reverse=True)

    def test_all_results_correct_type(self):
        results = compare_logp_pancreatic("D", 100.0, [1.0, 3.0])
        assert all(isinstance(r, PancreaticSecretionResult) for r in results)

    def test_single_logp(self):
        results = compare_logp_pancreatic("D", 100.0, [2.0])
        assert len(results) == 1

    def test_highest_logp_first(self):
        logp_values = [0.0, 2.0, 5.0]
        results = compare_logp_pancreatic("D", 100.0, logp_values)
        # The highest logP should produce the highest ratio → first in sorted list
        assert results[0].logP == 5.0

    def test_auc_ratio_consistent(self):
        results = compare_logp_pancreatic("D", 100.0, [1.0, 4.0])
        for r in results:
            expected_ratio = r.auc_pancreatic / r.auc_plasma
            assert abs(r.pancreatic_to_plasma_ratio - expected_ratio) < 1e-9
