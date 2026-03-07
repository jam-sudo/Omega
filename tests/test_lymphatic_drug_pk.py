"""
Tests for Phase 210 — Lymphatic Drug Transport
src/omega_pbpk/core/lymphatic_drug_pk.py
"""

import pytest

from omega_pbpk.core.lymphatic_drug_pk import (
    LymphaticTransportResult,
    compare_logp_effect,
    simulate_lymphatic_transport,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default(**overrides):
    """Run simulation with sensible defaults."""
    kwargs = dict(
        drug_name="DrugX",
        dose_mg=100.0,
        logP=3.0,
        cl_L_per_h=10.0,
        vd_L=50.0,
        t_end_h=24.0,
        dt_h=0.05,
    )
    kwargs.update(overrides)
    return simulate_lymphatic_transport(**kwargs)


# ---------------------------------------------------------------------------
# 1. Return type and dataclass structure
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_lymphatic_transport_result(self):
        r = _default()
        assert isinstance(r, LymphaticTransportResult)

    def test_has_list_fields(self):
        r = _default()
        assert isinstance(r.times_h, list)
        assert isinstance(r.c_systemic_mg_L, list)
        assert isinstance(r.a_lymph_thoracic_mg, list)

    def test_list_lengths_equal(self):
        r = _default()
        assert len(r.times_h) == len(r.c_systemic_mg_L) == len(r.a_lymph_thoracic_mg)

    def test_result_is_mutable(self):
        """Plain dataclass — should NOT raise on mutation."""
        r = _default()
        r.notes = "modified"
        assert r.notes == "modified"


# ---------------------------------------------------------------------------
# 2. Field preservation
# ---------------------------------------------------------------------------


class TestFieldPreservation:
    def test_drug_name_preserved(self):
        r = _default(drug_name="TestDrug")
        assert r.drug_name == "TestDrug"

    def test_dose_mg_preserved(self):
        r = _default(dose_mg=200.0)
        assert r.dose_mg == pytest.approx(200.0)

    def test_logP_preserved(self):
        r = _default(logP=4.5)
        assert r.logP == pytest.approx(4.5)

    def test_notes_passed_through(self):
        r = simulate_lymphatic_transport("D", 100.0, 3.0, 10.0, 50.0, notes="custom note")
        assert r.notes == "custom note"


# ---------------------------------------------------------------------------
# 3. Time grid
# ---------------------------------------------------------------------------


class TestTimeGrid:
    def test_time_starts_at_zero(self):
        r = _default()
        assert r.times_h[0] == pytest.approx(0.0)

    def test_time_ends_near_t_end(self):
        r = _default(t_end_h=12.0)
        assert r.times_h[-1] == pytest.approx(12.0, abs=0.1)

    def test_time_monotonically_increasing(self):
        r = _default()
        for i in range(1, len(r.times_h)):
            assert r.times_h[i] > r.times_h[i - 1]


# ---------------------------------------------------------------------------
# 4. Initial conditions
# ---------------------------------------------------------------------------


class TestInitialConditions:
    def test_c_systemic_starts_at_zero(self):
        r = _default()
        assert r.c_systemic_mg_L[0] == pytest.approx(0.0, abs=1e-10)

    def test_thoracic_starts_at_zero(self):
        r = _default()
        assert r.a_lymph_thoracic_mg[0] == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# 5. Cmax and Tmax
# ---------------------------------------------------------------------------


class TestCmaxTmax:
    def test_cmax_positive(self):
        r = _default()
        assert r.cmax_mg_L > 0.0

    def test_cmax_equals_max_trajectory(self):
        r = _default()
        assert r.cmax_mg_L == pytest.approx(max(r.c_systemic_mg_L), rel=1e-6)

    def test_tmax_within_simulation(self):
        r = _default(t_end_h=24.0)
        assert 0.0 <= r.tmax_h <= 24.0

    def test_auc_positive(self):
        r = _default()
        assert r.auc_mg_h_per_L > 0.0


# ---------------------------------------------------------------------------
# 6. logP-dependent lymphatic fraction
# ---------------------------------------------------------------------------


class TestLymphaticFraction:
    def test_high_logP_high_f_lymph(self):
        r = _default(logP=7.0)
        assert r.f_lymph_fraction > 0.5

    def test_low_logP_low_f_lymph(self):
        r = _default(logP=1.0)
        assert r.f_lymph_fraction < 0.2

    def test_midpoint_logP4_fraction_near_half(self):
        r = _default(logP=4.0)
        assert abs(r.f_lymph_fraction - 0.5) < 0.05

    def test_f_lymph_between_0_and_1(self):
        for lp in [-2.0, 0.0, 3.0, 5.0, 8.0]:
            r = _default(logP=lp)
            assert 0.0 <= r.f_lymph_fraction <= 1.0

    def test_high_logP_raises_lymphatic_auc_contribution(self):
        r_high = _default(logP=7.0)
        r_low = _default(logP=1.0)
        assert r_high.lymphatic_auc_contribution_pct > r_low.lymphatic_auc_contribution_pct

    def test_lymphatic_auc_contribution_pct_non_negative(self):
        r = _default(logP=3.0)
        assert r.lymphatic_auc_contribution_pct >= 0.0


# ---------------------------------------------------------------------------
# 7. compare_logp_effect
# ---------------------------------------------------------------------------


class TestCompareLogPEffect:
    def test_returns_list(self):
        result = compare_logp_effect("D", 100.0, 10.0, 50.0, [1.0, 3.0, 6.0])
        assert isinstance(result, list)

    def test_correct_length(self):
        logps = [1.0, 2.0, 4.0, 6.0, 8.0]
        result = compare_logp_effect("D", 100.0, 10.0, 50.0, logps)
        assert len(result) == len(logps)

    def test_sorted_by_f_lymph_descending(self):
        result = compare_logp_effect("D", 100.0, 10.0, 50.0, [1.0, 3.0, 5.0, 7.0])
        fracs = [r.f_lymph_fraction for r in result]
        assert fracs == sorted(fracs, reverse=True)

    def test_first_item_has_highest_f_lymph(self):
        result = compare_logp_effect("D", 100.0, 10.0, 50.0, [1.0, 4.0, 7.0])
        assert result[0].f_lymph_fraction >= result[-1].f_lymph_fraction

    def test_all_results_are_lymphatic_transport_result(self):
        result = compare_logp_effect("D", 100.0, 10.0, 50.0, [2.0, 5.0])
        for r in result:
            assert isinstance(r, LymphaticTransportResult)

    def test_single_logp_returns_one_item(self):
        result = compare_logp_effect("D", 100.0, 10.0, 50.0, [3.0])
        assert len(result) == 1


# ---------------------------------------------------------------------------
# 8. Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_dose_zero_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_lymphatic_transport("D", 0.0, 3.0, 10.0, 50.0)

    def test_dose_negative_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_lymphatic_transport("D", -50.0, 3.0, 10.0, 50.0)

    def test_cl_zero_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            simulate_lymphatic_transport("D", 100.0, 3.0, 0.0, 50.0)

    def test_cl_negative_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            simulate_lymphatic_transport("D", 100.0, 3.0, -5.0, 50.0)

    def test_compare_invalid_dose_raises(self):
        with pytest.raises(ValueError):
            compare_logp_effect("D", -10.0, 10.0, 50.0, [3.0])
