"""Tests for Phase 523 — Mesenteric Lymph Node Drug Distribution."""

import pytest

from omega_pbpk.core.mesenteric_lymph_node_pk import (
    MesentericLymphNodeResult,
    compare_logp_lymph,
    simulate_mesenteric_lymph_pk,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def default_result(**kwargs):
    params = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        logP=4.0,
    )
    params.update(kwargs)
    return simulate_mesenteric_lymph_pk(**params)


# ---------------------------------------------------------------------------
# Return type and field tests
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_mesenteric_result(self):
        r = default_result()
        assert isinstance(r, MesentericLymphNodeResult)

    def test_drug_name_stored(self):
        r = default_result(drug_name="DrugX")
        assert r.drug_name == "DrugX"

    def test_dose_stored(self):
        r = default_result(dose_mg=200.0)
        assert r.dose_mg == 200.0

    def test_logp_stored(self):
        r = default_result(logP=3.5)
        assert r.logP == 3.5

    def test_times_h_is_list(self):
        r = default_result()
        assert isinstance(r.times_h, list)

    def test_a_portal_mg_is_list(self):
        r = default_result()
        assert isinstance(r.a_portal_mg, list)

    def test_a_mln_mg_is_list(self):
        r = default_result()
        assert isinstance(r.a_mln_mg, list)

    def test_c_systemic_is_list(self):
        r = default_result()
        assert isinstance(r.c_systemic_mg_L, list)

    def test_notes_is_str(self):
        r = default_result()
        assert isinstance(r.notes, str)


# ---------------------------------------------------------------------------
# f_lymph / f_portal tests
# ---------------------------------------------------------------------------


class TestLymphaticFraction:
    def test_low_logp_zero_f_lymph(self):
        """logP <= 2 → f_lymph = 0."""
        r = default_result(logP=1.0)
        assert r.f_lymph == 0.0

    def test_logp_exactly_2_zero_f_lymph(self):
        r = default_result(logP=2.0)
        assert r.f_lymph == 0.0

    def test_high_logp_positive_f_lymph(self):
        """logP = 5 → f_lymph = 0.3."""
        r = default_result(logP=5.0)
        assert r.f_lymph > 0.0

    def test_f_lymph_capped_at_0_3(self):
        r = default_result(logP=10.0)
        assert r.f_lymph <= 0.3

    def test_f_portal_complements_f_lymph(self):
        r = default_result(logP=4.0)
        assert abs(r.f_lymph + r.f_portal - 1.0) < 1e-9

    def test_lymph_contribution_pct_matches_f_lymph(self):
        r = default_result(logP=4.0)
        assert abs(r.lymph_contribution_pct - r.f_lymph * 100.0) < 1e-9


# ---------------------------------------------------------------------------
# Time-series shape tests
# ---------------------------------------------------------------------------


class TestTimeSeries:
    def test_times_starts_at_zero(self):
        r = default_result()
        assert r.times_h[0] == 0.0

    def test_mln_amount_starts_at_zero(self):
        r = default_result()
        assert r.a_mln_mg[0] == 0.0

    def test_portal_amount_starts_at_zero(self):
        r = default_result()
        assert r.a_portal_mg[0] == 0.0

    def test_systemic_conc_starts_at_zero(self):
        r = default_result()
        assert r.c_systemic_mg_L[0] == 0.0

    def test_mln_rises_from_zero(self):
        """MLN amount should increase above zero for lipophilic drug."""
        r = default_result(logP=5.0)
        assert max(r.a_mln_mg) > 0.0

    def test_systemic_conc_rises_and_falls(self):
        r = default_result()
        # Should have a Tmax well before Tend
        assert r.tmax_systemic_h > 0.0
        assert r.tmax_systemic_h < 24.0
        # Concentration at end < Cmax
        assert r.c_systemic_mg_L[-1] < r.cmax_systemic


# ---------------------------------------------------------------------------
# Derived metric tests
# ---------------------------------------------------------------------------


class TestDerivedMetrics:
    def test_cmax_mln_nonneg(self):
        r = default_result()
        assert r.cmax_mln_mg >= 0.0

    def test_mln_peak_pct_between_0_and_100(self):
        r = default_result(logP=5.0)
        assert 0.0 <= r.mln_peak_pct <= 100.0

    def test_mln_peak_pct_zero_for_low_logp(self):
        """Low logP → no MLN accumulation."""
        r = default_result(logP=1.0)
        assert r.cmax_mln_mg == 0.0
        assert r.mln_peak_pct == 0.0

    def test_cmax_systemic_positive(self):
        r = default_result()
        assert r.cmax_systemic > 0.0

    def test_auc_systemic_positive(self):
        r = default_result()
        assert r.auc_systemic > 0.0

    def test_tmax_mln_after_zero(self):
        r = default_result(logP=5.0)
        assert r.tmax_mln_h > 0.0


# ---------------------------------------------------------------------------
# Dose linearity
# ---------------------------------------------------------------------------


class TestDoseLinearity:
    def test_cmax_doubles_with_doubled_dose(self):
        r1 = default_result(dose_mg=100.0)
        r2 = default_result(dose_mg=200.0)
        ratio = r2.cmax_systemic / r1.cmax_systemic
        assert abs(ratio - 2.0) < 0.01

    def test_auc_doubles_with_doubled_dose(self):
        r1 = default_result(dose_mg=100.0)
        r2 = default_result(dose_mg=200.0)
        ratio = r2.auc_systemic / r1.auc_systemic
        assert abs(ratio - 2.0) < 0.01

    def test_cmax_mln_doubles_with_doubled_dose(self):
        r1 = default_result(dose_mg=100.0, logP=5.0)
        r2 = default_result(dose_mg=200.0, logP=5.0)
        ratio = r2.cmax_mln_mg / r1.cmax_mln_mg
        assert abs(ratio - 2.0) < 0.01


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_mesenteric_lymph_pk("D", -10.0, logP=3.0)

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_mesenteric_lymph_pk("D", 0.0, logP=3.0)

    def test_zero_cl_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            simulate_mesenteric_lymph_pk("D", 100.0, logP=3.0, cl_L_per_h=0.0)

    def test_negative_vd_raises(self):
        with pytest.raises(ValueError, match="vd_sys_L"):
            simulate_mesenteric_lymph_pk("D", 100.0, logP=3.0, vd_sys_L=-5.0)


# ---------------------------------------------------------------------------
# compare_logp_lymph tests
# ---------------------------------------------------------------------------


class TestCompareLogp:
    def test_returns_list(self):
        result = compare_logp_lymph("Drug", 100.0, [1.0, 3.0, 5.0])
        assert isinstance(result, list)

    def test_correct_number_of_results(self):
        result = compare_logp_lymph("Drug", 100.0, [1.0, 3.0, 5.0])
        assert len(result) == 3

    def test_sorted_descending_by_lymph_contribution(self):
        result = compare_logp_lymph("Drug", 100.0, [1.0, 2.0, 4.0, 6.0])
        for i in range(len(result) - 1):
            assert result[i].lymph_contribution_pct >= result[i + 1].lymph_contribution_pct

    def test_highest_logp_first(self):
        result = compare_logp_lymph("Drug", 100.0, [1.0, 3.0, 5.0])
        # logP=5 has highest lymph contribution → should be first
        assert result[0].logP > result[-1].logP

    def test_all_results_are_mesenteric_type(self):
        results = compare_logp_lymph("Drug", 100.0, [2.0, 4.0])
        for r in results:
            assert isinstance(r, MesentericLymphNodeResult)
