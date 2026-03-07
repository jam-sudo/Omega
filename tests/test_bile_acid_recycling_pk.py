"""Tests for Phase 278 — Bile Acid Recycling PK Model."""

import pytest

from omega_pbpk.core.bile_acid_recycling_pk import (
    BileAcidRecyclingResult,
    compare_ehc_fractions,
    simulate_bile_acid_recycling,
)

# --- Helpers ---
DRUG = "Cholylsarcosine"

DEFAULT_SIM_KWARGS = dict(
    drug_name=DRUG,
    dose_mg=100.0,
    cl_hepatic_L_per_h=5.0,
    vd_sys_L=50.0,
    ka_per_h=1.0,
    f_ehc=0.5,
    k_bile_per_h=0.3,
    t_end_h=24.0,
    dt_h=0.05,
)


def make_result(**overrides) -> BileAcidRecyclingResult:
    kwargs = {**DEFAULT_SIM_KWARGS, **overrides}
    return simulate_bile_acid_recycling(**kwargs)


# --- Return type and structure ---


class TestReturnType:
    def test_returns_correct_type(self):
        result = make_result()
        assert isinstance(result, BileAcidRecyclingResult)

    def test_drug_name_preserved(self):
        result = make_result()
        assert result.drug_name == DRUG

    def test_dose_mg_preserved(self):
        result = make_result(dose_mg=200.0)
        assert result.dose_mg == 200.0

    def test_f_ehc_preserved(self):
        result = make_result(f_ehc=0.3)
        assert result.f_ehc == 0.3

    def test_k_bile_preserved(self):
        result = make_result(k_bile_per_h=0.4)
        assert result.k_bile_per_h == 0.4

    def test_notes_is_string(self):
        result = make_result()
        assert isinstance(result.notes, str)
        assert len(result.notes) > 0

    def test_secondary_peak_is_bool(self):
        result = make_result()
        assert isinstance(result.secondary_peak_present, bool)


# --- Time array consistency ---


class TestTimeArrayConsistency:
    def test_time_array_length_matches_concentration(self):
        result = make_result()
        assert len(result.times_h) == len(result.c_systemic_mg_L)

    def test_time_array_length_matches_bile_pool(self):
        result = make_result()
        assert len(result.times_h) == len(result.bile_pool_mg)

    def test_time_starts_at_zero(self):
        result = make_result()
        assert result.times_h[0] == 0.0

    def test_time_ends_at_t_end(self):
        result = make_result(t_end_h=12.0)
        assert abs(result.times_h[-1] - 12.0) < 0.1

    def test_time_array_monotonically_increasing(self):
        result = make_result()
        for i in range(1, len(result.times_h)):
            assert result.times_h[i] >= result.times_h[i - 1]


# --- Core PK statistics ---


class TestPKStatistics:
    def test_cmax_positive(self):
        result = make_result()
        assert result.cmax_systemic_mg_L > 0.0

    def test_auc_positive(self):
        result = make_result()
        assert result.auc_systemic_mg_h_per_L > 0.0

    def test_tmax_in_simulation_range(self):
        result = make_result()
        assert 0.0 <= result.tmax_systemic_h <= result.times_h[-1]

    def test_cmax_equals_max_concentration(self):
        result = make_result()
        assert abs(result.cmax_systemic_mg_L - max(result.c_systemic_mg_L)) < 1e-9

    def test_all_concentrations_nonnegative(self):
        result = make_result()
        for c in result.c_systemic_mg_L:
            assert c >= 0.0

    def test_all_bile_pool_nonnegative(self):
        result = make_result()
        for b in result.bile_pool_mg:
            assert b >= 0.0


# --- EHC effects ---


class TestEHCEffects:
    def test_higher_f_ehc_higher_auc(self):
        """More recirculation → higher systemic AUC."""
        low = make_result(f_ehc=0.0)
        high = make_result(f_ehc=0.9)
        assert high.auc_systemic_mg_h_per_L >= low.auc_systemic_mg_h_per_L

    def test_zero_f_ehc_near_zero_ehc_contribution(self):
        """With no reabsorption, EHC contribution should be ~0."""
        result = make_result(f_ehc=0.0)
        assert result.ehc_contribution_pct < 1.0

    def test_high_f_ehc_nonzero_ehc_contribution(self):
        """With high reabsorption, EHC contribution should be > 0."""
        result = make_result(f_ehc=0.9)
        assert result.ehc_contribution_pct > 0.0

    def test_ehc_contribution_pct_in_range(self):
        """EHC contribution percentage should be in [0, 100]."""
        for f in [0.0, 0.3, 0.6, 0.9]:
            result = make_result(f_ehc=f)
            assert 0.0 <= result.ehc_contribution_pct <= 100.0

    def test_higher_dose_higher_auc(self):
        """Higher dose → proportionally higher AUC."""
        low = make_result(dose_mg=50.0)
        high = make_result(dose_mg=200.0)
        assert high.auc_systemic_mg_h_per_L > low.auc_systemic_mg_h_per_L


# --- compare_ehc_fractions ---


class TestCompareEHCFractions:
    def test_returns_list(self):
        results = compare_ehc_fractions(DRUG, 100.0, [0.0, 0.3, 0.6, 0.9], 5.0, 50.0)
        assert isinstance(results, list)
        assert len(results) == 4

    def test_sorted_by_auc_descending(self):
        results = compare_ehc_fractions(DRUG, 100.0, [0.0, 0.2, 0.5, 0.8], 5.0, 50.0)
        for i in range(len(results) - 1):
            assert results[i].auc_systemic_mg_h_per_L >= results[i + 1].auc_systemic_mg_h_per_L

    def test_all_results_correct_type(self):
        results = compare_ehc_fractions(DRUG, 100.0, [0.1, 0.7], 5.0, 50.0)
        for r in results:
            assert isinstance(r, BileAcidRecyclingResult)

    def test_single_item_list(self):
        results = compare_ehc_fractions(DRUG, 100.0, [0.5], 5.0, 50.0)
        assert len(results) == 1


# --- Validation errors ---


class TestValidation:
    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            make_result(dose_mg=0.0)

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            make_result(dose_mg=-10.0)

    def test_zero_cl_hepatic_raises(self):
        with pytest.raises(ValueError, match="cl_hepatic"):
            make_result(cl_hepatic_L_per_h=0.0)

    def test_zero_vd_sys_raises(self):
        with pytest.raises(ValueError, match="vd_sys"):
            make_result(vd_sys_L=0.0)

    def test_zero_ka_raises(self):
        with pytest.raises(ValueError, match="ka_per_h"):
            make_result(ka_per_h=0.0)

    def test_f_ehc_above_one_raises(self):
        with pytest.raises(ValueError, match="f_ehc"):
            make_result(f_ehc=1.5)

    def test_negative_f_ehc_raises(self):
        with pytest.raises(ValueError, match="f_ehc"):
            make_result(f_ehc=-0.1)

    def test_zero_k_bile_raises(self):
        with pytest.raises(ValueError, match="k_bile"):
            make_result(k_bile_per_h=0.0)

    def test_zero_t_end_raises(self):
        with pytest.raises(ValueError, match="t_end"):
            make_result(t_end_h=0.0)
