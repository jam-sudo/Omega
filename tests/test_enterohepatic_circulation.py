"""Tests for Phase 380 — Enterohepatic Circulation Model."""

import pytest

from omega_pbpk.core.enterohepatic_circulation import (
    EnterohepaticResult,
    compare_ehc_fractions,
    simulate_enterohepatic_circulation,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_kwargs(**overrides):
    """Return minimal valid kwargs for simulate_enterohepatic_circulation."""
    kwargs = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        cl_hepatic_L_per_h=5.0,
        vd_L=50.0,
        ka_absorption_per_h=1.0,
        f_bile_secretion=0.3,
        k_bile_to_gut_per_h=0.5,
        f_gut_reabsorption=0.6,
        t_end_h=48.0,
        dt_h=0.05,
    )
    kwargs.update(overrides)
    return kwargs


# ---------------------------------------------------------------------------
# Return type tests
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_result_instance(self):
        result = simulate_enterohepatic_circulation(**_base_kwargs())
        assert isinstance(result, EnterohepaticResult)

    def test_all_fields_present(self):
        result = simulate_enterohepatic_circulation(**_base_kwargs())
        assert hasattr(result, "drug_name")
        assert hasattr(result, "dose_mg")
        assert hasattr(result, "times_h")
        assert hasattr(result, "c_plasma_mg_L")
        assert hasattr(result, "c_bile_mg_L")
        assert hasattr(result, "c_gut_lumen_mg_L")
        assert hasattr(result, "secondary_peaks")
        assert hasattr(result, "auc_total_mg_h_per_L")
        assert hasattr(result, "auc_without_ehc_mg_h_per_L")
        assert hasattr(result, "ehc_auc_increase_pct")
        assert hasattr(result, "t_half_apparent_h")
        assert hasattr(result, "cmax_mg_L")
        assert hasattr(result, "notes")

    def test_list_fields_are_lists(self):
        result = simulate_enterohepatic_circulation(**_base_kwargs())
        assert isinstance(result.times_h, list)
        assert isinstance(result.c_plasma_mg_L, list)
        assert isinstance(result.c_bile_mg_L, list)
        assert isinstance(result.c_gut_lumen_mg_L, list)


# ---------------------------------------------------------------------------
# EHC increases AUC vs no-EHC
# ---------------------------------------------------------------------------


class TestEhcAucIncrease:
    def test_ehc_increases_auc_vs_no_ehc(self):
        result = simulate_enterohepatic_circulation(**_base_kwargs(f_bile_secretion=0.4))
        assert result.auc_total_mg_h_per_L >= result.auc_without_ehc_mg_h_per_L

    def test_ehc_auc_increase_pct_positive_when_ehc_active(self):
        result = simulate_enterohepatic_circulation(**_base_kwargs(f_bile_secretion=0.4))
        assert result.ehc_auc_increase_pct >= 0.0

    def test_no_ehc_gives_zero_increase(self):
        result = simulate_enterohepatic_circulation(**_base_kwargs(f_bile_secretion=0.0))
        assert result.ehc_auc_increase_pct == pytest.approx(0.0, abs=1e-6)

    def test_higher_f_bile_gives_higher_auc(self):
        result_low = simulate_enterohepatic_circulation(**_base_kwargs(f_bile_secretion=0.1))
        result_high = simulate_enterohepatic_circulation(**_base_kwargs(f_bile_secretion=0.6))
        assert result_high.auc_total_mg_h_per_L > result_low.auc_total_mg_h_per_L


# ---------------------------------------------------------------------------
# Plasma profile
# ---------------------------------------------------------------------------


class TestPlasmaProfile:
    def test_plasma_profile_has_values(self):
        result = simulate_enterohepatic_circulation(**_base_kwargs())
        assert len(result.c_plasma_mg_L) > 0

    def test_plasma_starts_at_zero(self):
        result = simulate_enterohepatic_circulation(**_base_kwargs())
        assert result.c_plasma_mg_L[0] == pytest.approx(0.0, abs=1e-9)

    def test_cmax_positive(self):
        result = simulate_enterohepatic_circulation(**_base_kwargs())
        assert result.cmax_mg_L > 0.0

    def test_times_start_at_zero_and_monotone(self):
        result = simulate_enterohepatic_circulation(**_base_kwargs())
        assert result.times_h[0] == pytest.approx(0.0)
        for i in range(1, len(result.times_h)):
            assert result.times_h[i] > result.times_h[i - 1]


# ---------------------------------------------------------------------------
# Bile compartment
# ---------------------------------------------------------------------------


class TestBileCompartment:
    def test_bile_compartment_populated_when_ehc_active(self):
        result = simulate_enterohepatic_circulation(**_base_kwargs(f_bile_secretion=0.3))
        max_bile = max(result.c_bile_mg_L)
        assert max_bile > 0.0

    def test_bile_zero_when_no_ehc(self):
        result = simulate_enterohepatic_circulation(**_base_kwargs(f_bile_secretion=0.0))
        max_bile = max(result.c_bile_mg_L)
        assert max_bile == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Gut lumen compartment
# ---------------------------------------------------------------------------


class TestGutLumenCompartment:
    def test_gut_lumen_populated_when_ehc_active(self):
        result = simulate_enterohepatic_circulation(**_base_kwargs(f_bile_secretion=0.3))
        max_lumen = max(result.c_gut_lumen_mg_L)
        assert max_lumen > 0.0

    def test_gut_lumen_zero_initially(self):
        result = simulate_enterohepatic_circulation(**_base_kwargs())
        assert result.c_gut_lumen_mg_L[0] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Secondary peaks
# ---------------------------------------------------------------------------


class TestSecondaryPeaks:
    def test_secondary_peaks_non_negative(self):
        result = simulate_enterohepatic_circulation(**_base_kwargs())
        assert result.secondary_peaks >= 0

    def test_secondary_peaks_is_int(self):
        result = simulate_enterohepatic_circulation(**_base_kwargs())
        assert isinstance(result.secondary_peaks, int)


# ---------------------------------------------------------------------------
# Dose linearity (Cmax)
# ---------------------------------------------------------------------------


class TestDoseLinearity:
    def test_cmax_scales_with_dose(self):
        result_low = simulate_enterohepatic_circulation(**_base_kwargs(dose_mg=50.0))
        result_high = simulate_enterohepatic_circulation(**_base_kwargs(dose_mg=200.0))
        # 4x dose → Cmax should be proportionally larger
        ratio = result_high.cmax_mg_L / result_low.cmax_mg_L
        assert 1.5 < ratio < 6.0


# ---------------------------------------------------------------------------
# compare_ehc_fractions
# ---------------------------------------------------------------------------


class TestCompareEhcFractions:
    def test_compare_returns_list(self):
        results = compare_ehc_fractions("Drug", 100.0, 5.0, 50.0)
        assert isinstance(results, list)

    def test_compare_default_returns_five(self):
        results = compare_ehc_fractions("Drug", 100.0, 5.0, 50.0)
        assert len(results) == 5

    def test_compare_sorted_descending_by_auc(self):
        results = compare_ehc_fractions("Drug", 100.0, 5.0, 50.0)
        aucs = [r.auc_total_mg_h_per_L for r in results]
        assert aucs == sorted(aucs, reverse=True)

    def test_compare_all_are_result_instances(self):
        results = compare_ehc_fractions("Drug", 100.0, 5.0, 50.0)
        for r in results:
            assert isinstance(r, EnterohepaticResult)

    def test_compare_custom_f_bile_values(self):
        results = compare_ehc_fractions("Drug", 100.0, 5.0, 50.0, f_bile_values=[0.0, 0.2, 0.5])
        assert len(results) == 3


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_dose(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_enterohepatic_circulation(**_base_kwargs(dose_mg=0.0))

    def test_invalid_cl(self):
        with pytest.raises(ValueError, match="cl_hepatic"):
            simulate_enterohepatic_circulation(**_base_kwargs(cl_hepatic_L_per_h=-1.0))

    def test_invalid_vd(self):
        with pytest.raises(ValueError, match="vd_L"):
            simulate_enterohepatic_circulation(**_base_kwargs(vd_L=0.0))

    def test_invalid_ka(self):
        with pytest.raises(ValueError, match="ka_absorption"):
            simulate_enterohepatic_circulation(**_base_kwargs(ka_absorption_per_h=-0.1))

    def test_invalid_f_bile_out_of_range(self):
        with pytest.raises(ValueError, match="f_bile_secretion"):
            simulate_enterohepatic_circulation(**_base_kwargs(f_bile_secretion=1.5))

    def test_invalid_k_bile_to_gut(self):
        with pytest.raises(ValueError, match="k_bile_to_gut"):
            simulate_enterohepatic_circulation(**_base_kwargs(k_bile_to_gut_per_h=0.0))

    def test_invalid_f_gut_reabsorption(self):
        with pytest.raises(ValueError, match="f_gut_reabsorption"):
            simulate_enterohepatic_circulation(**_base_kwargs(f_gut_reabsorption=-0.1))
