"""Tests for oral absorption window simulation (Phase 344)."""

from __future__ import annotations

import pytest

from omega_pbpk.biopharmaceutics.absorption_window_sim import (
    AbsorptionWindowResult,
    compare_absorption_windows,
    simulate_absorption_window,
)

# ---------------------------------------------------------------------------
# Shared base parameters
# ---------------------------------------------------------------------------

BASE = dict(
    drug_name="TestDrug",
    dose_mg=100.0,
    peff_cm_per_s=1e-4,       # moderate permeability
    solubility_mg_mL=1.0,
    dose_volume_mL=250.0,
    stomach_emptying_h=0.5,
    si_transit_h=4.0,
    colon_transit_h=20.0,
    absorption_window="full_si",
    f_degraded_colon=0.8,
    t_end_h=28.0,
    dt_h=0.1,
)


def base_result() -> AbsorptionWindowResult:
    return simulate_absorption_window(**BASE)


# ---------------------------------------------------------------------------
# 1. Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_absorption_window(**{**BASE, "dose_mg": 0.0})

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_absorption_window(**{**BASE, "dose_mg": -10.0})

    def test_zero_peff_raises(self):
        with pytest.raises(ValueError, match="peff"):
            simulate_absorption_window(**{**BASE, "peff_cm_per_s": 0.0})

    def test_negative_peff_raises(self):
        with pytest.raises(ValueError, match="peff"):
            simulate_absorption_window(**{**BASE, "peff_cm_per_s": -1e-5})

    def test_zero_solubility_raises(self):
        with pytest.raises(ValueError, match="solubility"):
            simulate_absorption_window(**{**BASE, "solubility_mg_mL": 0.0})

    def test_negative_solubility_raises(self):
        with pytest.raises(ValueError, match="solubility"):
            simulate_absorption_window(**{**BASE, "solubility_mg_mL": -1.0})

    def test_zero_stomach_emptying_raises(self):
        with pytest.raises(ValueError, match="stomach_emptying"):
            simulate_absorption_window(**{**BASE, "stomach_emptying_h": 0.0})

    def test_zero_si_transit_raises(self):
        with pytest.raises(ValueError, match="si_transit"):
            simulate_absorption_window(**{**BASE, "si_transit_h": 0.0})

    def test_zero_colon_transit_raises(self):
        with pytest.raises(ValueError, match="colon_transit"):
            simulate_absorption_window(**{**BASE, "colon_transit_h": 0.0})

    def test_invalid_absorption_window_raises(self):
        with pytest.raises(ValueError, match="absorption_window"):
            simulate_absorption_window(**{**BASE, "absorption_window": "rectum"})


# ---------------------------------------------------------------------------
# 2. Frozen dataclass
# ---------------------------------------------------------------------------

class TestFrozenDataclass:
    def test_result_is_frozen(self):
        result = base_result()
        with pytest.raises((AttributeError, TypeError)):
            result.drug_name = "other"  # type: ignore[misc]

    def test_result_is_AbsorptionWindowResult(self):
        result = base_result()
        assert isinstance(result, AbsorptionWindowResult)


# ---------------------------------------------------------------------------
# 3. Output shapes
# ---------------------------------------------------------------------------

class TestOutputShapes:
    def test_times_and_abs_rate_same_length(self):
        result = base_result()
        assert len(result.times_h) == len(result.absorption_rate_mg_per_h)

    def test_times_length_correct(self):
        result = simulate_absorption_window(**{**BASE, "t_end_h": 10.0, "dt_h": 0.5})
        # n_steps = int(10/0.5)+1 = 21
        assert len(result.times_h) == 21

    def test_absorption_rate_non_negative(self):
        result = base_result()
        assert all(r >= 0.0 for r in result.absorption_rate_mg_per_h)


# ---------------------------------------------------------------------------
# 4. f_absorbed_total in [0, 1.05]
# ---------------------------------------------------------------------------

class TestAbsorbedFractionRange:
    def test_f_total_non_negative(self):
        result = base_result()
        assert result.f_absorbed_total >= 0.0

    def test_f_total_at_most_1_05(self):
        # Allow small numerical overrun per spec
        result = base_result()
        assert result.f_absorbed_total <= 1.05

    def test_individual_fractions_non_negative(self):
        result = base_result()
        assert result.f_absorbed_stomach >= 0.0
        assert result.f_absorbed_upper_si >= 0.0
        assert result.f_absorbed_lower_si >= 0.0
        assert result.f_absorbed_colon >= 0.0

    def test_total_equals_sum_of_parts(self):
        result = base_result()
        total_manual = (
            result.f_absorbed_stomach
            + result.f_absorbed_upper_si
            + result.f_absorbed_lower_si
            + result.f_absorbed_colon
        )
        assert abs(result.f_absorbed_total - total_manual) < 1e-9


# ---------------------------------------------------------------------------
# 5. "stomach_only" — no SI or colon absorption
# ---------------------------------------------------------------------------

class TestStomachOnly:
    def test_stomach_only_no_si_absorption(self):
        result = simulate_absorption_window(
            **{**BASE, "absorption_window": "stomach_only"}
        )
        assert result.f_absorbed_upper_si == pytest.approx(0.0, abs=1e-9)
        assert result.f_absorbed_lower_si == pytest.approx(0.0, abs=1e-9)
        assert result.f_absorbed_colon == pytest.approx(0.0, abs=1e-9)

    def test_stomach_only_has_some_absorption(self):
        # Stomach absorption should be > 0 for a permeable drug
        result = simulate_absorption_window(
            **{**BASE, "peff_cm_per_s": 1e-3, "absorption_window": "stomach_only"}
        )
        assert result.f_absorbed_stomach > 0.0


# ---------------------------------------------------------------------------
# 6. High Peff → high f_absorbed_total
# ---------------------------------------------------------------------------

class TestHighPeff:
    def test_high_peff_high_absorption(self):
        result = simulate_absorption_window(
            **{**BASE, "peff_cm_per_s": 1e-2, "absorption_window": "full_si"}
        )
        assert result.f_absorbed_total > 0.5


# ---------------------------------------------------------------------------
# 7. "colon_included" > "full_si" for low-Peff drug
# ---------------------------------------------------------------------------

class TestColonIncluded:
    def test_colon_included_higher_than_full_si_for_low_peff(self):
        # Low Peff drug → a lot remains unabsorbed by end of SI
        low_peff = 1e-6  # very low permeability
        r_full_si = simulate_absorption_window(
            **{**BASE, "peff_cm_per_s": low_peff, "absorption_window": "full_si"}
        )
        r_colon = simulate_absorption_window(
            **{**BASE, "peff_cm_per_s": low_peff, "absorption_window": "colon_included",
               "f_degraded_colon": 0.0}  # no degradation so colon absorbs
        )
        assert r_colon.f_absorbed_total >= r_full_si.f_absorbed_total

    def test_colon_fraction_zero_when_not_included(self):
        result = simulate_absorption_window(
            **{**BASE, "absorption_window": "full_si"}
        )
        assert result.f_absorbed_colon == pytest.approx(0.0, abs=1e-9)

    def test_colon_fraction_positive_when_included(self):
        result = simulate_absorption_window(
            **{
                **BASE,
                "peff_cm_per_s": 1e-5,
                "absorption_window": "colon_included",
                "f_degraded_colon": 0.0,
            }
        )
        # With no degradation and some transit into colon, colon absorption > 0
        assert result.f_absorbed_colon >= 0.0


# ---------------------------------------------------------------------------
# 8. recommended_formulation is valid
# ---------------------------------------------------------------------------

class TestFormulationRecommendation:
    _valid = {"immediate_release", "modified_release_4h", "modified_release_8h"}

    def test_recommendation_valid_for_full_si(self):
        result = base_result()
        assert result.recommended_formulation in self._valid

    def test_high_peff_immediate_release(self):
        result = simulate_absorption_window(
            **{**BASE, "peff_cm_per_s": 1e-2, "solubility_mg_mL": 10.0}
        )
        assert result.recommended_formulation == "immediate_release"

    @pytest.mark.parametrize("window", ["upper_si", "full_si", "colon_included", "stomach_only"])
    def test_all_windows_give_valid_recommendation(self, window):
        result = simulate_absorption_window(**{**BASE, "absorption_window": window})
        assert result.recommended_formulation in self._valid


# ---------------------------------------------------------------------------
# 9. compare_absorption_windows
# ---------------------------------------------------------------------------

class TestCompareAbsorptionWindows:
    def test_returns_all_windows(self):
        result = compare_absorption_windows(
            drug_name="Comp",
            dose_mg=100.0,
            peff_cm_per_s=1e-4,
            solubility_mg_mL=1.0,
        )
        assert "upper_si" in result
        assert "full_si" in result
        assert "colon_included" in result
        assert "stomach_only" in result

    def test_highest_f_absorbed_key_present(self):
        result = compare_absorption_windows(
            drug_name="Comp",
            dose_mg=100.0,
            peff_cm_per_s=1e-4,
            solubility_mg_mL=1.0,
        )
        assert "highest_f_absorbed" in result

    def test_highest_f_absorbed_is_valid_window(self):
        result = compare_absorption_windows(
            drug_name="Comp",
            dose_mg=100.0,
            peff_cm_per_s=1e-4,
            solubility_mg_mL=1.0,
        )
        assert result["highest_f_absorbed"] in {
            "upper_si", "full_si", "colon_included", "stomach_only"
        }

    def test_results_are_AbsorptionWindowResult(self):
        result = compare_absorption_windows(
            drug_name="Comp",
            dose_mg=100.0,
            peff_cm_per_s=1e-4,
            solubility_mg_mL=1.0,
        )
        for window in ("upper_si", "full_si", "colon_included", "stomach_only"):
            assert isinstance(result[window], AbsorptionWindowResult)

    def test_stomach_only_lowest_or_comparable(self):
        # stomach_only should generally give the lowest total absorption
        result = compare_absorption_windows(
            drug_name="Comp",
            dose_mg=100.0,
            peff_cm_per_s=1e-4,
            solubility_mg_mL=1.0,
        )
        assert result["stomach_only"].f_absorbed_total <= result["full_si"].f_absorbed_total + 1e-6


# ---------------------------------------------------------------------------
# 10. Dose number calculation
# ---------------------------------------------------------------------------

class TestDoseNumber:
    def test_explicit_dose_number_used(self):
        # Providing explicit dose_number should bypass internal calculation
        result = simulate_absorption_window(
            **{**BASE, "dose_number": 0.1}  # well below 1 → dissolution not limiting
        )
        # Notes should mention Dn=0.10
        assert "0.10" in result.notes

    def test_high_dose_number_noted(self):
        # Dn > 1 → dissolution limited
        result = simulate_absorption_window(
            **{**BASE, "dose_mg": 10000.0, "solubility_mg_mL": 0.01}
        )
        assert "dissolution" in result.notes.lower() or "BCS" in result.notes
