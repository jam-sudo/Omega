"""Tests for biopharmaceutics/crystal_form_pk.py — Phase 555."""

import pytest

from omega_pbpk.biopharmaceutics.crystal_form_pk import (
    compare_crystal_forms,
    crystal_conversion_risk,
    crystal_form_solubility,
    predict_f_oral_from_crystal,
)

# ---------------------------------------------------------------------------
# crystal_form_solubility
# ---------------------------------------------------------------------------


class TestCrystalFormSolubility:
    def test_all_forms_return_positive_solubility(self):
        forms = ["amorphous", "polymorph_I", "polymorph_II", "hydrate", "solvate"]
        for form in forms:
            result = crystal_form_solubility(form, 100.0, 25.0)
            assert result["solubility_mg_L"] > 0, f"Expected positive solubility for {form}"

    def test_amorphous_higher_than_polymorph_I(self):
        s_am = crystal_form_solubility("amorphous", 100.0, 25.0)["solubility_mg_L"]
        s_pi = crystal_form_solubility("polymorph_I", 100.0, 25.0)["solubility_mg_L"]
        assert s_am > s_pi

    def test_polymorph_I_higher_than_polymorph_II(self):
        s_pi = crystal_form_solubility("polymorph_I", 100.0, 25.0)["solubility_mg_L"]
        s_pii = crystal_form_solubility("polymorph_II", 100.0, 25.0)["solubility_mg_L"]
        assert s_pi > s_pii

    def test_amorphous_20x_multiplier_at_25C(self):
        intrinsic = 50.0
        result = crystal_form_solubility("amorphous", intrinsic, 25.0)
        assert abs(result["solubility_mg_L"] - 20.0 * intrinsic) < 1e-9

    def test_polymorph_I_reference_at_25C(self):
        intrinsic = 80.0
        result = crystal_form_solubility("polymorph_I", intrinsic, 25.0)
        assert abs(result["solubility_mg_L"] - intrinsic) < 1e-9

    def test_polymorph_II_0_6x_at_25C(self):
        intrinsic = 100.0
        result = crystal_form_solubility("polymorph_II", intrinsic, 25.0)
        assert abs(result["solubility_mg_L"] - 60.0) < 1e-9

    def test_hydrate_0_8x_at_25C(self):
        intrinsic = 100.0
        result = crystal_form_solubility("hydrate", intrinsic, 25.0)
        assert abs(result["solubility_mg_L"] - 80.0) < 1e-9

    def test_solvate_1_2x_at_25C(self):
        intrinsic = 100.0
        result = crystal_form_solubility("solvate", intrinsic, 25.0)
        assert abs(result["solubility_mg_L"] - 120.0) < 1e-9

    def test_higher_temperature_increases_solubility(self):
        s_25 = crystal_form_solubility("polymorph_I", 100.0, 25.0)["solubility_mg_L"]
        s_35 = crystal_form_solubility("polymorph_I", 100.0, 35.0)["solubility_mg_L"]
        assert s_35 > s_25

    def test_q10_doubles_per_10C(self):
        intrinsic = 100.0
        s_25 = crystal_form_solubility("polymorph_I", intrinsic, 25.0)["solubility_mg_L"]
        s_35 = crystal_form_solubility("polymorph_I", intrinsic, 35.0)["solubility_mg_L"]
        assert abs(s_35 / s_25 - 2.0) < 1e-9

    def test_return_keys(self):
        result = crystal_form_solubility("amorphous", 100.0, 25.0)
        assert set(result.keys()) == {
            "form",
            "solubility_mg_L",
            "temperature_C",
            "fold_vs_polymorph_I",
            "notes",
        }

    def test_invalid_form_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid crystal form"):
            crystal_form_solubility("unknown_form", 100.0, 25.0)

    def test_negative_solubility_raises_value_error(self):
        with pytest.raises(ValueError):
            crystal_form_solubility("polymorph_I", -1.0, 25.0)


# ---------------------------------------------------------------------------
# predict_f_oral_from_crystal
# ---------------------------------------------------------------------------


class TestPredictFOralFromCrystal:
    def test_do_less_than_1_gives_fa_1(self):
        # solubility=1000 mg/L, dose=100 mg, gi_vol=0.25 L → Do = 100/(1000*0.25)=0.4 < 1
        fa = predict_f_oral_from_crystal(1000.0, 100.0, 1e-5)
        assert fa == 1.0

    def test_do_exactly_1_gives_fa_1(self):
        # Do = dose/(sol*vol) = 250/(1000*0.25) = 1.0
        fa = predict_f_oral_from_crystal(1000.0, 250.0, 1e-5)
        assert fa == 1.0

    def test_do_greater_than_1_gives_fa_less_than_1(self):
        # Do = 500/(1000*0.25)=2.0 → fa=0.5
        fa = predict_f_oral_from_crystal(1000.0, 500.0, 1e-5)
        assert fa < 1.0
        assert abs(fa - 0.5) < 1e-9

    def test_double_dose_halves_fa_when_solubility_limited(self):
        sol = 100.0
        peff = 1e-5
        fa1 = predict_f_oral_from_crystal(sol, 100.0, peff)
        fa2 = predict_f_oral_from_crystal(sol, 200.0, peff)
        # Do1=100/(100*0.25)=4 → fa1=0.25; Do2=200/(100*0.25)=8 → fa2=0.125
        assert fa1 > 0 and fa1 < 1
        assert fa2 > 0 and fa2 < 1
        assert abs(fa1 - 2.0 * fa2) < 1e-9

    def test_fa_clamped_between_0_and_1(self):
        fa = predict_f_oral_from_crystal(0.001, 1000.0, 1e-5)
        assert 0.0 <= fa <= 1.0

    def test_zero_solubility_gives_zero_fa(self):
        fa = predict_f_oral_from_crystal(0.0, 100.0, 1e-5)
        assert fa == 0.0

    def test_negative_dose_raises_value_error(self):
        with pytest.raises(ValueError):
            predict_f_oral_from_crystal(100.0, -10.0, 1e-5)

    def test_negative_solubility_raises_value_error(self):
        with pytest.raises(ValueError):
            predict_f_oral_from_crystal(-100.0, 100.0, 1e-5)


# ---------------------------------------------------------------------------
# compare_crystal_forms
# ---------------------------------------------------------------------------


class TestCompareCrystalForms:
    def test_returns_5_entries(self):
        result = compare_crystal_forms("DrugX", 100.0, 1e-5, 50.0)
        assert len(result) == 5

    def test_sorted_by_fa_descending(self):
        result = compare_crystal_forms("DrugX", 100.0, 1e-5, 50.0)
        fas = [r["fa"] for r in result]
        assert fas == sorted(fas, reverse=True)

    def test_amorphous_has_highest_fa(self):
        result = compare_crystal_forms("DrugX", 500.0, 1e-5, 10.0)
        assert result[0]["form"] == "amorphous"

    def test_each_entry_has_required_keys(self):
        result = compare_crystal_forms("DrugX", 100.0, 1e-5, 50.0)
        for entry in result:
            assert "form" in entry
            assert "solubility_mg_L" in entry
            assert "fa" in entry
            assert "fold_vs_polymorph_I" in entry

    def test_all_fa_values_in_valid_range(self):
        result = compare_crystal_forms("DrugX", 100.0, 1e-5, 50.0)
        for entry in result:
            assert 0.0 <= entry["fa"] <= 1.0

    def test_negative_intrinsic_solubility_raises(self):
        with pytest.raises(ValueError):
            compare_crystal_forms("DrugX", 100.0, 1e-5, -10.0)


# ---------------------------------------------------------------------------
# crystal_conversion_risk
# ---------------------------------------------------------------------------


class TestCrystalConversionRisk:
    def test_high_rh_high_temp_is_high_risk(self):
        result = crystal_conversion_risk(80.0, 45.0, "amorphous")
        assert result["risk_level"] == "high"

    def test_high_rh_high_temp_non_amorphous_still_high_risk(self):
        result = crystal_conversion_risk(80.0, 45.0, "polymorph_I")
        assert result["risk_level"] == "high"

    def test_moderate_rh_is_moderate_risk(self):
        result = crystal_conversion_risk(65.0, 20.0, "polymorph_I")
        assert result["risk_level"] == "moderate"

    def test_moderate_temp_is_moderate_risk(self):
        result = crystal_conversion_risk(40.0, 35.0, "polymorph_II")
        assert result["risk_level"] == "moderate"

    def test_low_rh_low_temp_is_low_risk(self):
        result = crystal_conversion_risk(40.0, 20.0, "amorphous")
        assert result["risk_level"] == "low"

    def test_return_keys(self):
        result = crystal_conversion_risk(40.0, 20.0, "amorphous")
        assert "risk_level" in result
        assert "notes" in result

    def test_invalid_form_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid crystal form"):
            crystal_conversion_risk(40.0, 20.0, "bad_form")

    def test_boundary_rh_75_with_high_temp_not_high(self):
        # RH exactly 75 is NOT > 75, so not high risk (even if T > 40)
        result = crystal_conversion_risk(75.0, 45.0, "amorphous")
        assert result["risk_level"] in ("moderate", "low")

    def test_boundary_temp_40_with_high_rh_not_high(self):
        # T exactly 40 is NOT > 40, so not high risk
        result = crystal_conversion_risk(80.0, 40.0, "amorphous")
        assert result["risk_level"] in ("moderate", "low")
