"""Tests for colonic absorption model (Phase 431)."""

from __future__ import annotations

import pytest

from omega_pbpk.biopharmaceutics.colonic_absorption import (
    ColonicAbsorptionResult,
    modified_release_colonic_pk,
    simulate_colonic_absorption,
)

# ---------------------------------------------------------------------------
# Default test parameters
# ---------------------------------------------------------------------------
_DRUG = "test_drug"
_DOSE = 100.0  # mg
_SOL = 2.0  # mg/mL — high solubility (dose < sol * 40 mL = 80 mg... use 50 mg)
_DOSE_LO = 50.0  # mg — dissolved fraction = 1.0 with SOL=2.0, V=40mL
_PEFF = 1.0e-5  # cm/s — low colonic permeability
_CL = 5.0  # L/h
_VD = 50.0  # L
_T_TRANSIT = 20.0  # h — typical colonic transit


# ---------------------------------------------------------------------------
# 1. Input validation tests
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_empty_drug_name(self):
        with pytest.raises(ValueError, match="drug_name"):
            simulate_colonic_absorption("", _DOSE_LO, _SOL, _PEFF, _CL, _VD)

    def test_whitespace_drug_name(self):
        with pytest.raises(ValueError, match="drug_name"):
            simulate_colonic_absorption("   ", _DOSE_LO, _SOL, _PEFF, _CL, _VD)

    def test_dose_zero(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_colonic_absorption(_DRUG, 0.0, _SOL, _PEFF, _CL, _VD)

    def test_dose_negative(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_colonic_absorption(_DRUG, -10.0, _SOL, _PEFF, _CL, _VD)

    def test_solubility_zero(self):
        with pytest.raises(ValueError, match="solubility_mg_mL"):
            simulate_colonic_absorption(_DRUG, _DOSE_LO, 0.0, _PEFF, _CL, _VD)

    def test_peff_zero(self):
        with pytest.raises(ValueError, match="peff_colon_cm_s"):
            simulate_colonic_absorption(_DRUG, _DOSE_LO, _SOL, 0.0, _CL, _VD)

    def test_cl_zero(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            simulate_colonic_absorption(_DRUG, _DOSE_LO, _SOL, _PEFF, 0.0, _VD)

    def test_vd_zero(self):
        with pytest.raises(ValueError, match="vd_L"):
            simulate_colonic_absorption(_DRUG, _DOSE_LO, _SOL, _PEFF, _CL, 0.0)

    def test_t_transit_zero(self):
        with pytest.raises(ValueError, match="t_transit_h"):
            simulate_colonic_absorption(
                _DRUG, _DOSE_LO, _SOL, _PEFF, _CL, _VD, t_transit_h=0.0
            )

    def test_t_end_zero(self):
        with pytest.raises(ValueError, match="t_end_h"):
            simulate_colonic_absorption(
                _DRUG, _DOSE_LO, _SOL, _PEFF, _CL, _VD, t_end_h=0.0
            )

    def test_dt_too_large(self):
        with pytest.raises(ValueError, match="dt_h"):
            simulate_colonic_absorption(
                _DRUG, _DOSE_LO, _SOL, _PEFF, _CL, _VD, t_end_h=10.0, dt_h=15.0
            )


# ---------------------------------------------------------------------------
# 2. Result structure tests
# ---------------------------------------------------------------------------


class TestResultStructure:
    def setup_method(self):
        self.result = simulate_colonic_absorption(
            _DRUG, _DOSE_LO, _SOL, _PEFF, _CL, _VD,
            t_transit_h=_T_TRANSIT, t_end_h=48.0, dt_h=0.1,
        )

    def test_returns_correct_type(self):
        assert isinstance(self.result, ColonicAbsorptionResult)

    def test_drug_name_stored(self):
        assert self.result.drug_name == _DRUG

    def test_dose_stored(self):
        assert self.result.dose_mg == _DOSE_LO

    def test_solubility_stored(self):
        assert self.result.solubility_mg_mL == _SOL

    def test_transit_time_stored(self):
        assert self.result.t_transit_h == _T_TRANSIT

    def test_times_non_empty(self):
        assert len(self.result.times_h) > 0

    def test_times_starts_at_zero(self):
        assert self.result.times_h[0] == pytest.approx(0.0)

    def test_a_colon_same_length_as_times(self):
        assert len(self.result.a_colon_mg) == len(self.result.times_h)

    def test_c_plasma_same_length_as_times(self):
        assert len(self.result.c_plasma_mg_L) == len(self.result.times_h)

    def test_notes_is_list(self):
        assert isinstance(self.result.notes, list)


# ---------------------------------------------------------------------------
# 3. Physical / physiological correctness tests
# ---------------------------------------------------------------------------


class TestPhysiology:
    def setup_method(self):
        self.result = simulate_colonic_absorption(
            _DRUG, _DOSE_LO, _SOL, _PEFF, _CL, _VD,
            t_transit_h=_T_TRANSIT, t_end_h=48.0, dt_h=0.05,
        )

    def test_a_colon_starts_at_dose(self):
        assert self.result.a_colon_mg[0] == pytest.approx(_DOSE_LO)

    def test_a_colon_non_negative(self):
        for a in self.result.a_colon_mg:
            assert a >= -1e-9

    def test_c_plasma_non_negative(self):
        for c in self.result.c_plasma_mg_L:
            assert c >= -1e-9

    def test_c_plasma_starts_at_zero(self):
        assert self.result.c_plasma_mg_L[0] == pytest.approx(0.0)

    def test_cmax_positive(self):
        assert self.result.cmax > 0.0

    def test_auc_positive(self):
        assert self.result.auc > 0.0

    def test_cmax_equals_max_c_plasma(self):
        assert self.result.cmax == pytest.approx(max(self.result.c_plasma_mg_L))

    def test_f_absorbed_in_unit_interval(self):
        assert 0.0 <= self.result.f_absorbed_colon <= 1.0 + 1e-9

    def test_a_colon_decreases_during_transit(self):
        """Amount in colon should decrease before transit time ends."""
        idx_transit = min(
            int(_T_TRANSIT / 0.05), len(self.result.a_colon_mg) - 1
        )
        assert self.result.a_colon_mg[idx_transit] <= self.result.a_colon_mg[0]


class TestAbsorptionCeases:
    """After transit time, no more absorption — remaining drug leaves colon."""

    def test_absorption_stops_after_transit(self):
        result = simulate_colonic_absorption(
            _DRUG, _DOSE_LO, _SOL, _PEFF, _CL, _VD,
            t_transit_h=5.0, t_end_h=24.0, dt_h=0.1,
        )
        # After transit, a_colon should be constant (no further absorption)
        idx_post = int(6.0 / 0.1)  # 1h past transit
        idx_end = len(result.a_colon_mg) - 1
        # The colon depot should remain unchanged after transit
        assert result.a_colon_mg[idx_post] == pytest.approx(
            result.a_colon_mg[idx_end], rel=1e-6
        )


class TestSolubilityLimited:
    """Low solubility should limit dissolved fraction and reduce absorption."""

    def test_low_solubility_note(self):
        # dissolved_frac = min(1.0, dose / (sol * V_colon))
        # Note is added when dissolved_frac < 1.0, i.e., dose < sol * V_colon
        # dose=10 mg, sol=1.0 mg/mL, V=40 mL → 10/40 = 0.25 < 1 → note added
        result = simulate_colonic_absorption(
            _DRUG, 10.0, 1.0, _PEFF, _CL, _VD,
            t_transit_h=_T_TRANSIT, t_end_h=48.0, dt_h=0.1,
        )
        assert any("dissolved" in n.lower() for n in result.notes)

    def test_fully_dissolved_no_note(self):
        # dissolved_frac = min(1.0, dose/(sol*V))
        # For no solubility note: need dose >= sol*V
        # dose=400, sol=10, V=40 → 400/(10*40)=1.0 → dissolved_frac=1.0 → no note
        result = simulate_colonic_absorption(
            _DRUG, 400.0, 10.0, _PEFF, _CL, _VD,
            t_transit_h=_T_TRANSIT, t_end_h=48.0, dt_h=0.1,
        )
        assert not any("dissolved" in n.lower() for n in result.notes)


# ---------------------------------------------------------------------------
# 4. Transit time warning tests
# ---------------------------------------------------------------------------


class TestTransitTimeWarnings:
    def test_short_transit_warning(self):
        result = simulate_colonic_absorption(
            _DRUG, _DOSE_LO, _SOL, _PEFF, _CL, _VD,
            t_transit_h=4.0, t_end_h=24.0, dt_h=0.1,
        )
        assert any("shorter" in n for n in result.notes)

    def test_long_transit_warning(self):
        result = simulate_colonic_absorption(
            _DRUG, _DOSE_LO, _SOL, _PEFF, _CL, _VD,
            t_transit_h=40.0, t_end_h=72.0, dt_h=0.1,
        )
        assert any("longer" in n for n in result.notes)


# ---------------------------------------------------------------------------
# 5. Modified-release colonic PK tests
# ---------------------------------------------------------------------------


class TestModifiedReleaseColonicPK:
    def test_returns_dict(self):
        result = modified_release_colonic_pk(
            _DRUG, _DOSE, 0.5, _CL, _VD, solubility_mg_mL=_SOL
        )
        assert isinstance(result, dict)

    def test_required_keys_present(self):
        result = modified_release_colonic_pk(
            _DRUG, _DOSE, 0.5, _CL, _VD, solubility_mg_mL=_SOL
        )
        expected_keys = {
            "drug_name", "dose_mg", "si_release_pct", "colon_dose_mg",
            "si_cmax", "si_auc", "colon_result", "combined_cmax",
            "combined_auc", "combined_times_h", "combined_c_plasma",
            "f_si", "f_colon", "f_total", "notes",
        }
        assert expected_keys.issubset(result.keys())

    def test_colon_dose_complementary(self):
        result = modified_release_colonic_pk(
            _DRUG, _DOSE, 0.6, _CL, _VD, solubility_mg_mL=_SOL
        )
        assert result["colon_dose_mg"] == pytest.approx(0.4 * _DOSE, rel=1e-6)

    def test_f_total_between_0_and_1(self):
        result = modified_release_colonic_pk(
            _DRUG, _DOSE, 0.5, _CL, _VD, solubility_mg_mL=_SOL
        )
        assert 0.0 <= result["f_total"] <= 1.0 + 1e-9

    def test_combined_cmax_positive(self):
        result = modified_release_colonic_pk(
            _DRUG, _DOSE, 0.5, _CL, _VD, solubility_mg_mL=_SOL
        )
        assert result["combined_cmax"] > 0.0

    def test_combined_auc_positive(self):
        result = modified_release_colonic_pk(
            _DRUG, _DOSE, 0.5, _CL, _VD, solubility_mg_mL=_SOL
        )
        assert result["combined_auc"] > 0.0

    def test_all_si_release_no_colon_dose(self):
        result = modified_release_colonic_pk(
            _DRUG, _DOSE, 1.0, _CL, _VD, solubility_mg_mL=_SOL
        )
        assert result["colon_dose_mg"] == pytest.approx(0.0, abs=1e-9)

    def test_no_si_release_all_colon(self):
        result = modified_release_colonic_pk(
            _DRUG, _DOSE, 0.0, _CL, _VD, solubility_mg_mL=_SOL
        )
        assert result["si_cmax"] == pytest.approx(0.0, abs=1e-9)

    def test_invalid_si_release_pct(self):
        with pytest.raises(ValueError, match="si_release_pct"):
            modified_release_colonic_pk(_DRUG, _DOSE, 1.5, _CL, _VD)

    def test_invalid_cl(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            modified_release_colonic_pk(_DRUG, _DOSE, 0.5, 0.0, _VD)

    def test_high_si_note(self):
        result = modified_release_colonic_pk(
            _DRUG, _DOSE, 0.95, _CL, _VD, solubility_mg_mL=_SOL
        )
        assert any("SI" in n or "minor" in n for n in result["notes"])

    def test_colon_result_type(self):
        result = modified_release_colonic_pk(
            _DRUG, _DOSE, 0.5, _CL, _VD, solubility_mg_mL=_SOL
        )
        assert isinstance(result["colon_result"], ColonicAbsorptionResult)

    def test_combined_c_plasma_length_matches_times(self):
        result = modified_release_colonic_pk(
            _DRUG, _DOSE, 0.5, _CL, _VD, solubility_mg_mL=_SOL
        )
        assert len(result["combined_c_plasma"]) == len(result["combined_times_h"])
