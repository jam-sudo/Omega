"""Tests for renal excretion predictor (Phase 396)."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.prediction.excretion_predictor import (
    RenalExcretionResult,
    predict_renal_excretion,
    urine_pH_sensitivity,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _predict(**kwargs) -> RenalExcretionResult:
    defaults = dict(
        mw=250.0,
        logP=1.0,
        pka=7.4,
        molecule_type="neutral",
        fu_plasma=0.5,
        gfr_mL_per_min=120.0,
    )
    defaults.update(kwargs)
    return predict_renal_excretion(**defaults)


# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------

class TestRenalExcretionResultStructure:
    def test_returns_correct_type(self):
        result = _predict()
        assert isinstance(result, RenalExcretionResult)

    def test_frozen_dataclass(self):
        result = _predict()
        with pytest.raises((AttributeError, TypeError)):
            result.mw = 999.0  # type: ignore[misc]

    def test_fields_preserved(self):
        result = _predict(mw=300.0, logP=2.0, pka=8.0, molecule_type="base",
                          fu_plasma=0.3, gfr_mL_per_min=90.0)
        assert result.mw == 300.0
        assert result.logP == 2.0
        assert result.pka == 8.0
        assert result.molecule_type == "base"
        assert result.fu_plasma == 0.3
        assert result.gfr_mL_per_min == 90.0

    def test_notes_is_string(self):
        result = _predict()
        assert isinstance(result.notes, str)
        assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Filtration clearance
# ---------------------------------------------------------------------------

class TestFiltrationClearance:
    def test_cl_filtration_equals_gfr_times_fup(self):
        result = _predict(fu_plasma=0.4, gfr_mL_per_min=120.0)
        expected = 120.0 * 0.4
        assert abs(result.cl_filtration_mL_per_min - expected) < 1e-9

    def test_cl_filtration_proportional_to_gfr(self):
        r1 = _predict(gfr_mL_per_min=60.0)
        r2 = _predict(gfr_mL_per_min=120.0)
        assert abs(r2.cl_filtration_mL_per_min / r1.cl_filtration_mL_per_min - 2.0) < 1e-9

    def test_cl_filtration_proportional_to_fup(self):
        r1 = _predict(fu_plasma=0.2)
        r2 = _predict(fu_plasma=0.8)
        assert abs(r2.cl_filtration_mL_per_min / r1.cl_filtration_mL_per_min - 4.0) < 1e-9


# ---------------------------------------------------------------------------
# Reabsorption
# ---------------------------------------------------------------------------

class TestReabsorption:
    def test_reabsorption_between_0_and_1(self):
        result = _predict()
        assert 0.0 <= result.f_reabsorption <= 1.0

    def test_high_logp_high_reabsorption(self):
        result = _predict(logP=4.0, molecule_type="neutral")
        assert result.f_reabsorption > 0.7

    def test_low_logp_low_reabsorption(self):
        result = _predict(logP=-3.0, molecule_type="neutral")
        assert result.f_reabsorption < 0.3

    def test_reabsorption_increases_with_logP_neutral(self):
        r_low = _predict(logP=-2.0, molecule_type="neutral")
        r_high = _predict(logP=3.0, molecule_type="neutral")
        assert r_high.f_reabsorption > r_low.f_reabsorption


# ---------------------------------------------------------------------------
# Renal clearance
# ---------------------------------------------------------------------------

class TestRenalClearance:
    def test_cl_renal_nonnegative(self):
        result = _predict()
        assert result.cl_renal_mL_per_min >= 0.0

    def test_cl_renal_leq_filtration(self):
        result = _predict()
        assert result.cl_renal_mL_per_min <= result.cl_filtration_mL_per_min + 1e-9

    def test_cl_renal_L_per_h_conversion(self):
        result = _predict()
        expected = result.cl_renal_mL_per_min * 60.0 / 1000.0
        assert abs(result.cl_renal_L_per_h - expected) < 1e-9

    def test_high_logp_lower_cl_renal(self):
        r_low = _predict(logP=-2.0, molecule_type="neutral")
        r_high = _predict(logP=4.0, molecule_type="neutral")
        assert r_high.cl_renal_mL_per_min < r_low.cl_renal_mL_per_min


# ---------------------------------------------------------------------------
# Ionization effects
# ---------------------------------------------------------------------------

class TestIonizationEffects:
    def test_neutral_ionization_zero(self):
        result = _predict(molecule_type="neutral")
        assert result.ionization_fraction_at_urine_pH == 0.0

    def test_zwitterion_ionization_zero(self):
        result = _predict(molecule_type="zwitterion")
        assert result.ionization_fraction_at_urine_pH == 0.0

    def test_weak_acid_partial_ionization_at_urine_ph(self):
        # pKa = 6.0, urine pH = 6.0 → fi = 0.5
        result = _predict(pka=6.0, molecule_type="acid")
        assert abs(result.ionization_fraction_at_urine_pH - 0.5) < 1e-6

    def test_strong_acid_mostly_ionized(self):
        # pKa = 3.0, urine pH 6.0 → nearly fully ionized
        result = _predict(pka=3.0, molecule_type="acid")
        assert result.ionization_fraction_at_urine_pH > 0.99

    def test_weak_base_partial_ionization(self):
        # pKa = 6.0, urine pH = 6.0 → fi = 0.5
        result = _predict(pka=6.0, molecule_type="base")
        assert abs(result.ionization_fraction_at_urine_pH - 0.5) < 1e-6

    def test_ionization_reduces_reabsorption_acid(self):
        # At pKa=4, acid is very ionized at urine pH → lower reabsorption
        r_neutral = _predict(logP=3.0, molecule_type="neutral")
        r_acid = _predict(logP=3.0, pka=4.0, molecule_type="acid")
        assert r_acid.f_reabsorption < r_neutral.f_reabsorption


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestPredictValidation:
    def test_negative_mw_raises(self):
        with pytest.raises(ValueError, match="mw"):
            _predict(mw=-100.0)

    def test_zero_mw_raises(self):
        with pytest.raises(ValueError, match="mw"):
            _predict(mw=0.0)

    def test_invalid_molecule_type_raises(self):
        with pytest.raises(ValueError, match="molecule_type"):
            _predict(molecule_type="amine")

    def test_zero_fu_plasma_raises(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            _predict(fu_plasma=0.0)

    def test_fu_plasma_above_1_raises(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            _predict(fu_plasma=1.5)

    def test_zero_gfr_raises(self):
        with pytest.raises(ValueError, match="gfr_mL_per_min"):
            _predict(gfr_mL_per_min=0.0)

    def test_negative_gfr_raises(self):
        with pytest.raises(ValueError, match="gfr_mL_per_min"):
            _predict(gfr_mL_per_min=-10.0)

    def test_logp_out_of_range_raises(self):
        with pytest.raises(ValueError, match="logP"):
            _predict(logP=20.0)

    def test_non_finite_pka_raises(self):
        with pytest.raises(ValueError, match="pka"):
            _predict(pka=math.nan)


# ---------------------------------------------------------------------------
# urine_pH_sensitivity
# ---------------------------------------------------------------------------

class TestUrinePHSensitivity:
    def _sweep(self, **kwargs):
        defaults = dict(
            mw=250.0, logP=1.0, pka=5.0, molecule_type="acid",
            fu_plasma=0.5, gfr_mL_per_min=120.0,
            urine_pH_values=[4.5, 5.0, 6.0, 7.0, 8.0],
        )
        defaults.update(kwargs)
        return urine_pH_sensitivity(**defaults)

    def test_returns_list_of_dicts(self):
        result = self._sweep()
        assert isinstance(result, list)
        assert all(isinstance(r, dict) for r in result)

    def test_length_matches_ph_values(self):
        result = self._sweep(urine_pH_values=[5.0, 6.0, 7.0])
        assert len(result) == 3

    def test_result_keys_present(self):
        result = self._sweep()
        expected_keys = {
            "urine_pH", "cl_filtration_mL_per_min", "f_reabsorption",
            "cl_renal_mL_per_min", "cl_renal_L_per_h", "ionization_fraction",
        }
        for r in result:
            assert expected_keys <= set(r.keys())

    def test_acid_higher_clr_at_high_ph(self):
        # Acid more ionized at higher pH → less reabsorption → higher CLr
        result = self._sweep(pka=5.0, molecule_type="acid")
        cl_low_ph = next(r["cl_renal_mL_per_min"] for r in result if r["urine_pH"] == 4.5)
        cl_high_ph = next(r["cl_renal_mL_per_min"] for r in result if r["urine_pH"] == 8.0)
        assert cl_high_ph > cl_low_ph

    def test_neutral_drug_ph_insensitive(self):
        result = self._sweep(molecule_type="neutral", pka=5.0)
        cl_values = [r["cl_renal_mL_per_min"] for r in result]
        assert max(cl_values) - min(cl_values) < 1e-9

    def test_empty_ph_list_raises(self):
        with pytest.raises(ValueError, match="urine_pH_values"):
            urine_pH_sensitivity(
                mw=250.0, logP=1.0, pka=5.0, molecule_type="acid",
                fu_plasma=0.5, gfr_mL_per_min=120.0, urine_pH_values=[],
            )

    def test_zero_gfr_raises(self):
        with pytest.raises(ValueError, match="gfr_mL_per_min"):
            urine_pH_sensitivity(
                mw=250.0, logP=1.0, pka=5.0, molecule_type="acid",
                fu_plasma=0.5, gfr_mL_per_min=0.0, urine_pH_values=[6.0],
            )
