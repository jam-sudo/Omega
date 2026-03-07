"""Tests for clinical/hepatic_extraction.py — Phase 472."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.clinical.hepatic_extraction import (
    HepaticExtractionResult,
    compare_models,
    dispersion_model,
    first_pass_effect,
    parallel_tube_model,
    well_stirred_model,
)

# ---------------------------------------------------------------------------
# Helper constants
# ---------------------------------------------------------------------------

QH = 1500.0  # mL/min, default hepatic blood flow


# ---------------------------------------------------------------------------
# first_pass_effect
# ---------------------------------------------------------------------------


class TestFirstPassEffect:
    def test_eh_0_gives_fg_1(self):
        assert first_pass_effect(0.0) == pytest.approx(1.0)

    def test_eh_1_gives_fg_0(self):
        assert first_pass_effect(1.0) == pytest.approx(0.0)

    def test_eh_09_gives_fg_01(self):
        assert first_pass_effect(0.9) == pytest.approx(0.1)

    def test_eh_05_gives_fg_05(self):
        assert first_pass_effect(0.5) == pytest.approx(0.5)

    def test_invalid_eh_above_1_raises(self):
        with pytest.raises(ValueError):
            first_pass_effect(1.5)

    def test_invalid_eh_below_0_raises(self):
        with pytest.raises(ValueError):
            first_pass_effect(-0.1)


# ---------------------------------------------------------------------------
# well_stirred_model
# ---------------------------------------------------------------------------


class TestWellStirredModel:
    def test_returns_dict_with_required_keys(self):
        result = well_stirred_model(100.0, 0.5)
        assert "eh" in result
        assert "cl_h_mL_per_min" in result
        assert "fg" in result

    def test_high_clint_high_eh(self):
        result = well_stirred_model(50000.0, 1.0, QH)
        assert result["eh"] > 0.90

    def test_low_clint_low_eh(self):
        result = well_stirred_model(1.0, 0.1, QH)
        assert result["eh"] < 0.01

    def test_eh_between_0_and_1(self):
        result = well_stirred_model(500.0, 0.5, QH)
        assert 0.0 <= result["eh"] <= 1.0

    def test_cl_h_equals_eh_times_qh(self):
        result = well_stirred_model(500.0, 0.5, QH)
        assert result["cl_h_mL_per_min"] == pytest.approx(result["eh"] * QH, rel=1e-6)

    def test_fg_equals_1_minus_eh(self):
        result = well_stirred_model(500.0, 0.5, QH)
        assert result["fg"] == pytest.approx(1.0 - result["eh"], rel=1e-6)

    def test_lower_fup_lower_clh_low_extraction(self):
        # Low extraction regime: CL_h proportional to fu_p * CLint
        high_fup = well_stirred_model(10.0, 1.0, QH)
        low_fup = well_stirred_model(10.0, 0.1, QH)
        assert high_fup["cl_h_mL_per_min"] > low_fup["cl_h_mL_per_min"]

    def test_invalid_clint_raises(self):
        with pytest.raises(ValueError):
            well_stirred_model(0.0, 0.5)

    def test_invalid_fup_raises(self):
        with pytest.raises(ValueError):
            well_stirred_model(100.0, 0.0)

    def test_negative_fup_raises(self):
        with pytest.raises(ValueError):
            well_stirred_model(100.0, -0.1)


# ---------------------------------------------------------------------------
# parallel_tube_model
# ---------------------------------------------------------------------------


class TestParallelTubeModel:
    def test_returns_dict_with_required_keys(self):
        result = parallel_tube_model(100.0, 0.5)
        assert "eh" in result
        assert "cl_h_mL_per_min" in result
        assert "fg" in result

    def test_high_clint_approaches_1(self):
        result = parallel_tube_model(100000.0, 1.0, QH)
        assert result["eh"] > 0.99

    def test_low_clint_low_eh(self):
        result = parallel_tube_model(1.0, 0.1, QH)
        assert result["eh"] < 0.01

    def test_eh_between_0_and_1(self):
        result = parallel_tube_model(500.0, 0.5, QH)
        assert 0.0 <= result["eh"] <= 1.0

    def test_cl_h_equals_eh_times_qh(self):
        result = parallel_tube_model(500.0, 0.5, QH)
        assert result["cl_h_mL_per_min"] == pytest.approx(result["eh"] * QH, rel=1e-6)

    def test_parallel_tube_ge_well_stirred(self):
        # Parallel-tube Eh >= well-stirred Eh for the same inputs
        ws = well_stirred_model(500.0, 0.5, QH)
        pt = parallel_tube_model(500.0, 0.5, QH)
        assert pt["eh"] >= ws["eh"] - 1e-9

    def test_invalid_clint_raises(self):
        with pytest.raises(ValueError):
            parallel_tube_model(-1.0, 0.5)

    def test_invalid_fup_raises(self):
        with pytest.raises(ValueError):
            parallel_tube_model(100.0, 1.5)


# ---------------------------------------------------------------------------
# dispersion_model
# ---------------------------------------------------------------------------


class TestDispersionModel:
    def test_returns_dict_with_required_keys(self):
        result = dispersion_model(100.0, 0.5)
        assert "eh" in result
        assert "cl_h_mL_per_min" in result
        assert "fg" in result

    def test_eh_between_0_and_1(self):
        result = dispersion_model(500.0, 0.5, QH)
        assert 0.0 <= result["eh"] <= 1.0

    def test_high_clint_high_eh(self):
        result = dispersion_model(50000.0, 1.0, QH)
        assert result["eh"] > 0.90

    def test_low_clint_low_eh(self):
        result = dispersion_model(1.0, 0.1, QH)
        assert result["eh"] < 0.005

    def test_invalid_dn_raises(self):
        with pytest.raises(ValueError):
            dispersion_model(100.0, 0.5, dn=0.0)

    def test_invalid_clint_raises(self):
        with pytest.raises(ValueError):
            dispersion_model(0.0, 0.5)


# ---------------------------------------------------------------------------
# compare_models
# ---------------------------------------------------------------------------


class TestCompareModels:
    def test_returns_hepatic_extraction_result(self):
        result = compare_models(500.0, 0.5)
        assert isinstance(result, HepaticExtractionResult)

    def test_all_three_eh_present(self):
        result = compare_models(500.0, 0.5)
        assert 0.0 <= result.eh_well_stirred <= 1.0
        assert 0.0 <= result.eh_parallel_tube <= 1.0
        assert 0.0 <= result.eh_dispersion <= 1.0

    def test_all_three_clh_present(self):
        result = compare_models(500.0, 0.5)
        assert result.cl_h_well_stirred > 0
        assert result.cl_h_parallel_tube > 0
        assert result.cl_h_dispersion > 0

    def test_fg_sums_correctly(self):
        result = compare_models(500.0, 0.5)
        assert result.fg_well_stirred == pytest.approx(1.0 - result.eh_well_stirred, rel=1e-6)
        assert result.fg_parallel_tube == pytest.approx(1.0 - result.eh_parallel_tube, rel=1e-6)
        assert result.fg_dispersion == pytest.approx(1.0 - result.eh_dispersion, rel=1e-6)

    def test_high_extraction_flag(self):
        result = compare_models(50000.0, 1.0)
        assert result.high_extraction is True

    def test_low_extraction_flag(self):
        result = compare_models(1.0, 0.1)
        assert result.low_extraction is True

    def test_not_both_high_and_low_extraction(self):
        result = compare_models(500.0, 0.5)
        assert not (result.high_extraction and result.low_extraction)

    def test_notes_is_nonempty_string(self):
        result = compare_models(500.0, 0.5)
        assert isinstance(result.notes, str) and len(result.notes) > 0

    def test_fields_preserved(self):
        result = compare_models(200.0, 0.3, 1200.0)
        assert result.cl_int_mL_per_min == 200.0
        assert result.fu_p == 0.3
        assert result.qh_mL_per_min == 1200.0

    def test_parallel_tube_eh_ge_well_stirred(self):
        result = compare_models(500.0, 0.5)
        assert result.eh_parallel_tube >= result.eh_well_stirred - 1e-9

    def test_invalid_clint_raises(self):
        with pytest.raises(ValueError):
            compare_models(0.0, 0.5)

    def test_invalid_fup_raises(self):
        with pytest.raises(ValueError):
            compare_models(100.0, 0.0)

    def test_invalid_qh_raises(self):
        with pytest.raises(ValueError):
            compare_models(100.0, 0.5, qh_mL_per_min=0.0)
