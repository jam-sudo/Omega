"""Tests for prediction/intrinsic_cl_scaling.py — Phase 842."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.intrinsic_cl_scaling import (
    CLintScalingResult,
    compare_models,
    scale_clint,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default(**kwargs) -> CLintScalingResult:
    defaults = dict(
        drug_name="TestDrug",
        assay_type="microsomal",
        clint_in_vitro=1.0,
        species="human",
        fu_plasma=1.0,
        fu_mic_or_hep=1.0,
        prediction_method="well_stirred",
    )
    defaults.update(kwargs)
    return scale_clint(**defaults)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_dataclass(self):
        r = _default()
        assert isinstance(r, CLintScalingResult)

    def test_drug_name_preserved(self):
        r = _default(drug_name="Midazolam")
        assert r.drug_name == "Midazolam"

    def test_notes_is_nonempty_string(self):
        r = _default()
        assert isinstance(r.notes, str)
        assert len(r.notes) > 0

    def test_assay_type_stored(self):
        r = _default(assay_type="hepatocyte")
        assert r.assay_type == "hepatocyte"

    def test_species_stored(self):
        r = _default(species="rat")
        assert r.species == "rat"


# ---------------------------------------------------------------------------
# CL_H <= Q_H (bounded)
# ---------------------------------------------------------------------------


class TestCLBounded:
    def test_cl_h_le_qh_human(self):
        # Q_H human = 20.7 mL/min/kg
        r = _default(clint_in_vitro=100.0)
        assert r.cl_hepatic_mL_per_min_per_kg <= 20.7 + 1e-9

    def test_cl_h_le_qh_rat(self):
        r = _default(species="rat", clint_in_vitro=100.0)
        assert r.cl_hepatic_mL_per_min_per_kg <= 70.0 + 1e-9

    def test_cl_h_le_qh_mouse(self):
        r = _default(species="mouse", clint_in_vitro=100.0)
        assert r.cl_hepatic_mL_per_min_per_kg <= 90.0 + 1e-9

    def test_cl_h_le_qh_dog(self):
        r = _default(species="dog", clint_in_vitro=100.0)
        assert r.cl_hepatic_mL_per_min_per_kg <= 30.0 + 1e-9

    def test_very_high_clint_does_not_exceed_qh(self):
        r = _default(clint_in_vitro=1e9)
        assert r.cl_hepatic_mL_per_min_per_kg <= 20.7 + 1e-6


# ---------------------------------------------------------------------------
# Extraction ratio in [0, 1]
# ---------------------------------------------------------------------------


class TestExtractionRatio:
    def test_extraction_ratio_in_range_low_clint(self):
        r = _default(clint_in_vitro=0.001)
        assert 0.0 <= r.extraction_ratio <= 1.0

    def test_extraction_ratio_in_range_high_clint(self):
        r = _default(clint_in_vitro=1e6)
        assert 0.0 <= r.extraction_ratio <= 1.0

    def test_extraction_ratio_clamped_below_one(self):
        r = _default(clint_in_vitro=1e9)
        assert r.extraction_ratio <= 0.9999 + 1e-10

    def test_extraction_ratio_positive(self):
        r = _default(clint_in_vitro=1.0)
        assert r.extraction_ratio >= 0.0


# ---------------------------------------------------------------------------
# hepatic_bioavailability = 1 - extraction_ratio
# ---------------------------------------------------------------------------


class TestHepBioavailability:
    def test_hepatic_ba_formula(self):
        r = _default(clint_in_vitro=1.0)
        assert r.hepatic_bioavailability == pytest.approx(1.0 - r.extraction_ratio, rel=1e-9)

    def test_hepatic_ba_in_range(self):
        r = _default(clint_in_vitro=1.0)
        assert 0.0 <= r.hepatic_bioavailability <= 1.0

    def test_high_extraction_low_fh(self):
        r_high = _default(clint_in_vitro=1e6)
        assert r_high.hepatic_bioavailability < 0.1


# ---------------------------------------------------------------------------
# Higher CLint → higher CL_H → lower hepatic bioavailability
# ---------------------------------------------------------------------------


class TestCLintEffect:
    def test_higher_clint_higher_cl_h(self):
        r_low = _default(clint_in_vitro=0.1)
        r_high = _default(clint_in_vitro=10.0)
        assert r_high.cl_hepatic_mL_per_min_per_kg > r_low.cl_hepatic_mL_per_min_per_kg

    def test_higher_clint_lower_hepatic_ba(self):
        r_low = _default(clint_in_vitro=0.1)
        r_high = _default(clint_in_vitro=10.0)
        assert r_high.hepatic_bioavailability < r_low.hepatic_bioavailability

    def test_higher_fup_higher_cl_h(self):
        r_low = _default(fu_plasma=0.1)
        r_high = _default(fu_plasma=1.0)
        assert r_high.cl_hepatic_mL_per_min_per_kg > r_low.cl_hepatic_mL_per_min_per_kg


# ---------------------------------------------------------------------------
# Well-stirred model formula
# ---------------------------------------------------------------------------


class TestWellStirred:
    def test_well_stirred_formula(self):
        q_h = 20.7
        fu = 1.0
        clint_iv = 1.0  # after scaling: 1.0 * 963 = 963 mL/min/kg
        sf = 963.0
        clint_vivo = clint_iv * sf * (fu / 1.0)
        fu_clint = fu * clint_vivo
        expected_cl_h = q_h * fu_clint / (q_h + fu_clint)
        r = _default(clint_in_vitro=clint_iv, fu_plasma=fu, fu_mic_or_hep=1.0)
        # After clamping extraction ratio
        expected_er = min(expected_cl_h / q_h, 0.9999)
        assert r.extraction_ratio == pytest.approx(expected_er, rel=1e-6)

    def test_cl_h_l_per_h_per_kg_conversion(self):
        r = _default(clint_in_vitro=0.01)
        # mL/min/kg * 60 / 1000 = L/h/kg
        expected = r.cl_hepatic_mL_per_min_per_kg / 1000.0 * 60.0
        assert r.cl_hepatic_L_per_h_per_kg == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# compare_models returns 3 results
# ---------------------------------------------------------------------------


class TestCompareModels:
    def test_returns_three_results(self):
        results = compare_models("Drug", "microsomal", 1.0, "human", 1.0, 1.0)
        assert len(results) == 3

    def test_all_are_dataclass(self):
        results = compare_models("Drug", "microsomal", 1.0, "human", 1.0, 1.0)
        assert all(isinstance(r, CLintScalingResult) for r in results)

    def test_methods_are_distinct(self):
        results = compare_models("Drug", "microsomal", 1.0, "human", 1.0, 1.0)
        methods = {r.prediction_method for r in results}
        assert len(methods) == 3

    def test_parallel_tube_ne_well_stirred_intermediate_clint(self):
        # At intermediate CLint, parallel-tube != well-stirred
        results = compare_models("Drug", "microsomal", 0.01, "human", 0.5, 1.0)
        ws = next(r for r in results if r.prediction_method == "well_stirred")
        pt = next(r for r in results if r.prediction_method == "parallel_tube")
        assert ws.cl_hepatic_mL_per_min_per_kg != pytest.approx(
            pt.cl_hepatic_mL_per_min_per_kg, rel=1e-3
        )


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_negative_clint_raises(self):
        with pytest.raises(ValueError):
            _default(clint_in_vitro=-1.0)

    def test_zero_clint_raises(self):
        with pytest.raises(ValueError):
            _default(clint_in_vitro=0.0)

    def test_fu_plasma_zero_raises(self):
        with pytest.raises(ValueError):
            _default(fu_plasma=0.0)

    def test_fu_plasma_above_one_raises(self):
        with pytest.raises(ValueError):
            _default(fu_plasma=1.1)

    def test_fu_mic_zero_raises(self):
        with pytest.raises(ValueError):
            _default(fu_mic_or_hep=0.0)

    def test_fu_mic_above_one_raises(self):
        with pytest.raises(ValueError):
            _default(fu_mic_or_hep=1.5)

    def test_unknown_species_raises(self):
        with pytest.raises(ValueError):
            _default(species="rabbit")

    def test_unknown_assay_raises(self):
        with pytest.raises(ValueError):
            _default(assay_type="plasma")
