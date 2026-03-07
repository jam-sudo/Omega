"""Tests for hepatic clearance models — Phase 276."""
import pytest
from omega_pbpk.core.hepatic_clearance_models import (
    HepaticCLResult,
    well_stirred_model,
    parallel_tube_model,
    dispersion_model,
    compare_hepatic_models,
)

QH = 90.0  # L/h default hepatic blood flow


class TestWellStirredModel:
    def test_returns_result(self):
        r = well_stirred_model(20.0, 0.5, QH)
        assert isinstance(r, HepaticCLResult)
        assert r.model == "well_stirred"

    def test_low_extraction_drug(self):
        r = well_stirred_model(1.0, 1.0, QH)
        assert r.eh < 0.05
        assert r.cl_hepatic_L_per_h == pytest.approx(r.eh * QH, rel=1e-6)

    def test_high_extraction_drug(self):
        r = well_stirred_model(10000.0, 1.0, QH)
        assert r.eh > 0.99
        assert r.cl_hepatic_L_per_h == pytest.approx(QH, rel=0.01)

    def test_bioavailability_complement_of_eh(self):
        r = well_stirred_model(20.0, 0.5, QH)
        assert r.bioavailability_fg == pytest.approx(1.0 - r.eh, rel=1e-9)

    def test_clint_increases_eh(self):
        r1 = well_stirred_model(10.0, 0.5, QH)
        r2 = well_stirred_model(20.0, 0.5, QH)
        assert r2.eh > r1.eh

    def test_fu_effect(self):
        r1 = well_stirred_model(20.0, 0.1, QH)
        r2 = well_stirred_model(20.0, 1.0, QH)
        assert r2.eh > r1.eh

    def test_invalid_clint(self):
        with pytest.raises(ValueError):
            well_stirred_model(0.0, 0.5, QH)

    def test_invalid_fu_zero(self):
        with pytest.raises(ValueError):
            well_stirred_model(10.0, 0.0, QH)

    def test_invalid_fu_above_one(self):
        with pytest.raises(ValueError):
            well_stirred_model(10.0, 1.5, QH)

    def test_invalid_qh(self):
        with pytest.raises(ValueError):
            well_stirred_model(10.0, 0.5, 0.0)

    def test_eh_bounds(self):
        r = well_stirred_model(50.0, 0.3, QH)
        assert 0.0 < r.eh < 1.0

    def test_notes_nonempty(self):
        r = well_stirred_model(20.0, 0.5, QH)
        assert len(r.notes) > 0

    def test_cl_hepatic_equals_qh_times_eh(self):
        r = well_stirred_model(30.0, 0.6, QH)
        assert r.cl_hepatic_L_per_h == pytest.approx(QH * r.eh, rel=1e-6)


class TestParallelTubeModel:
    def test_returns_result(self):
        r = parallel_tube_model(20.0, 0.5, QH)
        assert isinstance(r, HepaticCLResult)
        assert r.model == "parallel_tube"

    def test_high_extraction(self):
        r = parallel_tube_model(10000.0, 1.0, QH)
        assert r.eh > 0.99

    def test_low_extraction(self):
        r = parallel_tube_model(1.0, 1.0, QH)
        assert r.eh < 0.05

    def test_bioavailability(self):
        r = parallel_tube_model(20.0, 0.5, QH)
        assert r.bioavailability_fg == pytest.approx(1.0 - r.eh, rel=1e-9)

    def test_invalid_clint(self):
        with pytest.raises(ValueError):
            parallel_tube_model(-1.0, 0.5, QH)

    def test_invalid_fu(self):
        with pytest.raises(ValueError):
            parallel_tube_model(10.0, 0.0, QH)

    def test_eh_in_range(self):
        r = parallel_tube_model(30.0, 0.4, QH)
        assert 0.0 < r.eh <= 1.0

    def test_cl_positive(self):
        r = parallel_tube_model(20.0, 0.5, QH)
        assert r.cl_hepatic_L_per_h > 0


class TestDispersionModel:
    def test_returns_result(self):
        r = dispersion_model(20.0, 0.5, QH)
        assert isinstance(r, HepaticCLResult)
        assert r.model == "dispersion"

    def test_eh_positive(self):
        r = dispersion_model(20.0, 0.5, QH)
        assert r.eh > 0

    def test_bioavailability(self):
        r = dispersion_model(20.0, 0.5, QH)
        assert r.bioavailability_fg == pytest.approx(1.0 - r.eh, rel=1e-9)

    def test_high_extraction(self):
        r = dispersion_model(5000.0, 1.0, QH)
        assert r.eh > 0.9

    def test_low_extraction(self):
        r = dispersion_model(1.0, 1.0, QH)
        assert r.eh < 0.05

    def test_invalid_clint(self):
        with pytest.raises(ValueError):
            dispersion_model(0.0, 0.5, QH)

    def test_invalid_fu(self):
        with pytest.raises(ValueError):
            dispersion_model(10.0, 1.5, QH)

    def test_cl_positive(self):
        r = dispersion_model(30.0, 0.4, QH)
        assert r.cl_hepatic_L_per_h > 0


class TestCompareHepaticModels:
    def test_returns_list_of_three(self):
        result = compare_hepatic_models(20.0, 0.5, QH)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_result_values_are_HepaticCLResult(self):
        result = compare_hepatic_models(20.0, 0.5, QH)
        for r in result:
            assert isinstance(r, HepaticCLResult)

    def test_models_are_well_stirred_parallel_tube_dispersion(self):
        result = compare_hepatic_models(20.0, 0.5, QH)
        models = [r.model for r in result]
        assert "well_stirred" in models
        assert "parallel_tube" in models
        assert "dispersion" in models

    def test_all_models_similar_for_intermediate_extraction(self):
        result = compare_hepatic_models(20.0, 0.5, QH)
        ehs = [r.eh for r in result]
        assert max(ehs) - min(ehs) < 0.3

    def test_invalid_clint(self):
        with pytest.raises(ValueError):
            compare_hepatic_models(0.0, 0.5, QH)

    def test_invalid_fu(self):
        with pytest.raises(ValueError):
            compare_hepatic_models(10.0, 0.0, QH)

    def test_high_clint_all_high_eh(self):
        result = compare_hepatic_models(10000.0, 1.0, QH)
        for r in result:
            assert r.eh > 0.9

    def test_low_clint_all_low_eh(self):
        result = compare_hepatic_models(1.0, 1.0, QH)
        for r in result:
            assert r.eh < 0.05
