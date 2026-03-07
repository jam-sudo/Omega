"""Tests for Phase 357 — Hepatic Clearance Scaling (IVIVE)."""

import pytest

from omega_pbpk.prediction.ivive_hepatic_scaling import (
    IVIVEHepaticScalingResult,
    scale_hepatic_clearance,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def default_result(**kwargs) -> IVIVEHepaticScalingResult:
    """Return a result with sensible defaults, overridable via kwargs."""
    params = dict(
        drug_name="TestDrug",
        clint_in_vitro=10.0,
        in_vitro_system="microsomal",
        incubation_protein_mg_mL=1.0,
        fu_inc=1.0,
        fu_plasma=1.0,
    )
    params.update(kwargs)
    return scale_hepatic_clearance(**params)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_ivive_result(self):
        result = default_result()
        assert isinstance(result, IVIVEHepaticScalingResult)

    def test_result_is_frozen(self):
        result = default_result()
        with pytest.raises((AttributeError, TypeError)):
            result.cl_hepatic_L_per_h = 999.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------


class TestBasicCorrectness:
    def test_cl_hepatic_positive(self):
        result = default_result()
        assert result.cl_hepatic_L_per_h > 0

    def test_extraction_ratio_in_open_unit_interval(self):
        result = default_result()
        assert 0 < result.extraction_ratio < 1

    def test_f_hepatic_in_open_unit_interval(self):
        result = default_result()
        assert 0 < result.f_hepatic < 1

    def test_f_hepatic_complement_of_extraction(self):
        result = default_result()
        assert abs(result.f_hepatic + result.extraction_ratio - 1.0) < 1e-12

    def test_ivive_success_flag_true(self):
        result = default_result()
        assert result.ivive_success is True

    def test_well_stirred_cl_le_q_hepatic(self):
        result = default_result()
        assert result.cl_hepatic_L_per_h <= 90.0

    def test_clint_corrected_equals_liver_when_fu_equal(self):
        """When fu_inc == fu_plasma, no binding correction."""
        result = default_result(fu_inc=0.5, fu_plasma=0.5)
        assert abs(result.clint_corrected_L_per_h - result.clint_liver_L_per_h) < 1e-12

    def test_drug_name_preserved(self):
        result = default_result(drug_name="Warfarin")
        assert result.drug_name == "Warfarin"

    def test_in_vitro_system_preserved(self):
        result = default_result(in_vitro_system="hepatocyte")
        assert result.in_vitro_system == "hepatocyte"


# ---------------------------------------------------------------------------
# Higher clint → higher CL
# ---------------------------------------------------------------------------


class TestMonotonicity:
    def test_higher_clint_higher_cl_hepatic(self):
        low = default_result(clint_in_vitro=1.0)
        high = default_result(clint_in_vitro=100.0)
        assert high.cl_hepatic_L_per_h > low.cl_hepatic_L_per_h

    def test_higher_clint_higher_extraction_ratio(self):
        low = default_result(clint_in_vitro=1.0)
        high = default_result(clint_in_vitro=500.0)
        assert high.extraction_ratio > low.extraction_ratio


# ---------------------------------------------------------------------------
# fu_inc correction
# ---------------------------------------------------------------------------


class TestBindingCorrection:
    def test_lower_fu_inc_lower_corrected_clint(self):
        high_fu = default_result(fu_inc=1.0)
        low_fu = default_result(fu_inc=0.1)
        assert low_fu.clint_corrected_L_per_h < high_fu.clint_corrected_L_per_h

    def test_fu_inc_ratio_applied_correctly(self):
        r1 = default_result(fu_inc=1.0, fu_plasma=1.0)
        r2 = default_result(fu_inc=0.5, fu_plasma=1.0)
        assert abs(r2.clint_corrected_L_per_h - 0.5 * r1.clint_corrected_L_per_h) < 1e-10


# ---------------------------------------------------------------------------
# Different in_vitro_system → different scaling
# ---------------------------------------------------------------------------


class TestSystemScaling:
    def test_microsomal_vs_hepatocyte_differ(self):
        micro = default_result(in_vitro_system="microsomal", incubation_protein_mg_mL=1.0)
        hepa = default_result(in_vitro_system="hepatocyte", cell_density=1e6)
        # Different scaling constants → different CLint_liver
        assert micro.clint_liver_L_per_h != hepa.clint_liver_L_per_h

    def test_microsomal_vs_s9_differ(self):
        micro = default_result(in_vitro_system="microsomal")
        s9 = default_result(in_vitro_system="s9")
        # MPPGL=45 vs S9PPGL=120 → different scaling
        assert micro.clint_liver_L_per_h != s9.clint_liver_L_per_h

    def test_s9_higher_scaling_than_microsomal(self):
        """S9PPGL (120) > MPPGL (45), so S9 gives higher CLint_liver."""
        micro = default_result(in_vitro_system="microsomal")
        s9 = default_result(in_vitro_system="s9")
        assert s9.clint_liver_L_per_h > micro.clint_liver_L_per_h

    def test_scaling_factor_positive(self):
        for system in ("microsomal", "hepatocyte", "s9"):
            r = default_result(in_vitro_system=system)
            assert r.scaling_factor > 0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidationErrors:
    def test_zero_clint_raises(self):
        with pytest.raises(ValueError, match="clint_in_vitro"):
            scale_hepatic_clearance("Drug", clint_in_vitro=0.0)

    def test_negative_clint_raises(self):
        with pytest.raises(ValueError, match="clint_in_vitro"):
            scale_hepatic_clearance("Drug", clint_in_vitro=-1.0)

    def test_fu_inc_zero_raises(self):
        with pytest.raises(ValueError, match="fu_inc"):
            scale_hepatic_clearance("Drug", clint_in_vitro=10.0, fu_inc=0.0)

    def test_fu_inc_above_one_raises(self):
        with pytest.raises(ValueError, match="fu_inc"):
            scale_hepatic_clearance("Drug", clint_in_vitro=10.0, fu_inc=1.5)

    def test_fu_plasma_zero_raises(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            scale_hepatic_clearance("Drug", clint_in_vitro=10.0, fu_plasma=0.0)

    def test_fu_plasma_above_one_raises(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            scale_hepatic_clearance("Drug", clint_in_vitro=10.0, fu_plasma=1.1)

    def test_invalid_system_raises(self):
        with pytest.raises(ValueError, match="in_vitro_system"):
            scale_hepatic_clearance("Drug", clint_in_vitro=10.0, in_vitro_system="plasma")
