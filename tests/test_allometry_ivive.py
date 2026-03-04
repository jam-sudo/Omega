"""Tests for allometric scaling and IVIVE modules."""

import pytest


@pytest.mark.unit
class TestAllometry:
    def test_single_species_cl_scales_correctly(self):
        from omega_pbpk.clinical.allometry import scale_single_species

        # Rat (0.25 kg) CL → Human (70 kg): ratio = (70/0.25)^0.75 ≈ 89.4
        pred = scale_single_species("rat", cl_L_per_h_per_kg=1.0, vd_L_per_kg=1.0)
        expected_ratio = (70.0 / 0.25) ** 0.75
        assert pred.cl_human_L_per_h == pytest.approx(0.25 * expected_ratio, rel=0.01)

    def test_vd_scales_linearly(self):
        from omega_pbpk.clinical.allometry import scale_single_species

        # Vd exponent=1.0: human Vd = rat_per_kg × 70
        pred = scale_single_species("rat", cl_L_per_h_per_kg=1.0, vd_L_per_kg=2.0)
        assert pred.vd_human_L == pytest.approx(2.0 * 70.0, rel=0.01)

    def test_t_half_positive(self):
        from omega_pbpk.clinical.allometry import scale_single_species

        pred = scale_single_species("rat", cl_L_per_h_per_kg=1.0, vd_L_per_kg=1.0)
        assert pred.t_half_human_h > 0

    def test_multi_species_returns_r_squared(self):
        from omega_pbpk.clinical.allometry import scale_multi_species

        data = {"rat": (1.0, 1.0), "dog": (3.0, 3.0)}
        pred = scale_multi_species(data)
        assert pred.r_squared is not None

    def test_multi_species_larger_cl_than_rat_alone(self):
        from omega_pbpk.clinical.allometry import scale_multi_species, scale_single_species

        # Dog has higher CL per kg → multi-species should predict higher human CL
        rat_only = scale_single_species("rat", 1.0, 1.0)
        multi = scale_multi_species({"rat": (1.0, 1.0), "dog": (5.0, 1.0)})
        assert multi.cl_human_L_per_h > rat_only.cl_human_L_per_h

    def test_predict_human_returns_dict(self):
        from omega_pbpk.clinical.allometry import predict_human_from_preclinical

        result = predict_human_from_preclinical(
            smiles="C", preclinical_data={"rat": (1.0, 1.0)}, dose_mg=100.0
        )
        assert "cl_L_per_h" in result
        assert "t_half_h" in result
        assert result["cl_L_per_h"] > 0

    def test_unknown_species_uses_default_bw(self):
        from omega_pbpk.clinical.allometry import scale_single_species

        # Unknown species defaults to rat BW (0.25 kg)
        pred = scale_single_species("unknown_species", 1.0, 1.0)
        assert pred.cl_human_L_per_h > 0


@pytest.mark.unit
class TestIVIVE:
    def test_microsomal_clint_scales_to_positive_clh(self):
        from omega_pbpk.clinical.ivive import scale_microsomal_clint

        result = scale_microsomal_clint(clint_uL_min_mg=5.0, fup=0.1)
        assert result.clh_well_stirred_L_per_h > 0

    def test_clh_bounded_by_q_liver(self):
        from omega_pbpk.clinical.ivive import scale_microsomal_clint

        # Even very high CLint should not exceed Q_liver
        result = scale_microsomal_clint(clint_uL_min_mg=10000.0, fup=1.0)
        assert result.clh_well_stirred_L_per_h < 100.0  # Q_liver ≈ 96.6 L/h

    def test_hepatocyte_clint_positive(self):
        from omega_pbpk.clinical.ivive import scale_hepatocyte_clint

        result = scale_hepatocyte_clint(clint_uL_min_million_cells=10.0, fup=0.1)
        assert result.clh_well_stirred_L_per_h > 0

    def test_fu_mic_correction(self):
        from omega_pbpk.clinical.ivive import scale_microsomal_clint

        # Lower fu_mic → higher corrected CLint → higher CLh
        r_low = scale_microsomal_clint(5.0, fup=0.1, fu_mic=0.1)
        r_high = scale_microsomal_clint(5.0, fup=0.1, fu_mic=1.0)
        assert r_low.clh_well_stirred_L_per_h > r_high.clh_well_stirred_L_per_h

    def test_backcalculate_clint(self):
        from omega_pbpk.clinical.ivive import estimate_clint_for_target_clh, scale_microsomal_clint

        # Round-trip: estimate CLint → scale to CLh → should match target
        target_clh = 20.0
        fup = 0.2
        clint_est = estimate_clint_for_target_clh(target_clh, fup=fup, system="microsomes")
        result = scale_microsomal_clint(clint_est, fup=fup)
        assert result.clh_well_stirred_L_per_h == pytest.approx(target_clh, rel=0.05)

    def test_import_from_clinical(self):
        from omega_pbpk.clinical import scale_microsomal_clint, scale_single_species

        assert scale_microsomal_clint is not None
        assert scale_single_species is not None
