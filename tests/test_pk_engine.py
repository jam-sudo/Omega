"""Tests for analytical 1-compartment PK engine."""

import numpy as np


class TestPredictPkAnalytical:
    def test_returns_positive_cmax(self):
        from omega_pbpk.pipeline.pk_engine import predict_pk_analytical

        result = predict_pk_analytical(dose_mg=100, cl_L_h=10.0, vd_L=50.0, f_oral=0.9, ka_h=2.0)
        assert result["cmax_mg_L"] > 0
        assert result["tmax_h"] > 0
        assert result["auc_mg_h_L"] > 0

    def test_higher_dose_higher_cmax(self):
        from omega_pbpk.pipeline.pk_engine import predict_pk_analytical

        r1 = predict_pk_analytical(dose_mg=100, cl_L_h=10.0, vd_L=50.0, f_oral=0.9, ka_h=2.0)
        r2 = predict_pk_analytical(dose_mg=200, cl_L_h=10.0, vd_L=50.0, f_oral=0.9, ka_h=2.0)
        assert r2["cmax_mg_L"] > r1["cmax_mg_L"]

    def test_higher_cl_lower_auc(self):
        from omega_pbpk.pipeline.pk_engine import predict_pk_analytical

        r1 = predict_pk_analytical(dose_mg=100, cl_L_h=5.0, vd_L=50.0, f_oral=0.9, ka_h=2.0)
        r2 = predict_pk_analytical(dose_mg=100, cl_L_h=20.0, vd_L=50.0, f_oral=0.9, ka_h=2.0)
        assert r2["auc_mg_h_L"] < r1["auc_mg_h_L"]

    def test_t_half_correct(self):
        from omega_pbpk.pipeline.pk_engine import predict_pk_analytical

        result = predict_pk_analytical(dose_mg=100, cl_L_h=10.0, vd_L=50.0, f_oral=0.9, ka_h=2.0)
        expected_t_half = 0.693 * 50.0 / 10.0  # 3.465h
        assert abs(result["t_half_h"] - expected_t_half) < 0.01

    def test_time_array_shape(self):
        from omega_pbpk.pipeline.pk_engine import predict_pk_analytical

        result = predict_pk_analytical(
            dose_mg=100, cl_L_h=10.0, vd_L=50.0, f_oral=0.9, ka_h=2.0, n_points=101
        )
        assert len(result["time_h"]) == 101
        assert len(result["cp_mg_L"]) == 101

    def test_no_negative_concentrations(self):
        from omega_pbpk.pipeline.pk_engine import predict_pk_analytical

        result = predict_pk_analytical(dose_mg=100, cl_L_h=10.0, vd_L=50.0, f_oral=0.9, ka_h=2.0)
        assert np.all(result["cp_mg_L"] >= 0)

    def test_ka_equals_ke_guard(self):
        """When ka ≈ ke, the function should not crash."""
        from omega_pbpk.pipeline.pk_engine import predict_pk_analytical

        # ke = cl/vd = 10/50 = 0.2, set ka = 0.2 (exactly equal)
        result = predict_pk_analytical(dose_mg=100, cl_L_h=10.0, vd_L=50.0, f_oral=0.9, ka_h=0.2)
        assert result["cmax_mg_L"] > 0

    def test_invalid_vd_returns_defaults(self):
        from omega_pbpk.pipeline.pk_engine import predict_pk_analytical

        result = predict_pk_analytical(dose_mg=100, cl_L_h=10.0, vd_L=0.0, f_oral=0.9, ka_h=2.0)
        assert result["cmax_mg_L"] == 1.0  # dose/100

    def test_invalid_cl_returns_defaults(self):
        from omega_pbpk.pipeline.pk_engine import predict_pk_analytical

        result = predict_pk_analytical(dose_mg=100, cl_L_h=0.0, vd_L=50.0, f_oral=0.9, ka_h=2.0)
        assert result["cmax_mg_L"] == 1.0

    def test_auc_proportional_to_dose(self):
        from omega_pbpk.pipeline.pk_engine import predict_pk_analytical

        r1 = predict_pk_analytical(dose_mg=100, cl_L_h=10.0, vd_L=50.0, f_oral=0.9, ka_h=2.0)
        r2 = predict_pk_analytical(dose_mg=200, cl_L_h=10.0, vd_L=50.0, f_oral=0.9, ka_h=2.0)
        ratio = r2["auc_mg_h_L"] / r1["auc_mg_h_L"]
        assert abs(ratio - 2.0) < 0.05


class TestComputePkParameters:
    def test_basic_computation(self):
        from omega_pbpk.pipeline.pk_engine import compute_pk_parameters

        adme = {
            "fup": 0.1,
            "logP": 2.5,
            "peff": 1.5,
            "clint_hepatocyte_uL_min": 20.0,
        }
        params = compute_pk_parameters(adme, dose_mg=100.0, smiles="CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
        assert params["cl_total"] > 0
        assert params["vd_L"] > 0
        assert 0 < params["f_oral"] <= 1.0
        assert params["ka_h"] > 0

    def test_high_clearance_drug(self):
        from omega_pbpk.pipeline.pk_engine import compute_pk_parameters

        adme = {
            "fup": 0.5,
            "logP": 1.0,
            "peff": 2.0,
            "clint_hepatocyte_uL_min": 200.0,
        }
        params = compute_pk_parameters(adme, dose_mg=500.0, smiles="C")
        # High CLint should give high hepatic clearance
        assert params["cl_hepatic"] > 10.0
        # Should have some renal clearance (logP < 2)
        assert params["cl_renal"] > 0

    def test_lipophilic_no_renal_clearance(self):
        from omega_pbpk.pipeline.pk_engine import compute_pk_parameters

        adme = {
            "fup": 0.05,
            "logP": 4.0,
            "peff": 3.0,
            "clint_hepatocyte_uL_min": 50.0,
        }
        params = compute_pk_parameters(adme, dose_mg=100.0, smiles="C")
        assert params["cl_renal"] == 0.0

    def test_hydrophilic_has_renal_clearance(self):
        from omega_pbpk.pipeline.pk_engine import compute_pk_parameters

        adme = {
            "fup": 0.9,
            "logP": 0.5,
            "peff": 0.5,
            "clint_hepatocyte_uL_min": 5.0,
        }
        params = compute_pk_parameters(adme, dose_mg=100.0, smiles="C")
        assert params["cl_renal"] > 0

    def test_minimum_vd(self):
        from omega_pbpk.pipeline.pk_engine import compute_pk_parameters

        adme = {
            "fup": 0.001,
            "logP": 3.0,
            "peff": 1.0,
            "clint_hepatocyte_uL_min": 10.0,
        }
        params = compute_pk_parameters(adme, dose_mg=100.0, smiles="C")
        # Minimum Vd = 0.04 * 70 = 2.8 L
        assert params["vd_L"] >= 0.04 * 70.0

    def test_ka_bounded(self):
        from omega_pbpk.pipeline.pk_engine import compute_pk_parameters

        # Very low peff → ka clamped to 0.3
        adme_low = {"fup": 0.1, "logP": 2.0, "peff": 0.01}
        params_low = compute_pk_parameters(adme_low, dose_mg=100.0, smiles="C")
        assert params_low["ka_h"] == 0.3

        # Very high peff → ka clamped to 5.0
        adme_high = {"fup": 0.1, "logP": 2.0, "peff": 10.0}
        params_high = compute_pk_parameters(adme_high, dose_mg=100.0, smiles="C")
        assert params_high["ka_h"] == 5.0

    def test_intermediate_values_present(self):
        from omega_pbpk.pipeline.pk_engine import compute_pk_parameters

        adme = {"fup": 0.1, "logP": 2.0, "peff": 1.0, "clint_hepatocyte_uL_min": 10.0}
        params = compute_pk_parameters(adme, dose_mg=100.0, smiles="C")
        # Check debugging intermediates are present
        assert "clint_hep" in params
        assert "clint_liver" in params
        assert "fa" in params
        assert "fh" in params
        assert "extraction_ratio" in params
