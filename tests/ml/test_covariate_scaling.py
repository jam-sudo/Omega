"""Tests for allometric covariate scaling."""
import pytest


def test_weight_scaling_clearance():
    from omega_pbpk.ml.models.foundation.covariate_scaling import scale_clearance
    assert scale_clearance(10.0, weight_kg=70.0) == pytest.approx(10.0, rel=1e-3)
    cl_35 = scale_clearance(10.0, weight_kg=35.0)
    assert cl_35 == pytest.approx(10.0 * (35 / 70) ** 0.75, rel=1e-3)


def test_weight_scaling_volume():
    from omega_pbpk.ml.models.foundation.covariate_scaling import scale_volume
    assert scale_volume(50.0, weight_kg=70.0) == pytest.approx(50.0, rel=1e-3)
    assert scale_volume(50.0, weight_kg=35.0) == pytest.approx(25.0, rel=1e-3)


def test_cyp2d6_scaling():
    from omega_pbpk.ml.models.foundation.covariate_scaling import cyp_genotype_factor
    assert cyp_genotype_factor("CYP2D6", "EM") == pytest.approx(1.0)
    assert cyp_genotype_factor("CYP2D6", "PM") == pytest.approx(0.1)
    assert cyp_genotype_factor("CYP2D6", "UM") == pytest.approx(1.5)


def test_cyp2c9_scaling():
    from omega_pbpk.ml.models.foundation.covariate_scaling import cyp_genotype_factor
    assert cyp_genotype_factor("CYP2C9", "*1/*1") == pytest.approx(1.0)
    assert cyp_genotype_factor("CYP2C9", "*1/*3") == pytest.approx(0.6)
    assert cyp_genotype_factor("CYP2C9", "*3/*3") == pytest.approx(0.1)


def test_apply_covariates():
    from omega_pbpk.ml.models.foundation.covariate_scaling import apply_covariates
    base = {"cl_L_h": 10.0, "vd_L": 50.0}
    adjusted = apply_covariates(base, {"weight_kg": 100.0})
    assert adjusted["cl_L_h"] > base["cl_L_h"]
    assert adjusted["vd_L"] > base["vd_L"]


def test_unknown_genotype_returns_1():
    from omega_pbpk.ml.models.foundation.covariate_scaling import cyp_genotype_factor
    assert cyp_genotype_factor("CYP_UNKNOWN", "XX") == pytest.approx(1.0)
