"""Tests for population_adapter and ParameterNetPlugin."""

from __future__ import annotations

import pytest

from omega_pbpk.adapters.population_adapter import (
    sample_patient_spec,
    subject_covariates_to_patient_spec,
)
from omega_pbpk.contracts.drug_spec import DrugSpec
from omega_pbpk.contracts.patient_spec import PatientSpec
from omega_pbpk.plugins import ParameterNetPlugin

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_drug() -> DrugSpec:
    return DrugSpec(
        name="test_drug",
        clint_hepatic_L_per_h=10.0,
        clr_L_per_h=2.0,
    )


@pytest.fixture
def em_patient() -> PatientSpec:
    """Extensive metabolizer — cyp3a4_activity=1.0."""
    return PatientSpec(cyp3a4_activity=1.0, hepatic_cl_factor=1.0)


@pytest.fixture
def pm_patient() -> PatientSpec:
    """Poor metabolizer — cyp3a4_activity=0.1."""
    return PatientSpec(cyp3a4_activity=0.1, hepatic_cl_factor=1.0)


# ---------------------------------------------------------------------------
# sample_patient_spec
# ---------------------------------------------------------------------------


def test_sample_patient_spec_count():
    specs = sample_patient_spec(n=5, seed=0)
    assert len(specs) == 5


def test_sample_patient_spec_types():
    specs = sample_patient_spec(n=5, seed=1)
    for s in specs:
        assert isinstance(s, PatientSpec)


def test_sample_patient_spec_validate():
    specs = sample_patient_spec(n=10, seed=42)
    for s in specs:
        s.validate()  # must not raise


def test_sample_patient_spec_sex_values():
    specs = sample_patient_spec(n=20, seed=7)
    for s in specs:
        assert s.sex in ("male", "female")


def test_sample_patient_spec_positive_bw():
    specs = sample_patient_spec(n=10, seed=5)
    for s in specs:
        assert s.body_weight_kg > 0


# ---------------------------------------------------------------------------
# subject_covariates_to_patient_spec field mapping
# ---------------------------------------------------------------------------


def test_sex_mapping_male():
    from omega_pbpk.population.physiology import SubjectCovariates

    cov = SubjectCovariates(
        body_weight_kg=70.0,
        age_years=30.0,
        sex="M",
        height_cm=175.0,
        bmi=22.9,
        gfr_mL_min=100.0,
        child_pugh="normal",
        cyp3a4_activity=1.0,
        cyp2d6_activity=1.0,
    )
    spec = subject_covariates_to_patient_spec(cov)
    assert spec.sex == "male"


def test_sex_mapping_female():
    from omega_pbpk.population.physiology import SubjectCovariates

    cov = SubjectCovariates(
        body_weight_kg=60.0,
        age_years=25.0,
        sex="F",
        height_cm=162.0,
        bmi=22.9,
        gfr_mL_min=110.0,
        child_pugh="normal",
        cyp3a4_activity=1.0,
        cyp2d6_activity=1.0,
    )
    spec = subject_covariates_to_patient_spec(cov)
    assert spec.sex == "female"


def test_hepatic_cl_factor_child_pugh_b():
    from omega_pbpk.population.physiology import SubjectCovariates

    cov = SubjectCovariates(
        body_weight_kg=70.0,
        age_years=50.0,
        sex="M",
        height_cm=175.0,
        bmi=22.9,
        gfr_mL_min=125.0,
        child_pugh="B",
        cyp3a4_activity=1.0,
        cyp2d6_activity=1.0,
    )
    spec = subject_covariates_to_patient_spec(cov)
    assert spec.hepatic_cl_factor == pytest.approx(0.50)


def test_renal_cl_factor_normal_gfr():
    from omega_pbpk.population.physiology import SubjectCovariates

    cov = SubjectCovariates(
        body_weight_kg=70.0,
        age_years=30.0,
        sex="M",
        height_cm=175.0,
        bmi=22.9,
        gfr_mL_min=125.0,
        child_pugh="normal",
        cyp3a4_activity=1.0,
        cyp2d6_activity=1.0,
    )
    spec = subject_covariates_to_patient_spec(cov)
    assert spec.renal_cl_factor == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# ParameterNetPlugin.predict()
# ---------------------------------------------------------------------------


def test_parameter_net_plugin_provides_keys(em_patient, base_drug):
    plugin = ParameterNetPlugin(patient=em_patient)
    result = plugin.predict(base_drug)
    assert "clint_hepatic_L_per_h" in result
    assert "clr_L_per_h" in result


def test_parameter_net_em_vs_pm_clint(em_patient, pm_patient, base_drug):
    em_plugin = ParameterNetPlugin(patient=em_patient)
    pm_plugin = ParameterNetPlugin(patient=pm_patient)

    em_result = em_plugin.predict(base_drug)
    pm_result = pm_plugin.predict(base_drug)

    # PM CLint should be ~10x lower than EM
    assert pm_result["clint_hepatic_L_per_h"] < em_result["clint_hepatic_L_per_h"]
    ratio = em_result["clint_hepatic_L_per_h"] / pm_result["clint_hepatic_L_per_h"]
    assert ratio == pytest.approx(10.0, rel=0.01)


def test_parameter_net_confidence(em_patient, base_drug):
    plugin = ParameterNetPlugin(patient=em_patient)
    assert plugin.confidence(base_drug) == pytest.approx(0.7)


def test_parameter_net_apply_updates_drug(em_patient, base_drug):
    plugin = ParameterNetPlugin(patient=em_patient)
    updated = plugin.apply(base_drug)
    assert isinstance(updated, DrugSpec)
    # Reference patient at 70 kg + EM: scales should be ~1.0
    assert updated.clint_hepatic_L_per_h == pytest.approx(base_drug.clint_hepatic_L_per_h, rel=0.05)


# ---------------------------------------------------------------------------
# ParameterNetPlugin.apply_pgx()
# ---------------------------------------------------------------------------


def test_apply_pgx_pm_em_um_differ(base_drug):
    plugin = ParameterNetPlugin()

    pm_spec = plugin.apply_pgx(base_drug, diplotype="*4/*4", enzyme="CYP2D6")
    em_spec = plugin.apply_pgx(base_drug, diplotype="*1/*1", enzyme="CYP2D6")
    um_spec = plugin.apply_pgx(base_drug, diplotype="*1x2/*1", enzyme="CYP2D6")

    # PM < EM < UM
    assert pm_spec.clint_hepatic_L_per_h < em_spec.clint_hepatic_L_per_h
    assert em_spec.clint_hepatic_L_per_h < um_spec.clint_hepatic_L_per_h


def test_apply_pgx_pm_zero_clint(base_drug):
    plugin = ParameterNetPlugin()
    pm_spec = plugin.apply_pgx(base_drug, diplotype="*4/*4", enzyme="CYP2D6")
    assert pm_spec.clint_hepatic_L_per_h == pytest.approx(0.0)


def test_apply_pgx_em_full_clint(base_drug):
    plugin = ParameterNetPlugin()
    em_spec = plugin.apply_pgx(base_drug, diplotype="*1/*1", enzyme="CYP2D6")
    # *1/*1 activity_score=2.0 → scaling=1.0 → CLint unchanged
    assert em_spec.clint_hepatic_L_per_h == pytest.approx(base_drug.clint_hepatic_L_per_h)


# ---------------------------------------------------------------------------
# Import check
# ---------------------------------------------------------------------------


def test_parameter_net_plugin_importable():
    from omega_pbpk.plugins import ParameterNetPlugin as PNP  # noqa: F401

    assert PNP is ParameterNetPlugin
