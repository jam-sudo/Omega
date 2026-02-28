"""Integration: Plugin → DrugSpec → SimulationEngine E2E."""
import pytest
import numpy as np
from omega_pbpk.adapters.yaml_loader import load_drug_spec_by_name
from omega_pbpk.plugins import HeuristicKpPlugin
from omega_pbpk.engine import WholeBodyPBPKEngine
from omega_pbpk.contracts import PatientSpec, Regimen


@pytest.fixture(scope="module")
def engine():
    return WholeBodyPBPKEngine()


@pytest.fixture(scope="module")
def default_patient():
    return PatientSpec()


@pytest.fixture(scope="module")
def oral_100mg():
    return Regimen(dose_mg=100.0, route="oral")


class TestPluginPipelineE2E:
    def test_kp_plugin_then_simulate(self, engine, default_patient, oral_100mg):
        spec = load_drug_spec_by_name("caffeine")
        # Kp 플러그인 적용
        kp_plugin = HeuristicKpPlugin()
        updated_spec = kp_plugin.apply(spec)
        updated_spec.validate()

        result = engine.run(updated_spec, default_patient, oral_100mg)
        assert result.success
        assert result.Cmax > 0
        assert result.Tmax >= 0

    def test_drugspec_to_drug_round_trip(self, engine, default_patient, oral_100mg):
        """DrugSpec → ODE Engine → SimulationResult 기본 검증."""
        spec = load_drug_spec_by_name("midazolam")
        result = engine.run(spec, default_patient, oral_100mg)
        assert result.success
        assert result.Cmax > 0
        plasma = result.plasma_concentration
        assert np.all(np.array(plasma) >= -1e-9), "음수 농도 발생"

    def test_pk_summary_keys(self, engine, default_patient, oral_100mg):
        spec = load_drug_spec_by_name("warfarin")
        result = engine.run(spec, default_patient, oral_100mg)
        summary = engine.pk_summary(result)
        assert "Cmax" in summary
        assert "Tmax" in summary
        assert "AUC" in summary
        assert summary["Cmax"] > 0
        assert summary["AUC"] > 0

    @pytest.mark.parametrize("drug_name", ["caffeine", "warfarin", "midazolam"])
    def test_three_drugs_simulate(self, engine, default_patient, oral_100mg, drug_name):
        spec = load_drug_spec_by_name(drug_name)
        result = engine.run(spec, default_patient, oral_100mg)
        assert result.success
        assert result.Cmax > 0
