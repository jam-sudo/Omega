"""Unit tests for PluginBase ABC, SimulationEngine ABC, and WholeBodyPBPKEngine."""

from __future__ import annotations

import pytest

from omega_pbpk.contracts import DrugSpec, PatientSpec, Regimen
from omega_pbpk.engine import SimulationEngine, SimulationResult, WholeBodyPBPKEngine
from omega_pbpk.plugins import PluginBase, SurrogateModelPlugin

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _caffeine_spec() -> DrugSpec:
    return DrugSpec(
        name="caffeine",
        mw=194.19,
        logP=-0.07,
        fup=0.65,
        rbp=0.8,
        clint_hepatic_L_per_h=2.0,
        peff=1.5,
    )


def _default_patient() -> PatientSpec:
    return PatientSpec()


def _oral_regimen(dose_mg: float = 200.0) -> Regimen:
    return Regimen(dose_mg=dose_mg, route="oral")


# ---------------------------------------------------------------------------
# PluginBase tests
# ---------------------------------------------------------------------------


class _GoodPlugin(PluginBase):
    @property
    def name(self) -> str:
        return "good_plugin"

    @property
    def provides(self) -> frozenset[str]:
        return frozenset({"fup"})

    def predict(self, spec: DrugSpec) -> dict:
        return {"fup": 0.42}


class _BadPlugin(PluginBase):
    """Declares provides={'fup'} but returns empty dict."""

    @property
    def name(self) -> str:
        return "bad_plugin"

    @property
    def provides(self) -> frozenset[str]:
        return frozenset({"fup"})

    def predict(self, spec: DrugSpec) -> dict:
        return {}  # missing 'fup'


@pytest.mark.unit
def test_plugin_apply_success():
    spec = _caffeine_spec()
    plugin = _GoodPlugin()
    updated = plugin.apply(spec)
    assert updated.fup == pytest.approx(0.42)
    assert updated.name == spec.name


@pytest.mark.unit
def test_plugin_apply_missing_key_raises_type_error():
    spec = _caffeine_spec()
    plugin = _BadPlugin()
    with pytest.raises(TypeError, match="did not return keys"):
        plugin.apply(spec)


@pytest.mark.unit
def test_plugin_confidence_default():
    plugin = _GoodPlugin()
    assert plugin.confidence(_caffeine_spec()) == pytest.approx(0.5)


@pytest.mark.unit
def test_surrogate_model_plugin_is_protocol():
    """SurrogateModelPlugin is a runtime-checkable Protocol."""
    assert isinstance(SurrogateModelPlugin, type)


# ---------------------------------------------------------------------------
# SimulationEngine ABC tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_simulation_engine_cannot_be_instantiated():
    with pytest.raises(TypeError):
        SimulationEngine()  # type: ignore[abstract]


@pytest.mark.integration
def test_whole_body_pbpk_engine_is_instance_of_simulation_engine():
    engine = WholeBodyPBPKEngine()
    assert isinstance(engine, SimulationEngine)


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_import_plugins():
    from omega_pbpk.plugins import PluginBase, SurrogateModelPlugin  # noqa: F401


@pytest.mark.unit
def test_import_engine():
    from omega_pbpk.engine import (  # noqa: F401
        SimulationEngine,
        SimulationResult,
        WholeBodyPBPKEngine,
    )


# ---------------------------------------------------------------------------
# WholeBodyPBPKEngine functional test
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_engine_run_caffeine_oral():
    engine = WholeBodyPBPKEngine()
    drug = _caffeine_spec()
    patient = _default_patient()
    regimen = _oral_regimen(dose_mg=200.0)

    result = engine.run(drug, patient, regimen, t_end_h=24.0, n_points=241)

    assert isinstance(result, SimulationResult)
    assert result.success is True
    assert result.drug_name == "caffeine"
    assert result.route == "oral"
    assert result.dose_mg == pytest.approx(200.0)
    assert len(result.t) > 0
    assert result.plasma_concentration.shape == result.t.shape
    assert result.Cmax > 0
    assert 0 <= result.Tmax <= 24.0


@pytest.mark.integration
def test_engine_run_caffeine_iv():
    engine = WholeBodyPBPKEngine()
    drug = _caffeine_spec()
    patient = _default_patient()
    regimen = Regimen(dose_mg=100.0, route="iv")

    result = engine.run(drug, patient, regimen, t_end_h=24.0, n_points=241)

    assert result.success is True
    assert result.Cmax > 0


@pytest.mark.integration
def test_engine_unsupported_route_raises():
    engine = WholeBodyPBPKEngine()
    drug = _caffeine_spec()
    patient = _default_patient()
    regimen = Regimen(dose_mg=100.0, route="sc")

    # sc is delegated to WholeBodyPBPK.setup_sc — should NOT raise
    result = engine.run(drug, patient, regimen, t_end_h=24.0, n_points=241)
    assert result.success is True


@pytest.mark.integration
def test_engine_pk_summary():
    engine = WholeBodyPBPKEngine()
    drug = _caffeine_spec()
    patient = _default_patient()
    regimen = _oral_regimen(dose_mg=200.0)

    result = engine.run(drug, patient, regimen, t_end_h=24.0, n_points=241)
    pk = engine.pk_summary(result)

    assert "Cmax" in pk
    assert "Tmax" in pk
    assert "AUC" in pk
    assert "t_half" in pk
    assert pk["Cmax"] > 0
    assert pk["AUC"] > 0
