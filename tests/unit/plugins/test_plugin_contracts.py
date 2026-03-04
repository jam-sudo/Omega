"""Plugin 공통 계약 단위 테스트."""

import pytest

from omega_pbpk.adapters.yaml_loader import load_drug_spec_by_name
from omega_pbpk.plugins import ADMEPredictorPlugin, HeuristicKpPlugin
from omega_pbpk.plugins.base import PluginBase


@pytest.fixture(scope="module")
def caffeine_spec():
    return load_drug_spec_by_name("caffeine")


@pytest.mark.unit
class TestADMEPredictorPlugin:
    def test_provides_set(self):
        p = ADMEPredictorPlugin()
        assert isinstance(p.provides, frozenset)
        assert len(p.provides) > 0

    def test_predict_returns_all_provides_keys(self, caffeine_spec):
        p = ADMEPredictorPlugin()
        result = p.predict(caffeine_spec)
        assert p.provides.issubset(set(result.keys()))

    def test_fup_in_range(self, caffeine_spec):
        p = ADMEPredictorPlugin()
        result = p.predict(caffeine_spec)
        assert 0 < result["fup"] <= 1.0

    def test_rbp_positive(self, caffeine_spec):
        p = ADMEPredictorPlugin()
        result = p.predict(caffeine_spec)
        assert result["rbp"] > 0

    def test_apply_returns_drugspec(self, caffeine_spec):
        p = ADMEPredictorPlugin()
        updated = p.apply(caffeine_spec)
        updated.validate()  # 물리 범위 검증

    def test_confidence_in_range(self, caffeine_spec):
        p = ADMEPredictorPlugin()
        c = p.confidence(caffeine_spec)
        assert 0.0 <= c <= 1.0


@pytest.mark.unit
class TestHeuristicKpPlugin:
    def test_kp_all_positive(self, caffeine_spec):
        p = HeuristicKpPlugin()
        result = p.predict(caffeine_spec)
        assert "kp" in result
        for v in result["kp"].values():
            assert v > 0, f"Negative kp: {v}"

    def test_rodgers_rowland(self, caffeine_spec):
        p = HeuristicKpPlugin(method="rodgers_rowland")
        result = p.predict(caffeine_spec)
        assert "kp" in result
        assert len(result["kp"]) > 0


@pytest.mark.unit
class TestPluginBaseContract:
    def test_provides_mismatch_raises(self, caffeine_spec):
        """predict()가 provides와 다른 키 반환 시 TypeError."""

        class BrokenPlugin(PluginBase):
            name = "broken"
            provides = frozenset({"nonexistent_field"})

            def predict(self, spec):
                return {"wrong_key": 1.0}

        p = BrokenPlugin()
        with pytest.raises(TypeError):
            p.apply(caffeine_spec)
