"""Tests for ADMEPredictorPlugin and HeuristicKpPlugin."""

from __future__ import annotations

import pytest

from omega_pbpk.contracts.drug_spec import DrugSpec
from omega_pbpk.plugins.adme_plugin import ADMEPredictorPlugin
from omega_pbpk.plugins.heuristic_kp import HeuristicKpPlugin

SPEC_WITH_SMILES = DrugSpec(
    name="aspirin",
    smiles="CC(=O)Oc1ccccc1C(=O)O",
    mw=180.16,
    logP=1.2,
    pka=[3.5],
    compound_type="acid",
    fup=0.3,
    rbp=0.9,
    peff=1.5,
)

SPEC_NO_SMILES = DrugSpec(
    name="test_drug",
    mw=300.0,
    logP=2.0,
    pka=[7.4],
    compound_type="base",
    fup=0.5,
)


# ---------------------------------------------------------------------------
# Import guard (works without torch)
# ---------------------------------------------------------------------------

def test_import_plugins():
    """플러그인이 torch 없이도 import 성공해야 한다."""
    from omega_pbpk.plugins import ADMEPredictorPlugin, HeuristicKpPlugin  # noqa: F401
    assert ADMEPredictorPlugin is not None
    assert HeuristicKpPlugin is not None


# ---------------------------------------------------------------------------
# ADMEPredictorPlugin
# ---------------------------------------------------------------------------

class TestADMEPredictorPlugin:
    def setup_method(self):
        self.plugin = ADMEPredictorPlugin()

    def test_predict_returns_provides_keys(self):
        result = self.plugin.predict(SPEC_WITH_SMILES)
        assert set(result.keys()) == self.plugin.provides

    def test_fup_in_range(self):
        result = self.plugin.predict(SPEC_WITH_SMILES)
        assert 0 < result["fup"] <= 1, f"fup out of range: {result['fup']}"

    def test_rbp_positive(self):
        result = self.plugin.predict(SPEC_WITH_SMILES)
        assert result["rbp"] > 0, f"rbp not positive: {result['rbp']}"

    def test_peff_non_negative(self):
        result = self.plugin.predict(SPEC_WITH_SMILES)
        assert result["peff"] >= 0, f"peff negative: {result['peff']}"

    def test_clint_non_negative(self):
        result = self.plugin.predict(SPEC_WITH_SMILES)
        assert result["clint_hepatic_L_per_h"] >= 0

    def test_predict_no_smiles_fallback(self):
        result = self.plugin.predict(SPEC_NO_SMILES)
        assert 0 < result["fup"] <= 1
        assert result["rbp"] > 0
        assert result["peff"] >= 0

    def test_apply_drugspec_validates(self):
        updated = self.plugin.apply(SPEC_WITH_SMILES)
        updated.validate()  # should not raise

    def test_apply_no_smiles_validates(self):
        updated = self.plugin.apply(SPEC_NO_SMILES)
        updated.validate()

    def test_confidence_with_smiles(self):
        conf = self.plugin.confidence(SPEC_WITH_SMILES)
        assert 0 <= conf <= 1

    def test_confidence_without_smiles(self):
        conf = self.plugin.confidence(SPEC_NO_SMILES)
        assert conf < self.plugin.confidence(SPEC_WITH_SMILES)


# ---------------------------------------------------------------------------
# HeuristicKpPlugin
# ---------------------------------------------------------------------------

class TestHeuristicKpPlugin:
    def setup_method(self):
        self.plugin = HeuristicKpPlugin()

    def test_predict_returns_provides_keys(self):
        result = self.plugin.predict(SPEC_WITH_SMILES)
        assert set(result.keys()) == self.plugin.provides

    def test_kp_is_dict(self):
        result = self.plugin.predict(SPEC_WITH_SMILES)
        assert isinstance(result["kp"], dict)

    def test_kp_all_positive(self):
        result = self.plugin.predict(SPEC_WITH_SMILES)
        kp = result["kp"]
        assert len(kp) > 0
        for tissue, val in kp.items():
            assert val > 0, f"kp[{tissue}] = {val} not positive"

    def test_kp_no_smiles(self):
        result = self.plugin.predict(SPEC_NO_SMILES)
        kp = result["kp"]
        assert all(v > 0 for v in kp.values())

    def test_apply_drugspec_validates(self):
        updated = self.plugin.apply(SPEC_WITH_SMILES)
        updated.validate()  # kp values must be >= 0

    def test_apply_updates_kp_field(self):
        updated = self.plugin.apply(SPEC_WITH_SMILES)
        assert len(updated.kp) > 0

    def test_rodgers_rowland_method(self):
        plugin_rr = HeuristicKpPlugin(method="rodgers_rowland")
        result = plugin_rr.predict(SPEC_WITH_SMILES)
        kp = result["kp"]
        assert isinstance(kp, dict)
        assert len(kp) > 0
        for tissue, val in kp.items():
            assert val > 0, f"kp[{tissue}] = {val} not positive"

    def test_rodgers_rowland_apply_validates(self):
        plugin_rr = HeuristicKpPlugin(method="rodgers_rowland")
        updated = plugin_rr.apply(SPEC_WITH_SMILES)
        updated.validate()

    def test_confidence(self):
        conf = self.plugin.confidence(SPEC_WITH_SMILES)
        assert 0 <= conf <= 1
