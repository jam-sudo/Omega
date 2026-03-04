"""Tests for reference drug compounds: warfarin, propranolol, metformin, caffeine.

Tests cover:
  - Import of Drug objects from omega_pbpk.drugs.<compound>
  - Molecular weight in physiologically plausible range
  - Fraction unbound (fup) in (0, 1]
  - logP within literature-expected range for each compound
  - Non-negative intrinsic clearance
  - YAML loading for compounds/warfarin.yaml, etc.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

# Root directory of the project (one level above this test file)
PROJECT_ROOT = Path(__file__).parent.parent
COMPOUNDS_DIR = PROJECT_ROOT / "compounds"
BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks" / "configs"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _try_import_drug(module_path: str, class_name: str):
    """Attempt to import a Drug class; skip test if module not yet available."""
    try:
        import importlib

        mod = importlib.import_module(module_path)
        return getattr(mod, class_name)
    except (ImportError, AttributeError):
        pytest.skip(f"{module_path}.{class_name} not yet implemented")


# ---------------------------------------------------------------------------
# Warfarin
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestWarfarin:
    SMILES_OR_CLASS = ("omega_pbpk.drugs.warfarin", "WARFARIN")

    @pytest.fixture(scope="class")
    def warfarin(self):
        return _try_import_drug(*self.SMILES_OR_CLASS)

    def test_warfarin_import(self):
        _try_import_drug(*self.SMILES_OR_CLASS)

    def test_warfarin_mw_reasonable(self, warfarin):
        assert 100 <= warfarin.mw <= 2000, f"MW out of range: {warfarin.mw}"

    def test_warfarin_fup_in_range(self, warfarin):
        assert 0 < warfarin.fup <= 1.0, f"fup out of range: {warfarin.fup}"

    def test_warfarin_logP_known_range(self, warfarin):
        assert 2.0 <= warfarin.logP <= 4.0, f"Warfarin logP should be 2-4, got {warfarin.logP}"

    def test_warfarin_clint_non_negative(self, warfarin):
        clint_total = sum(warfarin.clint.values()) if warfarin.clint else 0.0
        assert clint_total >= 0, f"clint must be non-negative, got {clint_total}"


@pytest.mark.unit
class TestWarfarinYaml:
    def _load_yaml(self, filename: str) -> dict:
        """Load a YAML file from the compounds or benchmarks directory."""
        for search_dir in (COMPOUNDS_DIR, BENCHMARKS_DIR):
            path = search_dir / filename
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    return yaml.safe_load(f)
        pytest.skip(f"{filename} not found in compounds/ or benchmarks/configs/")

    def test_warfarin_yaml_loads(self):
        """Warfarin YAML contains clint_hepatic_L_per_h or clearance section."""
        data = self._load_yaml("warfarin.yaml")
        assert data is not None
        # Accept either top-level clint_hepatic_L_per_h or nested clearance section
        has_clint = (
            "clint_hepatic_L_per_h" in data
            or "clearance" in data
            or any("clint" in str(k).lower() for k in data.keys())
        )
        assert has_clint, f"warfarin.yaml missing clearance-related key; keys={list(data.keys())}"

    def test_propranolol_yaml_loads(self):
        """Propranolol YAML loads and has a clearance-related key."""
        data = self._load_yaml("propranolol.yaml")
        assert data is not None
        has_clint = (
            "clint_hepatic_L_per_h" in data
            or "clearance" in data
            or any("clint" in str(k).lower() for k in data.keys())
        )
        assert has_clint, (
            f"propranolol.yaml missing clearance-related key; keys={list(data.keys())}"
        )

    def test_metformin_yaml_loads(self):
        """Metformin YAML loads and has a clearance-related key."""
        data = self._load_yaml("metformin.yaml")
        assert data is not None
        has_clint = (
            "clint_hepatic_L_per_h" in data
            or "clearance" in data
            or any("clint" in str(k).lower() for k in data.keys())
        )
        assert has_clint, f"metformin.yaml missing clearance-related key; keys={list(data.keys())}"

    def test_caffeine_yaml_loads(self):
        """Caffeine YAML loads and has a clearance-related key."""
        # caffeine.yaml exists in compounds/
        data = self._load_yaml("caffeine.yaml")
        assert data is not None
        # caffeine.yaml uses nested 'clearance' section
        has_clint = (
            "clint_hepatic_L_per_h" in data
            or "clearance" in data
            or any("clint" in str(k).lower() for k in data.keys())
        )
        assert has_clint, f"caffeine.yaml missing clearance-related key; keys={list(data.keys())}"


# ---------------------------------------------------------------------------
# Propranolol
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPropranolol:
    SMILES_OR_CLASS = ("omega_pbpk.drugs.propranolol", "PROPRANOLOL")

    @pytest.fixture(scope="class")
    def propranolol(self):
        return _try_import_drug(*self.SMILES_OR_CLASS)

    def test_propranolol_import(self):
        _try_import_drug(*self.SMILES_OR_CLASS)

    def test_propranolol_mw_reasonable(self, propranolol):
        assert 100 <= propranolol.mw <= 2000, f"MW out of range: {propranolol.mw}"

    def test_propranolol_fup_in_range(self, propranolol):
        assert 0 < propranolol.fup <= 1.0, f"fup out of range: {propranolol.fup}"

    def test_propranolol_logP_known_range(self, propranolol):
        assert 3.0 <= propranolol.logP <= 5.0, (
            f"Propranolol logP should be 3-5, got {propranolol.logP}"
        )

    def test_propranolol_clint_non_negative(self, propranolol):
        clint_total = sum(propranolol.clint.values()) if propranolol.clint else 0.0
        assert clint_total >= 0, f"clint must be non-negative, got {clint_total}"


# ---------------------------------------------------------------------------
# Metformin
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMetformin:
    SMILES_OR_CLASS = ("omega_pbpk.drugs.metformin", "METFORMIN")

    @pytest.fixture(scope="class")
    def metformin(self):
        return _try_import_drug(*self.SMILES_OR_CLASS)

    def test_metformin_import(self):
        _try_import_drug(*self.SMILES_OR_CLASS)

    def test_metformin_mw_reasonable(self, metformin):
        assert 100 <= metformin.mw <= 2000, f"MW out of range: {metformin.mw}"

    def test_metformin_fup_in_range(self, metformin):
        assert 0 < metformin.fup <= 1.0, f"fup out of range: {metformin.fup}"

    def test_metformin_logP_known_range(self, metformin):
        assert -3.0 <= metformin.logP <= 0.0, (
            f"Metformin logP should be -3 to 0, got {metformin.logP}"
        )

    def test_metformin_clint_non_negative(self, metformin):
        clint_total = sum(metformin.clint.values()) if metformin.clint else 0.0
        assert clint_total >= 0, f"clint must be non-negative, got {clint_total}"


# ---------------------------------------------------------------------------
# Caffeine
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCaffeine:
    SMILES_OR_CLASS = ("omega_pbpk.drugs.caffeine", "CAFFEINE")

    @pytest.fixture(scope="class")
    def caffeine(self):
        return _try_import_drug(*self.SMILES_OR_CLASS)

    def test_caffeine_import(self):
        _try_import_drug(*self.SMILES_OR_CLASS)

    def test_caffeine_mw_reasonable(self, caffeine):
        assert 100 <= caffeine.mw <= 2000, f"MW out of range: {caffeine.mw}"

    def test_caffeine_fup_in_range(self, caffeine):
        assert 0 < caffeine.fup <= 1.0, f"fup out of range: {caffeine.fup}"

    def test_caffeine_logP_known_range(self, caffeine):
        assert -1.0 <= caffeine.logP <= 1.0, f"Caffeine logP should be -1 to 1, got {caffeine.logP}"

    def test_caffeine_clint_non_negative(self, caffeine):
        clint_total = sum(caffeine.clint.values()) if caffeine.clint else 0.0
        assert clint_total >= 0, f"clint must be non-negative, got {clint_total}"
