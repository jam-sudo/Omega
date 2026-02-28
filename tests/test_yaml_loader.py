"""Tests for YAML → DrugSpec adapter (M0-3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from omega_pbpk.adapters.yaml_loader import (
    drug_to_spec,
    load_drug_spec,
    load_drug_spec_by_name,
    spec_to_drug,
)
from omega_pbpk.contracts import DrugSpec

REPO_ROOT = Path(__file__).parent.parent
COMPOUNDS_DIR = REPO_ROOT / "compounds"

REFERENCE_DRUGS = ["caffeine", "warfarin", "midazolam", "propranolol", "metformin"]


class TestLoadDrugSpecByName:
    @pytest.mark.parametrize("name", REFERENCE_DRUGS)
    def test_loads_without_error(self, name: str) -> None:
        spec = load_drug_spec_by_name(name)
        assert isinstance(spec, DrugSpec)

    @pytest.mark.parametrize("name", REFERENCE_DRUGS)
    def test_validate_passes(self, name: str) -> None:
        spec = load_drug_spec_by_name(name)
        spec.validate()  # should not raise

    def test_caffeine_known_values(self) -> None:
        spec = load_drug_spec_by_name("caffeine")
        assert spec.mw == pytest.approx(194.19, abs=0.5)
        assert spec.fup == pytest.approx(0.65, abs=0.05)
        assert spec.smiles is not None

    def test_warfarin_is_acid(self) -> None:
        spec = load_drug_spec_by_name("warfarin")
        assert spec.compound_type == "acid"

    def test_param_source_is_yaml(self) -> None:
        spec = load_drug_spec_by_name("caffeine")
        assert spec.param_source == "yaml"

    def test_unknown_drug_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_drug_spec_by_name("nonexistent_compound_xyz")


class TestDrugToSpec:
    def test_round_trip_preserves_name(self) -> None:
        spec = load_drug_spec_by_name("caffeine")
        drug = spec_to_drug(spec)
        spec2 = drug_to_spec(drug)
        assert spec2.name == spec.name

    def test_round_trip_preserves_mw(self) -> None:
        spec = load_drug_spec_by_name("midazolam")
        drug = spec_to_drug(spec)
        spec2 = drug_to_spec(drug)
        assert spec2.mw == pytest.approx(spec.mw, abs=0.01)

    def test_round_trip_preserves_fup(self) -> None:
        spec = load_drug_spec_by_name("propranolol")
        drug = spec_to_drug(spec)
        spec2 = drug_to_spec(drug)
        assert spec2.fup == pytest.approx(spec.fup, rel=0.01)

    def test_kp_dict_preserved(self) -> None:
        spec = load_drug_spec_by_name("caffeine")
        assert len(spec.kp) > 0
        for val in spec.kp.values():
            assert val > 0

    @pytest.mark.parametrize("name", REFERENCE_DRUGS)
    def test_full_round_trip(self, name: str) -> None:
        spec = load_drug_spec_by_name(name)
        drug = spec_to_drug(spec)
        spec2 = drug_to_spec(drug)
        spec2.validate()
        assert spec2.name == spec.name
        assert spec2.mw == pytest.approx(spec.mw, abs=0.01)


class TestLoadDrugSpecPath:
    def test_load_by_path(self) -> None:
        path = COMPOUNDS_DIR / "caffeine.yaml"
        spec = load_drug_spec(path)
        assert spec.name == "Caffeine"
        spec.validate()
