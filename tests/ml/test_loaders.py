"""Tests for clinical data loaders and dataset harmonization."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from omega_pbpk.ml.data.loaders import (
    FDALabelExtractor,
    PKDBLoader,
    TDCLoader,
    _SUPPORTED_TDC_ENDPOINTS,
)
from omega_pbpk.ml.data.datasets import (
    ClinicalPKDataset,
    convert_clearance,
    convert_concentration,
    convert_time,
    convert_volume,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture()
def tmp_cache(tmp_path: Path) -> Path:
    return tmp_path / "cache"


@pytest.fixture()
def pkdb_loader(tmp_cache: Path) -> PKDBLoader:
    return PKDBLoader(
        cache_dir=tmp_cache / "pkdb",
        base_url="https://pk-db.test/api/v1/",
        rate_limit=0.0,  # no delay in tests
    )


@pytest.fixture()
def fda_extractor(tmp_cache: Path) -> FDALabelExtractor:
    return FDALabelExtractor(
        cache_dir=tmp_cache / "fda",
        base_url="https://dailymed.test/services/v2/",
        rate_limit=0.0,
    )


@pytest.fixture()
def tdc_loader(tmp_cache: Path) -> TDCLoader:
    return TDCLoader(cache_dir=tmp_cache / "tdc")


# ===========================================================================
# PKDBLoader tests
# ===========================================================================


class TestPKDBLoader:
    def test_cache_hit(self, pkdb_loader: PKDBLoader) -> None:
        """Cached responses should be returned without HTTP call."""
        # Pre-populate cache
        from omega_pbpk.ml.data.loaders import _cache_key

        url = pkdb_loader.base_url + "studies/"
        params = {"page": 1}
        key = _cache_key(url, params)
        cache_file = pkdb_loader.cache_dir / f"{key}.json"
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        fake_data = {"count": 1, "next": None, "results": [{"sid": "S001"}]}
        cache_file.write_text(json.dumps(fake_data))

        studies = pkdb_loader.list_studies()
        assert len(studies) == 1
        assert studies[0]["sid"] == "S001"

    @patch("omega_pbpk.ml.data.loaders.PKDBLoader._get")
    def test_list_studies_pagination(self, mock_get: MagicMock, pkdb_loader: PKDBLoader) -> None:
        """Pagination should collect results from multiple pages."""
        mock_get.side_effect = [
            {"count": 3, "next": "page2", "results": [{"sid": "S001"}, {"sid": "S002"}]},
            {"count": 3, "next": None, "results": [{"sid": "S003"}]},
        ]
        studies = pkdb_loader.list_studies()
        assert len(studies) == 3
        assert studies[-1]["sid"] == "S003"

    @patch("omega_pbpk.ml.data.loaders.PKDBLoader._get")
    def test_list_studies_with_limit(self, mock_get: MagicMock, pkdb_loader: PKDBLoader) -> None:
        mock_get.return_value = {"count": 2, "next": None, "results": [{"sid": "S1"}, {"sid": "S2"}]}
        studies = pkdb_loader.list_studies(limit=1)
        assert len(studies) == 1

    @patch("omega_pbpk.ml.data.loaders.PKDBLoader._get")
    def test_get_pk_data(self, mock_get: MagicMock, pkdb_loader: PKDBLoader) -> None:
        mock_get.return_value = {
            "count": 1,
            "next": None,
            "results": [{"substance": "caffeine", "value": 5.2, "unit": "mg/L"}],
        }
        data = pkdb_loader.get_pk_data("caffeine")
        assert len(data) == 1
        assert data[0]["substance"] == "caffeine"

    @patch("omega_pbpk.ml.data.loaders.PKDBLoader._get")
    def test_get_concentration_time(self, mock_get: MagicMock, pkdb_loader: PKDBLoader) -> None:
        mock_get.return_value = {
            "count": 2,
            "next": None,
            "results": [
                {"time": 0, "value": 0},
                {"time": 1, "value": 3.5},
            ],
        }
        data = pkdb_loader.get_concentration_time("PKDB00001")
        assert len(data) == 2


# ===========================================================================
# FDALabelExtractor tests
# ===========================================================================


class TestFDALabelExtractor:
    @patch("omega_pbpk.ml.data.loaders.FDALabelExtractor._get")
    def test_search_drug(self, mock_get: MagicMock, fda_extractor: FDALabelExtractor) -> None:
        mock_get.return_value = {
            "data": [
                {"setid": "abc-123", "title": "ASPIRIN tablet"},
            ]
        }
        results = fda_extractor.search_drug("aspirin")
        assert len(results) == 1
        assert results[0]["setid"] == "abc-123"

    @patch("omega_pbpk.ml.data.loaders.FDALabelExtractor._get")
    def test_get_pk_section(self, mock_get: MagicMock, fda_extractor: FDALabelExtractor) -> None:
        mock_get.return_value = {
            "sections": [
                {
                    "loinc_code": "34090-1",
                    "text": "Cmax was 10 mg/L after oral administration.",
                }
            ]
        }
        text = fda_extractor.get_pk_section("abc-123")
        assert text is not None
        assert "Cmax" in text

    @patch("omega_pbpk.ml.data.loaders.FDALabelExtractor._get")
    def test_get_pk_section_nested(self, mock_get: MagicMock, fda_extractor: FDALabelExtractor) -> None:
        """PK section might be nested under children_sections."""
        mock_get.return_value = {
            "sections": [
                {
                    "loinc_code": "other",
                    "text": "unrelated",
                    "children_sections": [
                        {
                            "loinc_code": "43682-4",
                            "text": "half-life was 6.5 hours",
                        }
                    ],
                }
            ]
        }
        text = fda_extractor.get_pk_section("xyz")
        assert text is not None
        assert "half-life" in text

    @patch("omega_pbpk.ml.data.loaders.FDALabelExtractor._get")
    def test_get_pk_section_missing(self, mock_get: MagicMock, fda_extractor: FDALabelExtractor) -> None:
        mock_get.return_value = {
            "sections": [{"loinc_code": "other", "text": "nothing here"}]
        }
        assert fda_extractor.get_pk_section("xyz") is None

    def test_extract_pk_params_cmax(self) -> None:
        text = "The Cmax was 12.5 mg/L following a single 100 mg dose."
        result = FDALabelExtractor.extract_pk_params(text)
        assert "cmax" in result
        assert result["cmax"][0]["value"] == 12.5
        assert result["cmax"][0]["unit"] == "mg/L"

    def test_extract_pk_params_multiple(self) -> None:
        text = (
            "Cmax was 5.3 µg/mL. The elimination half-life is 4.5 hours. "
            "Bioavailability of 85 %. Volume of distribution is 45 L. "
            "Clearance was 3.2 L/h."
        )
        result = FDALabelExtractor.extract_pk_params(text)
        assert "cmax" in result
        assert result["cmax"][0]["value"] == 5.3
        assert "t_half" in result
        assert result["t_half"][0]["value"] == 4.5
        assert "bioavailability" in result
        assert result["bioavailability"][0]["value"] == 85.0
        assert "vd" in result
        assert result["vd"][0]["value"] == 45.0
        assert "clearance" in result
        assert result["clearance"][0]["value"] == 3.2

    def test_extract_pk_params_ng_ml(self) -> None:
        text = "Cmax of 250 ng/mL was observed."
        result = FDALabelExtractor.extract_pk_params(text)
        assert result["cmax"][0]["value"] == 250.0
        assert result["cmax"][0]["unit"] == "ng/mL"

    def test_extract_pk_params_empty(self) -> None:
        result = FDALabelExtractor.extract_pk_params("No PK data here.")
        assert result == {}


# ===========================================================================
# TDCLoader tests
# ===========================================================================


class TestTDCLoader:
    def test_supported_endpoints(self) -> None:
        endpoints = TDCLoader.supported_endpoints()
        assert "Caco2_Wang" in endpoints
        assert "PPBR_AZ" in endpoints
        assert "hERG" in endpoints
        assert len(endpoints) == len(_SUPPORTED_TDC_ENDPOINTS)

    def test_invalid_endpoint(self, tdc_loader: TDCLoader) -> None:
        with pytest.raises(ValueError, match="Unsupported endpoint"):
            tdc_loader.load_adme("NonExistent_Dataset")

    def test_load_adme_no_tdc(self, tdc_loader: TDCLoader) -> None:
        """If PyTDC is not installed, raise ImportError with message."""
        tdc_loader._tdc_available = False
        with pytest.raises(ImportError, match="PyTDC is required"):
            tdc_loader.load_adme("Caco2_Wang")


# ===========================================================================
# Unit conversion tests
# ===========================================================================


class TestUnitConversions:
    def test_concentration_mg_l(self) -> None:
        assert convert_concentration(10.0, "mg/L") == 10.0

    def test_concentration_ug_ml(self) -> None:
        # 1 µg/mL = 1 mg/L
        assert convert_concentration(5.0, "µg/mL") == 5.0

    def test_concentration_ng_ml(self) -> None:
        # 1 ng/mL = 0.001 mg/L
        result = convert_concentration(1000.0, "ng/mL")
        assert result is not None
        assert abs(result - 1.0) < 1e-9

    def test_concentration_unknown(self) -> None:
        assert convert_concentration(1.0, "stones/barrel") is None

    def test_time_hours(self) -> None:
        assert convert_time(2.0, "h") == 2.0

    def test_time_minutes(self) -> None:
        result = convert_time(120.0, "min")
        assert result is not None
        assert abs(result - 2.0) < 1e-9

    def test_time_days(self) -> None:
        assert convert_time(1.0, "day") == 24.0

    def test_clearance_l_h(self) -> None:
        assert convert_clearance(5.0, "L/h") == 5.0

    def test_clearance_ml_min(self) -> None:
        # 1 mL/min = 0.06 L/h
        result = convert_clearance(100.0, "mL/min")
        assert result is not None
        assert abs(result - 6.0) < 1e-9

    def test_volume_l(self) -> None:
        assert convert_volume(70.0, "L") == 70.0

    def test_volume_ml(self) -> None:
        result = convert_volume(1000.0, "mL")
        assert result is not None
        assert abs(result - 1.0) < 1e-9


# ===========================================================================
# ClinicalPKDataset tests
# ===========================================================================


class TestClinicalPKDataset:
    def test_empty_dataset(self) -> None:
        ds = ClinicalPKDataset()
        assert len(ds.records) == 0
        df = ds.to_dataframe()
        assert len(df) == 0

    def test_merge(self) -> None:
        ds1 = ClinicalPKDataset(records=[{"compound": "A", "value": 1}])
        ds2 = ClinicalPKDataset(records=[{"compound": "B", "value": 2}])
        merged = ds1.merge(ds2)
        assert len(merged.records) == 2
        # Originals unchanged
        assert len(ds1.records) == 1
        assert len(ds2.records) == 1

    def test_standardize_units_cmax(self) -> None:
        ds = ClinicalPKDataset(
            records=[
                {"parameter": "cmax", "value": 500.0, "unit": "ng/mL"},
            ]
        )
        result = ds.standardize_units()
        assert abs(result.records[0]["value"] - 0.5) < 1e-9
        assert result.records[0]["unit"] == "mg/L"

    def test_standardize_units_t_half(self) -> None:
        ds = ClinicalPKDataset(
            records=[
                {"parameter": "t_half", "value": 120.0, "unit": "min"},
            ]
        )
        result = ds.standardize_units()
        assert abs(result.records[0]["value"] - 2.0) < 1e-9
        assert result.records[0]["unit"] == "h"

    def test_standardize_units_clearance(self) -> None:
        ds = ClinicalPKDataset(
            records=[
                {"parameter": "clearance", "value": 100.0, "unit": "mL/min"},
            ]
        )
        result = ds.standardize_units()
        assert abs(result.records[0]["value"] - 6.0) < 1e-9
        assert result.records[0]["unit"] == "L/h"

    def test_standardize_preserves_unknown(self) -> None:
        """Records with unknown units should pass through unchanged."""
        ds = ClinicalPKDataset(
            records=[
                {"parameter": "cmax", "value": 10.0, "unit": "weird_unit"},
            ]
        )
        result = ds.standardize_units()
        assert result.records[0]["value"] == 10.0
        assert result.records[0]["unit"] == "weird_unit"

    def test_assign_splits(self) -> None:
        """All records should get a split assignment."""
        ds = ClinicalPKDataset(
            records=[
                {"compound": "A", "smiles": "C", "value": 1, "parameter": "cmax", "unit": "mg/L"},
                {"compound": "B", "smiles": "CC", "value": 2, "parameter": "cmax", "unit": "mg/L"},
                {"compound": "C", "smiles": None, "value": 3, "parameter": "cmax", "unit": "mg/L"},
            ]
        )
        result = ds.assign_splits(seed=42)
        splits = {r["split"] for r in result.records}
        assert splits.issubset({"train", "valid", "test"})
        # Record without SMILES should default to train
        assert result.records[2]["split"] == "train"

    def test_save_and_load(self, tmp_path: Path) -> None:
        ds = ClinicalPKDataset(
            records=[
                {"compound": "caffeine", "smiles": "Cn1cnc2c1c(=O)n(c(=O)n2C)C", "value": 5.0, "unit": "mg/L", "parameter": "cmax", "source": "test", "split": "train"},
            ]
        )
        out = ds.save(tmp_path / "test.parquet")
        assert out.exists()
        # May be .parquet or .csv depending on installed engines
        assert out.suffix in (".parquet", ".csv")

        loaded = ClinicalPKDataset.load(out)
        assert len(loaded.records) == 1
        assert loaded.records[0]["compound"] == "caffeine"
        assert loaded.records[0]["value"] == 5.0

    def test_immutability_of_standardize(self) -> None:
        """standardize_units should return a new dataset, not mutate."""
        original_records = [
            {"parameter": "cmax", "value": 500.0, "unit": "ng/mL"},
        ]
        ds = ClinicalPKDataset(records=original_records)
        result = ds.standardize_units()
        # Original should be unchanged
        assert ds.records[0]["value"] == 500.0
        assert result.records[0]["value"] != 500.0
