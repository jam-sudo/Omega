"""Tests for clinical data loaders and dataset harmonization."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

requests = pytest.importorskip("requests", reason="requests not installed")

from omega_pbpk.ml.data.datasets import (  # noqa: E402
    ClinicalPKDataset,
    convert_clearance,
    convert_concentration,
    convert_time,
    convert_volume,
    validate_dataset,
    validate_pk_record,
)
from omega_pbpk.ml.data.loaders import (  # noqa: E402
    _SUPPORTED_TDC_ENDPOINTS,
    FDALabelExtractor,
    OpenFDAExtractor,
    PKDBLoader,
    TDCLoader,
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
        mock_get.return_value = {
            "count": 2,
            "next": None,
            "results": [{"sid": "S1"}, {"sid": "S2"}],
        }
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
    def test_get_pk_section_nested(
        self, mock_get: MagicMock, fda_extractor: FDALabelExtractor
    ) -> None:
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
    def test_get_pk_section_missing(
        self, mock_get: MagicMock, fda_extractor: FDALabelExtractor
    ) -> None:
        mock_get.return_value = {"sections": [{"loinc_code": "other", "text": "nothing here"}]}
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
                {
                    "compound": "caffeine",
                    "smiles": "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
                    "value": 5.0,
                    "unit": "mg/L",
                    "parameter": "cmax",
                    "source": "test",
                    "split": "train",
                },
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


# ===========================================================================
# Data Quality Validation tests (Phase D.4)
# ===========================================================================


class TestDataQualityValidation:
    def test_validate_pass(self) -> None:
        rec = {"parameter": "cmax", "value": 10.0, "unit": "mg/L"}
        result = validate_pk_record(rec)
        assert result["qc_flag"] == "pass"

    def test_validate_out_of_range_high(self) -> None:
        rec = {"parameter": "cmax", "value": 5000.0, "unit": "mg/L"}
        result = validate_pk_record(rec)
        assert result["qc_flag"] == "out_of_range"
        assert "outside" in result["qc_detail"]

    def test_validate_out_of_range_low(self) -> None:
        rec = {"parameter": "cmax", "value": 0.00001, "unit": "mg/L"}
        result = validate_pk_record(rec)
        assert result["qc_flag"] == "out_of_range"

    def test_validate_missing_value(self) -> None:
        rec = {"parameter": "cmax", "value": None, "unit": "mg/L"}
        result = validate_pk_record(rec)
        assert result["qc_flag"] == "missing_value"

    def test_validate_non_numeric_value(self) -> None:
        rec = {"parameter": "t_half", "value": "N/A", "unit": "h"}
        result = validate_pk_record(rec)
        assert result["qc_flag"] == "missing_value"

    def test_validate_unknown_parameter_passes(self) -> None:
        """Parameters not in the range table should pass by default."""
        rec = {"parameter": "some_custom_param", "value": 999.0, "unit": ""}
        result = validate_pk_record(rec)
        assert result["qc_flag"] == "pass"

    def test_validate_t_half_range(self) -> None:
        # 0.05 to 1000 hours
        rec_ok = {"parameter": "t_half", "value": 6.0, "unit": "h"}
        assert validate_pk_record(rec_ok)["qc_flag"] == "pass"

        rec_too_long = {"parameter": "t_half", "value": 5000.0, "unit": "h"}
        assert validate_pk_record(rec_too_long)["qc_flag"] == "out_of_range"

    def test_validate_bioavailability_range(self) -> None:
        rec_ok = {"parameter": "bioavailability", "value": 0.85, "unit": "fraction"}
        assert validate_pk_record(rec_ok)["qc_flag"] == "pass"

        rec_bad = {"parameter": "bioavailability", "value": 1.5, "unit": "fraction"}
        assert validate_pk_record(rec_bad)["qc_flag"] == "out_of_range"

    def test_validate_dataset(self) -> None:
        ds = ClinicalPKDataset(
            records=[
                {"parameter": "cmax", "value": 10.0, "unit": "mg/L"},
                {"parameter": "cmax", "value": None, "unit": "mg/L"},
                {"parameter": "cmax", "value": 99999.0, "unit": "mg/L"},
            ]
        )
        result = validate_dataset(ds)
        assert len(result.records) == 3
        flags = [r["qc_flag"] for r in result.records]
        assert flags == ["pass", "missing_value", "out_of_range"]


# ===========================================================================
# PKDBLoader timecourse extraction tests (Phase D.2)
# ===========================================================================


class TestPKDBTimecourses:
    @patch("omega_pbpk.ml.data.loaders.PKDBLoader._get")
    def test_get_timecourses_for_substance(
        self, mock_get: MagicMock, pkdb_loader: PKDBLoader
    ) -> None:
        mock_get.return_value = {
            "count": 3,
            "next": None,
            "results": [
                {
                    "substance": "caffeine",
                    "study": "S001",
                    "group": "G1",
                    "route": "oral",
                    "dose": 100.0,
                    "dose_unit": "mg",
                    "time": 0.0,
                    "value": 0.0,
                    "unit": "mg/L",
                    "time_unit": "h",
                },
                {
                    "substance": "caffeine",
                    "study": "S001",
                    "group": "G1",
                    "route": "oral",
                    "dose": 100.0,
                    "dose_unit": "mg",
                    "time": 1.0,
                    "value": 5.2,
                    "unit": "mg/L",
                    "time_unit": "h",
                },
                {
                    "substance": "caffeine",
                    "study": "S001",
                    "group": "G1",
                    "route": "oral",
                    "dose": 100.0,
                    "dose_unit": "mg",
                    "time": 4.0,
                    "value": 2.1,
                    "unit": "mg/L",
                    "time_unit": "h",
                },
            ],
        }
        curves = pkdb_loader.get_timecourses_for_substance("caffeine")
        assert len(curves) == 1  # one study+group combo
        assert curves[0]["substance"] == "caffeine"
        assert len(curves[0]["timepoints"]) == 3
        # Should be sorted by time
        times = [tp["time_h"] for tp in curves[0]["timepoints"]]
        assert times == [0.0, 1.0, 4.0]

    @patch("omega_pbpk.ml.data.loaders.PKDBLoader._get")
    def test_timecourses_time_conversion(
        self, mock_get: MagicMock, pkdb_loader: PKDBLoader
    ) -> None:
        """Minutes should be converted to hours."""
        mock_get.return_value = {
            "count": 1,
            "next": None,
            "results": [
                {
                    "substance": "test",
                    "study": "S1",
                    "group": "G1",
                    "time": 30,
                    "value": 1.0,
                    "unit": "mg/L",
                    "time_unit": "min",
                },
            ],
        }
        curves = pkdb_loader.get_timecourses_for_substance("test")
        assert len(curves) == 1
        assert abs(curves[0]["timepoints"][0]["time_h"] - 0.5) < 1e-9

    @patch("omega_pbpk.ml.data.loaders.PKDBLoader._get")
    def test_timecourses_multiple_groups(
        self, mock_get: MagicMock, pkdb_loader: PKDBLoader
    ) -> None:
        """Different groups should produce separate curves."""
        mock_get.return_value = {
            "count": 2,
            "next": None,
            "results": [
                {
                    "substance": "drug",
                    "study": "S1",
                    "group": "G1",
                    "time": 1.0,
                    "value": 5.0,
                    "unit": "mg/L",
                    "time_unit": "h",
                },
                {
                    "substance": "drug",
                    "study": "S1",
                    "group": "G2",
                    "time": 1.0,
                    "value": 3.0,
                    "unit": "mg/L",
                    "time_unit": "h",
                },
            ],
        }
        curves = pkdb_loader.get_timecourses_for_substance("drug")
        assert len(curves) == 2


# ===========================================================================
# PK-DB pagination format tests
# ===========================================================================


class TestPKDBPagination:
    """Test _get_all_pages with the PK-DB data.data nested format."""

    @patch("omega_pbpk.ml.data.loaders.PKDBLoader._get")
    def test_nested_data_format(self, mock_get: MagicMock, pkdb_loader: PKDBLoader) -> None:
        """PK-DB returns {data: {count, data: [...]}, current_page, last_page, ...}."""
        mock_get.return_value = {
            "current_page": 1,
            "last_page": 1,
            "next_page_url": None,
            "prev_page_url": None,
            "data": {
                "count": 2,
                "data": [
                    {"substance": "caffeine", "measurement_type": "clearance", "mean": 5.0},
                    {"substance": "caffeine", "measurement_type": "concentration", "mean": 3.0},
                ],
            },
        }
        results = pkdb_loader._get_all_pages("pkdata/groups/")
        assert len(results) == 2
        assert results[0]["substance"] == "caffeine"

    @patch("omega_pbpk.ml.data.loaders.PKDBLoader._get")
    def test_nested_data_multi_page(self, mock_get: MagicMock, pkdb_loader: PKDBLoader) -> None:
        """Multi-page with next_page_url."""
        mock_get.side_effect = [
            {
                "current_page": 1,
                "last_page": 2,
                "next_page_url": "https://pk-db.com/api/v1/pkdata/groups/?page=2",
                "data": {"count": 3, "data": [{"id": 1}, {"id": 2}]},
            },
            {
                "current_page": 2,
                "last_page": 2,
                "next_page_url": None,
                "data": {"count": 3, "data": [{"id": 3}]},
            },
        ]
        results = pkdb_loader._get_all_pages("pkdata/groups/")
        assert len(results) == 3

    @patch("omega_pbpk.ml.data.loaders.PKDBLoader._get")
    def test_nested_data_empty(self, mock_get: MagicMock, pkdb_loader: PKDBLoader) -> None:
        mock_get.return_value = {
            "current_page": 1,
            "last_page": 1,
            "next_page_url": None,
            "data": {"count": 0, "data": []},
        }
        results = pkdb_loader._get_all_pages("pkdata/groups/")
        assert results == []

    @patch("omega_pbpk.ml.data.loaders.PKDBLoader._get")
    def test_legacy_format_still_works(self, mock_get: MagicMock, pkdb_loader: PKDBLoader) -> None:
        """Legacy {results, next} format should still work."""
        mock_get.return_value = {
            "count": 1,
            "next": None,
            "results": [{"sid": "S001"}],
        }
        results = pkdb_loader._get_all_pages("studies/")
        assert len(results) == 1


# ===========================================================================
# PKDBLoader bulk endpoint tests
# ===========================================================================


class TestPKDBBulkEndpoints:
    @patch("omega_pbpk.ml.data.loaders.PKDBLoader._get")
    def test_get_pkdata_groups(self, mock_get: MagicMock, pkdb_loader: PKDBLoader) -> None:
        mock_get.return_value = {
            "current_page": 1,
            "last_page": 1,
            "next_page_url": None,
            "data": {
                "count": 1,
                "data": [{"substance": "caffeine", "measurement_type": "clearance", "mean": 5.0}],
            },
        }
        results = pkdb_loader.get_pkdata_groups(substance="caffeine")
        assert len(results) == 1
        mock_get.assert_called_once()

    @patch("omega_pbpk.ml.data.loaders.PKDBLoader._get")
    def test_get_pkdata_individuals(self, mock_get: MagicMock, pkdb_loader: PKDBLoader) -> None:
        mock_get.return_value = {
            "current_page": 1,
            "last_page": 1,
            "next_page_url": None,
            "data": {"count": 0, "data": []},
        }
        results = pkdb_loader.get_pkdata_individuals(substance="caffeine")
        assert results == []

    @patch("omega_pbpk.ml.data.loaders.PKDBLoader._get")
    def test_get_pkdata_groups_no_filter(
        self, mock_get: MagicMock, pkdb_loader: PKDBLoader
    ) -> None:
        """No substance filter should still work."""
        mock_get.return_value = {
            "current_page": 1,
            "last_page": 1,
            "next_page_url": None,
            "data": {"count": 2, "data": [{"id": 1}, {"id": 2}]},
        }
        results = pkdb_loader.get_pkdata_groups(substance=None)
        assert len(results) == 2


# ===========================================================================
# OpenFDAExtractor tests
# ===========================================================================


@pytest.fixture()
def openfda_extractor(tmp_cache: Path) -> OpenFDAExtractor:
    return OpenFDAExtractor(cache_dir=tmp_cache / "openfda", rate_limit=0.0)


class TestOpenFDAExtractor:
    def test_extract_pk_params_cmax(self) -> None:
        text = "The Cmax was 250 ng/mL following oral administration."
        result = OpenFDAExtractor.extract_pk_params(text)
        assert "cmax" in result
        assert result["cmax"][0]["value"] == 250.0

    def test_extract_pk_params_half_life_natural(self) -> None:
        """Should match natural language like 'half-life of 6 hours'."""
        text = "The elimination half-life of metformin is approximately 6.2 hours."
        result = OpenFDAExtractor.extract_pk_params(text)
        assert "t_half" in result
        assert result["t_half"][0]["value"] == 6.2

    def test_extract_pk_params_bioavailability(self) -> None:
        text = "The absolute bioavailability is about 90%."
        result = OpenFDAExtractor.extract_pk_params(text)
        assert "bioavailability" in result
        assert result["bioavailability"][0]["value"] == 90.0

    def test_extract_pk_params_protein_binding(self) -> None:
        text = "Clonazepam is approximately 85% bound to plasma proteins."
        result = OpenFDAExtractor.extract_pk_params(text)
        assert "protein_binding" in result
        assert result["protein_binding"][0]["value"] == 85.0

    def test_extract_pk_params_protein_binding_alt(self) -> None:
        text = "The plasma protein binding of nifedipine is high (92%)."
        result = OpenFDAExtractor.extract_pk_params(text)
        assert "protein_binding" in result
        assert result["protein_binding"][0]["value"] == 92.0

    def test_extract_pk_params_tmax(self) -> None:
        text = "Time to peak concentration is 1.5 hours after dosing."
        result = OpenFDAExtractor.extract_pk_params(text)
        assert "tmax" in result
        assert result["tmax"][0]["value"] == 1.5

    def test_extract_pk_params_clearance(self) -> None:
        text = "The apparent clearance is 25.5 L/hr."
        result = OpenFDAExtractor.extract_pk_params(text)
        assert "clearance" in result
        assert result["clearance"][0]["value"] == 25.5

    def test_extract_pk_params_empty(self) -> None:
        result = OpenFDAExtractor.extract_pk_params("No pharmacokinetic data.")
        assert result == {}

    @patch("omega_pbpk.ml.data.loaders.OpenFDAExtractor._get")
    def test_get_pk_for_drug(
        self, mock_get: MagicMock, openfda_extractor: OpenFDAExtractor
    ) -> None:
        mock_get.return_value = {
            "results": [
                {
                    "pharmacokinetics": ["The half-life is 6 hours. Clearance is 10 L/h."],
                    "clinical_pharmacology": [],
                }
            ]
        }
        result = openfda_extractor.get_pk_for_drug("caffeine")
        assert result["found"] is True
        assert "t_half" in result["params"]
        assert "clearance" in result["params"]

    @patch("omega_pbpk.ml.data.loaders.OpenFDAExtractor._get")
    def test_get_pk_for_drug_not_found(
        self, mock_get: MagicMock, openfda_extractor: OpenFDAExtractor
    ) -> None:
        mock_get.return_value = {"results": []}
        result = openfda_extractor.get_pk_for_drug("nonexistent")
        assert result["found"] is False


# ===========================================================================
# ClinicalPKDataset new classmethod tests
# ===========================================================================


class TestClinicalPKDatasetFromMethods:
    @patch("omega_pbpk.ml.data.loaders.PKDBLoader._get")
    def test_from_pkdb_with_drug_list(self, mock_get: MagicMock, pkdb_loader: PKDBLoader) -> None:
        """from_pkdb should parse group/individual records."""
        mock_get.return_value = {
            "current_page": 1,
            "last_page": 1,
            "next_page_url": None,
            "data": {
                "count": 2,
                "data": [
                    {
                        "substance": "caffeine",
                        "measurement_type": "clearance",
                        "mean": 5.2,
                        "unit": "L/h",
                    },
                    {
                        "substance": "caffeine",
                        "measurement_type": "concentration",
                        "mean": 3.0,
                        "unit": "mg/L",
                    },
                ],
            },
        }
        ds = ClinicalPKDataset.from_pkdb(pkdb_loader, drug_list=["caffeine"])
        assert len(ds.records) >= 2
        params = {r["parameter"] for r in ds.records}
        assert "clearance" in params
        assert "concentration" in params

    @patch("omega_pbpk.ml.data.loaders.PKDBLoader._get")
    def test_from_pkdb_skips_demographics(
        self, mock_get: MagicMock, pkdb_loader: PKDBLoader
    ) -> None:
        """Demographics (age, sex, etc.) should be skipped."""
        mock_get.return_value = {
            "current_page": 1,
            "last_page": 1,
            "next_page_url": None,
            "data": {
                "count": 3,
                "data": [
                    {"substance": "caffeine", "measurement_type": "sex", "value": 1.0, "unit": ""},
                    {
                        "substance": "caffeine",
                        "measurement_type": "age",
                        "mean": 35,
                        "unit": "year",
                    },
                    {
                        "substance": "caffeine",
                        "measurement_type": "clearance",
                        "mean": 5.0,
                        "unit": "L/h",
                    },
                ],
            },
        }
        ds = ClinicalPKDataset.from_pkdb(pkdb_loader, drug_list=["caffeine"])
        # Only clearance should be kept (sex and age skipped)
        pk_params = [r["parameter"] for r in ds.records]
        assert "sex" not in pk_params
        assert "age" not in pk_params
        assert any(p == "clearance" for p in pk_params)

    def test_from_openfda(self) -> None:
        """from_openfda with empty drug list returns empty dataset."""
        extractor = MagicMock(spec=OpenFDAExtractor)
        ds = ClinicalPKDataset.from_openfda(extractor, drug_list=[])
        assert len(ds.records) == 0

    def test_from_openfda_with_data(self) -> None:
        extractor = MagicMock(spec=OpenFDAExtractor)
        extractor.get_pk_for_drug.return_value = {
            "drug": "caffeine",
            "found": True,
            "pk_text": "half-life 5h",
            "params": {"t_half": [{"value": 5.0, "unit": "h"}]},
        }
        ds = ClinicalPKDataset.from_openfda(extractor, drug_list=["caffeine"])
        assert len(ds.records) == 1
        assert ds.records[0]["parameter"] == "t_half"
        assert ds.records[0]["source"] == "openfda"
        assert ds.records[0]["smiles"] is not None  # caffeine is in BUILTIN_SMILES

    @patch("omega_pbpk.ml.data.loaders.PKDBLoader._get")
    def test_from_pkdb_timecourses(self, mock_get: MagicMock, pkdb_loader: PKDBLoader) -> None:
        """from_pkdb_timecourses should derive Cmax, Tmax, AUC from curves."""
        mock_get.return_value = {
            "count": 3,
            "next": None,
            "results": [
                {
                    "substance": "caffeine",
                    "study": "S1",
                    "group": "G1",
                    "route": "oral",
                    "dose": 100.0,
                    "dose_unit": "mg",
                    "time": 0.0,
                    "value": 0.0,
                    "unit": "mg/L",
                    "time_unit": "h",
                },
                {
                    "substance": "caffeine",
                    "study": "S1",
                    "group": "G1",
                    "time": 1.0,
                    "value": 10.0,
                    "unit": "mg/L",
                    "time_unit": "h",
                },
                {
                    "substance": "caffeine",
                    "study": "S1",
                    "group": "G1",
                    "time": 4.0,
                    "value": 2.0,
                    "unit": "mg/L",
                    "time_unit": "h",
                },
            ],
        }
        ds = ClinicalPKDataset.from_pkdb_timecourses(pkdb_loader, drug_list=["caffeine"])
        assert len(ds.records) == 3  # cmax, tmax, auc
        params = {r["parameter"] for r in ds.records}
        assert params == {"cmax", "tmax", "auc"}

        cmax_rec = next(r for r in ds.records if r["parameter"] == "cmax")
        assert cmax_rec["value"] == 10.0
        assert cmax_rec["source"] == "pkdb_timecourse"

        tmax_rec = next(r for r in ds.records if r["parameter"] == "tmax")
        assert tmax_rec["value"] == 1.0
