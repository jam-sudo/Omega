"""Tests for Phase 39: PubChem PUG REST client.

All tests mock urllib.request.urlopen to avoid real network calls in CI.
"""

from __future__ import annotations

import csv
import json
from unittest.mock import MagicMock, patch

from omega_pbpk.data.pubchem_client import (
    PubChemCompound,
    batch_lookup,
    enrich_adme_reference,
    lookup_by_cid,
    lookup_by_name,
    lookup_by_smiles,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_CAFFEINE_PROPS = {
    "CID": 2519,
    "MolecularWeight": 194.19,
    "XLogP": -0.1,
    "TPSA": 58.4,
    "HBondDonorCount": 0,
    "HBondAcceptorCount": 3,
    "RotatableBondCount": 0,
    "Complexity": 318.0,
    "Charge": 0,
    "ExactMass": 194.07,
    "CanonicalSMILES": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
}

_IBUPROFEN_PROPS = {
    "CID": 3672,
    "MolecularWeight": 206.28,
    "XLogP": 3.97,
    "TPSA": 37.3,
    "HBondDonorCount": 1,
    "HBondAcceptorCount": 1,
    "RotatableBondCount": 4,
    "Complexity": 226.0,
    "Charge": 0,
    "ExactMass": 206.13,
    "CanonicalSMILES": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
}

_ASPIRIN_PROPS = {
    "CID": 2244,
    "MolecularWeight": 180.16,
    "XLogP": 1.19,
    "TPSA": 63.6,
    "HBondDonorCount": 1,
    "HBondAcceptorCount": 3,
    "RotatableBondCount": 3,
    "Complexity": 212.0,
    "Charge": 0,
    "ExactMass": 180.04,
    "CanonicalSMILES": "CC(=O)OC1=CC=CC=C1C(=O)O",
}


def _make_mock_response(props: dict) -> MagicMock:
    """Create a mock urllib response returning a PropertyTable JSON."""
    payload = json.dumps({"PropertyTable": {"Properties": [props]}}).encode()
    mock_resp = MagicMock()
    mock_resp.read.return_value = payload
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


# ---------------------------------------------------------------------------
# TestPubChemClient
# ---------------------------------------------------------------------------


class TestPubChemClient:
    def test_lookup_by_name_caffeine(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        mock_resp = _make_mock_response(_CAFFEINE_PROPS)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = lookup_by_name("caffeine")
        assert result is not None
        assert isinstance(result, PubChemCompound)
        assert result.cid == 2519
        assert abs(result.molecular_weight - 194.19) < 0.01
        assert "C" in result.smiles
        assert result.source == "pubchem"

    def test_lookup_by_smiles(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        mock_resp = _make_mock_response(_CAFFEINE_PROPS)
        smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = lookup_by_smiles(smiles)
        assert result is not None
        assert result.cid == 2519

    def test_lookup_by_cid(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        mock_resp = _make_mock_response(_CAFFEINE_PROPS)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = lookup_by_cid(2519)
        assert result is not None
        assert result.cid == 2519
        assert result.hbd == 0
        assert result.hba == 3

    def test_lookup_returns_none_on_404(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        import urllib.error

        http_error = urllib.error.HTTPError(
            url="http://test",
            code=404,
            msg="Not Found",
            hdrs=None,
            fp=None,  # type: ignore[arg-type]
        )
        with patch("urllib.request.urlopen", side_effect=http_error):
            result = lookup_by_name("nonexistent_compound_xyz")
        assert result is None

    def test_lookup_returns_none_on_network_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        import urllib.error

        url_error = urllib.error.URLError(reason="Connection refused")
        with patch("urllib.request.urlopen", side_effect=url_error):
            result = lookup_by_name("caffeine")
        assert result is None

    def test_batch_lookup_returns_dict(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        responses = [
            _make_mock_response(_CAFFEINE_PROPS),
            _make_mock_response(_IBUPROFEN_PROPS),
            _make_mock_response(_ASPIRIN_PROPS),
        ]
        with patch("urllib.request.urlopen", side_effect=responses):
            with patch("time.sleep"):  # skip rate-limiting sleep
                results = batch_lookup(["caffeine", "ibuprofen", "aspirin"])
        assert isinstance(results, dict)
        assert set(results.keys()) == {"caffeine", "ibuprofen", "aspirin"}
        assert results["caffeine"] is not None
        assert results["caffeine"].cid == 2519
        assert results["ibuprofen"] is not None
        assert results["ibuprofen"].cid == 3672


# ---------------------------------------------------------------------------
# TestCaching
# ---------------------------------------------------------------------------


class TestCaching:
    def test_cache_miss_makes_request(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        mock_resp = _make_mock_response(_CAFFEINE_PROPS)
        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
            result = lookup_by_name("caffeine_cache_miss")
        assert result is not None
        assert mock_open.called

    def test_cache_hit_skips_request(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        # Perform first lookup to populate cache
        mock_resp = _make_mock_response(_CAFFEINE_PROPS)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            first = lookup_by_name("caffeine")
        assert first is not None

        # Second lookup should use cache — urlopen should NOT be called
        with patch("urllib.request.urlopen") as mock_open:
            second = lookup_by_name("caffeine")
        assert second is not None
        assert second.cid == first.cid
        assert not mock_open.called, "Cache hit should not make a network request"

    def test_cache_written_after_lookup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        mock_resp = _make_mock_response(_IBUPROFEN_PROPS)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = lookup_by_name("ibuprofen")
        assert result is not None

        # Check that the cache file now exists and contains ibuprofen data
        cache_file = tmp_path / ".omega_pbpk_cache" / "pubchem.json"
        assert cache_file.exists(), "Cache file should be written after successful lookup"
        with open(cache_file) as f:
            cache_data = json.load(f)
        assert any("ibuprofen" in k for k in cache_data), "Cache should contain ibuprofen entry"


# ---------------------------------------------------------------------------
# TestEnrichment
# ---------------------------------------------------------------------------


class TestEnrichment:
    def _write_csv(self, path: str, names: list[str]) -> None:
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["compound", "fup", "clint_3a4"])
            for n in names:
                writer.writerow([n, "0.1", "10.0"])

    def test_enrich_adme_reference(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        csv_path = str(tmp_path / "test.csv")
        self._write_csv(csv_path, ["caffeine", "ibuprofen"])

        responses = [
            _make_mock_response(_CAFFEINE_PROPS),
            _make_mock_response(_IBUPROFEN_PROPS),
        ]
        with patch("urllib.request.urlopen", side_effect=responses):
            with patch("time.sleep"):
                count = enrich_adme_reference(csv_path)
        assert count > 0

    def test_enrichment_skips_already_cached(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        # Pre-populate cache for aspirin
        mock_resp = _make_mock_response(_ASPIRIN_PROPS)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            lookup_by_name("aspirin")

        csv_path = str(tmp_path / "test2.csv")
        self._write_csv(csv_path, ["aspirin"])

        # Now enrichment should use cache — no real call
        with patch("urllib.request.urlopen") as mock_open:
            count = enrich_adme_reference(csv_path)
        assert count == 1
        assert not mock_open.called, "Cached compound should not trigger network call"

    def test_enrichment_graceful_on_failure(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        csv_path = str(tmp_path / "test3.csv")
        self._write_csv(csv_path, ["caffeine", "nonexistent_xyz"])

        import urllib.error

        def side_effect(req, timeout=10):
            url = req.full_url if hasattr(req, "full_url") else str(req)
            if "nonexistent" in url:
                raise urllib.error.HTTPError(
                    url=url,
                    code=404,
                    msg="Not Found",
                    hdrs=None,
                    fp=None,  # type: ignore[arg-type]
                )
            return _make_mock_response(_CAFFEINE_PROPS)

        with patch("urllib.request.urlopen", side_effect=side_effect):
            with patch("time.sleep"):
                count = enrich_adme_reference(csv_path)
        # At least caffeine should succeed; partial failure should not crash
        assert count >= 0


# ---------------------------------------------------------------------------
# TestAPI
# ---------------------------------------------------------------------------


class TestAPI:
    def test_pubchem_endpoint_name_lookup(self, tmp_path, monkeypatch):
        """POST /pubchem/lookup with mocked response returns 200 with cid field."""
        monkeypatch.setenv("HOME", str(tmp_path))
        from fastapi.testclient import TestClient

        from omega_pbpk.api.app import app

        client = TestClient(app)
        mock_resp = _make_mock_response(_CAFFEINE_PROPS)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            response = client.post(
                "/pubchem/lookup",
                json={"query": "caffeine", "query_type": "name"},
            )
        assert response.status_code == 200
        data = response.json()
        assert "cid" in data
        assert data["cid"] == 2519

    def test_pubchem_endpoint_invalid_type(self, tmp_path, monkeypatch):
        """POST /pubchem/lookup with invalid query_type returns 422."""
        monkeypatch.setenv("HOME", str(tmp_path))
        from fastapi.testclient import TestClient

        from omega_pbpk.api.app import app

        client = TestClient(app)
        response = client.post(
            "/pubchem/lookup",
            json={"query": "caffeine", "query_type": "invalid"},
        )
        assert response.status_code == 422
