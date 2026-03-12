"""Data loaders for clinical PK data sources.

This module provides clients for:
- PK-DB (pk-db.com) — curated pharmacokinetic studies
- DailyMed/FDA — drug label PK sections
- PyTDC — Therapeutics Data Commons ADME benchmarks
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_CACHE_ROOT = Path("data/ml")
_PKDB_BASE_URL = "https://pk-db.com/api/v1/"
_DAILYMED_BASE_URL = "https://dailymed.nlm.nih.gov/dailymed/services/v2/"
_OPENFDA_BASE_URL = "https://api.fda.gov/drug/label.json"
_RATE_LIMIT_DELAY = 1.0  # seconds between requests


def _make_session(retries: int = 3, backoff_factor: float = 1.0) -> requests.Session:
    """Create a requests session with retry and backoff."""
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _cache_key(url: str, params: dict[str, Any] | None = None) -> str:
    """Derive a filesystem-safe cache key from a URL and query params."""
    raw = url + json.dumps(params or {}, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


# ===========================================================================
# Task D.1 — PK-DB Connector
# ===========================================================================


class PKDBLoader:
    """REST API client for PK-DB (pk-db.com/api/v1/).

    All responses are cached as JSON to ``cache_dir`` so that repeated
    queries do not hit the remote server.

    Parameters
    ----------
    cache_dir:
        Directory for cached JSON responses.  Created automatically.
    base_url:
        PK-DB API root.  Override for testing.
    rate_limit:
        Minimum seconds between consecutive HTTP requests.
    """

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        base_url: str = _PKDB_BASE_URL,
        rate_limit: float = _RATE_LIMIT_DELAY,
    ) -> None:
        self.cache_dir = Path(cache_dir or _DEFAULT_CACHE_ROOT / "pkdb_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = base_url.rstrip("/") + "/"
        self.rate_limit = rate_limit
        self._session = _make_session()
        self._last_request_time: float = 0.0

    # -- internal helpers ---------------------------------------------------

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)

    def _get(self, endpoint: str, params: dict[str, Any] | None = None) -> Any:
        """GET with caching, rate-limiting, and error handling."""
        url = self.base_url + endpoint.lstrip("/")
        key = _cache_key(url, params)
        cache_path = self.cache_dir / f"{key}.json"

        if cache_path.exists():
            logger.debug("Cache hit: %s", cache_path.name)
            return json.loads(cache_path.read_text())

        self._throttle()
        logger.info("Fetching %s  params=%s", url, params)
        try:
            resp = self._session.get(url, params=params, timeout=30)
            resp.raise_for_status()
        except requests.RequestException:
            logger.exception("Request failed: %s", url)
            raise
        finally:
            self._last_request_time = time.monotonic()

        data = resp.json()
        cache_path.write_text(json.dumps(data, indent=2))
        return data

    def _get_all_pages(
        self, endpoint: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Follow ``next`` links to collect all paginated results.

        Handles two PK-DB response formats:
        - Legacy: ``{"count": N, "next": url|null, "results": [...]}``
        - Current: ``{"current_page": N, "last_page": M, "next_page_url": url|null, "data": {"count": N, "data": [...]}}``
        """
        params = dict(params or {})
        results: list[dict[str, Any]] = []
        page = 1

        while True:
            params["page"] = page
            data = self._get(endpoint, params)

            if not isinstance(data, dict):
                if isinstance(data, list):
                    results.extend(data)
                break

            # Format 1: {"results": [...], "next": ...}
            if "results" in data:
                results.extend(data["results"])
                if not data.get("next"):
                    break
                page += 1
            # Format 2: {"data": {"count": N, "data": [...]}, "next_page_url": ...}
            elif "data" in data:
                inner = data["data"]
                if isinstance(inner, dict) and "data" in inner:
                    records = inner["data"]
                    if isinstance(records, list):
                        results.extend(records)
                elif isinstance(inner, list):
                    results.extend(inner)

                # Check pagination
                next_url = data.get("next_page_url")
                last_page = data.get("last_page", 1)
                current_page = data.get("current_page", page)
                if not next_url or current_page >= last_page:
                    break
                page += 1
            else:
                results.append(data)
                break

        return results

    # -- public API ---------------------------------------------------------

    def list_studies(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Return a list of all PK study summaries.

        Parameters
        ----------
        limit:
            Maximum number of studies to return.  ``None`` means all.
        """
        studies = self._get_all_pages("studies/")
        if limit is not None:
            studies = studies[:limit]
        return studies

    def get_pk_data(self, drug_name: str) -> list[dict[str, Any]]:
        """Get PK output data filtered by substance/drug name.

        Parameters
        ----------
        drug_name:
            Case-insensitive compound name, e.g. ``"caffeine"``.
        """
        return self._get_all_pages("pkdata/", params={"substance": drug_name})

    def get_concentration_time(self, study_id: str) -> list[dict[str, Any]]:
        """Return concentration–time data for a given study.

        Parameters
        ----------
        study_id:
            PK-DB study identifier (e.g. ``"PKDB00001"``).
        """
        return self._get_all_pages("timecourses/", params={"study": study_id})

    def get_timecourses_for_substance(self, drug_name: str) -> list[dict[str, Any]]:
        """Get all concentration–time curves for a substance across studies.

        Returns a list of dicts, each containing::

            {
                "substance": str,
                "study": str,         # study identifier
                "group": str,         # group/individual identifier
                "route": str,         # e.g. "oral", "iv"
                "dose": float | None, # dose value
                "dose_unit": str,     # e.g. "mg"
                "timepoints": [       # list of (time, conc, unit) tuples
                    {"time_h": float, "conc": float, "conc_unit": str},
                    ...
                ],
            }

        Parameters
        ----------
        drug_name:
            Case-insensitive compound name, e.g. ``"caffeine"``.
        """
        raw = self._get_all_pages("timecourses/", params={"substance": drug_name})

        # Group raw timecourse records by study + group
        from collections import defaultdict

        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        meta: dict[tuple[str, str], dict[str, Any]] = {}

        for rec in raw:
            study = rec.get("study", rec.get("study_name", "unknown"))
            group = rec.get("group", rec.get("group_name", "unknown"))
            key = (str(study), str(group))
            grouped[key].append(rec)
            if key not in meta:
                meta[key] = {
                    "substance": rec.get("substance", drug_name),
                    "study": str(study),
                    "group": str(group),
                    "route": rec.get("route", ""),
                    "dose": rec.get("dose"),
                    "dose_unit": rec.get("dose_unit", "mg"),
                }

        results: list[dict[str, Any]] = []
        for key, records in grouped.items():
            timepoints = []
            for r in records:
                t = r.get("time") if r.get("time") is not None else r.get("time_value")
                c = r.get("value") if r.get("value") is not None else r.get("concentration")
                unit = r.get("unit", r.get("concentration_unit", "mg/L"))
                time_unit = r.get("time_unit", "h")
                if t is not None and c is not None:
                    try:
                        t_h = float(t)
                        if time_unit == "min":
                            t_h /= 60.0
                        elif time_unit == "day":
                            t_h *= 24.0
                        timepoints.append({"time_h": t_h, "conc": float(c), "conc_unit": unit})
                    except (TypeError, ValueError):
                        continue

            if timepoints:
                timepoints.sort(key=lambda x: x["time_h"])
                entry = dict(meta[key])
                entry["timepoints"] = timepoints
                results.append(entry)

        logger.info(
            "Found %d timecourse curves for %s across %d studies",
            len(results),
            drug_name,
            len({r["study"] for r in results}),
        )
        return results

    def get_pkdata_groups(self, substance: str | None = None) -> list[dict[str, Any]]:
        """Fetch group-level PK summary data from ``/pkdata/groups/``.

        This endpoint is publicly accessible (no auth required) and contains
        ~20K records with aggregated PK parameters (Cmax, AUC, t½, CL, Vd, etc.).

        Parameters
        ----------
        substance:
            Optional substance name filter (case-insensitive).
        """
        params: dict[str, Any] = {"page_size": 200}
        if substance:
            params["substance"] = substance
        return self._get_all_pages("pkdata/groups/", params)

    def get_pkdata_individuals(self, substance: str | None = None) -> list[dict[str, Any]]:
        """Fetch individual-level PK data from ``/pkdata/individuals/``.

        This endpoint is publicly accessible (no auth required) and contains
        ~167K records with individual subject PK parameters.

        Parameters
        ----------
        substance:
            Optional substance name filter (case-insensitive).
        """
        params: dict[str, Any] = {"page_size": 200}
        if substance:
            params["substance"] = substance
        return self._get_all_pages("pkdata/individuals/", params)

    def get_statistics(self) -> dict[str, Any]:
        """Return PK-DB database statistics (counts, version)."""
        return self._get("statistics/")


# ===========================================================================
# Task D.2 — FDA / DailyMed Label Extractor
# ===========================================================================

# Regex patterns for common PK parameters found in drug labels.
_PK_PATTERNS: dict[str, re.Pattern[str]] = {
    "cmax": re.compile(
        r"[Cc]\s*max\s*(?:(?:was|is|of|=|:)\s*)?(\d+\.?\d*)\s*(mg/[Ll]|[µu]g/m[Ll]|ng/m[Ll]|mg/m[Ll])",
    ),
    "auc": re.compile(
        r"AUC\s*(?:0-(?:inf|∞|24|t))?\s*(?:(?:was|is|of|=|:)\s*)?(\d+\.?\d*)\s*"
        r"((?:mg|[µu]g|ng)\s*[·*·]\s*h(?:r)?/(?:m[Ll]|[Ll])|h\s*[·*·]\s*(?:mg|[µu]g|ng)/(?:m[Ll]|[Ll]))",
    ),
    "t_half": re.compile(
        r"(?:t\s*½|t\s*1/2|half[- ]?life|elimination half[- ]?life)\s*"
        r"(?:(?:was|is|of|=|:)\s*)?(\d+\.?\d*)\s*(h(?:ours?)?|min(?:utes?)?|days?)",
    ),
    "bioavailability": re.compile(
        r"(?:[Bb]ioavailability|F)\s*(?:(?:was|is|of|=|:)\s*)?(\d+\.?\d*)\s*(%)",
    ),
    "vd": re.compile(
        r"(?:[Vv]olume of distribution|V\s*d|Vd?ss)\s*"
        r"(?:(?:was|is|of|=|:)\s*)?(\d+\.?\d*)\s*([Ll](?:/kg)?)",
    ),
    "clearance": re.compile(
        r"(?:[Cc]learance|CL|CL/F)\s*(?:(?:was|is|of|=|:)\s*)?(\d+\.?\d*)\s*"
        r"([Ll]/h(?:r)?(?:/kg)?|m[Ll]/min(?:/kg)?)",
    ),
}


class FDALabelExtractor:
    """Download and parse PK sections from DailyMed drug labels.

    Parameters
    ----------
    cache_dir:
        Directory for cached API responses.
    base_url:
        DailyMed services root.  Override for testing.
    rate_limit:
        Minimum seconds between HTTP requests.
    """

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        base_url: str = _DAILYMED_BASE_URL,
        rate_limit: float = _RATE_LIMIT_DELAY,
    ) -> None:
        self.cache_dir = Path(cache_dir or _DEFAULT_CACHE_ROOT / "fda_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = base_url.rstrip("/") + "/"
        self.rate_limit = rate_limit
        self._session = _make_session()
        self._last_request_time: float = 0.0

    # -- internal helpers ---------------------------------------------------

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)

    def _get(self, endpoint: str, params: dict[str, Any] | None = None) -> Any:
        url = self.base_url + endpoint.lstrip("/")
        key = _cache_key(url, params)
        cache_path = self.cache_dir / f"{key}.json"

        if cache_path.exists():
            logger.debug("Cache hit: %s", cache_path.name)
            return json.loads(cache_path.read_text())

        self._throttle()
        logger.info("Fetching %s  params=%s", url, params)
        try:
            resp = self._session.get(url, params=params, timeout=30)
            resp.raise_for_status()
        except requests.RequestException:
            logger.exception("Request failed: %s", url)
            raise
        finally:
            self._last_request_time = time.monotonic()

        data = resp.json()
        cache_path.write_text(json.dumps(data, indent=2))
        return data

    # -- public API ---------------------------------------------------------

    def search_drug(self, name: str) -> list[dict[str, Any]]:
        """Search DailyMed SPL resources by drug name.

        Returns a list of matching SPL entries (each has ``setid``, ``title``, etc.).
        """
        data = self._get("spls.json", params={"drug_name": name})
        results = data.get("data", []) if isinstance(data, dict) else data
        return results if isinstance(results, list) else []

    def get_pk_section(self, setid: str) -> str | None:
        """Retrieve the Clinical Pharmacology / Pharmacokinetics section text.

        Parameters
        ----------
        setid:
            DailyMed SPL set identifier.

        Returns
        -------
        str or None
            Raw section text, or ``None`` if no PK section is found.
        """
        data = self._get(f"spls/{setid}.json")
        if not isinstance(data, dict):
            return None

        # DailyMed nests sections; look for LOINC code 34090-1 (Clinical Pharmacology)
        # or 43682-4 (Pharmacokinetics).
        pk_codes = {"34090-1", "43682-4"}

        def _walk_sections(sections: list[dict[str, Any]]) -> str | None:
            for sec in sections:
                code = sec.get("loinc_code") or sec.get("code")
                if code in pk_codes:
                    return sec.get("text", "")
                # Recurse into children
                children = sec.get("children_sections", [])
                if children:
                    result = _walk_sections(children)
                    if result is not None:
                        return result
            return None

        sections = data.get("sections", [])
        return _walk_sections(sections)

    @staticmethod
    def extract_pk_params(text: str) -> dict[str, list[dict[str, Any]]]:
        """Extract PK parameters from free-text using regex.

        Parameters
        ----------
        text:
            Raw label text (e.g. from :meth:`get_pk_section`).

        Returns
        -------
        dict
            Mapping of parameter name → list of ``{"value": float, "unit": str}``
            dicts for every match found.
        """
        results: dict[str, list[dict[str, Any]]] = {}
        for param_name, pattern in _PK_PATTERNS.items():
            matches = pattern.findall(text)
            if matches:
                results[param_name] = [{"value": float(m[0]), "unit": m[1]} for m in matches]
        return results


# ===========================================================================
# Task D.2b — OpenFDA Label Extractor (api.fda.gov)
# ===========================================================================

# Regex patterns tuned for OpenFDA label text.
# These are intentionally looser than DailyMed patterns to capture natural-language
# descriptions like "bioavailability is about 90%" or "half-life of 6 hours".
# Flexible verb/connector pattern: matches phrases like "is approximately",
# "of metformin is about", "was", etc. between a PK keyword and a number.
_OPENFDA_VERB = (
    r"(?:[^.;]{0,60}?(?:was|is|of|=|:|approximately|about|around|roughly|nearly|estimated at)\s*)*"
)
_OPENFDA_PK_PATTERNS: dict[str, re.Pattern[str]] = {
    "cmax": re.compile(
        r"[Cc][\s_]?max\s*(?:\([^)]*\)\s*)?" + _OPENFDA_VERB + r"(\d+\.?\d*)\s*"
        r"(ng/mL|mcg/mL|µg/mL|mg/L|ug/mL|mg/mL)"
    ),
    "auc": re.compile(
        r"AUC\s*(?:0[-–](?:∞|inf(?:inity)?|t|tau|24h?))?\s*(?:\([^)]*\)\s*)?"
        + _OPENFDA_VERB
        + r"(\d+\.?\d*)\s*"
        r"(ng[·*·\s]h(?:r)?/mL|ng\*h/mL|mcg[·*·\s]h(?:r)?/mL|"
        r"h[·*·\s]ng/mL|ng/mL[·*·\s]h|"
        r"(?:mg|µg|ug|ng)\s*[·*·\s]*h(?:r)?/(?:mL|L)|h\s*[·*·\s]*(?:mg|µg|ug|ng)/(?:mL|L))"
    ),
    "t_half": re.compile(
        r"(?:t\s*½|t\s*1/2|half[- ]?life|elimination half[- ]?life)\s*"
        + _OPENFDA_VERB
        + r"(\d+\.?\d*)\s*(h(?:ours?)?|hr|min(?:utes?)?|days?)",
        re.IGNORECASE,
    ),
    "bioavailability": re.compile(
        r"(?:bioavailability|absolute bioavailability|oral bioavailability)\s*"
        + _OPENFDA_VERB
        + r"(\d+\.?\d*)\s*(%)",
        re.IGNORECASE,
    ),
    "vd": re.compile(
        r"(?:volume of distribution|V\s*d|Vd?ss)\s*"
        + _OPENFDA_VERB
        + r"(\d+\.?\d*)\s*([Ll](?:/kg)?)",
        re.IGNORECASE,
    ),
    "clearance": re.compile(
        r"(?:clearance|CL(?:/F)?|total clearance|oral clearance|apparent clearance)\s*"
        + _OPENFDA_VERB
        + r"(\d+\.?\d*)\s*"
        r"([Ll]/h(?:r)?(?:/kg)?|m[Ll]/min(?:/kg)?)",
        re.IGNORECASE,
    ),
    "protein_binding": re.compile(
        r"(?:"
        r"(?:plasma )?protein bind(?:ing)?[^.]{0,40}?(\d+\.?\d*)\s*(%)|"
        r"bound to (?:plasma |serum |human serum )?protein[s]?[^.]{0,40}?(\d+\.?\d*)\s*(%)|"
        r"(\d+\.?\d*)\s*(%)\s*(?:bound to (?:plasma |serum )?protein|protein[- ]?bound)"
        r")",
        re.IGNORECASE,
    ),
    "tmax": re.compile(
        r"(?:t\s*max|time to (?:peak|maximum) (?:plasma )?concentration[s]?)\s*"
        + _OPENFDA_VERB
        + r"(\d+\.?\d*)\s*(h(?:ours?)?|hr|min(?:utes?)?)",
        re.IGNORECASE,
    ),
}


class OpenFDAExtractor:
    """Extract PK parameters from OpenFDA drug label API (api.fda.gov).

    Complementary to :class:`FDALabelExtractor` (DailyMed). OpenFDA often
    returns different labels and additional formulations not found via DailyMed.

    Parameters
    ----------
    cache_dir:
        Directory for cached API responses.
    rate_limit:
        Minimum seconds between HTTP requests.
    """

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        rate_limit: float = _RATE_LIMIT_DELAY,
    ) -> None:
        self.cache_dir = Path(cache_dir or _DEFAULT_CACHE_ROOT / "openfda_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = _OPENFDA_BASE_URL
        self.rate_limit = rate_limit
        self._session = _make_session()
        self._last_request_time: float = 0.0

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)

    def _get(self, params: dict[str, Any]) -> Any:
        key = _cache_key(self.base_url, params)
        cache_path = self.cache_dir / f"{key}.json"

        if cache_path.exists():
            logger.debug("Cache hit: %s", cache_path.name)
            return json.loads(cache_path.read_text())

        self._throttle()
        logger.info("OpenFDA: fetching params=%s", params)
        try:
            resp = self._session.get(self.base_url, params=params, timeout=30)
            resp.raise_for_status()
        except requests.RequestException:
            logger.exception("OpenFDA request failed")
            raise
        finally:
            self._last_request_time = time.monotonic()

        data = resp.json()
        cache_path.write_text(json.dumps(data, indent=2))
        return data

    def get_pk_for_drug(self, drug_name: str) -> dict[str, Any]:
        """Search OpenFDA for a drug and extract PK parameters.

        Tries generic_name first, then brand_name. Extracts PK text from
        ``pharmacokinetics`` and ``clinical_pharmacology`` sections.

        Returns
        -------
        dict
            ``{"drug": str, "found": bool, "pk_text": str, "params": {...}}``
        """
        result: dict[str, Any] = {
            "drug": drug_name,
            "found": False,
            "pk_text": "",
            "params": {},
        }

        data = None
        for field in ("openfda.generic_name", "openfda.brand_name"):
            try:
                data = self._get({"search": f'{field}:"{drug_name}"', "limit": "3"})
                if data.get("results"):
                    break
            except requests.RequestException:
                continue

        if not data or not data.get("results"):
            return result

        # Find the best label with PK text
        pk_text = ""
        for label in data["results"]:
            pk_sections = label.get("pharmacokinetics", [])
            clin_pharm = label.get("clinical_pharmacology", [])
            combined_text = " ".join(pk_sections + clin_pharm)
            if combined_text:
                pk_text = combined_text
                break

        if not pk_text:
            return result

        result["found"] = True
        result["pk_text"] = pk_text[:10_000]
        result["params"] = self.extract_pk_params(pk_text)
        return result

    @staticmethod
    def extract_pk_params(text: str) -> dict[str, list[dict[str, Any]]]:
        """Extract PK parameters from OpenFDA label text using regex."""
        results: dict[str, list[dict[str, Any]]] = {}
        for param_name, pattern in _OPENFDA_PK_PATTERNS.items():
            matches = pattern.findall(text)
            if not matches:
                continue
            extracted = []
            for m in matches:
                # Some patterns (protein_binding) have multiple capture group
                # alternatives — find the first non-empty (value, unit) pair.
                if isinstance(m, tuple):
                    groups = list(m)
                    value_str = None
                    unit_str = None
                    for i in range(0, len(groups) - 1, 2):
                        if groups[i]:
                            value_str = groups[i]
                            unit_str = groups[i + 1]
                            break
                    if value_str:
                        extracted.append({"value": float(value_str), "unit": unit_str})
                else:
                    try:
                        extracted.append({"value": float(m), "unit": ""})
                    except (TypeError, ValueError):
                        continue
            if extracted:
                results[param_name] = extracted
        return results


# ===========================================================================
# Task D.3 — TDC (Therapeutics Data Commons) Loader
# ===========================================================================

_SUPPORTED_TDC_ENDPOINTS: dict[str, str] = {
    "Caco2_Wang": "Caco2_Wang",
    "Lipophilicity_AstraZeneca": "Lipophilicity_AstraZeneca",
    "Solubility_AqSolDB": "Solubility_AqSolDB",
    "PPBR_AZ": "PPBR_AZ",
    "Clearance_Hepatocyte_AZ": "Clearance_Hepatocyte_AZ",
    "Clearance_Microsome_AZ": "Clearance_Microsome_AZ",
    "Half_Life_Obach": "Half_Life_Obach",
    "VDss_Lombardo": "VDss_Lombardo",
    "Bioavailability_Ma": "Bioavailability_Ma",
    "hERG": "hERG",
}


class TDCLoader:
    """Wrapper around PyTDC for ADME benchmark datasets.

    Handles ``ImportError`` gracefully if PyTDC is not installed.

    Parameters
    ----------
    cache_dir:
        Directory where TDC caches its downloads.
    """

    def __init__(self, cache_dir: Path | str | None = None) -> None:
        self.cache_dir = Path(cache_dir or _DEFAULT_CACHE_ROOT / "tdc_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._tdc_available: bool | None = None

    @property
    def tdc_available(self) -> bool:
        if self._tdc_available is None:
            try:
                from tdc.single_pred import ADME  # noqa: F401

                self._tdc_available = True
            except ImportError:
                self._tdc_available = False
        return self._tdc_available

    @staticmethod
    def supported_endpoints() -> list[str]:
        """Return the list of supported TDC ADME endpoint names."""
        return list(_SUPPORTED_TDC_ENDPOINTS.keys())

    def load_adme(self, endpoint_name: str) -> pd.DataFrame:
        """Load a TDC ADME dataset and return a standardized DataFrame.

        Parameters
        ----------
        endpoint_name:
            One of :meth:`supported_endpoints`.

        Returns
        -------
        pandas.DataFrame
            Columns: ``smiles``, ``value``, ``split``
            where split is one of ``"train"``, ``"valid"``, ``"test"``
            (scaffold-based).

        Raises
        ------
        ValueError
            If *endpoint_name* is not supported.
        ImportError
            If PyTDC is not installed.
        """
        import pandas as pd

        if endpoint_name not in _SUPPORTED_TDC_ENDPOINTS:
            raise ValueError(
                f"Unsupported endpoint: {endpoint_name!r}. "
                f"Supported: {list(_SUPPORTED_TDC_ENDPOINTS)}"
            )

        if not self.tdc_available:
            raise ImportError(
                "PyTDC is required but not installed. Install it with: pip install PyTDC"
            )

        from tdc.single_pred import ADME

        tdc_name = _SUPPORTED_TDC_ENDPOINTS[endpoint_name]
        dataset = ADME(name=tdc_name, path=str(self.cache_dir))

        splits = dataset.get_split(method="scaffold")

        frames: list[pd.DataFrame] = []
        for split_name, df in splits.items():
            df = df.rename(columns={"Drug": "smiles", "Y": "value"})
            df = df[["smiles", "value"]].copy()
            df["split"] = split_name
            frames.append(df)

        return pd.concat(frames, ignore_index=True)
