"""PubChem PUG REST API client for live compound property lookup.

Uses only stdlib (urllib, json, time, os, pathlib).  All network errors
are caught and return None — the caller never sees an exception.

Cache: JSON file at ~/.omega_pbpk_cache/pubchem.json, keyed by
       "{query_type}:{query_value}".

Rate limit: 5 requests/second (0.2 s sleep between batch calls).
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "PubChemCompound",
    "lookup_by_name",
    "lookup_by_smiles",
    "lookup_by_cid",
    "batch_lookup",
    "enrich_adme_reference",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROPERTIES = (
    "MolecularWeight",
    "XLogP",
    "TPSA",
    "HBondDonorCount",
    "HBondAcceptorCount",
    "RotatableBondCount",
    "Complexity",
    "Charge",
    "ExactMass",
    "IsomericSMILES",
)
_PROP_STR = ",".join(_PROPERTIES)
_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"
_TIMEOUT = 10  # seconds
_RATE_SLEEP = 0.2  # seconds between batch calls

# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _cache_path() -> Path:
    cache_dir = Path(os.path.expanduser("~")) / ".omega_pbpk_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "pubchem.json"


def _load_cache() -> dict[str, Any]:
    p = _cache_path()
    if p.exists():
        try:
            with open(p) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_cache(cache: dict[str, Any]) -> None:
    try:
        with open(_cache_path(), "w") as f:
            json.dump(cache, f)
    except OSError as exc:
        logger.warning("PubChem cache write failed: %s", exc)


def _cache_get(key: str) -> dict[str, Any] | None:
    return _load_cache().get(key)


def _cache_set(key: str, value: dict[str, Any]) -> None:
    cache = _load_cache()
    cache[key] = value
    _save_cache(cache)


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PubChemCompound:
    """Compound properties retrieved from PubChem PUG REST."""

    cid: int
    name: str
    smiles: str
    molecular_weight: float
    xlogp: float
    tpsa: float
    hbd: int
    hba: int
    rotatable_bonds: int
    complexity: float
    charge: int
    exact_mass: float
    source: str = field(default="pubchem")


# ---------------------------------------------------------------------------
# Internal fetch
# ---------------------------------------------------------------------------


def _fetch_json(url: str) -> dict[str, Any] | None:
    """Fetch JSON from *url* with timeout; return None on any error."""
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "OmegaPBPK/1.0 (academic)"},
        )
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            raw = resp.read()
        return json.loads(raw)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            logger.debug("PubChem: compound not found at %s", url)
        else:
            logger.warning("PubChem HTTP %s for %s", exc.code, url)
        return None
    except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError) as exc:
        logger.warning("PubChem request failed: %s", exc)
        return None


def _parse_compound(data: dict[str, Any], name: str) -> PubChemCompound | None:
    """Parse PubChem PropertyTable JSON into PubChemCompound."""
    try:
        props_list = data["PropertyTable"]["Properties"]
    except (KeyError, IndexError, TypeError):
        return None

    if not props_list:
        return None

    p = props_list[0]
    try:
        # PubChem returns IsomericSMILES as "SMILES" and CanonicalSMILES as
        # "ConnectivitySMILES" in the PropertyTable JSON response.
        smiles_val = str(
            p.get("SMILES") or p.get("CanonicalSMILES") or p.get("ConnectivitySMILES") or ""
        )
        return PubChemCompound(
            cid=int(p["CID"]),
            name=name,
            smiles=smiles_val,
            molecular_weight=float(p.get("MolecularWeight", 0.0)),
            xlogp=float(p.get("XLogP", 0.0)),
            tpsa=float(p.get("TPSA", 0.0)),
            hbd=int(p.get("HBondDonorCount", 0)),
            hba=int(p.get("HBondAcceptorCount", 0)),
            rotatable_bonds=int(p.get("RotatableBondCount", 0)),
            complexity=float(p.get("Complexity", 0.0)),
            charge=int(p.get("Charge", 0)),
            exact_mass=float(p.get("ExactMass", 0.0)),
        )
    except (KeyError, ValueError, TypeError) as exc:
        logger.warning("PubChem parse error: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Public lookup functions
# ---------------------------------------------------------------------------


def lookup_by_name(name: str) -> PubChemCompound | None:
    """Look up a compound by common name.

    Args:
        name: Common or IUPAC name (e.g. "caffeine").

    Returns:
        PubChemCompound on success, None if not found or network error.
    """
    key = f"name:{name.lower()}"
    cached = _cache_get(key)
    if cached is not None:
        try:
            return PubChemCompound(**cached)
        except (TypeError, KeyError):
            pass

    encoded = urllib.parse.quote(name, safe="")
    url = f"{_BASE}/name/{encoded}/property/{_PROP_STR}/JSON"
    data = _fetch_json(url)
    if data is None:
        return None

    compound = _parse_compound(data, name)
    if compound is not None:
        _cache_set(
            key, {f.name: getattr(compound, f.name) for f in compound.__dataclass_fields__.values()}
        )  # type: ignore[attr-defined]
    return compound


def lookup_by_smiles(smiles: str) -> PubChemCompound | None:
    """Look up a compound by SMILES string.

    Args:
        smiles: Canonical or non-canonical SMILES.

    Returns:
        PubChemCompound on success, None if not found or network error.
    """
    key = f"smiles:{smiles}"
    cached = _cache_get(key)
    if cached is not None:
        try:
            return PubChemCompound(**cached)
        except (TypeError, KeyError):
            pass

    encoded = urllib.parse.quote(smiles, safe="")
    url = f"{_BASE}/smiles/{encoded}/property/{_PROP_STR}/JSON"
    data = _fetch_json(url)
    if data is None:
        return None

    compound = _parse_compound(data, smiles[:50])
    if compound is not None:
        _cache_set(
            key, {f.name: getattr(compound, f.name) for f in compound.__dataclass_fields__.values()}
        )  # type: ignore[attr-defined]
    return compound


def lookup_by_cid(cid: int) -> PubChemCompound | None:
    """Look up a compound by PubChem CID.

    Args:
        cid: Numeric PubChem Compound ID.

    Returns:
        PubChemCompound on success, None if not found or network error.
    """
    key = f"cid:{cid}"
    cached = _cache_get(key)
    if cached is not None:
        try:
            return PubChemCompound(**cached)
        except (TypeError, KeyError):
            pass

    url = f"{_BASE}/cid/{cid}/property/{_PROP_STR}/JSON"
    data = _fetch_json(url)
    if data is None:
        return None

    compound = _parse_compound(data, str(cid))
    if compound is not None:
        _cache_set(
            key, {f.name: getattr(compound, f.name) for f in compound.__dataclass_fields__.values()}
        )  # type: ignore[attr-defined]
    return compound


def batch_lookup(
    names: list[str],
    max_workers: int = 5,  # noqa: ARG001 — kept for API compat, lookup is sequential
) -> dict[str, PubChemCompound | None]:
    """Look up multiple compounds by name, sequentially with rate limiting.

    Args:
        names: List of compound names.
        max_workers: Ignored (sequential; kept for API compatibility).

    Returns:
        Dict mapping each name to PubChemCompound or None.
    """
    results: dict[str, PubChemCompound | None] = {}
    for i, name in enumerate(names):
        results[name] = lookup_by_name(name)
        if i < len(names) - 1:
            time.sleep(_RATE_SLEEP)
    return results


# ---------------------------------------------------------------------------
# ADME reference enrichment
# ---------------------------------------------------------------------------


def enrich_adme_reference(csv_path: str) -> int:
    """Count how many compounds in *csv_path* can be found in PubChem.

    Reads the first column (compound names) from the CSV and performs
    batch_lookup.  Does NOT write back to the CSV.

    Args:
        csv_path: Path to adme_reference.csv.

    Returns:
        Number of compounds successfully looked up from PubChem.
    """
    names: list[str] = []
    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                first_val = next(iter(row.values()), None)
                if first_val:
                    names.append(first_val.strip())
    except OSError as exc:
        logger.warning("enrich_adme_reference: cannot open %s: %s", csv_path, exc)
        return 0

    if not names:
        return 0

    results = batch_lookup(names)
    enriched = sum(1 for v in results.values() if v is not None)
    logger.info(
        "PubChem enrichment: %d/%d compounds found.",
        enriched,
        len(names),
    )
    return enriched
