"""P-gp and transporter substrate/inhibitor lookup from reference CSV.

Provides functions to check whether a drug is a P-gp substrate (or inhibitor)
based on curated transporter data in ``data/transporter_reference.csv``.
"""

from __future__ import annotations

import csv
import json
import logging
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

_DATA_PATH = Path(__file__).resolve().parents[5] / "data" / "transporter_reference.csv"
_REF_DB_PATH = Path(__file__).resolve().parents[5] / "data" / "clinical" / "reference_database.json"

_TRANSPORTER_COLUMNS = [
    "oatp1b1_substrate",
    "oatp1b1_inhibitor",
    "oatp1b3_substrate",
    "oatp1b3_inhibitor",
    "pgp_substrate",
    "pgp_inhibitor",
    "bcrp_substrate",
    "bcrp_inhibitor",
    "oct2_substrate",
    "oct2_inhibitor",
    "mrp2_substrate",
    "mrp2_inhibitor",
    "mate1_substrate",
    "mate1_inhibitor",
]


@lru_cache(maxsize=1)
def _load_transporter_data() -> dict[str, dict]:
    """Load transporter reference CSV into a name-keyed dictionary.

    Returns
    -------
    dict[str, dict]
        Mapping of ``drug_name.lower()`` to a dict containing MW, logP,
        charge_class, and all transporter flags (as int 0/1).
    """
    result: dict[str, dict] = {}
    if not _DATA_PATH.exists():
        logger.warning("Transporter reference CSV not found: %s", _DATA_PATH)
        return result

    with open(_DATA_PATH, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            name = row["name"].strip().lower()
            entry: dict = {
                "name": row["name"].strip(),
                "mw": float(row.get("mw", 0)),
                "logP": float(row.get("logP", 0)),
                "charge_class": row.get("charge_class", ""),
            }
            for col in _TRANSPORTER_COLUMNS:
                entry[col] = int(row.get(col, 0))
            result[name] = entry
    return result


def _canonicalize(smiles: str) -> str | None:
    """Canonicalize a SMILES string using RDKit."""
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol)
    except Exception:
        pass
    return None


@lru_cache(maxsize=1)
def _build_smiles_index() -> dict[str, str]:
    """Build a canonical SMILES → drug_name index from the clinical reference database.

    Uses RDKit canonicalization so different SMILES representations of the
    same molecule map to the same key.

    Returns
    -------
    dict[str, str]
        Mapping of canonical SMILES string to lowercase drug name.
    """
    index: dict[str, str] = {}
    if not _REF_DB_PATH.exists():
        logger.warning("Reference database not found: %s", _REF_DB_PATH)
        return index

    with open(_REF_DB_PATH) as fh:
        db = json.load(fh)

    drugs = db.get("drugs", {})
    for drug_name, info in drugs.items():
        smiles = info.get("smiles")
        if smiles:
            canonical = _canonicalize(smiles)
            if canonical:
                index[canonical] = drug_name.lower()
            # Also store raw SMILES as fallback
            index[smiles] = drug_name.lower()

    # Also index from transporter_reference.csv — some drugs there aren't in ref DB
    # We don't have SMILES in transporter CSV, but we have drug names that match
    return index


def _resolve_drug_name(drug_name: str | None = None, smiles: str | None = None) -> str | None:
    """Resolve a drug name from either an explicit name or a SMILES lookup."""
    if drug_name is not None:
        return drug_name.strip().lower()
    if smiles is not None:
        smiles_index = _build_smiles_index()
        # Try raw SMILES first, then canonical
        name = smiles_index.get(smiles)
        if name is None:
            canonical = _canonicalize(smiles)
            if canonical:
                name = smiles_index.get(canonical)
        return name
    return None


def get_transporter_flags(drug_name: str | None = None, smiles: str | None = None) -> dict | None:
    """Look up all transporter flags for a drug.

    Parameters
    ----------
    drug_name : str, optional
        Drug name (case-insensitive).
    smiles : str, optional
        SMILES string; resolved to a drug name via the reference database.

    Returns
    -------
    dict or None
        Dictionary of transporter flags if the drug is found, else ``None``.
    """
    name = _resolve_drug_name(drug_name=drug_name, smiles=smiles)
    if name is None:
        return None
    data = _load_transporter_data()
    return data.get(name)


def is_pgp_substrate(smiles: str | None = None, drug_name: str | None = None) -> bool:
    """Check whether a drug is a P-gp substrate.

    Parameters
    ----------
    smiles : str, optional
        SMILES string for the compound.
    drug_name : str, optional
        Drug name (case-insensitive).

    Returns
    -------
    bool
        ``True`` if the drug is a known P-gp substrate, ``False`` otherwise
        (including when the drug is not found in the reference data).
    """
    flags = get_transporter_flags(drug_name=drug_name, smiles=smiles)
    if flags is None:
        return False
    return bool(flags.get("pgp_substrate", 0))
