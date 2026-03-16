"""Applicability domain filter for Omega PBPK predictions.

Detects drugs that the PBPK model is unlikely to handle well — prodrugs,
pH-unstable compounds, high-MW biologics, P-gp substrates — and returns
a structured result with tractability, confidence, and flags.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SMARTS patterns for prodrug motifs
# ---------------------------------------------------------------------------

_PRODRUG_PATTERNS: dict[str, str] = {
    # Ester linkage that is NOT a carboxylic acid (R-C(=O)-OH).
    # The first C must not be part of C(=O)[OH]; the O-linked C must not be C=O.
    "prodrug_ester": "[C;!$(C(=O)[OH])](=O)[O][C;!$(C=O)]",
    # Phosphonate ester — two O-C branches off P=O
    "prodrug_phosphonate": "[P](=O)([O][C])([O][C])",
    # Carbamate linkage — O-C(=O)-N
    "prodrug_carbamate": "[O][C](=O)[N]",
}

# ---------------------------------------------------------------------------
# Manually curated untractable drugs
# ---------------------------------------------------------------------------

_UNTRACTABLE_NAMES: set[str] = {
    "sonidegib",
    "temozolomide",
    "sodium oxybate",
    "lanthanum carbonate",
}

# ---------------------------------------------------------------------------
# Hard flags → tractable=False; soft flags → tractable=True but lower confidence
# ---------------------------------------------------------------------------

_HARD_FLAGS: frozenset[str] = frozenset(
    {
        "prodrug_ester",
        "prodrug_phosphonate",
        "prodrug_carbamate",
        "known_untractable",
        "invalid_smiles",
    }
)

_SOFT_FLAGS: frozenset[str] = frozenset(
    {
        "pgp_substrate",
        "high_mw",
    }
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ApplicabilityResult:
    """Result of an applicability-domain check."""

    tractable: bool
    confidence: str  # "high", "medium", or "low"
    flags: tuple[str, ...]
    details: str


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_applicability(
    smiles: str,
    drug_name: str | None = None,
) -> ApplicabilityResult:
    """Check whether a compound falls within the PBPK model's applicability domain.

    Parameters
    ----------
    smiles : str
        SMILES string for the compound.
    drug_name : str, optional
        Drug name (case-insensitive). Used for the untractable-name check.

    Returns
    -------
    ApplicabilityResult
        Frozen dataclass with ``tractable``, ``confidence``, ``flags``, and
        ``details``.
    """
    flags: list[str] = []

    # -- Known untractable by name ----------------------------------------
    if drug_name is not None and drug_name.strip().lower() in _UNTRACTABLE_NAMES:
        flags.append("known_untractable")

    # -- Parse SMILES with RDKit ------------------------------------------
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
    except ImportError:  # pragma: no cover
        logger.warning("RDKit not available — skipping structural checks")
        return ApplicabilityResult(
            tractable=True,
            confidence="medium",
            flags=("rdkit_unavailable",),
            details="RDKit not installed; structural checks skipped.",
        )

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        flags.append("invalid_smiles")
        return ApplicabilityResult(
            tractable=False,
            confidence="low",
            flags=tuple(flags),
            details=f"Could not parse SMILES: {smiles}",
        )

    # -- SMARTS prodrug checks --------------------------------------------
    for flag_name, smarts in _PRODRUG_PATTERNS.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is not None and mol.HasSubstructMatch(pattern):
            flags.append(flag_name)

    # -- Molecular weight check -------------------------------------------
    mw = Descriptors.MolWt(mol)
    if mw > 900:
        flags.append("high_mw")

    # -- P-gp substrate check ---------------------------------------------
    try:
        from omega_pbpk.ml.models.adme.transporter_lookup import is_pgp_substrate

        if is_pgp_substrate(smiles=smiles, drug_name=drug_name):
            flags.append("pgp_substrate")
    except Exception:
        logger.debug("P-gp lookup failed — skipping", exc_info=True)

    # -- Determine tractability and confidence ----------------------------
    flag_tuple = tuple(flags)
    has_hard = any(f in _HARD_FLAGS for f in flags)
    has_soft = any(f in _SOFT_FLAGS for f in flags)

    if has_hard:
        tractable = False
        confidence = "low"
    elif has_soft:
        tractable = True
        confidence = "medium"
    else:
        tractable = True
        confidence = "high"

    # -- Build details string ---------------------------------------------
    if not flags:
        details = "Compound is within the applicability domain."
    else:
        details = "Flags: " + ", ".join(flags) + "."

    return ApplicabilityResult(
        tractable=tractable,
        confidence=confidence,
        flags=flag_tuple,
        details=details,
    )
